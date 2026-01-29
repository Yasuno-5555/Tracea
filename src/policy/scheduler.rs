use crate::policy::types::*;
use crate::policy::engine::{PolicyEngine};
use crate::core::cost::CostModel;
use std::collections::{HashMap, HashSet};

pub struct StandardScheduler;

impl StandardScheduler {
    pub fn schedule<E: PolicyEngine>(engine: &mut E, ctx: &GraphContext) -> PolicyDecision {
        let mut tile_policies = Vec::new();
        let mut exec_policies = Vec::new();

        // 0. Canonicalize Graph (Phase D-3)
        // We work on a local copy to ensure shapes are valid before CostModel sees them.
        let mut graph = ctx.graph.clone();
        crate::policy::transform::canonicalize_graph(&mut graph);
        
        // 1. Generate Base Policies
        for op in &graph.operators {
             let sub_context = PolicyContext {
                 device: ctx.device,
                 model: &ModelTopology { layer_count: 0 },
                 operators: std::slice::from_ref(op),
                 history: &ExecutionHistory { last_latency_us: None },
             };
             
             let decision = engine.propose(&sub_context);
             tile_policies.extend(decision.tile_policies);
             exec_policies.extend(decision.exec_policies);
        }

        // 2. Apply Fusion Strategy
        let mut op_map = HashMap::new();
        for (i, policy) in exec_policies.iter().enumerate() {
            op_map.insert(policy.operator_id, i);
        }

        for (producer_id, consumer_id) in &ctx.graph.dependencies {
            if let (Some(&p_idx), Some(&c_idx)) = (op_map.get(producer_id), op_map.get(consumer_id)) {
                exec_policies[p_idx].kernel_binding.fuse_with.push(*consumer_id);
            }
        }

        // 3. Memory Allocation (Liveness Analysis)
        let allocations = Self::analyze_liveness(&graph, ctx.device);
        for policy in &mut exec_policies {
            if let Some(&offset) = allocations.get(&policy.operator_id) {
                policy.memory_alias_hint.output_offset = Some(offset);
            }
        }

        PolicyDecision {
            tile_policies,
            exec_policies,
            global_hints: GlobalPolicyHints { 
                prefer_fusion: true,
                debug_flags: 0 
            },
        }
    }

    fn analyze_liveness(graph: &GraphTopology, device: &DeviceProfile) -> HashMap<u64, usize> {
        let mut allocations = HashMap::new();
        
        // Step A: Build Dependency Graph & Lifetime Map
        // last_use: tensor_id (op_id) -> last consumer index
        let mut op_index_map = HashMap::new();
        for (i, op) in graph.operators.iter().enumerate() {
            op_index_map.insert(op.op_id(), i);
        }

        let mut last_use = HashMap::new();
        // Initialize last_use with producer's own index (at least used by itself)
        for (i, op) in graph.operators.iter().enumerate() {
            last_use.insert(op.op_id(), i);
        }

        for (producer_id, consumer_id) in &graph.dependencies {
             if let Some(&c_idx) = op_index_map.get(consumer_id) {
                 let current_max = *last_use.get(producer_id).unwrap_or(&0);
                 if c_idx > current_max {
                     last_use.insert(*producer_id, c_idx);
                 }
             }
        }

        // Step B: Simulate Allocation (Linear Scan)
        // active_buffers: Vec<(end_time, start_offset, size)>
        let mut active_buffers: Vec<(usize, usize, usize)> = Vec::new();
        let mut free_blocks: Vec<(usize, usize)> = Vec::new(); // (start, size)
        // Naive bump allocator tracking max usage, but we want reuse.
        // Let's use a simple strategy: Find first fit in free_blocks, else bump.
        let mut pool_watermark = 0;

        for (i, op) in graph.operators.iter().enumerate() {
            // 1. Release expired buffers
            // If buffer.end_time < i, it's free.
            active_buffers.retain(|&(end_time, offset, size)| {
                if end_time < i {
                    // Reclaim block
                    free_blocks.push((offset, size));
                    false // Remove from active
                } else {
                    true // Keep
                }
            });

            // Merge adjacent free blocks (Optional optimization, skip for Phase D-1 simplicity)
            
            // 2. Allocate output for current op
            // Estimate size using CostModel helper
            let size = CostModel::estimate_memory_bytes(op);
            
            if size > 0 {
                // Try allocate
                let mut chosen_offset = pool_watermark;
                let mut best_fit_idx = None;

                // Simple First Fit
                for (idx, &(offset, block_size)) in free_blocks.iter().enumerate() {
                    if block_size >= size {
                        chosen_offset = offset;
                        best_fit_idx = Some(idx);
                        break;
                    }
                }

                if let Some(idx) = best_fit_idx {
                    // Update free block
                    let (offset, block_size) = free_blocks[idx];
                    let remaining = block_size - size;
                     free_blocks.remove(idx);
                    if remaining > 0 {
                        free_blocks.push((offset + size, remaining));
                    }
                } else {
                    // Bump pointer
                    pool_watermark += size;
                }
                
                let death_time = *last_use.get(&op.op_id()).unwrap_or(&i);
                active_buffers.push((death_time, chosen_offset, size));
                allocations.insert(op.op_id(), chosen_offset);
            }
        }
        
        allocations
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::policy::standard::StandardPolicyEngine;

    #[test]
    fn test_scheduler_fusion() {
        let mut engine = StandardPolicyEngine::new();
        let device = DeviceProfile::default();
        
        let conv_op = OperatorTopology::Conv2d {
            op_id: 1,
            name: "Conv1".to_string(),
            n: 1, c: 32, h: 64, w: 64, k: 32,
            r: 3, s: 3, stride: 1, padding: 1,
            epilogue: vec![],
        };
        
        let relu_op = OperatorTopology::Relu {
            op_id: 2,
            name: "Relu1".to_string(),
            n: 1 * 32 * 64 * 64,
        };

        
        let graph = GraphTopology {
            operators: vec![conv_op, relu_op],
            dependencies: vec![(1, 2)],
        };
        
        let ctx = GraphContext {
            device: &device,
            graph: &graph,
        };
        
        let decision = StandardScheduler::schedule(&mut engine, &ctx);
        
        // Find Conv policy (op_id 1)
        let conv_policy = decision.exec_policies.iter().find(|p| p.operator_id == 1).unwrap();
        
        // Assert it fused with Relu (op_id 2)
        assert!(conv_policy.kernel_binding.fuse_with.contains(&2));
    }

    #[test]
    fn test_memory_reuse() {
        let mut engine = StandardPolicyEngine::new();
        let device = DeviceProfile::default();
        
        // Chain 1: Op 1 -> Op 2
        // Tensor 1 (Op 1 out) used by Op 2.
        let op1 = OperatorTopology::Gemm {
            op_id: 1, name: "gemm1".into(), m: 102, n: 102, k: 102, batch: 1, kind: TopologyKind::Dense, epilogue: vec![]
        };
        let op2 = OperatorTopology::Relu { op_id: 2, name: "relu1".into(), n: 102 * 102 };

        // Chain 2: Op 3 -> Op 4 (Disjoint from Chain 1 time-wise if scheduled seq)
        let op3 = OperatorTopology::Gemm {
            op_id: 3, name: "gemm2".into(), m: 102, n: 102, k: 102, batch: 1, kind: TopologyKind::Dense, epilogue: vec![]
        };
        let op4 = OperatorTopology::Relu { op_id: 4, name: "relu2".into(), n: 102 * 102 };


        let graph = GraphTopology {
            operators: vec![op1, op2, op3, op4],
            dependencies: vec![(1, 2), (3, 4)],
        };

        let ctx = GraphContext { device: &device, graph: &graph };
        let decision = StandardScheduler::schedule(&mut engine, &ctx);

        // Check Allocations
        // Op 1 Output Offset
        let p1 = decision.exec_policies.iter().find(|p| p.operator_id == 1).unwrap();
        let off1 = p1.memory_alias_hint.output_offset.unwrap();

        // Op 3 Output Offset
        let p3 = decision.exec_policies.iter().find(|p| p.operator_id == 3).unwrap();
        let off3 = p3.memory_alias_hint.output_offset.unwrap();

        // Since Op 3 runs after Op 1's consumer (Op 2) is done (or at least Op 1 is done),
        // Ideally Op 3 reuses Op 1's buffer if Op 1 is dead.
        // Wait, Op 1 is used by Op 2.
        // Op 1 dies at index 1 (Op 2).
        // Op 3 is index 2.
        // So Op 1 is dead when Op 3 starts.
        // So off3 should equal off1 (assuming same size, efficient allocator).
        
        assert_eq!(off1, off3, "Op 3 should reuse Op 1's memory");
    }
}
