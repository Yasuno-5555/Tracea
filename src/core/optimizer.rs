use crate::policy::types::{GraphTopology, OperatorTopology};
use crate::core::op::EpilogueOp;
use std::collections::{HashMap, HashSet};

pub struct GraphOptimizer;

impl GraphOptimizer {
    pub fn optimize(graph: &mut GraphTopology) {
        Self::apply_fusion(graph);
    }

    fn apply_fusion(graph: &mut GraphTopology) {
        // 1. Build Adjacency List and Operator Map
        let mut adj = HashMap::new();
        let mut reverse_adj = HashMap::new();
        for &(src, dst) in &graph.dependencies {
            adj.entry(src).or_insert_with(Vec::new).push(dst);
            reverse_adj.entry(dst).or_insert_with(Vec::new).push(src);
        }

        let mut op_map: HashMap<u64, &OperatorTopology> = graph.operators.iter().map(|op| (op.op_id(), op)).collect();
        
        // 2. Identification Phase: Find Fusion Candidates
        // We look for patterns: 
        //   Conv2d -> BN -> ReLU
        //   Conv2d -> Add -> ReLU
        // Rules for Fusion:
        //   - Consumer must have ONLY ONE producer (the one we fuse with)
        //   - Producer must have ONLY ONE consumer (the one being fused) -> Except for certain cases like Add residual
        
        let mut ops_to_remove = HashSet::new();
        let mut new_ops = Vec::new();
        let mut fused_id_map: HashMap<u64, Vec<EpilogueOp>> = HashMap::new();

        // We iterate through operators and try to sink consumers into producer's epilogues
        // For simplicity, we process in a few passes or a single greedy greedy pass
        
        let mut current_ops = graph.operators.clone();
        let mut modified = true;
        
        while modified {
            modified = false;
            let mut next_ops = Vec::new();
            let mut i = 0;
            
            while i < current_ops.len() {
                let op = &current_ops[i];
                let op_id = op.op_id();
                
                if ops_to_remove.contains(&op_id) {
                    i += 1;
                    continue;
                }

                match op {
                    OperatorTopology::Conv2d { .. } | OperatorTopology::Linear { .. } | OperatorTopology::Gemm { .. } => {
                        let consumers = adj.get(&op_id).cloned().unwrap_or_default();
                        
                        if consumers.len() == 1 {
                            let consumer_id = consumers[0];
                            let consumer_op = current_ops.iter().find(|o| o.op_id() == consumer_id);
                            
                            if let Some(c_op) = consumer_op {
                                // Check if we can fuse this consumer
                                if let Some(epilogue_op) = Self::try_map_to_epilogue(c_op, &reverse_adj, op_id) {
                                    // Also check if consumer has other inputs (only Add allows 2 inputs)
                                    let producers = reverse_adj.get(&consumer_id).cloned().unwrap_or_default();
                                    
                                    let can_fuse = match epilogue_op {
                                        EpilogueOp::BatchNorm { .. } | EpilogueOp::ReLU => producers.len() == 1,
                                        EpilogueOp::ResidualAdd { .. } => producers.len() == 2,
                                        _ => false,
                                    };

                                    if can_fuse {
                                        // Merge into producer
                                        let mut fused_op = op.clone();
                                        Self::push_epilogue(&mut fused_op, epilogue_op);
                                        
                                        // Update graph connectivity for subsequent passes
                                        // The new fused_op now "becomes" the consumer in terms of output
                                        // But it's easier to keep op_id of producer and just skip consumer
                                        next_ops.push(fused_op);
                                        ops_to_remove.insert(consumer_id);
                                        
                                        // Update adj: producer's new consumers are consumer's consumers
                                        let consumers_of_consumer = adj.get(&consumer_id).cloned().unwrap_or_default();
                                        adj.insert(op_id, consumers_of_consumer);
                                        
                                        modified = true;
                                        i += 1;
                                        continue;
                                    }
                                }
                            }
                        }
                    }
                    _ => {}
                }
                
                next_ops.push(op.clone());
                i += 1;
            }
            current_ops = next_ops;
        }

        // 3. Finalize: Update operators and dependencies
        graph.operators = current_ops;
        
        // Remove dependencies that are now internal to fused kernels
        graph.dependencies.retain(|(src, dst)| {
            !ops_to_remove.contains(dst)
        });
        
        // Re-bridge dependencies: if src -> fused_dst -> final_dst, it becomes src -> final_dst
        // (This is already handled by our adj update logic above if we were careful)
        // Let's rebuild dependencies from the modified adj list
        let mut final_deps = Vec::new();
        for (&src, dsts) in &adj {
            if !ops_to_remove.contains(&src) {
                for &dst in dsts {
                    final_deps.push((src, dst));
                }
            }
        }
        graph.dependencies = final_deps;
    }

    fn try_map_to_epilogue(op: &OperatorTopology, reverse_adj: &HashMap<u64, Vec<u64>>, producer_id: u64) -> Option<EpilogueOp> {
        match op {
            OperatorTopology::Relu { .. } => Some(EpilogueOp::ReLU),
            OperatorTopology::BatchNorm { op_id, epsilon, .. } => {
                Some(EpilogueOp::BatchNorm { 
                    op_id: *op_id, 
                    epsilon: *epsilon,
                    gamma_id: 0, beta_id: 0, mean_id: 0, var_id: 0, // Resolved by RuntimeManager via op_id
                })
            }
            OperatorTopology::Elementwise { op_id, kind, .. } if kind == "Add" => {
                let producers = reverse_adj.get(op_id)?;
                let skip_id = producers.iter().find(|&&id| id != producer_id)?;
                Some(EpilogueOp::ResidualAdd { residual_ptr: *skip_id as usize }) 
            }
            _ => None,
        }
    }

    fn push_epilogue(op: &mut OperatorTopology, epilogue: EpilogueOp) {
        match op {
            OperatorTopology::Conv2d { epilogue: e, .. } => e.push(epilogue),
            OperatorTopology::Linear { epilogue: e, .. } => e.push(epilogue),
            OperatorTopology::Gemm { epilogue: e, .. } => e.push(epilogue),
            _ => {}
        }
    }
}
