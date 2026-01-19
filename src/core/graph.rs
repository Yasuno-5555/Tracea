use serde::{Serialize, Deserialize};
use crate::core::op::GemmOp;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Operation {
    Gemm(GemmOp),
    FusedGemm(crate::core::op::FusedGemmOp),
    Softmax(crate::core::op::SoftmaxOp),
    FusedAttention(crate::core::op::FusedAttentionOp),
    NN(crate::core::op::NNOp),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Node {
    pub id: usize,
    pub op: Operation,
    pub dependencies: Vec<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Graph {
    pub nodes: Vec<Node>,
}

pub trait ScheduleStrategy {
    fn schedule(&self, graph: &Graph) -> Vec<usize>;
}

pub struct PrioritySchedule;

impl ScheduleStrategy for PrioritySchedule {
    fn schedule(&self, graph: &Graph) -> Vec<usize> {
        let mut node_ids: Vec<usize> = (0..graph.nodes.len()).collect();
        // Sort by complexity (M*N*K) descending
        node_ids.sort_by_key(|&id| {
            let node = &graph.nodes[id];
            match &node.op {
                Operation::Gemm(gemm) => {
                    let m = gemm.m.0 as u64;
                    let n = gemm.n.0 as u64;
                    let k = gemm.k.0 as u64;
                    std::cmp::Reverse(m * n * k)
                },
                Operation::FusedGemm(fused) => {
                    let m = fused.base.m.0 as u64;
                    let n = fused.base.n.0 as u64;
                    let k = fused.base.k.0 as u64;
                    std::cmp::Reverse(m * n * k)
                },
                Operation::Softmax(_) => {
                    // Softmax is relatively light but critical path
                    std::cmp::Reverse(1000)
                },
                Operation::FusedAttention(attn) => {
                    // Complexity: Batch * SeqLen^2 * HeadDim * 2 + Batch * SeqLen^2 (softmax)
                    let complexity = (attn.b as u64) * (attn.s as u64) * (attn.s as u64) * (attn.dh as u64) * 2;
                    std::cmp::Reverse(complexity)
                },
                Operation::NN(nn_op) => {
                    match nn_op {
                        crate::core::op::NNOp::Linear(lin) => {
                            // Rough estimate: In * Out * Batch(assume 128)
                            let m = 128; 
                            let n = lin.out_features.0 as u64;
                            let k = lin.in_features.0 as u64;
                            std::cmp::Reverse(m * n * k)
                        },
                        crate::core::op::NNOp::Attention(attn) => {
                             // Heavy
                             std::cmp::Reverse(u64::MAX)
                        }
                    }
                }
            }
        });
        node_ids
    }
}

impl Graph {
    pub fn new() -> Self {
        Self { nodes: Vec::new() }
    }

    pub fn add_gemm(&mut self, m: u32, n: u32, k: u32, deps: Vec<usize>) -> usize {
        let id = self.nodes.len();
        self.nodes.push(Node {
            id,
            op: Operation::Gemm(GemmOp::new(m, n, k)),
            dependencies: deps,
        });
        id
    }

    pub fn add_fused_gemm(&mut self, m: u32, n: u32, k: u32, epilogue: Vec<crate::core::op::EpilogueOp>, deps: Vec<usize>) -> usize {
        let id = self.nodes.len();
        let base = GemmOp::new(m, n, k);
        self.nodes.push(Node {
            id,
            op: Operation::FusedGemm(crate::core::op::FusedGemmOp { base, epilogue }),
            dependencies: deps,
        });
        id
    }

    pub fn add_linear(&mut self, in_features: u32, out_features: u32, bias: bool, activation: Vec<crate::core::op::EpilogueOp>, deps: Vec<usize>) -> usize {
        let id = self.nodes.len();
        let op = crate::core::op::LinearOp {
             in_features: crate::core::op::Dim(in_features),
             out_features: crate::core::op::Dim(out_features),
             bias,
             activation,
        };
        self.nodes.push(Node {
            id,
            op: Operation::NN(crate::core::op::NNOp::Linear(op)),
            dependencies: deps,
        });
        id
    }

    pub fn add_softmax(&mut self, axis: i32, deps: Vec<usize>) -> usize {
        let id = self.nodes.len();
        self.nodes.push(Node {
            id,
            op: Operation::Softmax(crate::core::op::SoftmaxOp { axis }),
            dependencies: deps,
        });
        id
    }

    pub fn add_attention(&mut self, embed_dim: u32, num_heads: u32, head_dim: u32, causal: bool, deps: Vec<usize>) -> usize {
        let id = self.nodes.len();
        let op = crate::core::op::AttentionOp {
            embed_dim: crate::core::op::Dim(embed_dim),
            num_heads: crate::core::op::Dim(num_heads),
            head_dim: crate::core::op::Dim(head_dim),
            causal,
        };
        self.nodes.push(Node {
            id,
            op: Operation::NN(crate::core::op::NNOp::Attention(op)),
            dependencies: deps,
        });
        id
    }

    pub fn add_fused_attention(&mut self, b: u32, s: u32, d: u32, h: u32, dh: u32, causal: bool, deps: Vec<usize>) -> usize {
        let id = self.nodes.len();
        self.nodes.push(Node {
            id,
            op: Operation::FusedAttention(crate::core::op::FusedAttentionOp { b, s, d, h, dh, causal, scale_inv_sqrt_d: true }),
            dependencies: deps,
        });
        id
    }

    /// Detects and fuses patterns in the graph (e.g., FlashAttention).
    pub fn optimize_fusion(&self) -> Self {
        let mut optimized = Self::new();
        let mut id_map = std::collections::HashMap::new();
        let mut fused_source_nodes = std::collections::HashSet::new();
        let mut fusion_mapping = std::collections::HashMap::new(); // Original Out-Gemm ID -> Fused Params

        // 1. First Pass: Identify all fusion patterns
        for node in &self.nodes {
            if let Operation::Gemm(ref _out_gemm) = node.op {
                if node.dependencies.len() == 2 {
                    let maybe_sm_id = node.dependencies[0];
                    let maybe_v_id = node.dependencies[1];
                    
                    if let Some(sm_node) = self.nodes.get(maybe_sm_id) {
                        if let Operation::Softmax(_) = sm_node.op {
                            let maybe_qk_id = sm_node.dependencies[0];
                            if let Some(qk_node) = self.nodes.get(maybe_qk_id) {
                                if let Operation::Gemm(ref qk_gemm) = qk_node.op {
                                    // Found Attention Pattern!
                                    let s = qk_gemm.n.0;
                                    let dh = qk_gemm.k.0;
                                    let b_h_s = qk_gemm.m.0;
                                    
                                    // Robustness Fix: Instead of assuming h=8, we flatten B and H.
                                    // Treat effective batch size as B_eff = B * H.
                                    // Set h=1 for the fused operator.
                                    let b_eff = b_h_s / s;
                                    let h_eff = 1;
                                    
                                    let causal = true; // Heuristic for now

                                    let q_id = qk_node.dependencies[0];
                                    let k_id = qk_node.dependencies[1];
                                    let v_id = maybe_v_id;

                                    fusion_mapping.insert(node.id, (b_eff, s, 512, h_eff, dh, causal, vec![q_id, k_id, v_id]));
                                    
                                    // Mark nodes to be swallowed by fusion
                                    fused_source_nodes.insert(node.id);
                                    fused_source_nodes.insert(maybe_sm_id);
                                    fused_source_nodes.insert(maybe_qk_id);
                                }
                            }
                        }
                    }
                }
            }
        }

        // 2. Second Pass: Construct the optimized graph
        for node in &self.nodes {
            if fusion_mapping.contains_key(&node.id) {
                let (b, s, d, h, dh, causal, orig_deps) = fusion_mapping.get(&node.id).unwrap();
                let deps = orig_deps.iter().map(|d| *id_map.get(d).unwrap_or(d)).collect();
                let new_id = optimized.add_fused_attention(*b, *s, *d, *h, *dh, *causal, deps);
                id_map.insert(node.id, new_id);
                continue;
            }

            if fused_source_nodes.contains(&node.id) {
                // This node is part of a fusion and is NOT the trigger node (Out-Gemm), so skip it
                continue;
            }

            // Normal node copy
            let deps: Vec<usize> = node.dependencies.iter()
                .map(|d| *id_map.get(d).unwrap_or(d))
                .collect();

            let new_id = match &node.op {
                Operation::Gemm(gemm) => optimized.add_gemm(gemm.m.0, gemm.n.0, gemm.k.0, deps),
                Operation::FusedGemm(fused) => optimized.add_fused_gemm(fused.base.m.0, fused.base.n.0, fused.base.k.0, fused.epilogue.clone(), deps),
                Operation::Softmax(s) => optimized.add_softmax(s.axis, deps),
                Operation::FusedAttention(f) => {
                    let mut id = optimized.add_fused_attention(f.b, f.s, f.d, f.h, f.dh, f.causal, deps);
                    // Hack: add_fused_attention sets scale to true by default, but we should copy if we had it
                    // Since add_fused_attention doesn't take scale arg yet, let's just leave it as true for now
                    // Or explicit update:
                    if let Operation::FusedAttention(ref mut op) = optimized.nodes[id].op {
                        op.scale_inv_sqrt_d = f.scale_inv_sqrt_d;
                    }
                    id
                },
                Operation::NN(nn) => {
                     match nn {
                        crate::core::op::NNOp::Linear(lin) => optimized.add_linear(lin.in_features.0, lin.out_features.0, lin.bias, lin.activation.clone(), deps),
                        crate::core::op::NNOp::Attention(attn) => optimized.add_attention(attn.embed_dim.0, attn.num_heads.0, attn.head_dim.0, attn.causal, deps),
                     }
                }
            };
            id_map.insert(node.id, new_id);
        }
        optimized
    }

    /// Lowers High-Level NN Ops into primitive Gemm/FusedGemm ops.
    pub fn lower(&self) -> Self {
        let mut lowered = Self::new();
        let mut id_map = std::collections::HashMap::new();

        for node in &self.nodes {
            let deps: Vec<usize> = node.dependencies.iter()
                .map(|d| *id_map.get(d).expect("Dependency should be lowered first"))
                .collect();

            let new_id = match &node.op {
                Operation::Gemm(gemm) => {
                    lowered.add_gemm(gemm.m.0, gemm.n.0, gemm.k.0, deps)
                },
                Operation::FusedGemm(fused) => {
                    lowered.add_fused_gemm(fused.base.m.0, fused.base.n.0, fused.base.k.0, fused.epilogue.clone(), deps)
                },
                Operation::Softmax(s) => {
                    lowered.add_softmax(s.axis, deps)
                },
                Operation::FusedAttention(f) => {
                    lowered.add_fused_attention(f.b, f.s, f.d, f.h, f.dh, f.causal, deps)
                },
                Operation::NN(nn_op) => {
                    match nn_op {
                        crate::core::op::NNOp::Linear(lin) => {
                            // Expand Linear to FusedGemm
                            // Batch size is currently not tracked in LinearOp, assuming 128 for optimization
                            let batch_size = 128; 
                            lowered.add_fused_gemm(
                                batch_size, 
                                lin.out_features.0, 
                                lin.in_features.0, 
                                lin.activation.clone(), 
                                deps
                            )
                        },
                        crate::core::op::NNOp::Attention(attn) => {
                            // Decompose Attention (MVP: Single Batch/Head for demo logic)
                            let b = 1; // Batch
                            let s = 128; // SeqLen
                            let d = attn.embed_dim.0;
                            let h = attn.num_heads.0;
                            let dh = attn.head_dim.0;

                            // 1. Q, K, V Projections (FusedGemm with no activation)
                            let id_q = lowered.add_fused_gemm(b * s, d, d, vec![], deps.clone());
                            let id_k = lowered.add_fused_gemm(b * s, d, d, vec![], deps.clone());
                            let id_v = lowered.add_fused_gemm(b * s, d, d, vec![], deps);

                            // 2. Q @ K^T (Scaled Dot Product)
                            // Tracea doesn't have transpose yet, assuming K is pre-transposed or we handle it in emitter later.
                            // For IR lowering, we emit Gemm(Q, K, dh)
                            let id_qk = lowered.add_gemm(b * h * s, s, dh, vec![id_q, id_k]);

                            // 3. Softmax
                            let id_sm = lowered.add_softmax(-1, vec![id_qk]);

                            // 4. (Softmax @ V)
                            let id_out = lowered.add_gemm(b * h * s, dh, s, vec![id_sm, id_v]);
                            
                            id_out
                        }
                    }
                }
            };
            id_map.insert(node.id, new_id);
        }
        lowered
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::op::EpilogueOp;

    #[test]
    fn test_linear_lowering() {
        let mut graph = Graph::new();
        let id_in = graph.add_gemm(128, 128, 128, vec![]);
        let id_lin = graph.add_linear(128, 64, true, vec![EpilogueOp::ReLU], vec![id_in]);
        
        let lowered = graph.lower();
        assert_eq!(lowered.nodes.len(), 2);
        
        let node0 = &lowered.nodes[0];
        match &node0.op {
            Operation::Gemm(gemm) => {
                assert_eq!(gemm.m.0, 128);
            },
            _ => panic!("Node 0 should be Gemm"),
        }

        let node1 = &lowered.nodes[1];
        match node1.op {
            Operation::FusedGemm(ref fused) => {
                assert_eq!(fused.base.n.0, 64);
                assert_eq!(fused.base.k.0, 128);
                assert_eq!(fused.epilogue.len(), 1);
            },
            _ => panic!("Node 1 should be FusedGemm"),
        }
    }

    #[test]
    fn test_attention_lowering() {
        let mut graph = Graph::new();
        let id_in = graph.add_gemm(128, 128, 128, vec![]);
        // 512 dim, 8 heads -> 64 head_dim
        let _id_attn = graph.add_attention(512, 8, 64, true, vec![id_in]);
        
        let lowered = graph.lower();
        // Expect: 1 (input gemm) + 3 (projections) + 1 (QK GEMM) + 1 (Softmax) + 1 (Out GEMM) = 7 nodes
        assert_eq!(lowered.nodes.len(), 7);
        
        // Check types
        assert!(matches!(lowered.nodes[1].op, Operation::FusedGemm(_))); // Q
        assert!(matches!(lowered.nodes[4].op, Operation::Gemm(_)));      // QK
        assert!(matches!(lowered.nodes[5].op, Operation::Softmax(_)));   // Softmax
        assert!(matches!(lowered.nodes[6].op, Operation::Gemm(_)));      // Out
    }

    #[test]
    fn test_attention_fusion() {
        let mut graph = Graph::new();
        let id_in = graph.add_gemm(128, 512, 512, vec![]);
        let _id_attn = graph.add_attention(512, 8, 64, true, vec![id_in]);
        
        let lowered = graph.lower();
        assert_eq!(lowered.nodes.len(), 7); // Input + 3proj + QK + Sm + Out
        
        let fused = lowered.optimize_fusion();
        // Expect: Input + 3 projections + 1 FusedAttention = 5 nodes
        // (QK, Softmax, Out are collapsed into FusedAttention)
        assert_eq!(fused.nodes.len(), 5);
        
        let last_node = &fused.nodes[4];
        match last_node.op {
            Operation::FusedAttention(ref attn) => {
                assert_eq!(attn.s, 128);
                assert_eq!(attn.h, 8);
                assert_eq!(attn.dh, 64);
            },
            _ => panic!("Last node should be FusedAttention"),
        }
    }
}
