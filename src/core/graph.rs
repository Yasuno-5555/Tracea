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
                    let m = gemm.m.as_static().unwrap_or(1024) as u64;
                    let n = gemm.n.as_static().unwrap_or(1024) as u64;
                    let k = gemm.k.as_static().unwrap_or(1024) as u64;
                    std::cmp::Reverse(m * n * k)
                },
                Operation::FusedGemm(fused) => {
                    let m = fused.base.m.as_static().unwrap_or(1024) as u64;
                    let n = fused.base.n.as_static().unwrap_or(1024) as u64;
                    let k = fused.base.k.as_static().unwrap_or(1024) as u64;
                    std::cmp::Reverse(m * n * k)
                },
                Operation::Softmax(_) => {
                    std::cmp::Reverse(1000)
                },
                Operation::FusedAttention(attn) => {
                    let b = attn.b.as_static().unwrap_or(1) as u64;
                    let s = attn.s.as_static().unwrap_or(1024) as u64;
                    let dh = attn.dh.as_static().unwrap_or(64) as u64;
                    let complexity = b * s * s * dh * 2;
                    std::cmp::Reverse(complexity)
                },
                Operation::NN(nn_op) => {
                    match nn_op {
                        crate::core::op::NNOp::Linear(lin) => {
                            let m = lin.batch_size.as_static().unwrap_or(128) as u64; 
                            let n = lin.out_features.as_static().unwrap_or(1024) as u64;
                            let k = lin.in_features.as_static().unwrap_or(1024) as u64;
                            std::cmp::Reverse(m * n * k)
                        },
                        crate::core::op::NNOp::Attention(_attn) => {
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

    pub fn add_gemm<M: Into<crate::core::op::DimExpr>, N: Into<crate::core::op::DimExpr>, K: Into<crate::core::op::DimExpr>>(&mut self, m: M, n: N, k: K, deps: Vec<usize>) -> usize {
        let id = self.nodes.len();
        self.nodes.push(Node {
            id,
            op: Operation::Gemm(GemmOp::new(m, n, k)),
            dependencies: deps,
        });
        id
    }

    pub fn add_fused_gemm<M: Into<crate::core::op::DimExpr>, N: Into<crate::core::op::DimExpr>, K: Into<crate::core::op::DimExpr>>(&mut self, m: M, n: N, k: K, epilogue: Vec<crate::core::op::EpilogueOp>, deps: Vec<usize>) -> usize {
        let id = self.nodes.len();
        let base = GemmOp::new(m, n, k);
        self.nodes.push(Node {
            id,
            op: Operation::FusedGemm(crate::core::op::FusedGemmOp { base, epilogue }),
            dependencies: deps,
        });
        id
    }

    pub fn add_linear<B: Into<crate::core::op::DimExpr>, I: Into<crate::core::op::DimExpr>, O: Into<crate::core::op::DimExpr>>(&mut self, batch_size: B, in_features: I, out_features: O, bias: bool, activation: Vec<crate::core::op::EpilogueOp>, deps: Vec<usize>) -> usize {
        let id = self.nodes.len();
        let op = crate::core::op::LinearOp {
             batch_size: batch_size.into(),
             in_features: in_features.into(),
             out_features: out_features.into(),
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

    pub fn add_attention<E: Into<crate::core::op::DimExpr>, H: Into<crate::core::op::DimExpr>, HD: Into<crate::core::op::DimExpr>>(&mut self, embed_dim: E, num_heads: H, head_dim: HD, causal: bool, deps: Vec<usize>) -> usize {
        let id = self.nodes.len();
        let op = crate::core::op::AttentionOp {
            embed_dim: embed_dim.into(),
            num_heads: num_heads.into(),
            head_dim: head_dim.into(),
            causal,
        };
        self.nodes.push(Node {
            id,
            op: Operation::NN(crate::core::op::NNOp::Attention(op)),
            dependencies: deps,
        });
        id
    }

    pub fn add_fused_attention<B, S, D, H, DH>(&mut self, b: B, s: S, d: D, h: H, dh: DH, causal: bool, deps: Vec<usize>) -> usize 
    where B: Into<crate::core::op::DimExpr>, S: Into<crate::core::op::DimExpr>, D: Into<crate::core::op::DimExpr>, H: Into<crate::core::op::DimExpr>, DH: Into<crate::core::op::DimExpr> {
        let id = self.nodes.len();
        self.nodes.push(Node {
            id,
            op: Operation::FusedAttention(crate::core::op::FusedAttentionOp { 
                b: b.into(), s: s.into(), d: d.into(), h: h.into(), dh: dh.into(), 
                causal, scale_inv_sqrt_d: true 
            }),
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
                                     let s = qk_gemm.n.clone();
                                     let dh = qk_gemm.k.clone();
                                     let b_h_s = qk_gemm.m.clone();
                                     
                                     // Flatten B and H (Effective batch size)
                                     let b_eff = b_h_s; 
                                     let h_eff = crate::core::op::DimExpr::Static(1);
                                     
                                     let causal = true; 

                                     let q_id = qk_node.dependencies[0];
                                     let k_id = qk_node.dependencies[1];
                                     let v_id = maybe_v_id;

                                     fusion_mapping.insert(node.id, (b_eff, s, crate::core::op::DimExpr::Static(512), h_eff, dh, causal, vec![q_id, k_id, v_id]));
                                    
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
                let new_id = optimized.add_fused_attention(b.clone(), s.clone(), d.clone(), h.clone(), dh.clone(), *causal, deps);
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
                Operation::Gemm(gemm) => optimized.add_gemm(gemm.m.clone(), gemm.n.clone(), gemm.k.clone(), deps),
                Operation::FusedGemm(fused) => optimized.add_fused_gemm(fused.base.m.clone(), fused.base.n.clone(), fused.base.k.clone(), fused.epilogue.clone(), deps),
                Operation::Softmax(s) => optimized.add_softmax(s.axis, deps),
                Operation::FusedAttention(f) => {
                    let id = optimized.add_fused_attention(f.b.clone(), f.s.clone(), f.d.clone(), f.h.clone(), f.dh.clone(), f.causal, deps);
                    if let Operation::FusedAttention(ref mut op) = optimized.nodes[id].op {
                        op.scale_inv_sqrt_d = f.scale_inv_sqrt_d;
                    }
                    id
                },
                Operation::NN(nn) => {
                     match nn {
                        crate::core::op::NNOp::Linear(lin) => optimized.add_linear(lin.batch_size.clone(), lin.in_features.clone(), lin.out_features.clone(), lin.bias, lin.activation.clone(), deps),
                        crate::core::op::NNOp::Attention(attn) => optimized.add_attention(attn.embed_dim.clone(), attn.num_heads.clone(), attn.head_dim.clone(), attn.causal, deps),
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
                    lowered.add_gemm(gemm.m.clone(), gemm.n.clone(), gemm.k.clone(), deps)
                },
                Operation::FusedGemm(fused) => {
                    lowered.add_fused_gemm(fused.base.m.clone(), fused.base.n.clone(), fused.base.k.clone(), fused.epilogue.clone(), deps)
                },
                Operation::Softmax(s) => {
                    lowered.add_softmax(s.axis, deps)
                },
                Operation::FusedAttention(f) => {
                    lowered.add_fused_attention(f.b.clone(), f.s.clone(), f.d.clone(), f.h.clone(), f.dh.clone(), f.causal, deps)
                },
                Operation::NN(nn_op) => {
                    match nn_op {
                        crate::core::op::NNOp::Linear(lin) => {
                            // Expand Linear to FusedGemm
                            lowered.add_fused_gemm(
                                lin.batch_size.clone(), 
                                lin.out_features.clone(), 
                                lin.in_features.clone(), 
                                lin.activation.clone(), 
                                deps
                            )
                        },
                        crate::core::op::NNOp::Attention(attn) => {
                            let b = 1; // Batch Symbol might be better
                            let s = 128; // SeqLen Symbol
                            let d = attn.embed_dim.clone();
                            let h_val = attn.num_heads.as_static().unwrap_or(8);
                            let dh = attn.head_dim.clone();

                            // 1. Q, K, V Projections
                            let id_q = lowered.add_fused_gemm(d.clone(), d.clone(), d.clone(), vec![], deps.clone());
                            let id_k = lowered.add_fused_gemm(d.clone(), d.clone(), d.clone(), vec![], deps.clone());
                            let id_v = lowered.add_fused_gemm(d.clone(), d.clone(), d.clone(), vec![], deps);

                            // 2. Q @ K^T (Scaled Dot Product)
                            // Tracea doesn't have transpose yet, assuming K is pre-transposed or we handle it in emitter later.
                            // For IR lowering, we emit Gemm(Q, K, dh)
                            let id_qk = lowered.add_gemm(b * h_val * s, s, dh.clone(), vec![id_q, id_k]);

                            // 3. Softmax
                            let id_sm = lowered.add_softmax(-1, vec![id_qk]);

                            // 4. (Softmax @ V)
                            let id_out = lowered.add_gemm(b * h_val * s, dh, s, vec![id_sm, id_v]);
                            
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
        let id_lin = graph.add_linear(128, 128, 64, true, vec![EpilogueOp::ReLU], vec![id_in]);
        
        let lowered = graph.lower();
        assert_eq!(lowered.nodes.len(), 2);
        
        let node0 = &lowered.nodes[0];
        match &node0.op {
            Operation::Gemm(gemm) => {
                assert_eq!(gemm.m.as_static().unwrap(), 128);
            },
            _ => panic!("Node 0 should be Gemm"),
        }

        let node1 = &lowered.nodes[1];
        match node1.op {
            Operation::FusedGemm(ref fused) => {
                assert_eq!(fused.base.n.as_static().unwrap(), 64);
                assert_eq!(fused.base.k.as_static().unwrap(), 128);
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
                assert_eq!(attn.s.as_static().unwrap(), 128);
                assert_eq!(attn.h.as_static().unwrap(), 8);
                assert_eq!(attn.dh.as_static().unwrap(), 64);
            },
            _ => panic!("Last node should be FusedAttention"),
        }
    }
}
