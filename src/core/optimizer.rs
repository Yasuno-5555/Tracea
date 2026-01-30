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

        let mut current_ops = graph.operators.clone();
        let mut ops_to_remove = HashSet::new();
        let mut fused_id_map: HashMap<u64, u64> = HashMap::new(); // maps original_id -> fused_into_id

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
                                if let Some(epilogue_op) = Self::try_map_to_epilogue(c_op, &reverse_adj, op_id, &fused_id_map) {
                                    let producers = reverse_adj.get(&consumer_id).cloned().unwrap_or_default();
                                    
                                    let can_fuse = match epilogue_op {
                                        EpilogueOp::ReLU => producers.len() == 1,
                                        EpilogueOp::BatchNorm { .. } => producers.len() == 5,
                                        EpilogueOp::ResidualAdd { .. } => producers.len() == 2,
                                        _ => false,
                                    };

                                    if can_fuse {
                                        println!("[Fusion] Merging {} into {}", c_op.name(), op.name());
                                        let mut fused_op = op.clone();
                                        Self::push_epilogue(&mut fused_op, epilogue_op);
                                        
                                        next_ops.push(fused_op);
                                        ops_to_remove.insert(consumer_id);
                                        fused_id_map.insert(consumer_id, op_id);
                                        
                                        // Update adj/reverse_adj for next iterations
                                        let consumers_of_consumer = adj.get(&consumer_id).cloned().unwrap_or_default();
                                        adj.insert(op_id, consumers_of_consumer.clone());
                                        for &c_of_c in &consumers_of_consumer {
                                            if let Some(mut rev) = reverse_adj.get_mut(&c_of_c) {
                                                rev.retain(|&x| x != consumer_id);
                                                rev.push(op_id);
                                            }
                                        }

                                        // Update dependencies for side-inputs (params)
                                        if let Some(consumer_producers) = reverse_adj.get(&consumer_id).cloned() {
                                            for &prod_id in &consumer_producers {
                                                if prod_id != op_id {
                                                    // This is a side-input (e.g. Gamma). Redirect to Fused Op.
                                                    adj.entry(prod_id).or_default().push(op_id);
                                                    reverse_adj.entry(op_id).or_default().push(prod_id);
                                                }
                                            }
                                        }

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

        graph.operators = current_ops;
        
        // Final dependency cleanup
        let mut final_deps = Vec::new();
        for (src, dsts) in adj {
            if !ops_to_remove.contains(&src) {
                for dst in dsts {
                    if !ops_to_remove.contains(&dst) {
                        final_deps.push((src, dst));
                    }
                }
            }
        }
        graph.dependencies = final_deps;
    }

    fn try_map_to_epilogue(op: &OperatorTopology, reverse_adj: &HashMap<u64, Vec<u64>>, producer_id: u64, fused_id_map: &HashMap<u64, u64>) -> Option<EpilogueOp> {
        match op {
            OperatorTopology::Relu { .. } => Some(EpilogueOp::ReLU),
            OperatorTopology::BatchNorm { op_id, epsilon, .. } => {
                let producers = reverse_adj.get(op_id)?;
                let gamma_id = producers.get(1).cloned().unwrap_or(0);
                let beta_id = producers.get(2).cloned().unwrap_or(0);
                let mean_id = producers.get(3).cloned().unwrap_or(0);
                let var_id = producers.get(4).cloned().unwrap_or(0);

                Some(EpilogueOp::BatchNorm { 
                    op_id: *op_id, 
                    epsilon: *epsilon,
                    gamma_id, beta_id, mean_id, var_id,
                })
            }
            OperatorTopology::Elementwise { op_id, kind, .. } if kind == "Add" => {
                let producers = reverse_adj.get(op_id)?;
                let skip_id = producers.iter().find(|&&id| id != producer_id)?;
                // If skip_id was fused, we need its current representative
                let mut current_skip_id = *skip_id;
                while let Some(&next) = fused_id_map.get(&current_skip_id) {
                    current_skip_id = next;
                }
                Some(EpilogueOp::ResidualAdd { residual_ptr: current_skip_id as usize }) 
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
