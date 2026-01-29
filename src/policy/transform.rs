use crate::policy::types::{GraphTopology, OperatorTopology};

pub fn canonicalize_graph(graph: &mut GraphTopology) {
    // 1. Basic normalization
    for op in &mut graph.operators {
        match op {
            OperatorTopology::Gemm { m, n, k, batch, .. } => {
                if *m == 0 { *m = 1; }
                if *n == 0 { *n = 1; }
                if *k == 0 { *k = 1; }
                if *batch == 0 { *batch = 1; }
            },
            OperatorTopology::Conv2d { n, .. } => {
                 if *n == 0 { *n = 1; }
            },
            _ => {}
        }
    }

    // 2. Expand Attention nodes (Phase I)
    expand_attention(graph);
}

pub fn expand_attention(graph: &mut GraphTopology) {
    let mut new_operators = Vec::new();
    let mut new_dependencies = Vec::new();
    
    // 1. Group dependencies by consumer
    let mut consumers_to_producers: std::collections::HashMap<u64, Vec<u64>> = std::collections::HashMap::new();
    for &(prod, cons) in &graph.dependencies {
        consumers_to_producers.entry(cons).or_default().push(prod);
    }
    // Sort producers for deterministic [Q, K, V] order
    for producers in consumers_to_producers.values_mut() {
        producers.sort();
    }

    // 2. Identify which dependencies are *not* going into an Attention node
    // because we will rebuild Attention dependencies from scratch.
    let attention_ids: std::collections::HashSet<u64> = graph.operators.iter()
        .filter_map(|op| if let OperatorTopology::Attention { op_id, .. } = op { Some(*op_id) } else { None })
        .collect();

    for &(prod, cons) in &graph.dependencies {
        if !attention_ids.contains(&cons) {
            new_dependencies.push((prod, cons));
        }
    }

    // 3. Max ID for fresh op generation
    let mut next_id = graph.operators.iter().map(|o| o.op_id()).max().unwrap_or(0) + 1;

    // 4. Transform Operators
    for op in &graph.operators {
        match op {
            OperatorTopology::Attention { op_id, name, b, s, h, d } => {
                let op_id = *op_id;
                let (b, s, h, d) = (*b, *s, *h, *d);
                let producers = consumers_to_producers.get(&op_id).cloned().unwrap_or_default();
                
                if producers.len() < 3 {
                    // Fallback to original if we can't find Q, K, V
                    new_operators.push(op.clone());
                    // Re-add dependencies
                    for &p in &producers { new_dependencies.push((p, op_id)); }
                    continue;
                }
                
                let q_id = producers[0];
                let k_id = producers[1];
                let v_id = producers[2];

                // Decomposition: QK -> Softmax -> PV
                // A. QK Gemm
                let qk_id = next_id; next_id += 1;
                new_operators.push(OperatorTopology::Gemm {
                    op_id: qk_id,
                    name: format!("{}_qk", name),
                    m: s, n: s, k: d,
                    batch: b * h,
                    kind: crate::policy::types::TopologyKind::Dense,
                    epilogue: vec![],
                });

                new_dependencies.push((q_id, qk_id));
                new_dependencies.push((k_id, qk_id));

                // B. Softmax
                let sm_id = next_id; next_id += 1;
                new_operators.push(OperatorTopology::Softmax {
                    op_id: sm_id,
                    name: format!("{}_softmax", name),
                    axis: -1,
                });
                new_dependencies.push((qk_id, sm_id));

                // C. PV Gemm (Reusable Tiled GEMM)
                // We keep original op_id for final result
                new_operators.push(OperatorTopology::Gemm {
                    op_id,
                    name: format!("{}_pv", name),
                    m: s, n: d, k: s,
                    batch: b * h,
                    kind: crate::policy::types::TopologyKind::Dense,
                    epilogue: vec![],
                });

                new_dependencies.push((sm_id, op_id));
                new_dependencies.push((v_id, op_id));
            },
            _ => {
                new_operators.push(op.clone());
            }
        }
    }

    graph.operators = new_operators;
    graph.dependencies = new_dependencies;
}
