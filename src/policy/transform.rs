use crate::policy::types::{GraphTopology, OperatorTopology};

pub fn canonicalize_graph(graph: &mut GraphTopology) {
    for op in &mut graph.operators {
        match op {
            OperatorTopology::Gemm { m, n, k, .. } => {
                // Ensure dimensions are non-zero (avoid divide by zero in cost model)
                if *m == 0 { *m = 1; }
                if *n == 0 { *n = 1; }
                if *k == 0 { *k = 1; }
            },
            OperatorTopology::Conv2d { n, .. } => {
                 if *n == 0 { *n = 1; }
            },
            _ => {}
        }
    }
}
