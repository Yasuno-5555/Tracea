// src/doctor/polyhedral.rs
use crate::core::polyhedral::{PolyhedralInfo, Constraint, AffineFunction};
use crate::policy::types::OperatorTopology;

#[derive(Debug, Clone)]
pub struct PolyhedralAudit {
    pub op_name: String,
    pub info: crate::core::polyhedral::PolyhedralInfo,
    pub strategy: crate::core::polyhedral::TilingStrategy,
    pub parallel_dims: Vec<String>,
    pub issues: Vec<String>,
}

impl PolyhedralAudit {
    pub fn analyze(op: &OperatorTopology, caps: &crate::doctor::capabilities::TraceaCapabilities) -> Option<Self> {
        match op {
            OperatorTopology::Conv2d { n, c, h, w, k, r, s, stride, padding, .. } => {
                let mut audit = Self::from_conv2d(*n, *c, *h, *w, *k, *r, *s, *stride, *padding);
                audit.op_name = op.name().to_string();
                
                // Synthesis: Run Polyhedral Optimizer
                let optimizer = crate::core::polyhedral::PolyhedralOptimizer;
                audit.strategy = optimizer.optimize(&audit.info, caps);
                
                Some(audit)
            }
            _ => None,
        }
    }

    fn from_conv2d(n: u32, c: u32, h: u32, w: u32, k: u32, r: u32, s: u32, stride: u32, padding: u32) -> Self {
        // Dimensions: [n, oc, oh, ow, ic, kh, kw]
        let dim_names = vec![
            "n".to_string(), "oc".to_string(), "oh".to_string(), "ow".to_string(),
            "ic".to_string(), "kh".to_string(), "kw".to_string()
        ];
        let mut info = PolyhedralInfo::new(dim_names.clone());

        // Domain Constraints: 0 <= dim < LIMIT
        let limits = [n, k, h, w, c, r, s];
        for (i, &limit) in limits.iter().enumerate() {
            let mut coeffs = vec![0; 7];
            coeffs[i] = 1;
            // dim >= 0
            info.domain.push(Constraint::geq(coeffs.clone(), 0));
            // dim < limit  => -dim + limit - 1 >= 0
            let mut coeffs_neg = vec![0; 7];
            coeffs_neg[i] = -1;
            info.domain.push(Constraint::geq(coeffs_neg, (limit as i32) - 1));
        }

        // Access Maps (Simplified)
        // Reads (Input): Input[n, ic, oh*stride + kh - pad, ow*stride + kw - pad]
        // This is a bit complex for a simple affine mapping if we include padding in the index directly,
        // but the polyhedral model handles it by adding the padding to the domain or index function.
        // For now, let's just map the flat indices if possible or use logical dims.

        let mut issues = Vec::new();
        // Boundary Check (USER's Highlight): RGB case
        if c % 8 != 0 {
            issues.push(format!("Input channel alignment mismatch: C={} is not a multiple of 8. Vectorized loads will fail.", c));
        }

        Self {
            op_name: String::new(),
            info,
            strategy: crate::core::polyhedral::TilingStrategy::default(),
            parallel_dims: vec!["n".to_string(), "oc".to_string(), "oh".to_string(), "ow".to_string()],
            issues,
        }
    }
}
