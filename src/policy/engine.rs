use crate::policy::types::{PolicyContext, PolicyDecision, PolicyContext as _PC}; // Alias if needed

pub struct PolicyFeedback {
    pub operator_id: u64,
    pub latency_us: f32,
    pub error_msg: Option<String>,
}

pub trait PolicyEngine {
    fn propose(&mut self, ctx: &PolicyContext) -> PolicyDecision;
    fn propose_graph(&mut self, _ctx: &crate::policy::types::GraphContext) -> PolicyDecision {
        // Default implementation returns empty decision or panics if not implemented
        // For backwards compatibility, we can return empty decision for now
        PolicyDecision {
            tile_policies: vec![],
            exec_policies: vec![],
            global_hints: crate::policy::types::GlobalPolicyHints { prefer_fusion: false, debug_flags: 0 },
        }
    }
    fn update(&mut self, feedback: &PolicyFeedback);
}
