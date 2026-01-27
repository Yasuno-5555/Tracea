use crate::policy::types::{PolicyContext, PolicyDecision, PolicyContext as _PC}; // Alias if needed

pub struct PolicyFeedback {
    pub operator_id: u64,
    pub latency_us: f32,
    pub error_msg: Option<String>,
}

pub trait PolicyEngine {
    fn propose(&mut self, ctx: &PolicyContext) -> PolicyDecision;
    fn update(&mut self, feedback: &PolicyFeedback);
}
