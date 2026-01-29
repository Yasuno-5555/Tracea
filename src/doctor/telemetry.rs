use serde::Serialize;
use super::engine::CompileStrategy;

#[derive(Debug, Serialize)]
pub struct PtxasSummary {
    pub registers: u32,
    pub smem: u32,
    pub cmem: u32,
}

use super::registry::BackendKind;

#[derive(Debug, Serialize)]
pub struct ExecutionRecord {
    pub env_id: [u8; 32],
    pub backend: BackendKind,
    pub kernel_id: String,
    pub variant_id: String,
    pub compile_strategy: CompileStrategy,
    pub driver_api_result: i32,
    pub ptxas_log_summary: Option<PtxasSummary>,
    pub success: bool,
    pub error_detail: Option<String>,
    pub timestamp: u64,
}

pub fn log_execution(record: ExecutionRecord) {
    let json = serde_json::to_string(&record).unwrap_or_default();
    println!("[TraceaDoctor][Telemetry] {}", json);
}

pub fn log_message(level: u32, message: &str) {
    if level == 0 {
        println!("[TraceaDoctor][INFO] {}", message);
    }
}
