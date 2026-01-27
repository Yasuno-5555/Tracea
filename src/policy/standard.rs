use crate::policy::engine::{PolicyEngine, PolicyFeedback};
use crate::policy::types::*;
use crate::runtime::DeviceBackend;

pub struct StandardPolicyEngine {
    // History storage placeholder
}

impl StandardPolicyEngine {
    pub fn new() -> Self {
        Self {}
    }

    fn choose_tile_shape(&self, device: &DeviceProfile, op: &OperatorTopology) -> [u32; 3] {
        match device.backend {
            DeviceBackend::Metal => {
                // Threadgroup Memory priority (e.g. 32KB limit)
                // [32, 64, 1] is a safe default for Metal
                [32, 64, 1]
            },
            DeviceBackend::Cuda => {
                // TensorCore occupancy priority
                // Ampere/Hopper usually like 64x128 or 128x128. 
                // KT must be at least 16 for MMA.
                [64, 128, 16]
            },
            DeviceBackend::Rocm => {
                // MI300 Wave64 preference
                [64, 64, 1]
            },
            _ => [32, 32, 1]
        }
    }

    fn choose_tiling_kind(&self, op: &OperatorTopology) -> TilingKind {
        match &op.kind {
            TopologyKind::LowRank { r } => {
                 TilingKind::LowRank { r: *r, tile_m: 64, tile_n: 64 }
            }
            TopologyKind::Dense => {
                match op.op_type.as_str() {
                    "Attention" => TilingKind::Dense,
                    "Gemm" => TilingKind::Dense,
                    "LowRankMlp" => TilingKind::LowRank { r: 64, tile_m: 64, tile_n: 64 }, 
                    _ => TilingKind::Dense,
                }
            }
        }
    }

    fn choose_execution_order(&self, _device: &DeviceProfile) -> ExecutionOrder {
        // Default to DiagonalWavefront for better concurrency in pipelines
        ExecutionOrder::DiagonalWavefront
    }

    fn choose_activity_pattern(&self, op: &OperatorTopology) -> ActivityPattern {
        // Can read op annotations or use random for testing
        ActivityPattern::AllActive
    }
}

impl PolicyEngine for StandardPolicyEngine {
    fn propose(&mut self, ctx: &PolicyContext) -> PolicyDecision {
        let mut tile_policies = Vec::new();
        let mut exec_policies = Vec::new();

        for op in ctx.operators {
            // 1. Tile Policy
            let tile_shape = self.choose_tile_shape(ctx.device, op);
            let tiling_kind = self.choose_tiling_kind(op);
            let activity_pattern = self.choose_activity_pattern(op);
            
            tile_policies.push(TilePolicy {
                operator_id: op.op_id,
                tile_shape,
                tiling_kind,
                activity_pattern,
            });

            // 2. Exec Policy
            let execution_order = self.choose_execution_order(ctx.device);
            
            // Backend Hint Logic
            let backend_hint = match ctx.device.backend {
                DeviceBackend::Cuda => BackendExecHint {
                    preferred_block_dim: (160, 1, 1), // 5 warps (1 Prod + 4 Cons)
                    max_registers_per_thread: Some(255), 
                    use_async_copy: true,
                },
                DeviceBackend::Metal => BackendExecHint {
                    preferred_block_dim: (128, 1, 1), // SIMDGROUP size align
                    max_registers_per_thread: None,
                    use_async_copy: false,
                },
                _ => BackendExecHint {
                    preferred_block_dim: (64, 1, 1),
                    max_registers_per_thread: None,
                    use_async_copy: false,
                }
            };

            exec_policies.push(ExecPolicy {
                operator_id: op.op_id,
                execution_order,
                kernel_binding: KernelBindingPolicy { 
                    kernel_kind: KernelKind::Generic, 
                    fuse_with: vec![] 
                },
                backend_hint,
            });
        }

        PolicyDecision {
            tile_policies,
            exec_policies,
            global_hints: GlobalPolicyHints { 
                prefer_fusion: true,
                debug_flags: 0 
            },
        }
    }

    fn update(&mut self, _feedback: &PolicyFeedback) {
        // Simple no-op or logging for Phase B
    }
}
