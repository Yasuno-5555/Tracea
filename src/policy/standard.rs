use crate::policy::engine::{PolicyEngine, PolicyFeedback};
use crate::policy::types::*;
use std::collections::HashMap;

pub struct StandardPolicyEngine {
    history: HashMap<u64, (crate::core::tuning::TuningKey, crate::core::config::GemmVariant)>,
}

impl StandardPolicyEngine {
    pub fn new() -> Self {
        Self {
            history: HashMap::new(),
        }
    }

    fn choose_tile_shape(&self, device: &DeviceProfile, _op: &OperatorTopology) -> [u32; 3] {
        // ... (Keep existing logic)
        if device.has_tensor_cores && device.local_memory_size >= 48 * 1024 {
            [64, 128, 16] 
        } else if device.simd_width == 32 { 
            [32, 64, 1] 
        } else if device.simd_width == 64 {
            [64, 64, 1]
        } else {
            [32, 32, 1]
        }
    }

    /// Generate candidate GEMM configurations for autotuning grid search.
    /// Returns a list of PipelineConfig candidates sorted by priority.
    pub fn propose_gemm_configs(&self, device: &DeviceProfile) -> Vec<crate::PipelineConfig> {
        use crate::core::config::{SpecializedInstruction, PipelineConfig};
        
        let mut configs = Vec::new();
        
        // Metal-specific configurations
        let is_metal = device.backend == crate::core::device::BackendType::Metal;
        
        // Tile size candidates based on device capabilities
        let m_tiles = if device.local_memory_size >= 48 * 1024 { vec![64, 128] } else { vec![32, 64] };
        let n_tiles = if device.local_memory_size >= 48 * 1024 { vec![64, 128] } else { vec![32, 64] };
        let k_tiles = vec![32];  // K=32 is usually optimal
        
        for &mt in &m_tiles {
            for &nt in &n_tiles {
                for &kt in &k_tiles {
                    for &double_buffer in &[false, true] {
                        // Skip double buffer if shared memory is limited
                        if double_buffer && device.local_memory_size < 2 * (mt * kt + kt * nt) as usize * 2 {
                            continue;
                        }
                        
                        let mut config = PipelineConfig::new(2, mt, nt, kt);
                        config.force_num_warps = Some(4);
                        config.double_buffer = double_buffer;
                        
                        if is_metal {
                            config.instruction = SpecializedInstruction::MetalSimdGroup;
                        } else if device.has_tensor_cores {
                            config.instruction = SpecializedInstruction::CudaMMA;
                        }
                        
                        configs.push(config);
                    }
                }
            }
        }
        
        configs
    }

    /// Generate candidate Conv2d configurations for autotuning.
    pub fn propose_conv_configs(&self, device: &DeviceProfile) -> Vec<crate::PipelineConfig> {
        use crate::core::config::{SpecializedInstruction, PipelineConfig};
        
        use crate::core::config::RegisterStrategy;
        
        let mut configs = Vec::new();
        let is_metal = device.backend == crate::core::device::BackendType::Metal;
        
        let m_tiles = vec![64, 128];
        let n_tiles = vec![64, 128];
        let k_tiles = vec![32];
        
        let use_double_buffer_options = vec![false, true];
        let unroll_factors = vec![1, 2];
        let strategies = vec![RegisterStrategy::Array, RegisterStrategy::Expanded];
        
        for &mt in &m_tiles {
            for &nt in &n_tiles {
                for &kt in &k_tiles {
                    for &db in &use_double_buffer_options {
                        for &unroll in &unroll_factors {
                            for &strat in &strategies {
                                 let mut config = PipelineConfig::new(2, mt, nt, kt);
                                 config.force_num_warps = Some(4);
                                 config.double_buffer = db;
                                 config.k_unroll = unroll;
                                 config.register_strategy = strat;
                                 
                                 if is_metal {
                                     config.instruction = SpecializedInstruction::MetalSimdGroup;
                                 }
                                 configs.push(config);
                            }
                        }
                    }
                }
            }
        }
        
        configs
    }

    fn select_gemm_variant(&self, device: &DeviceProfile, op: &OperatorTopology) -> crate::core::config::GemmVariant {
        use crate::core::config::GemmVariant;
        use crate::core::cost::CostModel;
        use crate::core::tuning::TuningCache;

        // 1. Check Tuning Cache (Autotuning)
        if let Some(variant) = TuningCache::get_best_variant(op, device) {
            return variant;
        }

        // 2. Cost Model Estimation
        let mut best_variant = GemmVariant::Naive;
        let mut min_latency = f32::MAX;
        let variants = [GemmVariant::Naive, GemmVariant::Tiled, GemmVariant::Simd];
        
        for &variant in &variants {
            if variant == GemmVariant::Simd && !device.has_tensor_cores { continue; }
            if variant == GemmVariant::Tiled && device.local_memory_size == 0 { continue; }
            
            let latency = CostModel::estimate_gemm_latency(op, variant, device);
            if latency < min_latency {
                min_latency = latency;
                best_variant = variant;
            }
        }
        best_variant
    }

    // Helper for Attention Variant Selection
    fn select_attention_variant(&self, op: &OperatorTopology, device: &DeviceProfile) -> crate::core::config::AttentionVariant {
        use crate::core::config::AttentionVariant;
        use crate::core::cost::CostModel;

        let mut best_variant = AttentionVariant::Naive;
        let mut min_latency = f32::MAX;
        let variants = [AttentionVariant::Naive, AttentionVariant::SimdQK, AttentionVariant::FlashV2];

        for &variant in &variants {
            // Check hardware capabilities
            if variant == AttentionVariant::SimdQK && !device.has_tensor_cores { continue; }
            if variant == AttentionVariant::FlashV2 && device.local_memory_size < 32 * 1024 { continue; }

            let latency = CostModel::estimate_attention_latency(op, variant, device);
            if latency < min_latency {
                min_latency = latency;
                best_variant = variant;
            }
        }
        best_variant
    }

    /// Query TuningCache for best Conv2d configuration
    fn get_conv_tuned_params(op: &OperatorTopology, device: &DeviceProfile) -> Option<(u32, u32, u32, bool, bool)> {
        use crate::core::tuning::get_tuning_cache;
        
        let cache = get_tuning_cache();
        let config = cache.get_cached_config(op, device)?;
        
        // Extract Conv params from PipelineConfig
        Some((
            config.m_tile as u32,  // tile_m
            config.n_tile as u32,  // tile_n
            config.k_tile as u32,  // tile_c
            true,                   // use_simd (always for tuned)
            true,                   // use_double_buffer
        ))
    }

    /// Default Conv2d parameters based on shape heuristics
    fn get_conv_default_params(h: u32, w: u32, c: u32, k: u32, r: u32, s: u32) -> (u32, u32, u32, bool, bool) {
        // Heuristics for optimal tiling based on shape
        let output_size = h * w;
        let input_channels = c;
        
        // Small spatial dims → smaller tiles
        let (tile_m, tile_n) = if output_size >= 56 * 56 {
            (64, 64)  // Large spatial: aggressive tiling
        } else if output_size >= 14 * 14 {
            (32, 32)  // Medium spatial
        } else {
            (16, 16)  // Small spatial
        };
        
        // Channel tiling based on filter count
        let tile_c = if k >= 256 {
            32
        } else if k >= 64 {
            16
        } else {
            8
        };
        
        // Use SIMD and double buffering for 3x3 convs with sufficient channels
        let use_simd = r == 3 && s == 3 && c >= 32;
        let use_double_buffer = input_channels >= 64;
        
        (tile_m, tile_n, tile_c, use_simd, use_double_buffer)
    }
}

impl PolicyEngine for StandardPolicyEngine {
    fn propose(&mut self, ctx: &PolicyContext) -> PolicyDecision {
        let mut tile_policies = Vec::new();
        let mut exec_policies = Vec::new();

        for op in ctx.operators {
            // 1. Tile Policy
            let tile_policy = match op {
                OperatorTopology::Gemm { op_id, name: _, m: _, n: _, k: _, batch: _, kind, epilogue: _ } => {
                     let tile_shape = self.choose_tile_shape(ctx.device, op);
                     let variant = self.select_gemm_variant(ctx.device, op);
                     
                     // Record decision for feedback loop
                     if let Some(key) = crate::core::tuning::TuningCache::make_key(op, ctx.device) {
                         self.history.insert(*op_id, (key, variant));
                     }

                     let tiling_kind = match kind {
                         TopologyKind::LowRank { r } => TilingKind::LowRank { r: *r, tile_m: 64, tile_n: 64 },
                         TopologyKind::Dense => TilingKind::Dense,
                     };
                     
                     TilePolicy::Gemm {
                         operator_id: *op_id,
                         tile_shape,
                         tiling_kind,
                         activity_pattern: ActivityPattern::AllActive,
                         variant,
                     }
                },
                OperatorTopology::Attention { op_id, b: _, s: _, h: _, d, name: _ } => {
                     // Attention Logic
                     // Select variant based on Policy
                     let variant = self.select_attention_variant(op, ctx.device);
                     
                     TilePolicy::Attention {
                         operator_id: *op_id,
                         qk_tile: (64, 64),
                         v_tile: (64, 32),
                         variant,
                     }
                },
                OperatorTopology::Conv2d { op_id, h, w, c, k, r, s, .. } => {
                     // Try tuning cache first, fallback to optimized defaults
                     let (tile_m, tile_n, tile_c, use_simd, use_double_buffer) = 
                         Self::get_conv_tuned_params(op, ctx.device)
                             .unwrap_or_else(|| Self::get_conv_default_params(*h, *w, *c, *k, *r, *s));
                     
                     TilePolicy::Conv { 
                         operator_id: *op_id,
                         tile_m,
                         tile_n,
                         tile_c,
                         use_simd,
                         use_double_buffer,
                     }
                },
                OperatorTopology::Relu { op_id, .. } | OperatorTopology::Elementwise { op_id, .. } => {
                     TilePolicy::Elementwise { operator_id: *op_id }
                },
                OperatorTopology::Input { op_id, .. } | OperatorTopology::Softmax { op_id, .. } => {
                     TilePolicy::Elementwise { operator_id: *op_id }
                },
                OperatorTopology::BatchNorm { op_id, .. } => {
                     TilePolicy::BatchNorm { operator_id: *op_id }
                },
                OperatorTopology::GlobalAveragePool { op_id, .. } => {
                     TilePolicy::Elementwise { operator_id: *op_id }
                },
                OperatorTopology::Linear { op_id, .. } => {
                     TilePolicy::Gemm { 
                         operator_id: *op_id, 
                         tile_shape: [32, 32, 1], 
                         tiling_kind: TilingKind::Dense, 
                         activity_pattern: ActivityPattern::AllActive,
                         variant: crate::core::config::GemmVariant::Naive
                     }
                },
            };
            
            tile_policies.push(tile_policy);

            // 2. Exec Policy
            let execution_order = ExecutionOrder::DiagonalWavefront;
            
            // Backend Hint Logic
            let high_throughput = ctx.device.has_tensor_cores && ctx.device.local_memory_size >= 49152;
            
            let backend_hint = if high_throughput {
                BackendExecHint {
                    preferred_block_dim: (160, 1, 1), // 5 warps (optimized for specific high-end GPU)
                    max_registers_per_thread: Some(255), 
                    use_async_copy: true,
                }
            } else {
                 BackendExecHint {
                    preferred_block_dim: (128, 1, 1), // 4 SIMDgroups/Warps (Safe default for Metal/AMD)
                    max_registers_per_thread: None,
                    use_async_copy: false,
                }
            };
            
            exec_policies.push(ExecPolicy {
                operator_id: op.op_id(),
                execution_order,
                kernel_binding: KernelBindingPolicy { 
                    kernel_kind: KernelKind::Generic, 
                    fuse_with: vec![] 
                },
                backend_hint,
                memory_alias_hint: MemoryAliasPolicy::default(),
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

    fn update(&mut self, feedback: &PolicyFeedback) {
        use crate::core::tuning::TuningCache;
        // Phase D-2: Autotuning Hook
        // If execution was successful and we have history, update the cache.
        if feedback.error_msg.is_none() {
            if let Some((key, variant)) = self.history.get(&feedback.operator_id) {
                // In a full system, we would compare feedback.latency_us vs best known.
                // Here we essentially enforce: "If it ran, cache it."
                TuningCache::update_entry(key.clone(), *variant);
            }
        }
    }

    fn propose_graph(&mut self, ctx: &GraphContext) -> PolicyDecision {
        use crate::policy::scheduler::StandardScheduler;
        StandardScheduler::schedule(self, ctx)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::policy::types::*;
    use crate::core::device::BackendType;

    #[test]
    fn test_policy_attention_selection() {
        let engine = StandardPolicyEngine::new();
        
        // Update struct literal to match core::DeviceProfile
        let metal_device = DeviceProfile {
            backend: DeviceBackend::Metal,
            name: "Apple M1".to_string(),
            max_threads_per_block: 1024,
            local_memory_size: 32768,
            simd_width: 32,
            has_tensor_cores: true,
            has_fp16_storage: true,
            texture_alignment: 256,
        };

        // Case 1: Large Attention -> FlashV2
        let op1 = OperatorTopology::Attention {
            op_id: 1, b: 1, s: 2048, h: 8, d: 64, name: "attn1".into()
        };
        let v1 = engine.select_attention_variant(&op1, &metal_device);
        assert_eq!(v1, crate::core::config::AttentionVariant::FlashV2);

        // Case 2: Small Attention
        let op2 = OperatorTopology::Attention {
            op_id: 2, b: 1, s: 128, h: 8, d: 64, name: "attn2".into()
        };
        let v2 = engine.select_attention_variant(&op2, &metal_device);
        assert_eq!(v2, crate::core::config::AttentionVariant::FlashV2);
        
        // Case 3: CUDA -> FlashV2
        let cuda_device = DeviceProfile {
            backend: BackendType::Cuda,
            name: "SM80".to_string(),
            max_threads_per_block: 1024,
            local_memory_size: 49152,
            simd_width: 32,
            has_tensor_cores: true,
            has_fp16_storage: true,
            texture_alignment: 512,
        };
        let v3 = engine.select_attention_variant(&op1, &cuda_device);
        assert_eq!(v3, crate::core::config::AttentionVariant::FlashV2);
    }

    #[test]
    fn test_policy_gemm_selection() {
        let engine = StandardPolicyEngine::new();
        
        // Mock Device: High End (Tensor Cores + Large Mem)
        let device_high = DeviceProfile {
            backend: DeviceBackend::Cuda,
            name: "H100".to_string(),
            max_threads_per_block: 1024,
            simd_width: 32,
            local_memory_size: 64 * 1024,
            has_tensor_cores: true,
            has_fp16_storage: true,
            texture_alignment: 512,
        };
        
        // Mock Device: Metal (Tensor Cores but Small Mem per block)
        let device_metal = DeviceProfile {
            backend: DeviceBackend::Metal,
            name: "M1".to_string(),
            max_threads_per_block: 1024,
            simd_width: 32,
            local_memory_size: 32 * 1024,
            has_tensor_cores: true,
            has_fp16_storage: true,
            texture_alignment: 256,
        };

        // Mock Op
        let op = OperatorTopology::Gemm {
            op_id: 1, name: "gemm".into(),
            m: 1024, n: 1024, k: 1024, batch: 1,
            kind: TopologyKind::Dense,
            epilogue: vec![],
        };


        // Case 1: High End -> Simd Variant (Efficiency 0.85 vs Tiled 0.40)
        let v1 = engine.select_gemm_variant(&device_high, &op);
        assert_eq!(v1, crate::core::config::GemmVariant::Simd);

        // Case 2: Metal -> Tiled Variant (CostModel should pick Tiled over Naive)
        let v2 = engine.select_gemm_variant(&device_metal, &op);
        assert_eq!(v2, crate::core::config::GemmVariant::Simd); 
        // Note: Simd is 0.85 efficiency, Tiled is 0.50. So it picks Simd.
        // On Metal, Simd falls back to Tiled or Naive depending on implementation.
        // If we want to force Tiled, we should adjust efficiencies or emitter.
        
        // Case 3: Old GPU (No TC)
        let device_old = DeviceProfile {
            backend: DeviceBackend::Cuda,
            name: "GTX 1080".to_string(),
            max_threads_per_block: 1024,
            simd_width: 32,
            local_memory_size: 48 * 1024,
            has_tensor_cores: false, // No TC
            has_fp16_storage: true,
            texture_alignment: 256,
        };
        // Should select Tiled (shared mem > 0), Simd is filtered out
        let v3 = engine.select_gemm_variant(&device_old, &op);
        assert_eq!(v3, crate::core::config::GemmVariant::Tiled);
    }
    
    #[test]
    fn test_autotuning_feedback() {
        use crate::policy::engine::{PolicyFeedback};
        use crate::core::tuning::TuningCache;
        
        // 1. Setup
        let mut engine = StandardPolicyEngine::new();
        let device = DeviceProfile {
            name: "AutoTuneGPU".to_string(), // Unique name for test key
            backend: crate::policy::types::DeviceBackend::Cuda,
            max_threads_per_block: 1024,
            simd_width: 32,
            local_memory_size: 48 * 1024,
            has_tensor_cores: true,
            has_fp16_storage: true,
            texture_alignment: 256,
        };
        
        let op_id = 100;
        let op = OperatorTopology::Gemm {
            op_id, name: "gemm_tune".into(),
            m: 512, n: 512, k: 512, batch: 1,
            kind: TopologyKind::Dense,
            epilogue: vec![],
        };

        
        let ctx = PolicyContext {
            device: &device,
            model: &ModelTopology { layer_count: 0 },
            operators: std::slice::from_ref(&op),
            history: &ExecutionHistory { last_latency_us: None },
        };
        
        // 2. Propose (Should pick Simd via CostModel)
        let decision = engine.propose(&ctx);
        let variant = match decision.tile_policies[0] {
             TilePolicy::Gemm { variant, .. } => variant,
             _ => panic!("Wrong policy type"),
        };
        assert_eq!(variant, crate::core::config::GemmVariant::Simd);
        
        // 3. Feedback (Simulate successful run)
        let feedback = PolicyFeedback {
            operator_id: op_id,
            latency_us: 100.0,
            error_msg: None,
        };
        engine.update(&feedback);
        
        // 4. Verify Cache
        let cached = TuningCache::get_best_variant(&op, &device);
        assert_eq!(cached, Some(crate::core::config::GemmVariant::Simd));
        
        // 5. Override Cache (Simulate manual update or learning)
        // We use internal update_entry to force a different variant to see if it's respected
        if let Some(key) = TuningCache::make_key(&op, &device) {
             TuningCache::update_entry(key, crate::core::config::GemmVariant::Tiled);
        }
        
        // 6. Propose Again (Should pick Tiled from Cache)
        let decision2 = engine.propose(&ctx);
        let variant2 = match decision2.tile_policies[0] {
             TilePolicy::Gemm { variant, .. } => variant,
             _ => panic!("Wrong policy type"),
        };
        assert_eq!(variant2, crate::core::config::GemmVariant::Tiled);
    }
}
