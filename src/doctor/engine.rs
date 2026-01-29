use super::capabilities::TraceaCapabilities;
use super::registry::{KernelVariant, Requirement, Preference, BackendKind};

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum PrecisionPolicy {
    FP32,
    FP16,
    BF16,
    INT8,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum CompileStrategy {
    JIT,
    Precompiled,
    AOT,
}

#[derive(Debug, Clone)]
pub struct KernelRequestContext {
    pub precision_policy: PrecisionPolicy,
    pub latency_vs_throughput: f32, // 0.0 to 1.0
    pub allow_fallback: bool,
}

#[derive(Debug, Clone)]
pub struct FallbackEntry {
    pub variant_id: &'static str,
    pub reason: String,
}

#[derive(Debug, Clone)]
pub enum DecisionReason {
    MatchesAllRequirements,
    FallbackDueToCapability,
    NoSuitableVariantFound,
}

#[derive(Debug, Clone)]
pub struct Decision {
    pub selected_variant: Option<&'static str>,
    pub compile_strategy: CompileStrategy,
    pub fallback_plan: Vec<FallbackEntry>,
    pub reason: DecisionReason,
}

pub fn select_variant(
    caps: &TraceaCapabilities,
    kernel_id: &str,
    request: KernelRequestContext,
) -> Decision {
    let variants = super::registry::get_variants_for(kernel_id);
    
    let mut candidates: Vec<(KernelVariant, f32)> = Vec::new();
    let mut fallback_plan = Vec::new();

    for variant in variants {
        // 1. Check if the variant's backend is available
        let backend_caps = if let Some(bc) = caps.get_backend(variant.backend) {
            bc
        } else {
            fallback_plan.push(FallbackEntry {
                variant_id: variant.id,
                reason: format!("Backend {:?} not available", variant.backend),
            });
            continue;
        };

        let mut meets_requirements = true;
        for req in &variant.hard_requirements {
            match req {
                Requirement::BackendIs(kind) => {
                    if variant.backend != *kind {
                        // This should theoretically not happen if registry is consistent, but check anyway
                        meets_requirements = false; 
                    }
                }
                Requirement::SmAtLeast(sm) => {
                    // Start of legacy/mapped requirement handling
                    if variant.backend == BackendKind::Cuda {
                        if backend_caps.arch_code < *sm { // simple integer comparison for SM
                             meets_requirements = false;
                             fallback_plan.push(FallbackEntry {
                                variant_id: variant.id,
                                reason: format!("Required SM >= {}, but found {}", sm, backend_caps.arch_code),
                            });
                            break;
                        }
                    }
                }
                Requirement::HasTensorCoreLike => {
                    if !backend_caps.has_tensor_core_like {
                        meets_requirements = false;
                        fallback_plan.push(FallbackEntry {
                            variant_id: variant.id,
                            reason: "Required Tensor Core-like capability not supported".to_string(),
                        });
                        break;
                    }
                }
                Requirement::MaxSharedMemAtLeast(bytes) => {
                    if backend_caps.max_shared_mem < *bytes {
                        meets_requirements = false;
                        fallback_plan.push(FallbackEntry {
                            variant_id: variant.id,
                            reason: format!("Required Shared Mem >= {}, but found {}", bytes, backend_caps.max_shared_mem),
                        });
                        break;
                    }
                }
                Requirement::WarpOrWavefrontIs(size) => {
                    if backend_caps.warp_or_wavefront != *size {
                        meets_requirements = false;
                        fallback_plan.push(FallbackEntry {
                            variant_id: variant.id,
                            reason: format!("Required Warp/Wavefront size {}, but found {}", size, backend_caps.warp_or_wavefront),
                        });
                        break;
                    }
                }
                Requirement::SimdWidthAtLeast(width) => {
                    if backend_caps.simd_width_bits < *width {
                        meets_requirements = false;
                        fallback_plan.push(FallbackEntry {
                            variant_id: variant.id,
                            reason: format!("Required SIMD width >= {}, but found {}", width, backend_caps.simd_width_bits),
                        });
                        break;
                     }
                }
                Requirement::Precision(p) => {
                    if request.precision_policy != *p {
                        meets_requirements = false;
                        fallback_plan.push(FallbackEntry {
                            variant_id: variant.id,
                            reason: format!("Required Precision {:?}, but requested {:?}", p, request.precision_policy),
                        });
                        break;
                    }
                }
            }
        }

        if meets_requirements {
            // Scoring logic
            let mut score = variant.priority as f32;
            
            for pref in &variant.soft_preferences {
                match pref {
                    Preference::PreferTensorCoreLike => {
                        if backend_caps.has_tensor_core_like {
                            score += 50.0;
                        }
                    }
                    Preference::PreferLargerSharedMem => {
                         score += (backend_caps.max_shared_mem as f32) / 1024.0; // rudimentary boost
                    }
                    Preference::PreferLowerRegUsage => {
                        score += 5.0;
                    }
                }
            }
            
            candidates.push((variant, score));
        }
    }

    // Sort by score descending
    candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    if let Some((best, _)) = candidates.first() {
        Decision {
            selected_variant: Some(best.id),
            compile_strategy: match best.backend {
                BackendKind::Cuda => CompileStrategy::JIT,
                BackendKind::Rocm => CompileStrategy::AOT,
                BackendKind::Metal => CompileStrategy::AOT,
                BackendKind::Cpu => CompileStrategy::AOT, 
                BackendKind::Vulkan => CompileStrategy::JIT,
            },
            fallback_plan,
            reason: DecisionReason::MatchesAllRequirements,
        }
    } else {
        Decision {
            selected_variant: None,
            compile_strategy: CompileStrategy::JIT,
            fallback_plan,
            reason: DecisionReason::NoSuitableVariantFound,
        }
    }
}

