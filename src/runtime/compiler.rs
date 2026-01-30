use crate::runtime::plan::{ExecutionPlan, ExecutionStep, KernelArgSpec};
use crate::policy::types::{GraphTopology, OperatorTopology, PolicyDecision};
use crate::core::op::EpilogueOp;
use crate::runtime::manager::{DeviceBackend, RuntimeManager};
use std::collections::{HashMap, HashSet};

#[derive(Debug)]
pub struct GraphCompiler {
    // Compiler might hold state or config
}

impl GraphCompiler {
    pub fn new() -> Self {
        Self {}
    }

    /// Consumes the graph and produces an ExecutionPlan.
    /// This is the "Slow Path".
    pub fn compile(&self, graph: GraphTopology, manager: &RuntimeManager, backend: DeviceBackend) -> Result<ExecutionPlan, String> {
        // 1. Optimize & Select Kernels (Policy)
        let (decision, optimized_graph) = self.optimize_graph(&graph, backend)?;
        let sorted_nodes = self.schedule(&decision);
        let (arena_offsets, arena_size) = self.plan_memory(&optimized_graph, &sorted_nodes, &decision)?;

        let mut steps = Vec::new();
        
        // 2. Generate Plan Steps
        for &op_id in &sorted_nodes {
            let tile_policy = decision.tile_policies.iter().find(|t| t.operator_id() == op_id);
            
            let operator = optimized_graph.operators.iter().find(|n| n.op_id() == op_id)
                .ok_or_else(|| format!("Node {} not found in graph", op_id))?;

            // Skip Input nodes for execution steps (inputs are handled by Executor binding)
            if let OperatorTopology::Input { .. } = operator {
                continue;
            }

            // Compile Kernel (JIT)
            let kernel_id = self.compile_kernel(operator, tile_policy.cloned(), manager, backend)?;
            
            // Build Arguments
            let args = self.build_arguments(operator, &optimized_graph, &arena_offsets, op_id)?;

            let (grid_size, block_size) = self.calculate_launch_dims(operator, tile_policy.cloned());
            
            // Unified Shared Memory Calculation via Emitters (Temporary transition)
            let shared_mem_bytes = match operator {
                OperatorTopology::Conv2d { .. } => {
                    let ir = self.build_ir(operator, tile_policy, None).unwrap();
                    crate::emitter::conv::calculate_smem_usage(&ir)
                },
                _ => 0,
            };

            // Prototype: Create Manifold & Lattice (Not used yet in execution, but verifying path)
            let lattice = manager.doctor.synthesize_hardware_lattice();
            let _atom = match operator {
                OperatorTopology::Conv2d { n, c, h, w, k, r, s, stride, padding, .. } => {
                    Some(crate::core::manifold::ComputeAtom::from_conv2d(*n, *c, *h, *w, *k, *r, *s, *stride, *padding, 1))
                },
                OperatorTopology::Gemm { m, n, k, batch, .. } => {
                    Some(crate::core::manifold::ComputeAtom::from_gemm(*m, *n, *k, *batch))
                },
                _ => None,
            };

            steps.push(ExecutionStep::LaunchKernel {
                kernel_id,
                grid_size,
                block_size,
                shared_mem_bytes: shared_mem_bytes as u32,
                args,
            });
        }

        // 3. Build I/O Maps (Output Map contains Size for allocation)
        // Optimization: Only return leaf nodes (not consumed by others)
        let producers: HashSet<u64> = optimized_graph.dependencies.iter().map(|(p, _)| *p).collect();
        let mut output_map = HashMap::new();
         for node in &optimized_graph.operators {
            // If node is NOT a producer, it is an output (leaf)
            if !producers.contains(&node.op_id()) {
                if let Some(&offset) = arena_offsets.get(&node.op_id()) {
                    let size = Self::estimate_output_size(node);
                    output_map.insert(node.op_id(), (offset, size));
                }
            }
        }
        
        Ok(ExecutionPlan {
            steps,
            arena_size,
            output_map,
        })
    }
    
    // Helper methods
    fn optimize_graph(&self, graph: &GraphTopology, backend: DeviceBackend) -> Result<(PolicyDecision, GraphTopology), String> {
        // 1. Canonicalize & Optimize
        let mut optimized_graph = graph.clone();
        crate::policy::transform::canonicalize_graph(&mut optimized_graph);
        crate::core::optimizer::GraphOptimizer::optimize(&mut optimized_graph);

        let device = crate::core::device::DeviceProfile::from_backend(backend);
        let mut engine = crate::policy::standard::StandardPolicyEngine::new();
        let ctx = crate::policy::types::GraphContext {
            device: &device,
            graph: &optimized_graph,
        };

        let decision = crate::policy::scheduler::StandardScheduler::schedule(&mut engine, &ctx);
        Ok((decision, optimized_graph))
    }

    fn schedule(&self, decision: &PolicyDecision) -> Vec<u64> {
         // Sort by priority (which implies topo sort in standard scheduler)
         let mut policies: Vec<_> = decision.exec_policies.iter().collect();
         policies.sort_by_key(|p| p.operator_id);
         policies.into_iter().map(|p| p.operator_id).collect()
    }
    
    fn plan_memory(
        &self, 
        graph: &GraphTopology, 
        sorted: &[u64], 
        decision: &PolicyDecision
    ) -> Result<(HashMap<u64, usize>, usize), String> {
        let mut max_offset: usize = 0;
        let mut arena_offsets = HashMap::new();
        
        // Iterate through sorted ops and assign offsets based on decision or simple bump
        for &op_id in sorted {
            // Find policy for this op
            let exec_policy = decision.exec_policies.iter().find(|p| p.operator_id == op_id);
            if let Some(policy) = exec_policy {
                 if let Some(offset) = policy.memory_alias_hint.output_offset {
                    arena_offsets.insert(op_id, offset);
                    let op = graph.operators.iter().find(|o| o.op_id() == op_id);
                    if let Some(op) = op {
                        let size = Self::estimate_output_size(op);
                        let aligned_end = ((offset + size) + 255) & !255;
                        max_offset = max_offset.max(aligned_end);
                    }
                 }
            }
        }
        
        // Add TTG Workspace
        const TTG_WORKSPACE_SIZE: usize = 64 * 1024;
        max_offset += TTG_WORKSPACE_SIZE;
        
        // 10% padding
        let total_size = (max_offset as f64 * 1.1) as usize;

        Ok((arena_offsets, total_size))
    }


    fn compile_kernel(
        &self, 
        operator: &OperatorTopology, 
        tile_policy: Option<crate::policy::types::TilePolicy>,
        manager: &RuntimeManager,
        backend: DeviceBackend,
    ) -> Result<usize, String> {
        // Overwrite policy if Tuner has better config
        let mut final_policy = tile_policy.clone();
        
        if let Some(config) = manager.tuner.tune_operator(operator, backend, 5) {
             use crate::policy::types::TilePolicy;
             match operator {
                OperatorTopology::Gemm { .. } => {
                     final_policy = Some(TilePolicy::Gemm {
                         operator_id: operator.op_id(),
                         tile_shape: [config.m_tile, config.n_tile, config.k_tile],
                         variant: config.gemm_variant,
                         tiling_kind: crate::policy::types::TilingKind::Dense,
                         activity_pattern: crate::policy::types::ActivityPattern::AllActive,
                     });
                     eprintln!("[Compiler] 🚀 Applied Tuned Config for {}", operator.name());
                },
                OperatorTopology::Conv2d { .. } => {
                     final_policy = Some(TilePolicy::Conv {
                         operator_id: operator.op_id(),
                         tile_m: config.m_tile,
                         tile_n: config.n_tile,
                         tile_c: config.k_tile,
                         use_simd: matches!(config.instruction, crate::core::config::SpecializedInstruction::MetalSimdGroup),
                         use_double_buffer: config.double_buffer,
                     });
                     eprintln!("[Compiler] 🚀 Applied Tuned Config for {}", operator.name());
                },
                _ => {}
            }
        }

        let audit = manager.doctor.perform_polyhedral_audit(operator);
        let ir = self.build_ir(operator, final_policy.as_ref(), audit.map(|a| a.strategy))?;
        
        use crate::emitter::traits::Emitter;
        let (source, kernel_name) = match backend {
            DeviceBackend::Metal => {
                let emitter = crate::emitter::metal::MetalEmitter::detect();
                let src = emitter.generate_from_ir(&ir);
                let name = match operator {
                    OperatorTopology::Gemm { .. } => {
                         if let Some(crate::policy::types::TilePolicy::Gemm { variant: crate::core::config::GemmVariant::Tiled, .. }) = tile_policy {
                             "gemm_tiled_kernel"
                         } else {
                             "gemm_metal_kernel"
                         }
                    }, 
                    OperatorTopology::Attention { .. } => "flash_attention_v2_kernel",
                    OperatorTopology::Conv2d { .. } => "conv2d_implicit_gemm",
                    OperatorTopology::BatchNorm { .. } => "batchnorm_forward",
                    OperatorTopology::Elementwise { .. } => "elementwise_add",
                    OperatorTopology::Relu { .. } => "elementwise_relu",
                    OperatorTopology::GlobalAveragePool { .. } => "global_avg_pool_kernel",
                    OperatorTopology::Linear { .. } => "linear_kernel",
                    OperatorTopology::Softmax { .. } => "softmax_kernel",
                    _ => "kernel_main",
                };
                (src, name)
            },
            DeviceBackend::Cuda => {
                 let emitter = crate::emitter::cuda::CUDAEmitter::new();
                 let src = emitter.generate_from_ir(&ir);
                 (src, "kernel_main")
            },
            _ => return Err("Unsupported backend".into()),
        };

        // cast result to usize for Plan
        manager.compile(&source, kernel_name, backend)
            .map(|id| id.0 as usize) 
            // Assuming KernelId is a newtype around u64 or usize. Manager returns KernelId.
            // I need to check RuntimeManager::compile signature. It returns Result<KernelId, String>.
    }

    fn calculate_launch_dims(
        &self,
        operator: &OperatorTopology,
        tile_policy: Option<crate::policy::types::TilePolicy>,
    ) -> ((u32, u32, u32), (u32, u32, u32)) {
         match operator {
            OperatorTopology::Gemm { m, n, .. } => {
                if let Some(crate::policy::types::TilePolicy::Gemm { tile_shape, .. }) = tile_policy {
                     let m_tile = tile_shape[0] as u32;
                     let n_tile = tile_shape[1] as u32;
                     let grid = (
                         (*n as u32 + n_tile - 1) / n_tile,
                         (*m as u32 + m_tile - 1) / m_tile,
                         1
                     );
                     let block = (128, 1, 1);
                     (grid, block)
                } else {
                    ((1, 1, 1), (32, 1, 1)) // Default
                }
            },
            OperatorTopology::Conv2d { n, h, w, padding, r, s, stride, .. } => {
                let h_out = (h + 2 * padding - 1 * (r - 1) - 1) / stride + 1;
                let w_out = (w + 2 * padding - 1 * (s - 1) - 1) / stride + 1;
                 if let Some(crate::policy::types::TilePolicy::Conv { tile_m, tile_n, .. }) = tile_policy {
                     let _m_tile = tile_m as u32; 
                     let grid = (
                         (*n * h_out * w_out + 31) as u32 / 32, 
                         1,
                         1
                     );
                     (grid, (256, 1, 1))
                 } else {
                      ((1, 1, 1), (256, 1, 1))
                 }
            },
             OperatorTopology::Elementwise { n, .. } | OperatorTopology::Relu { n, .. } => {
                 let grid = ((*n as u32 + 255) / 256, 1, 1);
                 (grid, (256, 1, 1))
             },
             OperatorTopology::BatchNorm { n, c, h, w, .. } => {
                 let count = *n * *c * *h * *w;
                 let grid = ((count as u32 + 255) / 256, 1, 1);
                 (grid, (256, 1, 1))
             },
            _ => ((1, 1, 1), (32, 1, 1))
        }
    }

    fn build_ir(
        &self,
        operator: &OperatorTopology,
        tile_policy: Option<&crate::policy::types::TilePolicy>,
        polyhedral_strategy: Option<crate::core::polyhedral::TilingStrategy>,
    ) -> Result<crate::emitter::traits::UnifiedOpIR, String> {
        // ... (match blocks) ...
        let op_type = match operator {
            OperatorTopology::Gemm { m, n, k, batch, epilogue, .. } => {
                crate::emitter::traits::UnifiedOpType::Gemm { 
                    m: *m, n: *n, k: *k, batch: *batch,
                    epilogue: epilogue.clone(),
                }
            },
             OperatorTopology::Attention { b, s, h, d, .. } => {
                crate::emitter::traits::UnifiedOpType::FusedAttention {
                    b: *b, s: *s, d: *d, h: *h, dh: *d,
                    causal: false,
                }
            },
            OperatorTopology::Conv2d { n, c, h, w, k, r, s, stride, padding, epilogue, .. } => {
                crate::emitter::traits::UnifiedOpType::Conv2d {
                    n: *n as usize, c: *c as usize, h: *h as usize, w: *w as usize, k: *k as usize,
                    r: *r as usize, s: *s as usize, stride: *stride as usize, pad: *padding as usize,
                    dilation: 1,
                    layout: crate::core::config::LayoutPolicy::NHWC,
                    epilogue: epilogue.clone(),
                }
            },
             OperatorTopology::BatchNorm { n, c, h, w, epsilon, momentum, .. } => {
                crate::emitter::traits::UnifiedOpType::BatchNorm {
                    n: *n, c: *c, h: *h, w: *w, epsilon: *epsilon, momentum: *momentum 
                }
            },
            OperatorTopology::Elementwise { kind, n, .. } => {
                 let op_type = match kind.as_str() {
                     "Add" => crate::core::op::ElementwiseType::Add,
                     "Mul" => crate::core::op::ElementwiseType::Mul,
                     _ => crate::core::op::ElementwiseType::Add,
                 };
                 crate::emitter::traits::UnifiedOpType::Elementwise { op_type, n: *n }
            },
            OperatorTopology::GlobalAveragePool { n, c, h, w, .. } => {
                 crate::emitter::traits::UnifiedOpType::GlobalAveragePool {
                    n: *n, c: *c, h: *h, w: *w,
                }
            },
            OperatorTopology::Linear { batch, m, n, k, epilogue, .. } => {
                crate::emitter::traits::UnifiedOpType::Linear {
                    batch: *batch, m: *m, n: *n, k: *k,
                    epilogue: epilogue.clone(),
                }
            },
             OperatorTopology::Softmax { axis, .. } => {
                crate::emitter::traits::UnifiedOpType::Softmax { 
                    axis: *axis,
                    dim_size: 1024, stride: 1, total_elements: 1024 * 1024 
                }
            },
            OperatorTopology::Relu { n, .. } => {
                 crate::emitter::traits::UnifiedOpType::Elementwise { 
                     op_type: crate::core::op::ElementwiseType::Relu, 
                     n: *n 
                 }
            },
            _ => return Err("Unsupported op type for IR".into()),
        };
        
        let tiling = match tile_policy {
             Some(crate::policy::types::TilePolicy::Gemm { tile_shape, variant, .. }) => {
                crate::PipelineConfig {
                    m_tile: tile_shape[0], n_tile: tile_shape[1], k_tile: tile_shape[2],
                    gemm_variant: *variant, ..Default::default()
                }
            },
             Some(crate::policy::types::TilePolicy::Attention { qk_tile, variant, .. }) => {
                crate::PipelineConfig {
                    m_tile: qk_tile.0, n_tile: qk_tile.1, k_tile: 64,
                    attention_variant: *variant, ..Default::default()
                }
            },
            Some(crate::policy::types::TilePolicy::Conv { tile_m, tile_n, tile_c, use_simd, use_double_buffer, .. }) => {
                crate::PipelineConfig {
                    m_tile: *tile_m, n_tile: *tile_n, k_tile: *tile_c,
                    instruction: if *use_simd { crate::core::config::SpecializedInstruction::MetalSimdGroup } else { crate::core::config::SpecializedInstruction::None },
                    double_buffer: *use_double_buffer,
                    ..Default::default()
                }
            },
            _ => crate::PipelineConfig::default(),
        };
 
        Ok(crate::emitter::traits::UnifiedOpIR {
            op_type,
            precison: "fp16".to_string(),
            tiling,
            conv_magic_strategy: None,
            polyhedral_strategy,
        })
    }

    fn build_arguments(
        &self,
        op: &OperatorTopology,
        graph: &GraphTopology,
        offsets: &HashMap<u64, usize>,
        op_id: u64
    ) -> Result<Vec<KernelArgSpec>, String> {
        let mut args = Vec::new();
        
        // 1. Resolve Producers (Inputs)
        let producers: Vec<u64> = graph.dependencies.iter()
            .filter(|(_, consumer)| *consumer == op_id)
            .map(|(producer, _)| *producer)
            .collect();
        
        // Identify Epilogue Inputs
        let mut epilogue_input_ids = Vec::new();
        match op {
            OperatorTopology::Conv2d { epilogue, .. } | OperatorTopology::Linear { epilogue, .. } | OperatorTopology::Gemm { epilogue, .. } => {
                for epi in epilogue {
                    match epi {
                         crate::core::op::EpilogueOp::BatchNorm { gamma_id, beta_id, mean_id, var_id, .. } => {
                             epilogue_input_ids.push(*gamma_id);
                             epilogue_input_ids.push(*beta_id);
                             epilogue_input_ids.push(*mean_id);
                             epilogue_input_ids.push(*var_id);
                         },
                         crate::core::op::EpilogueOp::ResidualAdd { residual_ptr } => {
                             epilogue_input_ids.push(*residual_ptr as u64);
                         },
                         _ => {}
                    }
                }
            },
            _ => {}
        }

        // Core Producers: Input, Weight (excluding epilogue params)
        let mut core_producers: Vec<u64> = producers.iter().cloned()
            .filter(|p| !epilogue_input_ids.contains(p))
            .collect();
        core_producers.sort(); // Ensure deterministic order (Input < Weight usually)

        for p_id in core_producers {
             let producer_node = graph.operators.iter().find(|n| n.op_id() == p_id)
                 .ok_or_else(|| format!("Producer {} not found", p_id))?;
                 
             if let OperatorTopology::Input { .. } = producer_node {
                 args.push(KernelArgSpec::ExternalInput(p_id));
             } else {
                 if let Some(&offset) = offsets.get(&p_id) {
                     args.push(KernelArgSpec::ArenaOffset(offset));
                 } else {
                     return Err(format!("Producer {} has no arena offset", p_id));
                 }
             }
        }
        
        // 2. Output Buffer (Self)
        if let Some(&offset) = offsets.get(&op_id) {
            args.push(KernelArgSpec::ArenaOffset(offset));
        } else {
             return Err(format!("Op {} has no arena offset", op_id));
        }
        
        // 3. Op-Specific Scalars/Params
        match op {
            OperatorTopology::Elementwise { n, .. } | OperatorTopology::Relu { n, .. } => {
                args.push(KernelArgSpec::ScalarInt(*n as i32));
            },
            OperatorTopology::BatchNorm { epsilon, .. } => {
                args.push(KernelArgSpec::ScalarFloat(*epsilon));
            },
             OperatorTopology::Gemm { .. } => {
                  // Standard Gemm args 
             },
             OperatorTopology::Conv2d { n, h, w, c, k, r, s, stride, padding, .. } => {
                  let h_out = (h + 2 * padding - 1 * (r - 1) - 1) / stride + 1;
                  let w_out = (w + 2 * padding - 1 * (s - 1) - 1) / stride + 1;
                  
                  let (hw_m, hw_s) = crate::emitter::conv::magic_u32(h_out * w_out);
                  let (w_m, w_s) = crate::emitter::conv::magic_u32(w_out);
                  let (sic_m, sic_s) = crate::emitter::conv::magic_u32(*s * *c);
                  let (c_m, c_s) = crate::emitter::conv::magic_u32(*c);

                  let params = crate::runtime::manager::MetalConvParams {
                         batch: *n, h_in: *h, w_in: *w, c_in: *c, k_out: *k,
                         h_out, w_out, r_sz: *r, s_sz: *s,
                         stride: *stride, pad: *padding, dilation: 1,
                         hw_m, hw_s,
                         w_m, w_s,
                         sic_m, sic_s,
                         c_m, c_s,
                     };
                     let mut bytes = Vec::new();
                     let ptr = &params as *const crate::runtime::manager::MetalConvParams as *const u8;
                     let len = std::mem::size_of::<crate::runtime::manager::MetalConvParams>();
                     unsafe { bytes.extend_from_slice(std::slice::from_raw_parts(ptr, len)); }
                     args.push(KernelArgSpec::Bytes(bytes));

                     // L1/L2 Maps (Placeholders for now if not used, or point to existing input?)
                     // Metal Conv expects: buffer(4) l1, buffer(5) l2. 
                     // If we don't provide them, validation fails.
                     // IMPORTANT: We must push valid placeholders or actual data.
                     // Since `network_bench` doesn't populate l1/l2, we can push a dummy offset (e.g. 0).
                     // However, ArenaOffset(0) points to buffer 0 (Input)? No, it points to offset 0 in Arena.
                     // The Executor binds Arena Buffer + Offset.
                     // Let's assume offset 0 is safe/readable.
                     args.push(KernelArgSpec::ArenaOffset(0)); // L1 Map
                     args.push(KernelArgSpec::ArenaOffset(0)); // L2 Table
              },
              OperatorTopology::Linear { m, n, k, batch, .. } => {
                  args.push(KernelArgSpec::ScalarInt(*m as i32));
                  args.push(KernelArgSpec::ScalarInt(*n as i32));
                  args.push(KernelArgSpec::ScalarInt(*k as i32));
                  args.push(KernelArgSpec::ScalarInt(*batch as i32));
              },
             _ => {}
        }

        // 4. Epilogue Inputs (In order of epilogue vector)
        // MUST match `generate_epilogue_code` iteration order.
        for id in epilogue_input_ids { // This list was built in order of epilogue
             let producer_node = graph.operators.iter().find(|n| n.op_id() == id)
                 .ok_or_else(|| format!("Epilogue input {} not found", id))?;
                 
             if let OperatorTopology::Input { .. } = producer_node {
                 args.push(KernelArgSpec::ExternalInput(id));
             } else {
                 if let Some(&offset) = offsets.get(&id) {
                     args.push(KernelArgSpec::ArenaOffset(offset));
                 } else {
                     return Err(format!("Epilogue input {} has no arena offset", id));
                 }
             }
        }
        
        Ok(args)
    }

    fn estimate_output_size(op: &OperatorTopology) -> usize {
        match op {
            OperatorTopology::Gemm { m, n, batch, .. } => (*m as usize) * (*n as usize) * (*batch as usize) * 4,
            OperatorTopology::Attention { b, s, h, d, .. } => (*b as usize) * (*h as usize) * (*s as usize) * (*d as usize) * 2,
            OperatorTopology::Conv2d { n, k, h, w, .. } => (*n as usize) * (*k as usize) * (*h as usize) * (*w as usize) * 4,
            OperatorTopology::BatchNorm { n, c, h, w, .. } => (*n as usize) * (*c as usize) * (*h as usize) * (*w as usize) * 4,
            OperatorTopology::Elementwise { n, .. } | OperatorTopology::Relu { n, .. } => (*n as usize) * 4,
            OperatorTopology::GlobalAveragePool { n, c, .. } => (*n as usize) * (*c as usize) * 4,
            OperatorTopology::Linear { m, n, batch, .. } => (*m as usize) * (*n as usize) * (*batch as usize) * 4,
            _ => 64 * 1024 * 1024,
        }
    }
}
