use std::collections::HashMap;
use crate::runtime::manager::{RuntimeManager, BufferId, DeviceBackend, MetalConvParams};
use crate::policy::types::{GraphTopology, OperatorTopology, TilePolicy, ExecPolicy, MemoryAliasPolicy};
use crate::policy::engine::PolicyEngine;
use crate::policy::standard::StandardPolicyEngine;
use crate::policy::scheduler::StandardScheduler;
use crate::core::device::DeviceProfile;
use crate::core::op::EpilogueOp;
use crate::emitter::traits::UnifiedOpIR;

impl RuntimeManager {
    pub fn execute_graph(
        &self,
        graph: &GraphTopology,
        input_buffers: &HashMap<u64, BufferId>,
        backend: DeviceBackend,
    ) -> Result<HashMap<u64, BufferId>, String> {
        // 0. Canonicalize Graph & Optimize for Fusion
        let mut graph = graph.clone();
        crate::policy::transform::canonicalize_graph(&mut graph);
        crate::core::optimizer::GraphOptimizer::optimize(&mut graph);

        // 1. Get Device Profile
        let device = DeviceProfile::from_backend(backend);

        // 2. Policy Decision
        let mut engine = StandardPolicyEngine::new();
        let ctx = crate::policy::types::GraphContext {
            device: &device,
            graph: &graph,
        };

        // 3. Schedule
        let decision = StandardScheduler::schedule(&mut engine, &ctx);

        // 4. Memory Pool (alias-aware allocation)
        let mut memory_pool: HashMap<usize, BufferId> = HashMap::new();
        let mut output_buffers: HashMap<u64, BufferId> = HashMap::new();

        // 5. Execute Loop
        let mut sorted_policies: Vec<_> = decision.exec_policies.iter().collect();
        sorted_policies.sort_by_key(|p| p.operator_id);

        for exec_policy in sorted_policies {
            let op_id = exec_policy.operator_id;
            
            let tile_policy = decision.tile_policies.iter()
                .find(|t| t.operator_id() == op_id);
            let operator = graph.operators.iter()
                .find(|o| o.op_id() == op_id);

            let (tile_policy, operator) = match (tile_policy, operator) {
                (Some(t), Some(o)) => (t, o),
                _ => continue,
            };

            if let OperatorTopology::Input { op_id, .. } = operator {
                if let Some(&buf) = input_buffers.get(op_id) {
                    output_buffers.insert(*op_id, buf);
                    continue;
                } else {
                    return Err(format!("Input node {} missing from external_inputs", op_id));
                }
            }

            let buf_id = self.resolve_buffer_with_alias(
                op_id,
                &exec_policy.memory_alias_hint,
                &mut memory_pool,
                Self::estimate_output_size(operator),
                backend,
            )?;
            output_buffers.insert(op_id, buf_id);

            let layout = crate::runtime::ttg_builder::TTGBuilder::from_policy(operator, tile_policy);
            
            let ir = UnifiedOpIR {
                op_type: match operator {
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
                    OperatorTopology::Softmax { axis, .. } => {
                        crate::emitter::traits::UnifiedOpType::Softmax { 
                            axis: *axis,
                            dim_size: 1024, stride: 1, total_elements: 1024 * 1024 
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
                    _ => continue,
                },
                precison: "fp16".to_string(),
                tiling: match tile_policy {
                    TilePolicy::Gemm { tile_shape, variant, .. } => {
                        crate::PipelineConfig {
                            m_tile: tile_shape[0],
                            n_tile: tile_shape[1],
                            k_tile: tile_shape[2],
                            gemm_variant: *variant,
                            ..Default::default()
                        }
                    },
                    TilePolicy::Attention { qk_tile, variant, .. } => {
                        crate::PipelineConfig {
                            m_tile: qk_tile.0,
                            n_tile: qk_tile.1,
                            k_tile: 64,
                            attention_variant: *variant,
                            ..Default::default()
                        }
                    },
                    _ => {
                        let mut config = crate::PipelineConfig::default();
                        config.m_tile = 64; config.n_tile = 64; config.k_tile = 32;
                        config
                    },
                },
                conv_magic_strategy: None,
            };

            use crate::emitter::traits::Emitter;
            let (source, kernel_name) = match backend {
                DeviceBackend::Metal => {
                    let emitter = crate::emitter::metal::MetalEmitter::detect();
                    let src = emitter.generate_from_ir(&ir);
                    let name = match operator {
                        OperatorTopology::Gemm { .. } => {
                            match ir.tiling.gemm_variant {
                                crate::core::config::GemmVariant::Tiled => "gemm_tiled_kernel",
                                _ => "gemm_metal_kernel",
                            }
                        },
                        OperatorTopology::Attention { .. } => "flash_attention_v2_kernel",
                        OperatorTopology::Conv2d { .. } => "conv2d_implicit_gemm",
                        OperatorTopology::BatchNorm { .. } => "batchnorm_forward",
                        OperatorTopology::Elementwise { .. } => "elementwise_add",
                        OperatorTopology::Relu { .. } => "elementwise_relu",
                        OperatorTopology::GlobalAveragePool { .. } => "global_avg_pool_kernel",
                        OperatorTopology::Linear { .. } => "linear_kernel",
                        _ => "kernel_main",
                    };
                    (src, name)
                },
                DeviceBackend::Cuda => {
                    let emitter = crate::emitter::cuda::CUDAEmitter::new();
                    let src = emitter.generate_from_ir(&ir);
                    let name = match operator {
                        OperatorTopology::Gemm { .. } => "gemm_mma_kernel",
                        OperatorTopology::Attention { .. } => "flash_attention_v2_kernel",
                        _ => "kernel_main",
                    };
                    (src, name)
                },
                _ => return Err("Unsupported backend for JIT".into()),
            };

            let kernel_id = self.compile(&source, kernel_name, backend)?;

            let mut args = Vec::new();
            match operator {
                OperatorTopology::Gemm { .. } => {
                    let inputs = self.get_op_inputs(op_id, &graph, input_buffers, &output_buffers);
                    if inputs.len() >= 2 {
                        args.push(crate::runtime::manager::KernelArg::Buffer(inputs[0]));
                        args.push(crate::runtime::manager::KernelArg::Buffer(inputs[1]));
                    }
                    args.push(crate::runtime::manager::KernelArg::Buffer(buf_id));

                    if let OperatorTopology::Gemm { epilogue, .. } = operator {
                        for op in epilogue {
                            match op {
                                EpilogueOp::BatchNorm { gamma_id, beta_id, mean_id, var_id, .. } => {
                                    let g_buf = input_buffers.get(gamma_id).or(output_buffers.get(gamma_id)).cloned().unwrap_or(crate::runtime::manager::BufferId(0));
                                    let b_buf = input_buffers.get(beta_id).or(output_buffers.get(beta_id)).cloned().unwrap_or(crate::runtime::manager::BufferId(0));
                                    let m_buf = input_buffers.get(mean_id).or(output_buffers.get(mean_id)).cloned().unwrap_or(crate::runtime::manager::BufferId(0));
                                    let v_buf = input_buffers.get(var_id).or(output_buffers.get(var_id)).cloned().unwrap_or(crate::runtime::manager::BufferId(0));
                                    args.push(crate::runtime::manager::KernelArg::Buffer(g_buf));
                                    args.push(crate::runtime::manager::KernelArg::Buffer(b_buf));
                                    args.push(crate::runtime::manager::KernelArg::Buffer(m_buf));
                                    args.push(crate::runtime::manager::KernelArg::Buffer(v_buf));
                                }
                                EpilogueOp::ResidualAdd { residual_ptr } => {
                                    let res_buf = input_buffers.get(&(*residual_ptr as u64)).or(output_buffers.get(&(*residual_ptr as u64))).cloned().unwrap_or(crate::runtime::manager::BufferId(0));
                                    args.push(crate::runtime::manager::KernelArg::Buffer(res_buf));
                                }
                                EpilogueOp::BiasAdd { bias_ptr } => {
                                    let b_buf = input_buffers.get(&(*bias_ptr as u64)).or(output_buffers.get(&(*bias_ptr as u64))).cloned().unwrap_or(crate::runtime::manager::BufferId(0));
                                    args.push(crate::runtime::manager::KernelArg::Buffer(b_buf));
                                }
                                _ => {}
                            }
                        }
                    }
                },
                OperatorTopology::Attention { d, .. } => {
                    let inputs = self.get_op_inputs(op_id, &graph, input_buffers, &output_buffers);
                    if inputs.len() >= 3 {
                        args.push(crate::runtime::manager::KernelArg::Buffer(inputs[0]));
                        args.push(crate::runtime::manager::KernelArg::Buffer(inputs[1]));
                        args.push(crate::runtime::manager::KernelArg::Buffer(inputs[2]));
                    }
                    args.push(crate::runtime::manager::KernelArg::Buffer(buf_id));
                    args.push(crate::runtime::manager::KernelArg::Float(1.0 / (*d as f32).sqrt()));
                },
                OperatorTopology::Conv2d { n, h, w, c, k, r, s, stride, padding, .. } => {
                    let inputs = self.get_op_inputs(op_id, &graph, input_buffers, &output_buffers);
                    if inputs.len() >= 2 {
                         args.push(crate::runtime::manager::KernelArg::Buffer(inputs[0]));
                         args.push(crate::runtime::manager::KernelArg::Buffer(inputs[1]));
                    }
                    args.push(crate::runtime::manager::KernelArg::Buffer(buf_id));
                    
                    let h_out = (h + 2 * padding - 1 * (r - 1) - 1) / stride + 1;
                    let w_out = (w + 2 * padding - 1 * (s - 1) - 1) / stride + 1;

                    let epilogue = match operator {
                        OperatorTopology::Conv2d { epilogue, .. } => epilogue,
                        _ => &vec![],
                    };
                    
                    for op in epilogue {
                        match op {
                            EpilogueOp::BatchNorm { gamma_id, beta_id, mean_id, var_id, .. } => {
                                let g_buf = input_buffers.get(gamma_id).or(output_buffers.get(gamma_id)).cloned().unwrap_or(crate::runtime::manager::BufferId(0));
                                let b_buf = input_buffers.get(beta_id).or(output_buffers.get(beta_id)).cloned().unwrap_or(crate::runtime::manager::BufferId(0));
                                let m_buf = input_buffers.get(mean_id).or(output_buffers.get(mean_id)).cloned().unwrap_or(crate::runtime::manager::BufferId(0));
                                let v_buf = input_buffers.get(var_id).or(output_buffers.get(var_id)).cloned().unwrap_or(crate::runtime::manager::BufferId(0));
                                args.push(crate::runtime::manager::KernelArg::Buffer(g_buf));
                                args.push(crate::runtime::manager::KernelArg::Buffer(b_buf));
                                args.push(crate::runtime::manager::KernelArg::Buffer(m_buf));
                                args.push(crate::runtime::manager::KernelArg::Buffer(v_buf));
                            }
                            EpilogueOp::ResidualAdd { residual_ptr } => {
                                let res_buf = input_buffers.get(&(*residual_ptr as u64)).or(output_buffers.get(&(*residual_ptr as u64))).cloned().unwrap_or(crate::runtime::manager::BufferId(0));
                                args.push(crate::runtime::manager::KernelArg::Buffer(res_buf));
                            }
                            EpilogueOp::BiasAdd { bias_ptr } => {
                                let b_buf = input_buffers.get(&(*bias_ptr as u64)).or(output_buffers.get(&(*bias_ptr as u64))).cloned().unwrap_or(crate::runtime::manager::BufferId(0));
                                args.push(crate::runtime::manager::KernelArg::Buffer(b_buf));
                            }
                             EpilogueOp::BiasAddSiLU { bias_ptr } => {
                                let b_buf = input_buffers.get(&(*bias_ptr as u64)).or(output_buffers.get(&(*bias_ptr as u64))).cloned().unwrap_or(crate::runtime::manager::BufferId(0));
                                args.push(crate::runtime::manager::KernelArg::Buffer(b_buf));
                            }
                            _ => {}
                        }
                    }

                    let params = MetalConvParams {
                        batch: *n, h_in: *h, w_in: *w, c_in: *c, k_out: *k,
                        h_out, w_out, r_sz: *r, s_sz: *s,
                        stride: *stride, pad: *padding, dilation: 1,
                    };

                    let mut bytes = Vec::new();
                    let ptr = &params as *const MetalConvParams as *const u8;
                    let len = std::mem::size_of::<MetalConvParams>();
                    unsafe { bytes.extend_from_slice(std::slice::from_raw_parts(ptr, len)); }
                    args.push(crate::runtime::manager::KernelArg::Bytes(bytes));
                },
                OperatorTopology::BatchNorm { epsilon, .. } => {
                    let inputs = self.get_op_inputs(op_id, &graph, input_buffers, &output_buffers);
                    for buf in inputs { args.push(crate::runtime::manager::KernelArg::Buffer(buf)); }
                    args.push(crate::runtime::manager::KernelArg::Buffer(buf_id));
                    args.push(crate::runtime::manager::KernelArg::Float(*epsilon));
                },
                OperatorTopology::Elementwise { n, .. } | OperatorTopology::Relu { n, .. } => {
                    let inputs = self.get_op_inputs(op_id, &graph, input_buffers, &output_buffers);
                    for buf in inputs { args.push(crate::runtime::manager::KernelArg::Buffer(buf)); }
                    args.push(crate::runtime::manager::KernelArg::Buffer(buf_id));
                    args.push(crate::runtime::manager::KernelArg::Int(*n as i32));
                },
                OperatorTopology::GlobalAveragePool { .. } => {
                    let inputs = self.get_op_inputs(op_id, &graph, input_buffers, &output_buffers);
                    for buf in inputs { args.push(crate::runtime::manager::KernelArg::Buffer(buf)); }
                    args.push(crate::runtime::manager::KernelArg::Buffer(buf_id));
                },
                OperatorTopology::Linear { m, n, k, batch, .. } => {
                    let inputs = self.get_op_inputs(op_id, &graph, input_buffers, &output_buffers);
                    for buf in inputs { args.push(crate::runtime::manager::KernelArg::Buffer(buf)); }
                    args.push(crate::runtime::manager::KernelArg::Buffer(buf_id));

                    if let OperatorTopology::Linear { epilogue, .. } = operator {
                        for op in epilogue {
                            match op {
                                EpilogueOp::BatchNorm { gamma_id, beta_id, mean_id, var_id, .. } => {
                                    let g_buf = input_buffers.get(gamma_id).or(output_buffers.get(gamma_id)).cloned().unwrap_or(crate::runtime::manager::BufferId(0));
                                    let b_buf = input_buffers.get(beta_id).or(output_buffers.get(beta_id)).cloned().unwrap_or(crate::runtime::manager::BufferId(0));
                                    let m_buf = input_buffers.get(mean_id).or(output_buffers.get(mean_id)).cloned().unwrap_or(crate::runtime::manager::BufferId(0));
                                    let v_buf = input_buffers.get(var_id).or(output_buffers.get(var_id)).cloned().unwrap_or(crate::runtime::manager::BufferId(0));
                                    args.push(crate::runtime::manager::KernelArg::Buffer(g_buf));
                                    args.push(crate::runtime::manager::KernelArg::Buffer(b_buf));
                                    args.push(crate::runtime::manager::KernelArg::Buffer(m_buf));
                                    args.push(crate::runtime::manager::KernelArg::Buffer(v_buf));
                                }
                                EpilogueOp::ResidualAdd { residual_ptr } => {
                                    let res_buf = input_buffers.get(&(*residual_ptr as u64)).or(output_buffers.get(&(*residual_ptr as u64))).cloned().unwrap_or(crate::runtime::manager::BufferId(0));
                                    args.push(crate::runtime::manager::KernelArg::Buffer(res_buf));
                                }
                                _ => {}
                            }
                        }
                    }

                    args.push(crate::runtime::manager::KernelArg::Int(*m as i32));
                    args.push(crate::runtime::manager::KernelArg::Int(*n as i32));
                    args.push(crate::runtime::manager::KernelArg::Int(*k as i32));
                    args.push(crate::runtime::manager::KernelArg::Int(*batch as i32));
                },
                _ => {}
            }

            if !args.is_empty() {
                self.launch_with_policy(
                    kernel_id,
                    args,
                    operator,
                    tile_policy,
                    exec_policy,
                    vec![],
                    backend,
                )?;
            }
        }

        Ok(output_buffers)
    }

    fn get_op_inputs(
        &self,
        op_id: u64,
        graph: &GraphTopology,
        external_inputs: &HashMap<u64, BufferId>,
        internal_outputs: &HashMap<u64, BufferId>,
    ) -> Vec<BufferId> {
        let mut producers: Vec<u64> = graph.dependencies.iter()
            .filter(|(_, consumer)| *consumer == op_id)
            .map(|(producer, _)| *producer)
            .collect();
        producers.sort();

        let mut inputs = Vec::new();
        for p_id in producers {
            if let Some(&buf) = internal_outputs.get(&p_id) { inputs.push(buf); }
        }

        if inputs.is_empty() {
            if let Some(&buf) = external_inputs.get(&op_id) { inputs.push(buf); }
        }
        inputs
    }

    fn resolve_buffer_with_alias(
        &self,
        _op_id: u64,
        alias_hint: &MemoryAliasPolicy,
        pool: &mut HashMap<usize, BufferId>,
        size: usize,
        backend: DeviceBackend,
    ) -> Result<BufferId, String> {
        if let Some(offset) = alias_hint.output_offset {
            if let Some(&existing_id) = pool.get(&offset) { return Ok(existing_id); }
            let new_id = self.alloc(size, backend)?;
            pool.insert(offset, new_id);
            Ok(new_id)
        } else {
            self.alloc(size, backend)
        }
    }

    fn estimate_output_size(op: &OperatorTopology) -> usize {
        match op {
            OperatorTopology::Gemm { m, n, batch, .. } => (*m as usize) * (*n as usize) * (*batch as usize) * 4,
            OperatorTopology::Attention { b, s, h, d, .. } => (*b as usize) * (*h as usize) * (*s as usize) * (*d as usize) * 2,
            OperatorTopology::Conv2d { n, k, h, w, .. } => (*n as usize) * (*k as usize) * (*h as usize) * (*w as usize) * 4,
            OperatorTopology::Linear { m, n, batch, .. } => (*m as usize) * (*n as usize) * (*batch as usize) * 4,
            _ => 1024 * 1024,
        }
    }
}
