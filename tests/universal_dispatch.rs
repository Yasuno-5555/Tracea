use tracea::emitter::universal::UniversalEmitter;
use tracea::emitter::traits::{UnifiedOpIR, UnifiedOpType};
use tracea::runtime::manager::DeviceBackend;
use tracea::PipelineConfig;

#[test]
fn test_universal_emitter_dispatch() {
    let ir = UnifiedOpIR {
        op_type: UnifiedOpType::Gemm {
            m: 16,
            n: 16,
            k: 16,
            batch: 1,
            epilogue: vec![],
        },
        precison: "f16".to_string(),
        tiling: PipelineConfig::new(1, 16, 16, 16),
        conv_magic_strategy: None,
        polyhedral_strategy: None,
    };

    // 1. Test Metal Dispatch
    let emitter_metal = UniversalEmitter::new(DeviceBackend::Metal);
    let code_metal = emitter_metal.generate(ir.clone()).expect("Metal dispatch failed");
    println!("Metal Code Generated (sample):\n{}", &code_metal[0..150]);
    assert!(code_metal.contains("kernel void"), "Metal output must contain 'kernel void'");
    assert!(code_metal.contains("[[buffer("), "Metal output must contain parameter buffers decoration");

    // 2. Test CUDA Dispatch
    let emitter_cuda = UniversalEmitter::new(DeviceBackend::Cuda);
    let code_cuda = emitter_cuda.generate(ir.clone()).expect("CUDA dispatch failed");
    println!("CUDA Code Generated (sample):\n{}", &code_cuda[0..150]);
    assert!(code_cuda.contains("extern \"C\" __global__ void") || code_cuda.contains("__global__"), "CUDA output must contain '__global__'");

    // 3. Test CPU Dispatch (Fallback / verification)
    let emitter_cpu = UniversalEmitter::new(DeviceBackend::Cpu);
    let code_cpu = emitter_cpu.generate(ir.clone());
    match code_cpu {
        Ok(code) => {
            assert!(code.contains("CPU") || code.contains("static"), "CPU fallback must contain functional definition or fallback comment");
        }
        Err(e) => {
            println!("CPU returned expected emission error for GEMM: {:?}", e);
        }
    }
}
