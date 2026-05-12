use tracea::emitter::traits::{Emitter, UnifiedOpIR, UnifiedOpType};
use tracea::emitter::cuda::CUDAEmitter;
use tracea::core::config::PipelineConfig;

#[test]
fn test_cuda_emitter_gemm_generation() {
    let emitter = CUDAEmitter::new();
    
    // M=128/N=128/K=128, tile=64x64x16
    let config = PipelineConfig::new(2, 64, 64, 16);
    
    let ir = UnifiedOpIR {
        op_type: UnifiedOpType::Gemm {
            m: 128,
            n: 128,
            k: 128,
            batch: 1,
            epilogue: vec![],
        },
        precison: "f16".to_string(),
        tiling: config,
        conv_magic_strategy: None,
        polyhedral_strategy: None,
    };
    
    let code = emitter.generate_from_ir(&ir).unwrap();
    
    assert!(code.contains("extern \"C\" __global__"), "Missing global keyword");
    assert!(code.contains("__syncthreads()"), "Missing syncthreads");
    assert!(code.contains("gemm_mma_kernel"), "Missing kernel name");
    assert!(code.contains("#define MT 64"), "Tile M mismatch");
    assert!(code.contains("#define NT 64"), "Tile N mismatch");
    assert!(code.contains("#define KT 16"), "Tile K mismatch");
}

#[test]
fn test_cuda_emitter_invalid_tile_size_no_panic() {
    let emitter = CUDAEmitter::new();
    
    // mt=40 is not divisible by 16 (40 % 16 != 0) but warp partitioning is valid (mt / warp_m >= 16)
    // We verify this does not panic
    let config = PipelineConfig::new(2, 40, 64, 16);
    
    let ir = UnifiedOpIR {
        op_type: UnifiedOpType::Gemm {
            m: 128,
            n: 128,
            k: 128,
            batch: 1,
            epilogue: vec![],
        },
        precison: "f16".to_string(),
        tiling: config,
        conv_magic_strategy: None,
        polyhedral_strategy: None,
    };
    
    let code = emitter.generate_from_ir(&ir).unwrap();
    assert!(code.contains("#define MT 40"), "Should contain defined MT 40");
}

#[test]
fn test_cuda_emitter_invalid_warp_partitioning_returns_err() {
    let emitter = CUDAEmitter::new();
    
    // mt = 16, warp_m = consumers = 4 (force_num_warps = Some(5) => consumers = 4 => warp_m = 4)
    // mt / warp_m = 16 / 4 = 4 < 16, which violates the constraint
    let mut config = PipelineConfig::new(2, 16, 64, 16);
    config.force_num_warps = Some(5);
    
    let ir = UnifiedOpIR {
        op_type: UnifiedOpType::Gemm {
            m: 128,
            n: 128,
            k: 128,
            batch: 1,
            epilogue: vec![],
        },
        precison: "f16".to_string(),
        tiling: config,
        conv_magic_strategy: None,
        polyhedral_strategy: None,
    };
    
    let result = emitter.emit(&ir);
    assert!(result.is_err());
    if let Err(tracea::emitter::traits::EmissionError::InvalidTileConfiguration { reason }) = result {
        assert!(reason.contains("Invalid Warp Partitioning"), "Unexpected error reason: {}", reason);
    } else {
        panic!("Expected InvalidTileConfiguration error, got: {:?}", result);
    }
}
