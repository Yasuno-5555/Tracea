use std::fs::write;
use tracea::core::config::PipelineConfig;
use tracea::emitter::traits::{UnifiedOpIR, UnifiedOpType, Emitter};

fn dump(label: &str, config: PipelineConfig, m: u32, n: u32, k: u32) {
    let ir = UnifiedOpIR {
        op_type: UnifiedOpType::Gemm { m, n, k, batch: 1, epilogue: vec![] },
        precison: "fp16".to_string(),
        tiling: config,
        conv_magic_strategy: None,
        polyhedral_strategy: None,
    };
    let emitter = tracea::emitter::metal::MetalEmitter::detect();
    let source = emitter.generate_from_ir(&ir).expect("Codegen failed");
    write(format!("msl_{}.metal", label), &source).unwrap();
    println!("{}: {} bytes", label, source.len());
}

fn main() {
    for &(name, mt, nt, kt, stages, warps, db) in &[
        ("tuned",  32, 32, 16, 2, 4, true),
        ("wide64", 64, 64, 16, 2, 4, true),
        ("naive",  64, 64, 32, 2, 4, false),
    ] {
        let mut cfg = PipelineConfig::new(stages, mt, nt, kt);
        cfg.double_buffer = db;
        cfg.force_num_warps = Some(warps);
        dump(name, cfg, 2048, 2048, 2048);
    }
}
