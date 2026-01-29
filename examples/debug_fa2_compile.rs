use tracea::core::config::PipelineConfig;
use tracea::kernels::attention::cuda_emitter::FlashAttentionEmitter;
use tracea::runtime::manager::{RuntimeManager, DeviceBackend};
use std::sync::Arc;

fn main() {
    let mut config = PipelineConfig::new(2, 64, 64, 32);
    config.force_num_warps = Some(4);
    
    let emitter = FlashAttentionEmitter::new(config);
    let source = emitter.generate_kernel(8, 64, false); // H=8, D=64

    println!("Generated Source:\n{}", source);

    let runtime = RuntimeManager::new();
    println!("Compiling...");
    match runtime.compile(&source, "flash_attention_v2_kernel", DeviceBackend::Cuda) {
        Ok(_) => println!("Compilation Successful!"),
        Err(e) => println!("Compilation Failed: {}", e),
    }
}
