
use tracea::frontend::auto_architect::{AutoArchitect, ModelType};
use safetensors::SafeTensors;
use std::fs;
use memmap2::MmapOptions;

fn main() {
    println!("Verifying Meta-Architect...");

    // 1. Verify Llama (Universal Transformer)
    let file = fs::File::open("llama_dummy.safetensors").expect("Failed to open llama_dummy.safetensors");
    let mmap = unsafe { MmapOptions::new().map(&file).expect("Failed to mmap") };
    let weights = SafeTensors::deserialize(&mmap).expect("Failed to deserialize Llama weights");
    
    let model_type = AutoArchitect::identify(&weights);
    println!("Identified llama_dummy.safetensors as: {:?}", model_type);

    if let ModelType::UniversalTransformer(config) = model_type {
        println!("  - Detected Config: {:?}", config);
        // Verify key Llama dimensions
        assert_eq!(config.hidden_size, 4096);
        assert_eq!(config.vocab_size, 32000); // 32000 from dummy file
    } else {
        panic!("Expected UniversalTransformer for Llama weights, got {:?}", model_type);
    }

    // 2. Verify SD (should still be SD or Universal?)
    // Note: SD is a UNet, not a Transformer in the traditional sense (though it has transformer blocks).
    // Our Meta-Architect is specific to "Universal Transformer".
    // AutoArchitect::identify checks for "down_blocks" BEFORE calling Meta-Architect fallback.
    // So it should still return StableDiffusion.
    let file_sd = fs::File::open("sd_dummy.safetensors").expect("Failed to open sd_dummy.safetensors");
    let mmap_sd = unsafe { MmapOptions::new().map(&file_sd).expect("Failed to mmap") };
    let weights_sd = SafeTensors::deserialize(&mmap_sd).expect("Failed to deserialize SD weights");
    
    let model_type_sd = AutoArchitect::identify(&weights_sd);
    println!("Identified sd_dummy.safetensors as: {:?}", model_type_sd);
    assert!(matches!(model_type_sd, ModelType::StableDiffusion));

    println!("SUCCESS: Meta-Architect verification passed!");
}
