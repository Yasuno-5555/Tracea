// src/bin/self_evolution_demo.rs

use tracea::runtime::manager::RuntimeManager;
use tracea::core::manifold::ComputeAtom;
use tracea::runtime::manager::DeviceBackend;
use std::sync::Arc;

fn main() {
    println!("--- Tracea Self-Evolution Demo ---");
    
    let manager = RuntimeManager::new();
    let backend = if cfg!(target_os = "macos") { DeviceBackend::Metal } else { DeviceBackend::Cuda };

    // 1. Define a "DNA" (ComputeAtom)
    // We'll use a standard Conv2d atom
    let atom = ComputeAtom::from_conv2d(
        1,  // Batch
        64, // Input Channels
        56, // Input Height
        56, // Input Width
        64, // Output Channels
        3,  // Kernel R
        3,  // Kernel S
        1,  // Stride
        1,  // Padding
        1,  // Dilation
    );

    println!("[Demo] Atom created: {}", atom.name);
    
    // 2. Start Evolution
    let generations = 20;
    println!("[Demo] Starting evolution for {} generations...", generations);
    
    let best_strategy = manager.tuner.tune_atom(&atom, backend, generations).expect("Evolution failed");

    println!("\n--- Evolution Results ---");
    println!("Optimal Tile Sizes: {:?}", best_strategy.tile_sizes);
    println!("Loop Order: {:?}", best_strategy.loop_order);
    println!("Spatial Mapping: {:?}", best_strategy.spatial_map);

    // 3. Verify Persistence
    println!("\n[Demo] Verifying persistence...");
    let db_path = "dna_database.json";
    if std::path::Path::new(db_path).exists() {
        println!("[Demo] Found dna_database.json. Self-learning confirmed! ✅");
    } else {
        println!("[Demo] ⚠️ dna_database.json not found in current directory.");
    }

    println!("\n[Demo] Success. Tracea is now evolving.");
}
