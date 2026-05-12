use tracea::runtime::manager::{RuntimeManager, DeviceBackend};
use tracea::policy::types::OperatorTopology;
use tracea::optimizer::tuner::MetaTuner;
use std::sync::Arc;

fn main() {
    let runtime = Arc::new(RuntimeManager::new());
    let tuner = MetaTuner::new(Arc::downgrade(&runtime));

    // Define a GEMM operator for Metal
    let op = OperatorTopology::Gemm {
        op_id: 0,
        name: "metal_gemm_2048".into(),
        m: 2048, n: 2048, k: 2048,
        batch: 1,
        kind: tracea::policy::types::TopologyKind::Dense,
        epilogue: vec![],
    };

    println!("[MetaTuner] Tuning Metal GEMM 2048x2048x2048...\n");
    let best = tuner.tune_operator(&op, DeviceBackend::Metal, 5);

    match best {
        Some(config) => {
            println!("\n=== Best Config ===");
            println!("  Tile: {}x{}x{}", config.m_tile, config.n_tile, config.k_tile);
            println!("  Stages: {}", config.num_stages);
            println!("  Warps: {:?}", config.force_num_warps);
            println!("  Double Buffer: {}", config.double_buffer);
            println!("  GG: {:?}", config.gemm_variant);
        }
        None => {
            println!("[MetaTuner] No suitable config found.");
        }
    }

    // Print stats
    for (backend, stats) in tuner.get_tuning_stats() {
        println!("  {:?}: {} trials, {} pruned", backend, stats.total_trials, stats.pruned_count);
    }
}
