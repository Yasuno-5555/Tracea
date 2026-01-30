use tracea::core::config::PipelineConfig;
use tracea::policy::types::{GraphTopology, OperatorTopology};
use tracea::runtime::manager::{RuntimeManager, DeviceBackend};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

fn main() {
    let manager = RuntimeManager::new();
    println!("Backend initialized.");

    // 1. Define Graph
    let mut graph = GraphTopology {
        operators: Vec::new(),
        dependencies: Vec::new(),
    };

    // Input -> Conv -> Output
    graph.operators.push(OperatorTopology::Input { op_id: 1, name: "input".into() });
    graph.operators.push(OperatorTopology::Input { op_id: 2, name: "weight".into() });

    graph.operators.push(OperatorTopology::Conv2d { 
        op_id: 3, name: "conv".into(), 
        n: 1, c: 64, h: 56, w: 56, 
        k: 64, 
        r: 3, s: 3, 
        stride: 1, padding: 1,
        epilogue: vec![]
    });

    graph.dependencies.push((1, 3));
    graph.dependencies.push((2, 3));

    // 2. Compile Plan
    println!("Compiling plan...");
    let backend = if cfg!(target_os = "macos") { DeviceBackend::Metal } else { DeviceBackend::Cuda };
    
    let plan = {
        let compiler = manager.compiler.lock().unwrap();
        compiler.compile(graph.clone(), &manager, backend).expect("Compilation failed")
    };
    println!("Plan compiled. Steps: {}", plan.steps.len());
    
    // 3. Alloc Memory
    let arena_size = 256 * 1024 * 1024;
    manager.init_arena(arena_size, backend).expect("Arena init failed");
    
    // Alloc inputs manually
    let input_size = 1 * 64 * 56 * 56 * 2; // FP16
    let weight_size = 64 * 64 * 3 * 3 * 2; // FP16
    
    let buf_in = manager.alloc(input_size, backend).expect("Alloc input failed");
    let buf_w = manager.alloc(weight_size, backend).expect("Alloc weight failed");
    
    let mut inputs = HashMap::new();
    inputs.insert(1, buf_in);
    inputs.insert(2, buf_w);

    // 4. Capture
    println!("Capturing graph...");
    let captured = manager.executor.capture(&plan, &inputs, &manager).expect("Capture failed");
    
    #[cfg(target_os = "macos")]
    println!("Graph captured. ICB present: {:?}", captured.icb.is_some());

    // 5. Exec Loop
    println!("Warming up...");
    for _ in 0..10 {
        captured.execute(&manager, &inputs).expect("Exec failed");
    }
    manager.synchronize();

    println!("Benchmarking 100 iterations...");
    let start = Instant::now();
    for _ in 0..100 {
        captured.execute(&manager, &inputs).expect("Exec failed");
    }
    manager.synchronize();
    let duration = start.elapsed();
    println!("100 runs took: {:?}", duration);
    println!("Avg Latency: {:?}", duration / 100);
}
