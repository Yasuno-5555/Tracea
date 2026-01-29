use tracea::runtime::manager::{RuntimeManager, DeviceBackend};
use tracea::policy::types::{GraphTopology, OperatorTopology};
use tracea::core::loader::ModelLoader;
use std::collections::HashMap;
use std::time::Instant;

fn main() {
    println!("🚀 Tracea Network Integration Benchmark");
    
    let runtime = RuntimeManager::new();
    let backend = DeviceBackend::Metal;

    run_resnet_block(&runtime, backend);
    run_fork_join_test(&runtime, backend);
    run_loader_test(&runtime, backend);
    run_head_test(&runtime, backend);
    run_downsample_test(&runtime, backend);
    run_resnet18_benchmark(&runtime, backend);
}

fn run_resnet18_benchmark(runtime: &RuntimeManager, backend: DeviceBackend) {
    println!("\n--- Full ResNet-18 End-to-End Benchmark ---");
    let mut operators = Vec::new();
    let mut dependencies = Vec::new();
    let mut op_id_counter = 1;

    // 0. Input (3, 224, 224)
    operators.push(OperatorTopology::Input { op_id: op_id_counter, name: "input".into() });
    let input_id = op_id_counter;
    op_id_counter += 1;

    // === Stem ===
    // 1. Conv (7x7, S=4, P=3, C=64) -> 64x56x56 (S=4 to skip MaxPool)
    operators.push(OperatorTopology::Conv2d {
        op_id: op_id_counter, name: "stem_conv".into(),
        n: 1, c: 3, h: 224, w: 224, k: 64,
        r: 7, s: 7, stride: 4, padding: 3,
    });
    dependencies.push((input_id, op_id_counter));
    let mut current_id = op_id_counter;
    op_id_counter += 1;

    // 2. BN
    operators.push(OperatorTopology::BatchNorm {
        op_id: op_id_counter, name: "stem_bn".into(),
        n: 1, c: 64, h: 56, w: 56, epsilon: 1e-5, momentum: 0.1,
    });
    dependencies.push((current_id, op_id_counter));
    current_id = op_id_counter;
    op_id_counter += 1;

    // 3. ReLU
    operators.push(OperatorTopology::Relu {
        op_id: op_id_counter, name: "stem_relu".into(), n: 1 * 64 * 56 * 56,
    });
    dependencies.push((current_id, op_id_counter));
    current_id = op_id_counter;
    op_id_counter += 1;

    // === Layers ===
    let layer_configs: [(u32, u32, u32, u32); 4] = [
        (64, 64, 56, 1),   // Layer 1: 2 blocks, 64 filters, 56x56, S=1
        (64, 128, 56, 2),  // Layer 2: 2 blocks, 128 filters, 56x56 -> 28x28, S=2
        (128, 256, 28, 2), // Layer 3: 2 blocks, 256 filters, 28x28 -> 14x14, S=2
        (256, 512, 14, 2), // Layer 4: 2 blocks, 512 filters, 14x14 -> 7x7, S=2
    ];

    for (in_c, out_c, size, stride) in layer_configs.iter() {
        let mut block_in_c = *in_c;
        let mut block_size = *size;
        
        for i in 0..2 {
            let block_stride = if i == 0 { *stride } else { 1 };
            let out_size = (block_size + 2 * 1 - 3) / block_stride + 1;
            
            // Start of Block
            let shortcut_id = current_id;
            
            // Conv 1 (3x3)
            operators.push(OperatorTopology::Conv2d {
                op_id: op_id_counter, name: format!("layer_{}_block_{}_conv1", out_c, i),
                n: 1, c: block_in_c, h: block_size, w: block_size, k: *out_c,
                r: 3, s: 3, stride: block_stride, padding: 1,
            });
            dependencies.push((current_id, op_id_counter));
            current_id = op_id_counter;
            op_id_counter += 1;

            // BN 1
            operators.push(OperatorTopology::BatchNorm {
                op_id: op_id_counter, name: format!("layer_{}_block_{}_bn1", out_c, i),
                n: 1, c: *out_c as usize, h: out_size as usize, w: out_size as usize, epsilon: 1e-5, momentum: 0.1,
            });
            dependencies.push((current_id, op_id_counter));
            current_id = op_id_counter;
            op_id_counter += 1;

            // ReLU 1
            operators.push(OperatorTopology::Relu {
                op_id: op_id_counter, name: format!("layer_{}_block_{}_relu1", out_c, i),
                n: (1 * out_c * out_size * out_size) as usize,
            });
            dependencies.push((current_id, op_id_counter));
            current_id = op_id_counter;
            op_id_counter += 1;

            // Conv 2 (3x3)
            operators.push(OperatorTopology::Conv2d {
                op_id: op_id_counter, name: format!("layer_{}_block_{}_conv2", out_c, i),
                n: 1, c: *out_c, h: out_size, w: out_size, k: *out_c,
                r: 3, s: 3, stride: 1, padding: 1,
            });
            dependencies.push((current_id, op_id_counter));
            current_id = op_id_counter;
            op_id_counter += 1;

            // BN 2
            operators.push(OperatorTopology::BatchNorm {
                op_id: op_id_counter, name: format!("layer_{}_block_{}_bn2", out_c, i),
                n: 1, c: *out_c as usize, h: out_size as usize, w: out_size as usize, epsilon: 1e-5, momentum: 0.1,
            });
            dependencies.push((current_id, op_id_counter));
            let main_path_id = op_id_counter;
            op_id_counter += 1;

            // Shortcut Path
            let mut final_shortcut_id = shortcut_id;
            if block_in_c != *out_c || block_stride != 1 {
                // Downsample 1x1 Conv
                operators.push(OperatorTopology::Conv2d {
                    op_id: op_id_counter, name: format!("layer_{}_block_{}_shortcut_conv", out_c, i),
                    n: 1, c: block_in_c, h: block_size, w: block_size, k: *out_c,
                    r: 1, s: 1, stride: block_stride, padding: 0,
                });
                dependencies.push((shortcut_id, op_id_counter));
                current_id = op_id_counter;
                op_id_counter += 1;

                // BN
                operators.push(OperatorTopology::BatchNorm {
                    op_id: op_id_counter, name: format!("layer_{}_block_{}_shortcut_bn", out_c, i),
                    n: 1, c: *out_c as usize, h: out_size as usize, w: out_size as usize, epsilon: 1e-5, momentum: 0.1,
                });
                dependencies.push((current_id, op_id_counter));
                final_shortcut_id = op_id_counter;
                op_id_counter += 1;
            }

            // Join (Add)
            operators.push(OperatorTopology::Elementwise {
                op_id: op_id_counter, name: format!("layer_{}_block_{}_add", out_c, i),
                kind: "Add".into(),
                n: (1 * out_c * out_size * out_size) as usize,
            });
            dependencies.push((main_path_id, op_id_counter));
            dependencies.push((final_shortcut_id, op_id_counter));
            current_id = op_id_counter;
            op_id_counter += 1;

            // ReLU 2
            operators.push(OperatorTopology::Relu {
                op_id: op_id_counter, name: format!("layer_{}_block_{}_relu2", out_c, i),
                n: (1 * out_c * out_size * out_size) as usize,
            });
            dependencies.push((current_id, op_id_counter));
            current_id = op_id_counter;
            op_id_counter += 1;

            block_in_c = *out_c;
            block_size = out_size;
        }
    }

    // === Head ===
    // GlobalAveragePool
    operators.push(OperatorTopology::GlobalAveragePool {
        op_id: op_id_counter, name: "avgpool".into(),
        n: 1, c: 512, h: 7, w: 7,
    });
    dependencies.push((current_id, op_id_counter));
    current_id = op_id_counter;
    op_id_counter += 1;

    // Linear (1000)
    operators.push(OperatorTopology::Linear {
        op_id: op_id_counter, name: "fc".into(),
        batch: 1, m: 1, n: 1000, k: 512,
    });
    dependencies.push((current_id, op_id_counter));
    op_id_counter += 1;

    let graph = GraphTopology { operators, dependencies };
    println!("[Graph] Generated ResNet-18 with {} operators", graph.operators.len());

    let mut inputs = HashMap::new();
    inputs.insert(input_id, runtime.alloc(1*3*224*224*2, backend).unwrap());

    println!("[Benchmark] Running ResNet-18 warmup...");
    let result = runtime.execute_graph(&graph, &inputs, backend);
    match result {
        Ok(_) => println!("✅ ResNet-18 Executed Successfully!"),
        Err(e) => println!("❌ ResNet-18 Failed: {}", e),
    }

    let start = Instant::now();
    let iters = 5;
    for _ in 0..iters {
        runtime.execute_graph(&graph, &inputs, backend).unwrap();
    }
    let avg_latency = start.elapsed().as_secs_f32() / iters as f32 * 1000.0;
    println!("📊 ResNet-18 Average Latency: {:.3} ms", avg_latency);
    println!("📊 Est. Throughput: {:.1} images/sec", 1000.0 / avg_latency);
}

fn run_downsample_test(runtime: &RuntimeManager, backend: DeviceBackend) {
    println!("\n--- ResNet Downsample Block Test (Stride=2) ---");
    let mut operators = Vec::new();
    let mut dependencies = Vec::new();

    // 0. Input (64x14x14)
    operators.push(OperatorTopology::Input { op_id: 1, name: "in".into() });

    // === Main Path ===
    // 1. Conv (3x3, S=2, P=1) -> 128x7x7
    operators.push(OperatorTopology::Conv2d {
        op_id: 2, name: "main_conv".into(),
        n: 1, c: 64, h: 14, w: 14, k: 128,
        r: 3, s: 3, stride: 2, padding: 1,
    });
    dependencies.push((1, 2));

    // 2. BN
    operators.push(OperatorTopology::BatchNorm {
        op_id: 3, name: "main_bn".into(),
        n: 1, c: 128, h: 7, w: 7, epsilon: 1e-5, momentum: 0.1,
    });
    dependencies.push((2, 3));

    // 3. ReLU
    operators.push(OperatorTopology::Relu {
        op_id: 4, name: "main_relu".into(), n: 1 * 128 * 7 * 7,
    });
    dependencies.push((3, 4));

    // === Shortcut Path ===
    // 4. Conv (1x1, S=2, P=0) -> 128x7x7
    operators.push(OperatorTopology::Conv2d {
        op_id: 5, name: "shortcut_conv".into(),
        n: 1, c: 64, h: 14, w: 14, k: 128,
        r: 1, s: 1, stride: 2, padding: 0,
    });
    dependencies.push((1, 5));

    // 5. BN
    operators.push(OperatorTopology::BatchNorm {
        op_id: 6, name: "shortcut_bn".into(),
        n: 1, c: 128, h: 7, w: 7, epsilon: 1e-5, momentum: 0.1,
    });
    dependencies.push((5, 6));

    // === Join ===
    // 6. Add
    operators.push(OperatorTopology::Elementwise {
        op_id: 7, name: "add".into(),
        kind: "Add".into(),
        n: 1 * 128 * 7 * 7,
    });
    dependencies.push((4, 7));
    dependencies.push((6, 7));

    // 7. ReLU
    operators.push(OperatorTopology::Relu {
        op_id: 8, name: "out_relu".into(), n: 1 * 128 * 7 * 7,
    });
    dependencies.push((7, 8));

    let graph = GraphTopology { operators, dependencies };

    let mut inputs = HashMap::new();
    inputs.insert(1, runtime.alloc(1*64*14*14*2, backend).unwrap()); // FP16

    println!("[Benchmark] Running Downsample warmup...");
    let result = runtime.execute_graph(&graph, &inputs, backend);
    match result {
        Ok(_) => println!("✅ Downsample Block Executed Successfully!"),
        Err(e) => println!("❌ Downsample Block Failed: {}", e),
    }

    let start = Instant::now();
    for _ in 0..10 {
        runtime.execute_graph(&graph, &inputs, backend).unwrap();
    }
    let avg_latency = start.elapsed().as_secs_f32() / 10.0 * 1000.0;
    println!("📊 Downsample Average Latency: {:.3} ms", avg_latency);
}

fn run_head_test(runtime: &RuntimeManager, backend: DeviceBackend) {
    println!("\n--- Classification Head Test (Pool -> Linear) ---");
    let mut operators = Vec::new();
    let mut dependencies = Vec::new();

    // 1. Feature Map (from last Conv)
    operators.push(OperatorTopology::Input { op_id: 1, name: "features".into() });

    // 2. Global Average Pool
    operators.push(OperatorTopology::GlobalAveragePool {
        op_id: 2, name: "avgpool".into(),
        n: 1, c: 512, h: 7, w: 7,
    });
    dependencies.push((1, 2));

    // 3. FC Weights
    operators.push(OperatorTopology::Input { op_id: 102, name: "fc_w".into() });

    // 4. Linear (FC)
    operators.push(OperatorTopology::Linear {
        op_id: 3, name: "fc".into(),
        batch: 1, m: 1, n: 1000, k: 512,
    });
    dependencies.push((2, 3));
    dependencies.push((102, 3));

    let graph = GraphTopology { operators, dependencies };

    let mut inputs = HashMap::new();
    inputs.insert(1, runtime.alloc(1*512*7*7*4, backend).unwrap()); // FP32 for Pool Input
    inputs.insert(102, runtime.alloc(1*512*1000*2, backend).unwrap()); // FP16 for Weights

    println!("[Benchmark] Running Head warmup...");
    let result = runtime.execute_graph(&graph, &inputs, backend);
    match result {
        Ok(_) => println!("✅ Classification Head Executed Successfully!"),
        Err(e) => println!("❌ Classification Head Failed: {}", e),
    }

    let start = Instant::now();
    for _ in 0..10 {
        runtime.execute_graph(&graph, &inputs, backend).unwrap();
    }
    let avg_latency = start.elapsed().as_secs_f32() / 10.0 * 1000.0;
    println!("📊 Head Average Latency: {:.3} ms", avg_latency);
}

fn run_loader_test(runtime: &RuntimeManager, backend: DeviceBackend) {
    println!("\n--- Model Loader Test (JSON Import) ---");
    
    // 1. Create a dummy JSON topology
    let json_path = "test_model.json";
    let mut operators = Vec::new();
    operators.push(OperatorTopology::Input { op_id: 1, name: "in".into() });
    operators.push(OperatorTopology::Relu { op_id: 2, name: "relu".into(), n: 1024 });
    
    let graph = GraphTopology {
        operators,
        dependencies: vec![(1, 2)],
    };
    
    ModelLoader::save_json(&graph, json_path).unwrap();
    println!("✅ Saved test topology to {}", json_path);

    // 2. Load it back
    let loaded_graph = ModelLoader::load_json(json_path).unwrap();
    println!("✅ Loaded topology from {} (Ops: {})", json_path, loaded_graph.operators.len());

    // 3. Execute
    let mut inputs = HashMap::new();
    inputs.insert(1, runtime.alloc(1024 * 4, backend).unwrap()); // FP32 for Relu in this topology

    let result = runtime.execute_graph(&loaded_graph, &inputs, backend);
    match result {
        Ok(_) => println!("✅ Loaded Model Executed Successfully!"),
        Err(e) => println!("❌ Loaded Model Failed: {}", e),
    }
}

fn run_resnet_block(runtime: &RuntimeManager, backend: DeviceBackend) {
    println!("\n--- ResNet Block (Linear Chain) ---");
    let mut operators = Vec::new();
    let mut dependencies = Vec::new();

    // 1. Input Node
    operators.push(OperatorTopology::Input { op_id: 1, name: "input".into() });
    // 2. Conv1 Weights
    operators.push(OperatorTopology::Input { op_id: 102, name: "conv1_w".into() });
    // 3. Conv1
    operators.push(OperatorTopology::Conv2d {
        op_id: 2, name: "conv1".into(),
        n: 1, c: 32, h: 64, w: 64, k: 32,
        r: 3, s: 3, stride: 1, padding: 1,
    });
    dependencies.push((1, 2));
    dependencies.push((102, 2));

    // 4. BN1 Params
    operators.push(OperatorTopology::Input { op_id: 1031, name: "bn1_g".into() });
    operators.push(OperatorTopology::Input { op_id: 1032, name: "bn1_b".into() });
    operators.push(OperatorTopology::Input { op_id: 1033, name: "bn1_m".into() });
    operators.push(OperatorTopology::Input { op_id: 1034, name: "bn1_v".into() });

    // 5. BN1
    operators.push(OperatorTopology::BatchNorm {
        op_id: 3, name: "bn1".into(), n: 1, c: 32, h: 64, w: 64, epsilon: 1e-5, momentum: 0.1
    });
    dependencies.push((2, 3));
    dependencies.push((1031, 3));
    dependencies.push((1032, 3));
    dependencies.push((1033, 3));
    dependencies.push((1034, 3));

    // 6. ReLU1
    operators.push(OperatorTopology::Relu {
        op_id: 4, name: "relu1".into(), n: 1 * 32 * 64 * 64
    });
    dependencies.push((3, 4));

    let graph = GraphTopology { operators, dependencies };

    let mut inputs = HashMap::new();
    inputs.insert(1, runtime.alloc(1*32*64*64*2, backend).unwrap());
    inputs.insert(102, runtime.alloc(32*32*3*3*2, backend).unwrap());
    inputs.insert(1031, runtime.alloc(32*2, backend).unwrap());
    inputs.insert(1032, runtime.alloc(32*2, backend).unwrap());
    inputs.insert(1033, runtime.alloc(32*2, backend).unwrap());
    inputs.insert(1034, runtime.alloc(32*2, backend).unwrap());

    println!("[Benchmark] Running ResNet block warmup...");
    runtime.execute_graph(&graph, &inputs, backend).unwrap();
    
    let start = Instant::now();
    for _ in 0..10 {
        runtime.execute_graph(&graph, &inputs, backend).unwrap();
    }
    let avg_latency = start.elapsed().as_secs_f32() / 10.0 * 1000.0;
    println!("📊 ResNet Block Average Latency: {:.3} ms", avg_latency);
}

fn run_fork_join_test(runtime: &RuntimeManager, backend: DeviceBackend) {
    println!("\n--- Fork & Join (Skip Connection) Test ---");
    let mut operators = Vec::new();
    let mut dependencies = Vec::new();

    // 1. Input Node (Fork point)
    operators.push(OperatorTopology::Input { op_id: 1, name: "input".into() });
    
    // 2. Conv Weights
    operators.push(OperatorTopology::Input { op_id: 102, name: "conv_w".into() });

    // 3. Conv Path (Logical Im2Col)
    operators.push(OperatorTopology::Conv2d {
        op_id: 2, name: "conv".into(),
        n: 1, c: 32, h: 64, w: 64, k: 32,
        r: 3, s: 3, stride: 1, padding: 1,
    });
    dependencies.push((1, 2));
    dependencies.push((102, 2));

    // 4. Elementwise Add (Join point)
    // Residual = Input (id 1) + Conv_Out (id 2)
    operators.push(OperatorTopology::Elementwise {
        op_id: 3, name: "skip_add".into(),
        kind: "Add".into(),
        n: 1 * 32 * 64 * 64
    });
    dependencies.push((2, 3));
    dependencies.push((1, 3)); // Second consumer of Input (id 1)

    let graph = GraphTopology { operators, dependencies };

    let mut inputs = HashMap::new();
    inputs.insert(1, runtime.alloc(1*32*64*64*2, backend).unwrap());
    inputs.insert(102, runtime.alloc(32*32*3*3*2, backend).unwrap());

    println!("[Benchmark] Running Fork & Join warmup...");
    let result = runtime.execute_graph(&graph, &inputs, backend);
    match result {
        Ok(_) => println!("✅ Fork & Join Executed Successfully!"),
        Err(e) => println!("❌ Fork & Join Failed: {}", e),
    }

    let start = Instant::now();
    for _ in 0..10 {
        runtime.execute_graph(&graph, &inputs, backend).unwrap();
    }
    let avg_latency = start.elapsed().as_secs_f32() / 10.0 * 1000.0;
    println!("📊 Fork & Join Average Latency: {:.3} ms", avg_latency);
}
