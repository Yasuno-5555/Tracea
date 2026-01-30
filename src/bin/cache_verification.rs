use tracea::runtime::manager::{RuntimeManager, DeviceBackend};
use tracea::policy::types::{GraphTopology, OperatorTopology};
use std::collections::HashMap;

fn main() {
    println!("🔍 Tracea Graph Cache Verification");
    let runtime = RuntimeManager::new();
    let backend = DeviceBackend::Metal; // Or default available

    // Scenario 1: Basic Cache Hit
    println!("\n--- Scenario 1: Repeated Execution ---");
    let graph1 = create_dummy_graph(1);
    let inputs1 = create_dummy_inputs(&runtime, backend);
    
    println!("Run 1 (Expect MISS):");
    runtime.execute_graph(&graph1, &inputs1, backend).unwrap();
    
    println!("Run 2 (Expect HIT):");
    runtime.execute_graph(&graph1, &inputs1, backend).unwrap();

    // Scenario 2: Different Graph (Structurally identical but different ID)
    println!("\n--- Scenario 2: New Graph Topology ---");
    let graph2 = create_dummy_graph(100); // Different Op IDs
    let inputs2 = create_dummy_inputs(&runtime, backend); // Re-use buffer size but meaningful valid IDs

    println!("Run 3 (Expect MISS):");
    runtime.execute_graph(&graph2, &inputs2, backend).unwrap();

    // Scenario 3: Original Graph Again
    println!("\n--- Scenario 3: Revert to Original ---");
    println!("Run 4 (Expect HIT):");
    runtime.execute_graph(&graph1, &inputs1, backend).unwrap();
}

fn create_dummy_graph(start_id: u64) -> GraphTopology {
    // Simple Input -> Relu
    let ops = vec![
        OperatorTopology::Input { op_id: start_id, name: "input".into() },
        OperatorTopology::Relu { op_id: start_id + 1, name: "relu".into(), n: 1024 }
    ];
    let deps = vec![(start_id, start_id + 1)];
    GraphTopology { operators: ops, dependencies: deps }
}

fn create_dummy_inputs(runtime: &RuntimeManager, backend: DeviceBackend) -> HashMap<u64, tracea::runtime::manager::BufferId> {
    let mut map = HashMap::new();
    // In our dummy graph, IDs are start_id and start_id+1.
    // The executor binds inputs by Op ID.
    // Ideally we match the IDs used in the graph.
    // But since create_dummy_graph is dynamic, we need to handle that.
    // For simplicity, let's just allocate enough for any test case here if we knew IDs.
    // Actually, execute_graph expects inputs map to contain keys matching Input nodes.
    // So we need to return inputs specifically for the graph.
    
    // Hack: We will just alloc one buffer and assume the caller knows how to map it?
    // No, execute_graph takes &HashMap<u64, BufferId>.
    // So create_dummy_inputs needs to know the IDs.
    // We'll simplisticly return a map with keys 1 and 100 which are the start_ids.
    
    let buf = runtime.alloc(1024 * 4, backend).unwrap();
    map.insert(1, buf);
    map.insert(100, buf);
    map
}
