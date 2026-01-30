use tracea::policy::types::{GraphTopology, OperatorTopology};
use tracea::core::optimizer::GraphOptimizer;
use tracea::core::op::EpilogueOp;

#[test]
fn test_conv_relu_fusion() {
    let mut operators = Vec::new();
    let mut dependencies = Vec::new();

    // 1. Conv
    operators.push(OperatorTopology::Conv2d {
        op_id: 1,
        name: "conv".into(),
        n: 1, c: 3, h: 224, w: 224, k: 64,
        r: 3, s: 3, stride: 1, padding: 1,
        epilogue: vec![],
    });

    // 2. ReLU
    operators.push(OperatorTopology::Relu {
        op_id: 2,
        name: "relu".into(),
        n: 1 * 64 * 224 * 224,
    });

    dependencies.push((1, 2));

    let mut graph = GraphTopology { operators, dependencies };

    // Optimize
    GraphOptimizer::optimize(&mut graph);

    // Verify
    // Should have 1 operator now (the fused Conv)
    assert_eq!(graph.operators.len(), 1, "Should have fused Conv and ReLU");
    
    match &graph.operators[0] {
        OperatorTopology::Conv2d { epilogue, .. } => {
            assert_eq!(epilogue.len(), 1, "Fused Conv should have 1 epilogue op");
            assert!(matches!(epilogue[0], EpilogueOp::ReLU), "Epilogue should be ReLU");
        },
        _ => panic!("Expected Conv2d as the remaining operator"),
    }
}

#[test]
fn test_conv_bn_relu_fusion() {
    let mut operators = Vec::new();
    let mut dependencies = Vec::new();

    // 1. Conv
    operators.push(OperatorTopology::Conv2d {
        op_id: 1,
        name: "conv".into(),
        n: 1, c: 3, h: 224, w: 224, k: 64,
        r: 3, s: 3, stride: 1, padding: 1,
        epilogue: vec![],
    });

    // 2. BN
    operators.push(OperatorTopology::BatchNorm {
        op_id: 2,
        name: "bn".into(),
        n: 1, c: 64, h: 224, w: 224,
        epsilon: 1e-5,
        momentum: 0.1,
    });

    // 3. ReLU
    operators.push(OperatorTopology::Relu {
        op_id: 3,
        name: "relu".into(),
        n: 1 * 64 * 224 * 224,
    });

    dependencies.push((1, 2));
    dependencies.push((2, 3));

    let mut graph = GraphTopology { operators, dependencies };

    // Optimize
    GraphOptimizer::optimize(&mut graph);

    // Verify
    // Should have 1 operator now
    assert_eq!(graph.operators.len(), 1, "Should have fused Conv, BN, and ReLU");
    
    match &graph.operators[0] {
        OperatorTopology::Conv2d { epilogue, .. } => {
            assert_eq!(epilogue.len(), 2, "Fused Conv should have 2 epilogue ops");
            assert!(matches!(epilogue[0], EpilogueOp::BatchNorm { .. }), "First epilogue should be BN");
            assert!(matches!(epilogue[1], EpilogueOp::ReLU), "Second epilogue should be ReLU");
        },
        _ => panic!("Expected Conv2d as the remaining operator"),
    }
}

#[test]
fn test_residual_fusion() {
    let mut operators = Vec::new();
    let mut dependencies = Vec::new();

    // 1. Producer
    operators.push(OperatorTopology::Conv2d {
        op_id: 1,
        name: "main_path".into(),
        n: 1, c: 64, h: 56, w: 56, k: 64,
        r: 3, s: 3, stride: 1, padding: 1,
        epilogue: vec![],
    });

    // 2. Add (Residual)
    operators.push(OperatorTopology::Elementwise {
        op_id: 2,
        name: "add".into(),
        kind: "Add".into(),
        n: 1 * 64 * 56 * 56,
    });

    // 3. ReLU
    operators.push(OperatorTopology::Relu {
        op_id: 3,
        name: "relu".into(),
        n: 1 * 64 * 56 * 56,
    });

    // Dependence: 
    //   0 (Input) -> 1 (Conv)
    //   0 (Input) -> 2 (Add) (Skip connection)
    //   1 -> 2 (Main path)
    //   2 -> 3 (Final ReLU)
    
    // We mock Input as op_id 0
    operators.push(OperatorTopology::Input { op_id: 0, name: "input".into() });

    dependencies.push((0, 1));
    dependencies.push((0, 2)); // Skip
    dependencies.push((1, 2)); // Main
    dependencies.push((2, 3));

    let mut graph = GraphTopology { operators, dependencies };

    // Optimize
    GraphOptimizer::optimize(&mut graph);

    // Verify
    // Should have: Input (0) -> Conv (1, fused with Add and ReLU)
    assert_eq!(graph.operators.len(), 2, "Should have 2 operators (Input and fused Conv)");
    
    let conv = graph.operators.iter().find(|o| o.op_id() == 1).expect("Conv should exist");
    match conv {
        OperatorTopology::Conv2d { epilogue, .. } => {
            assert_eq!(epilogue.len(), 2, "Fused Conv should have 2 epilogue ops (Add, ReLU)");
            assert!(matches!(epilogue[0], EpilogueOp::ResidualAdd { .. }), "First epilogue should be ResidualAdd");
            assert!(matches!(epilogue[1], EpilogueOp::ReLU), "Second epilogue should be ReLU");
        },
        _ => panic!("Expected Conv2d"),
    }
}
