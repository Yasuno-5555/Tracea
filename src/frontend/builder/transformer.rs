
use crate::core::graph::Graph;
use safetensors::SafeTensors;
use std::collections::HashMap;

/// Configuration deduced from the weight file topology.
#[derive(Debug, Clone, PartialEq)]
pub struct TransformerConfig {
    pub hidden_size: usize,
    pub num_layers: usize,
    pub vocab_size: usize,
    pub intermediate_size: usize, // FFN hidden dim
    pub num_heads: usize,         // Deduced or defaulted
    pub num_kv_heads: usize,      // For GQA
    pub head_dim: usize,          // Deduced or defaulted
    pub is_gated_mlp: bool,       // SwiGLU (Gate + Up) vs MLP (Up)
}

pub struct MetaArchitect;

impl MetaArchitect {
    /// Detects the topology of the Transformer from the weight shapes.
    pub fn detect_topology(weights: &SafeTensors) -> TransformerConfig {
        let mut dim_counts: HashMap<usize, usize> = HashMap::new();
        let mut max_layer_index = 0;
        
        // 1. Scan all tensors to gather statistics
        for name in weights.names() {
            let tensor = weights.tensor(name).unwrap();
            let shape = tensor.shape();
            
            // Count dimension frequencies to find Hidden Size (H)
            // H is usually present in almost every layer's weight matrix (square or rectangular).
            for &dim in shape {
                *dim_counts.entry(dim).or_insert(0) += 1;
            }

            // Detect number of layers by parsing numbers in keys
            // e.g. "model.layers.31.self_attn..."
            // We look for the largest integer segment in the key.
            for component in name.split('.') {
                if let Ok(idx) = component.parse::<usize>() {
                    if idx > max_layer_index && idx < 1000 { // Sanity check < 1000
                        max_layer_index = idx;
                    }
                }
            }
        }

        // 2. Deduce Hidden Size (H)
        // Heuristic: The most frequent dimension that is not 1 (and maybe even/large) is likely H.
        // Actually, typical H (4096, 512, etc.) appears VERY frequently (Input, Output, Q, K, V, O, MLP-Up, MLP-Down...).
        let hidden_size = dim_counts.iter()
            .filter(|(&d, _)| d > 64) // Filter out small dims like heads (32) or batch/time placeholders if any? Actually head_dim can be 64/128.
            .max_by_key(|(_, &count)| count)
            .map(|(&d, _)| d)
            .unwrap_or(4096); // Fallback

        // 3. Deduce Vocab Size (V)
        // Look for [V, H] or [H, V] embedding/head weight.
        // V is usually large (32000, 128000, etc.) and appears 1-2 times (Embed, Head).
        let vocab_size = dim_counts.iter()
            .filter(|(&d, _)| d > 1000 && d != hidden_size) // V is usually > H and distinct
            .max_by_key(|(&d, _)| d) // Should be one of the largest dims
            .map(|(d, _)| *d)
            .unwrap_or(32000); // Fallback

        // 4. Deduce FFN Intermediate Size & Type
        // Inspect one layer to find FFN weights.
        // W_up: [Intermediate, H] or [H, Intermediate]
        let mut intermediate_size = 0;
        let mut is_gated_mlp = false;
        
        // Try to find a tensor that connects to H but is not H itself (MLP expansion).
        // Common ratio is 4*H (GPT) or 2.7*H (SwiGLU).
        // We look for a shape [X, H] where X != H and X != V.
        for name in weights.names() {
            if name.contains(&max_layer_index.to_string()) || name.contains("layers.0") {
                let tensor = weights.tensor(name).unwrap();
                let shape = tensor.shape();
                if shape.len() == 2 {
                    if shape.contains(&hidden_size) {
                        let other_dim = if shape[0] == hidden_size { shape[1] } else { shape[0] };
                        if other_dim != hidden_size && other_dim != vocab_size && other_dim > hidden_size {
                             intermediate_size = other_dim;
                             // Heuristic: If we see "gate" in name, it's Gated MLP.
                             if name.contains("gate") {
                                 is_gated_mlp = true;
                             }
                        }
                    }
                }
            }
        }
        if intermediate_size == 0 { intermediate_size = hidden_size * 4; } // Fallback

        // 5. Deduce Attention Heads
        // Defaulting for now as it's hard to distinguish from shape alone without known head_dim.
        // If H=4096, Heads=32 -> D_head=128.
        let head_dim = 128; 
        let num_heads = hidden_size / head_dim;
        let num_kv_heads = num_heads; // Assuming MHA default, GQA detection needs name inspection (k_proj vs q_proj shapes).

        TransformerConfig {
            hidden_size,
            num_layers: max_layer_index + 1,
            vocab_size,
            intermediate_size,
            num_heads,
            num_kv_heads,
            head_dim,
            is_gated_mlp,
        }
    }

    pub fn build_universal_graph(weights: &SafeTensors) -> Graph {
        let config = Self::detect_topology(weights);
        let mut graph = Graph::new();
        
        // Placeholder for graph construction using the config.
        // In a real implementation, we would iterate 0..config.num_layers and add ops.
        // Since we are simulating the behavior in this task, we will add one "Super Block" 
        // or a chain of GEMMs representing the detected structure.

        // 1. Embedding
        // graph.add_embedding(...) - Tracea Graph doesn't have explicit Embedding op, usually it's lookup or handled externally.
        // Let's assume input comes in, and we apply the layers.
        
        let mut prev_node_id = 0; // Dummy input ID (usually would be provided or created)
        
        // Create an input placeholder node (e.g. identify transformation) or just start from first layer
        // For this demo, let's create a dummy input GEMM (Identity)
        let input_id = graph.add_gemm(
            1u32, 
            config.hidden_size as u32, 
            config.hidden_size as u32, 
            vec![]
        );
        prev_node_id = input_id;

        for i in 0..config.num_layers {
            // Attention Block
            // Fused Attention Node
            let attn_id = graph.add_fused_attention(
                1u32, // Batch
                128u32, // Seq Len (placeholder)
                config.hidden_size as u32,
                config.num_heads as u32,
                config.head_dim as u32,
                true, // Causal
                vec![prev_node_id] // Dependency: Input from prev layer
            );
            
            // Residual Add (Post-Attention)
            // Graph currently doesn't have explicit Add. 
            // In Tracea, FusedGemm might handle it or we need a new Op::Elementwise.
            // For now, let's chain them.
            
            // FFN Block
            // Up Proj
            let up_id = graph.add_gemm(
                1u32, 
                config.intermediate_size as u32, 
                config.hidden_size as u32, 
                vec![attn_id]
            );
            
            // Down Proj
            let down_id = graph.add_gemm(
                1u32, 
                config.hidden_size as u32, 
                config.intermediate_size as u32, 
                vec![up_id]
            );
            
            prev_node_id = down_id;
        }
        
        // Final Head (Output Projection)
        let _head_id = graph.add_gemm(
            1u32, 
            config.vocab_size as u32, 
            config.hidden_size as u32, 
            vec![prev_node_id]
        );

        graph
    }
}
