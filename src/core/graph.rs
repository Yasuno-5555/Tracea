use serde::{Serialize, Deserialize};
use crate::core::op::GemmOp;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Operation {
    Gemm(GemmOp),
    // Add more ops here: Attention, Conv, etc.
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Node {
    pub id: usize,
    pub op: Operation,
    pub dependencies: Vec<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Graph {
    pub nodes: Vec<Node>,
}

pub trait ScheduleStrategy {
    fn schedule(&self, graph: &Graph) -> Vec<usize>;
}

pub struct PrioritySchedule;

impl ScheduleStrategy for PrioritySchedule {
    fn schedule(&self, graph: &Graph) -> Vec<usize> {
        let mut node_ids: Vec<usize> = (0..graph.nodes.len()).collect();
        // Sort by complexity (M*N*K) descending
        node_ids.sort_by_key(|&id| {
            let node = &graph.nodes[id];
            match &node.op {
                Operation::Gemm(gemm) => {
                    let m = gemm.m.0 as u64;
                    let n = gemm.n.0 as u64;
                    let k = gemm.k.0 as u64;
                    std::cmp::Reverse(m * n * k)
                }
            }
        });
        node_ids
    }
}

impl Graph {
    pub fn new() -> Self {
        Self { nodes: Vec::new() }
    }

    pub fn add_gemm(&mut self, m: u32, n: u32, k: u32, deps: Vec<usize>) -> usize {
        let id = self.nodes.len();
        self.nodes.push(Node {
            id,
            op: Operation::Gemm(GemmOp::new(m, n, k)),
            dependencies: deps,
        });
        id
    }
}
