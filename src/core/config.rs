use serde::{Serialize, Deserialize};
use crate::semantic::fusion::EpilogueOp;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineConfig {
    pub num_stages: u32,
    pub m_tile: u32,
    pub n_tile: u32,
    pub k_tile: u32,
    pub epilogue: Vec<EpilogueOp>,
}

impl PipelineConfig {
    pub fn new(num_stages: u32, m_tile: u32, n_tile: u32, k_tile: u32) -> Self {
        Self {
            num_stages,
            m_tile,
            n_tile,
            k_tile,
            epilogue: Vec::new(),
        }
    }

    pub fn to_vector(&self) -> Vec<f32> {
        vec![
            self.num_stages as f32,
            self.m_tile as f32,
            self.n_tile as f32,
            self.k_tile as f32,
        ]
    }

    pub fn from_vector(vec: &[f32]) -> Self {
        Self {
            num_stages: vec[0] as u32,
            m_tile: vec[1] as u32,
            n_tile: vec[2] as u32,
            k_tile: vec[3] as u32,
            epilogue: Vec::new(),
        }
    }
}
