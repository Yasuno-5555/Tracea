// src/core/polyhedral.rs

use serde::{Serialize, Deserialize};

/// Affine representation of a linear expression: c0*x0 + c1*x1 + ... + cn*xn + constant
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AffineFunction {
    /// Coefficients for each loop variable (dimension)
    pub coeffs: Vec<i32>,
    /// Constant term
    pub constant: i32,
}

impl AffineFunction {
    pub fn new(coeffs: Vec<i32>, constant: i32) -> Self {
        Self { coeffs, constant }
    }

    /// Evaluate the function for a given point (vector of loop indices)
    pub fn eval(&self, point: &[i32]) -> i32 {
        let mut sum = self.constant;
        for (i, &val) in point.iter().enumerate() {
            if i < self.coeffs.len() {
                sum += self.coeffs[i] * val;
            }
        }
        sum
    }
}

/// A geometric constraint defining a boundary of the iteration domain
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Constraint {
    pub function: AffineFunction,
    /// true: function == 0, false: function >= 0
    pub is_equality: bool,
}

impl Constraint {
    pub fn geq(coeffs: Vec<i32>, constant: i32) -> Self {
        Self {
            function: AffineFunction::new(coeffs, constant),
            is_equality: false,
        }
    }

    pub fn eq(coeffs: Vec<i32>, constant: i32) -> Self {
        Self {
            function: AffineFunction::new(coeffs, constant),
            is_equality: true,
        }
    }
}

/// SCoP (Static Control Part) representation of a loop nest
#[derive(Debug, Clone)]
pub struct PolyhedralInfo {
    /// Iteration Domain: The set of constraints defining the multi-dimensional iteration space
    /// Variables are ordered as [v0, v1, ..., vn]
    pub domain: Vec<Constraint>,
    
    /// Access Maps: For each tensor, the affine mapping from iteration space to index space
    pub reads: Vec<AffineFunction>, 
    pub writes: Vec<AffineFunction>,
    
    /// Names of the dimensions (for debugging)
    pub dim_names: Vec<String>,
}

impl PolyhedralInfo {
    pub fn new(dim_names: Vec<String>) -> Self {
        Self {
            domain: Vec::new(),
            reads: Vec::new(),
            writes: Vec::new(),
            dim_names,
        }
    }

    pub fn get_dim_index(&self, name: &str) -> Option<usize> {
        self.dim_names.iter().position(|n| n == name)
    }
}

/// A concrete strategy for executing a loop nest on a specific hardware
#[derive(Debug, Clone, Default)]
pub struct TilingStrategy {
    /// Tile sizes for each dimension in the domain
    pub tile_sizes: Vec<u32>,
    /// Number of elements to pad for each dimension to satisfy alignment
    pub padding_needed: Vec<(usize, u32)>,
    /// Expected occupancy (0.0 to 1.0)
    pub expected_occupancy: f32,
}

pub struct PolyhedralOptimizer;

impl PolyhedralOptimizer {
    /// Optimize the polyhedral domain for the given hardware capabilities
    pub fn optimize(
        &self,
        poly: &PolyhedralInfo,
        _hardware: &crate::doctor::capabilities::TraceaCapabilities,
    ) -> TilingStrategy {
        let mut strategy = TilingStrategy::default();
        strategy.tile_sizes = vec![1; poly.dim_names.len()];

        // 1. Alignment & Padding Synthesis (RGB Case: C=3 -> C=4 or 8)
        if let Some(c_idx) = poly.get_dim_index("ic") {
            // Find the constant limit for C if it exists in constraints
            // (Simple heuristic for now)
            let mut c_limit = 0;
            for c in &poly.domain {
                if !c.is_equality && c.function.coeffs.get(c_idx) == Some(&-1) {
                    c_limit = (c.function.constant + 1) as u32;
                }
            }

            if c_limit > 0 && c_limit % 8 != 0 {
                // Synthesize Padding: Distort space to next multiple of 8
                let pad = 8 - (c_limit % 8);
                strategy.padding_needed.push((c_idx, pad));
            }
        }

        // 2. Tile Size Selection (Occupancy Maximization)
        // For now, use a "Golden Square" heuristic if it fits
        // In real polyhedral optimizers, this involves solving for cache/register pressure
        for i in 0..strategy.tile_sizes.len() {
            if poly.dim_names[i] == "oh" || poly.dim_names[i] == "ow" {
                strategy.tile_sizes[i] = 32;
            } else if poly.dim_names[i] == "oc" {
                strategy.tile_sizes[i] = 128;
            }
        }

        strategy.expected_occupancy = 0.85; // Placeholder
        strategy
    }
}
