//! # Core Abstractions
//!
//! This module defines the "Platonic Ideals" of the operations Tracea optimizes.
//!
//! - **[`op`]:** Generic operation definitions (GEMM, Fused Epilogues). Independent of hardware.
//! - **[`graph`]:** Directed Acyclic Graph (DAG) of operations.
//! - **[`config`]:** The concrete "Implementation Strategy" (Tiling, Stages) chosen by the Optimizer.

pub mod op;
pub mod config;
pub mod graph;
pub mod cache;
pub mod tuning;
pub mod loader;
pub mod backend;
pub mod device;
pub mod ttg;
pub mod cost;
pub mod optimizer;
pub mod polyhedral;
pub mod manifold;
pub mod lattice;
pub mod mapper;
pub mod evolution;
