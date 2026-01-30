use std::collections::HashMap;

/// A self-contained execution plan that requires no further logic to run.
///
/// # Critical Constraints
/// 1. Must NOT reference `GraphTopology`, `NodeID`, `OperatorType`, or `Shape`.
/// 2. Must be pure data (Instruction Stream).
/// 3. Assumed valid by the Executor (no panic checks).
#[derive(Debug, Clone)]
pub struct ExecutionPlan {
    /// Linearized sequence of operations to execute.
    pub steps: Vec<ExecutionStep>,
    /// Total size in bytes required for the Arena.
    pub arena_size: usize,
    /// Mapping of LogicalID (Graph Node ID) to Arena Offset (Bytes).
    /// Used to retrieve output buffers from the arena.
    pub output_map: HashMap<u64, (usize, usize)>, // (offset, size_in_bytes)
}

/// A single step in the execution sequence.
#[derive(Debug, Clone)]
pub enum ExecutionStep {
    /// Launch a compiled kernel.
    LaunchKernel {
        kernel_id: usize,
        grid_size: (u32, u32, u32),
        block_size: (u32, u32, u32),
        shared_mem_bytes: u32,
        args: Vec<KernelArgSpec>,
    },
    /// Copy data within the arena (e.g., for double buffering or layout transforms).
    Memcpy {
        src_offset: usize,
        dst_offset: usize,
        size: usize,
    },
}

/// Specification for a kernel argument.
/// The Compiler pre-calculates everything; the Executor just binds it.
#[derive(Debug, Clone)]
pub enum KernelArgSpec {
    /// A pointer to an offset within the Arena.
    ArenaOffset(usize),
    /// An external input buffer provided by the user (Graph Node ID).
    ExternalInput(u64), 
    /// A raw integer scalar (passed by value).
    ScalarInt(i32),
    /// A raw float scalar (passed by value).
    ScalarFloat(f32),
    /// A raw byte buffer (e.g., a struct of parameters).
    Bytes(Vec<u8>),
}
