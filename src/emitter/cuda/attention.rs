use crate::emitter::traits::{UnifiedOpIR, EmissionError};
pub use crate::kernels::attention::cuda_emitter::FlashAttentionEmitter;

pub fn generate_attention(h: u32, dh: u32, causal: bool, ir: &UnifiedOpIR) -> Result<String, EmissionError> {
    let emitter = FlashAttentionEmitter::new(ir.tiling.clone());
    Ok(emitter.generate_kernel(h as usize, dh as usize, causal))
}
