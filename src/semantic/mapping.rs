use std::collections::HashSet;

#[derive(Debug, Clone, Copy)]
pub enum MatrixLayout {
    Amd16x16F32,
    Amd32x32F32,
    IntelXmx16x16F32,
    IntelXmx8x8F32,
}

#[derive(Debug, Clone, Copy)]
pub enum LaneMapping {
    Linear,
    TensorCore { m_per_lane: u32, n_per_lane: u32 },
    MatrixCore { layout: MatrixLayout },
}

impl LaneMapping {
    /// Map a global thread ID (lane_id) to (M, N) logical index within a fragment
    pub fn get_addr(&self, row: u32, col: u32) -> String {
        // Implementation logic...
        format!("(row * 16 + col)") // Simplified
    }

    /// 検証用：すべての (row, col) が一意なレジスタインデックスにマップされているか確認
    pub fn verify_injectivity(&self) -> Result<(), String> {
        let mut seen = std::collections::HashSet::new();
        // 典型的なタイルサイズ内での検証
        for r in 0..16 {
            for c in 0..16 {
                let addr = self.get_addr(r, c);
                if !seen.insert(addr.clone()) {
                    return Err(format!("Collision detected at ({}, {}) for addr {}", r, c, addr));
                }
            }
        }
        Ok(())
    }

    /// Map a global thread ID (lane_id) to (M, N) logical index within a fragment
    pub fn map(&self, lane_id: u32) -> (u32, u32) {
        match self {
            LaneMapping::Linear => (lane_id, 0),
            LaneMapping::TensorCore { m_per_lane, n_per_lane } => {
                let warp_id = lane_id / 32;
                let lane_in_warp = lane_id % 32;
                let m = (lane_in_warp / 4) + (warp_id * m_per_lane);
                let n = (lane_in_warp % 4) * n_per_lane;
                (m, n)
            }
            LaneMapping::MatrixCore { layout } => match layout {
                MatrixLayout::Amd16x16F32 => {
                    // AMD 16x16x16 v_mfma layout:
                    // Wavefront of 64 threads. 
                    // Each thread corresponds to 4 elements in a 16x16 result matrix.
                    // (Actually v_mfma_f32_16x16x16f32 produces 4 registers per thread)
                    let lane_in_wave = lane_id % 64;
                    let thread_row = lane_in_wave % 16;
                    let thread_col_group = lane_in_wave / 16; // 0..3
                    // In MFMA, one thread might handle (row, col_group*4 + 0..3)
                    (thread_row, thread_col_group * 4) 
                }
                MatrixLayout::Amd32x32F32 => {
                    // AMD 32x32x8 v_mfma layout:
                    let lane_in_wave = lane_id % 64;
                    let thread_row = lane_in_wave % 32;
                    let thread_col_group = lane_in_wave / 32; // 0..1
                    (thread_row, thread_col_group * 16)
                }
                MatrixLayout::IntelXmx16x16F32 => {
                    // Intel XMX 16x16 layout: 
                    // Usually 16 lanes per sub-group
                    let lane_in_sg = lane_id % 16;
                    (lane_in_sg, 0) // Simplified: Intel handles inner col mapping in hardware
                }
                MatrixLayout::IntelXmx8x8F32 => {
                    let lane_in_sg = lane_id % 8;
                    (lane_in_sg, 0)
                }
            },
        }
    }
}
