use std::collections::HashSet;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SwizzleMode {
    None,
    /// (row ^ (col >> 3)) << 4 (Simplified XOR for 128B boundaries)
    Xor128,
}

#[derive(Debug)]
pub struct BankConflictSimulator {
    pub num_banks: u32,
    pub bank_width: u32,
}

impl BankConflictSimulator {
    pub fn new(num_banks: u32, bank_width: u32) -> Self {
        Self { num_banks, bank_width }
    }

    /// Simulate memory access and return true if conflicts are detected
    pub fn has_conflicts(&self, thread_indices: &[(u32, u32)], row_stride: u32, mode: SwizzleMode) -> bool {
        let mut banks = HashSet::new();
        for &(row, col) in thread_indices {
            let addr = match mode {
                SwizzleMode::None => (row * row_stride + col) * self.bank_width,
                SwizzleMode::Xor128 => {
                    // Ampere logic: (row ^ (col >> 3)) << 4
                    ((row ^ (col >> 3)) << 4) | (col & 0xF) // Heuristic
                }
            };
            let bank = (addr / self.bank_width) % self.num_banks;
            if !banks.insert(bank) {
                return true; // Conflict
            }
        }
        false
    }
}
