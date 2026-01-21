use rayon::prelude::*;
use wide::*;
use std::convert::TryInto;
use crate::optimizer::ProblemDescriptor;
use crate::core::config::PipelineConfig;
use core::arch::asm;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::{_mm_prefetch, _MM_HINT_T0};

/// CPU GEMM with Register Blocking and Optimized Packing
/// This version attempts to reach matrixmultiply levels of performance.

#[target_feature(enable = "avx2,fma")]
unsafe fn micro_kernel_generic<const K_UNROLL: usize, const PF_DIST: usize, const MICRO_M: usize>(
    a: &[f32], // Packed A: [mr x k]
    b: &[f32], // Packed B: [k x nr]
    c: &mut [f32], // Output C: [mr x stride_c]
    k: usize,
    stride_c: usize,
    mr: usize,
) {
    use core::arch::x86_64::{__m256, _mm256_setzero_ps};

    // Accumulators in registers
    // We assume MICRO_M is at most 6 for these 12 YMMs (6 rows x 2 YMMs per row for 16-wide N)
    let mut c00: __m256 = _mm256_setzero_ps(); let mut c01: __m256 = _mm256_setzero_ps();
    let mut c10: __m256 = _mm256_setzero_ps(); let mut c11: __m256 = _mm256_setzero_ps();
    let mut c20: __m256 = _mm256_setzero_ps(); let mut c21: __m256 = _mm256_setzero_ps();
    let mut c30: __m256 = _mm256_setzero_ps(); let mut c31: __m256 = _mm256_setzero_ps();
    let mut c40: __m256 = _mm256_setzero_ps(); let mut c41: __m256 = _mm256_setzero_ps();
    let mut c50: __m256 = _mm256_setzero_ps(); let mut c51: __m256 = _mm256_setzero_ps();

    let mut a_ptr = a.as_ptr();
    let mut b_ptr = b.as_ptr();
    
    let k_main = k / K_UNROLL;
    let k_rem = k % K_UNROLL;

    for _ in 0..k_main {
        for _ in 0..K_UNROLL {
            unsafe {
                if PF_DIST > 0 {
                    _mm_prefetch(b_ptr.add(PF_DIST) as *const i8, _MM_HINT_T0);
                }

                match MICRO_M {
                    6 => {
                        asm!(
                            "vmovups ymm12, ymmword ptr [{b_curr}]",
                            "vmovups ymm13, ymmword ptr [{b_curr} + 32]",
                            "add {b_curr}, 64",
                            "vbroadcastss ymm14, dword ptr [{a_curr}]",
                            "vfmadd231ps ymm0, ymm14, ymm12", "vfmadd231ps ymm1, ymm14, ymm13",
                            "vbroadcastss ymm14, dword ptr [{a_curr} + 4]",
                            "vfmadd231ps ymm2, ymm14, ymm12", "vfmadd231ps ymm3, ymm14, ymm13",
                            "vbroadcastss ymm14, dword ptr [{a_curr} + 8]",
                            "vfmadd231ps ymm4, ymm14, ymm12", "vfmadd231ps ymm5, ymm14, ymm13",
                            "vbroadcastss ymm14, dword ptr [{a_curr} + 12]",
                            "vfmadd231ps ymm6, ymm14, ymm12", "vfmadd231ps ymm7, ymm14, ymm13",
                            "vbroadcastss ymm14, dword ptr [{a_curr} + 16]",
                            "vfmadd231ps ymm8, ymm14, ymm12", "vfmadd231ps ymm9, ymm14, ymm13",
                            "vbroadcastss ymm14, dword ptr [{a_curr} + 20]",
                            "vfmadd231ps ymm10, ymm14, ymm12", "vfmadd231ps ymm11, ymm14, ymm13",
                            "add {a_curr}, 24",
                            a_curr = inout(reg) a_ptr, b_curr = inout(reg) b_ptr,
                            inout("ymm0") c00, inout("ymm1") c01,
                            inout("ymm2") c10, inout("ymm3") c11,
                            inout("ymm4") c20, inout("ymm5") c21,
                            inout("ymm6") c30, inout("ymm7") c31,
                            inout("ymm8") c40, inout("ymm9") c41,
                            inout("ymm10") c50, inout("ymm11") c51,
                            out("ymm12") _, out("ymm13") _, out("ymm14") _,
                        );
                    },
                    4 => {
                        asm!(
                            "vmovups ymm12, ymmword ptr [{b_curr}]",
                            "vmovups ymm13, ymmword ptr [{b_curr} + 32]",
                            "add {b_curr}, 64",
                            "vbroadcastss ymm14, dword ptr [{a_curr}]",
                            "vfmadd231ps ymm0, ymm14, ymm12", "vfmadd231ps ymm1, ymm14, ymm13",
                            "vbroadcastss ymm14, dword ptr [{a_curr} + 4]",
                            "vfmadd231ps ymm2, ymm14, ymm12", "vfmadd231ps ymm3, ymm14, ymm13",
                            "vbroadcastss ymm14, dword ptr [{a_curr} + 8]",
                            "vfmadd231ps ymm4, ymm14, ymm12", "vfmadd231ps ymm5, ymm14, ymm13",
                            "vbroadcastss ymm14, dword ptr [{a_curr} + 12]",
                            "vfmadd231ps ymm6, ymm14, ymm12", "vfmadd231ps ymm7, ymm14, ymm13",
                            "add {a_curr}, 16",
                            a_curr = inout(reg) a_ptr, b_curr = inout(reg) b_ptr,
                            inout("ymm0") c00, inout("ymm1") c01,
                            inout("ymm2") c10, inout("ymm3") c11,
                            inout("ymm4") c20, inout("ymm5") c21,
                            inout("ymm6") c30, inout("ymm7") c31,
                            out("ymm12") _, out("ymm13") _, out("ymm14") _,
                        );
                    },
                    _ => unreachable!(),
                }
            }
        }
    }

    // Remainder loop
    if k_rem > 0 {
        for _ in 0..k_rem {
            unsafe {
                 match MICRO_M {
                    6 => {
                        asm!(
                            "vmovups ymm12, ymmword ptr [{b_curr}]",
                            "vmovups ymm13, ymmword ptr [{b_curr} + 32]",
                            "add {b_curr}, 64",
                            "vbroadcastss ymm14, dword ptr [{a_curr}]",
                            "vfmadd231ps ymm0, ymm14, ymm12", "vfmadd231ps ymm1, ymm14, ymm13",
                            "vbroadcastss ymm14, dword ptr [{a_curr} + 4]",
                            "vfmadd231ps ymm2, ymm14, ymm12", "vfmadd231ps ymm3, ymm14, ymm13",
                            "vbroadcastss ymm14, dword ptr [{a_curr} + 8]",
                            "vfmadd231ps ymm4, ymm14, ymm12", "vfmadd231ps ymm5, ymm14, ymm13",
                            "vbroadcastss ymm14, dword ptr [{a_curr} + 12]",
                            "vfmadd231ps ymm6, ymm14, ymm12", "vfmadd231ps ymm7, ymm14, ymm13",
                            "vbroadcastss ymm14, dword ptr [{a_curr} + 16]",
                            "vfmadd231ps ymm8, ymm14, ymm12", "vfmadd231ps ymm9, ymm14, ymm13",
                            "vbroadcastss ymm14, dword ptr [{a_curr} + 20]",
                            "vfmadd231ps ymm10, ymm14, ymm12", "vfmadd231ps ymm11, ymm14, ymm13",
                            "add {a_curr}, 24",
                            a_curr = inout(reg) a_ptr, b_curr = inout(reg) b_ptr,
                            inout("ymm0") c00, inout("ymm1") c01,
                            inout("ymm2") c10, inout("ymm3") c11,
                            inout("ymm4") c20, inout("ymm5") c21,
                            inout("ymm6") c30, inout("ymm7") c31,
                            inout("ymm8") c40, inout("ymm9") c41,
                            inout("ymm10") c50, inout("ymm11") c51,
                            out("ymm12") _, out("ymm13") _, out("ymm14") _,
                        );
                    },
                    4 => {
                        asm!(
                            "vmovups ymm12, ymmword ptr [{b_curr}]",
                            "vmovups ymm13, ymmword ptr [{b_curr} + 32]",
                            "add {b_curr}, 64",
                            "vbroadcastss ymm14, dword ptr [{a_curr}]",
                            "vfmadd231ps ymm0, ymm14, ymm12", "vfmadd231ps ymm1, ymm14, ymm13",
                            "vbroadcastss ymm14, dword ptr [{a_curr} + 4]",
                            "vfmadd231ps ymm2, ymm14, ymm12", "vfmadd231ps ymm3, ymm14, ymm13",
                            "vbroadcastss ymm14, dword ptr [{a_curr} + 8]",
                            "vfmadd231ps ymm4, ymm14, ymm12", "vfmadd231ps ymm5, ymm14, ymm13",
                            "vbroadcastss ymm14, dword ptr [{a_curr} + 12]",
                            "vfmadd231ps ymm6, ymm14, ymm12", "vfmadd231ps ymm7, ymm14, ymm13",
                            "add {a_curr}, 16",
                            a_curr = inout(reg) a_ptr, b_curr = inout(reg) b_ptr,
                            inout("ymm0") c00, inout("ymm1") c01,
                            inout("ymm2") c10, inout("ymm3") c11,
                            inout("ymm4") c20, inout("ymm5") c21,
                            inout("ymm6") c30, inout("ymm7") c31,
                            out("ymm12") _, out("ymm13") _, out("ymm14") _,
                        );
                    },
                    _ => unreachable!(),
                 }
            }
        }
    }

    // Store back to C
    let mut write_c = |row: usize, r0: __m256, r1: __m256| {
        if row < mr {
            let row_ptr = &mut c[row * stride_c..];
            let res0: [f32; 8] = core::mem::transmute(r0);
            let res1: [f32; 8] = core::mem::transmute(r1);
            for j in 0..8 { row_ptr[j] += res0[j]; }
            for j in 0..8 { row_ptr[j+8] += res1[j]; }
        }
    };

    write_c(0, c00, c01);
    write_c(1, c10, c11);
    write_c(2, c20, c21);
    write_c(3, c30, c31);
    if MICRO_M >= 5 { write_c(4, c40, c41); }
    if MICRO_M >= 6 { write_c(5, c50, c51); }
}

#[macro_export]
macro_rules! dispatch_kernel {
    ($cfg:expr, $a:expr, $b:expr, $c:expr, $k:expr, $sc:expr, $mr:expr) => {
        match ($cfg.k_unroll, $cfg.prefetch_distance, $cfg.micro_m) {
            (1, 0, 6) => unsafe { micro_kernel_generic::<1, 0, 6>($a, $b, $c, $k, $sc, $mr) },
            (2, 64, 6) => unsafe { micro_kernel_generic::<2, 64, 6>($a, $b, $c, $k, $sc, $mr) },
            (4, 128, 6) => unsafe { micro_kernel_generic::<4, 128, 6>($a, $b, $c, $k, $sc, $mr) },
            (2, 64, 4) => unsafe { micro_kernel_generic::<2, 64, 4>($a, $b, $c, $k, $sc, $mr) },
            _ => unsafe { micro_kernel_generic::<1, 0, 6>($a, $b, $c, $k, $sc, $mr) },
        }
    };
}

// Optimized packing for A (Mc x Kc) -> Panels of Mr x Kc
fn pack_a_panel(src: &[f32], dst: &mut [f32], mc: usize, kc: usize, stride_a: usize, mr: usize) {
    let mut dst_idx = 0;
    for mi in (0..mc).step_by(mr) {
        for l in 0..kc {
            for row in 0..mr {
                if mi + row < mc {
                    dst[dst_idx] = src[(mi + row) * stride_a + l];
                } else {
                    dst[dst_idx] = 0.0;
                }
                dst_idx += 1;
            }
        }
    }
}

// Optimized packing for B (Kc x Nc) -> Panels of Kc x Nr
fn pack_b_panel(src: &[f32], dst: &mut [f32], kc: usize, nc: usize, stride_b: usize) {
    let mut dst_idx = 0;
    for ni in (0..nc).step_by(16) {
        for l in 0..kc {
            for col in 0..16 {
                if ni + col < nc {
                    dst[dst_idx] = src[l * stride_b + (ni + col)];
                } else {
                    dst[dst_idx] = 0.0;
                }
                dst_idx += 1;
            }
        }
    }
}

pub fn gemm_cpu_packed(
    problem: &ProblemDescriptor,
    config: &PipelineConfig,
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
) {
    let m = problem.shape.m as usize;
    let n = problem.shape.n as usize;
    let k = problem.shape.k as usize;
    
    let mc = config.m_tile as usize;
    let nc = config.n_tile as usize;
    let kc = config.k_tile as usize;
    let mr = config.micro_m as usize;
    
    c.par_chunks_mut(mc * n).enumerate().for_each(|(mb, c_block)| {
        let m_start = mb * mc;
        let actual_mc = (mc).min(m - m_start);
        
        let mut packed_a = vec![0.0f32; ((mc + mr - 1) / mr * mr) * kc];
        let mut packed_b = vec![0.0f32; ((nc + 15) / 16 * 16) * kc];

        for k_idx in (0..k).step_by(kc) {
            let current_kc = (kc).min(k - k_idx);
            
            // Pack A once per K-step for this M-block
            pack_a_panel(&a[m_start * k + k_idx..], &mut packed_a, actual_mc, current_kc, k, mr);

            for nb in (0..n).step_by(nc) {
                let actual_nc = (nc).min(n - nb);
                
                // Pack B for this N-block
                pack_b_panel(&b[k_idx * n + nb..], &mut packed_b, current_kc, actual_nc, n);

                for mi in (0..actual_mc).step_by(mr) {
                    let current_mr = (mr).min(actual_mc - mi);
                    let a_panel = &packed_a[ (mi/mr) * current_kc * mr .. ];
                    
                    for ni in (0..actual_nc).step_by(16) {
                        let b_panel = &packed_b[ (ni/16) * current_kc * 16 .. ];
                        let c_idx = mi * n + ni;
                        
                        dispatch_kernel!(
                            config,
                            a_panel,
                            b_panel,
                            &mut c_block[c_idx..],
                            current_kc,
                            n,
                            current_mr
                        );
                    }
                }
            }
        }
    });
}
