use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::fs;
use serde::{Serialize, Deserialize};

// --- Core Enums ---

use super::registry::BackendKind;

// --- Configuration ---

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DoctorConfig {
    pub enable_logging: bool,
    pub save_artifacts: bool, // Corresponds to save_sources, save_ptx etc. unified
    pub strict_mode: bool,
    pub log_dir: PathBuf,
}

impl Default for DoctorConfig {
    fn default() -> Self {
        Self {
            enable_logging: true,
            save_artifacts: true,
            strict_mode: false,
            log_dir: PathBuf::from(".tracea/doctor"),
        }
    }
}

// --- Data Structures ---

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JitResultInfo {
    pub backend: BackendKind,
    pub kernel_name: String,
    pub return_code: i32,
    pub source: String,
    pub stdout: String,
    pub stderr: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssemblerResultInfo {
    pub backend: BackendKind,
    pub arch: String,
    pub return_code: i32,
    pub stderr: String,
    pub ptx_content: String,
    pub cubin_size: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModuleLoadInfo {
    pub backend: BackendKind,
    pub kernel_name: String,
    pub return_code: i32,
    pub error_msg: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KernelLaunchInfo {
    pub backend: BackendKind,
    pub kernel_name: String,
    pub return_code: i32,
    pub last_runtime_error: Option<String>,
    pub grid: (u32, u32, u32),
    pub block: (u32, u32, u32),
    pub smem: u32,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum DoctorErrorKind {
    JitFailure,
    AssemblerFailure,
    ModuleLoadFailure,
    KernelLaunchFailure,
    IllegalMemoryAccess,
    EnvIncompatible,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DoctorArtifacts {
    pub source_path: Option<PathBuf>,
    pub ptx_path: Option<PathBuf>,
    pub asm_log_path: Option<PathBuf>,
    pub cubin_path: Option<PathBuf>,
    pub launch_snapshot_path: Option<PathBuf>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DoctorErrorReport {
    pub kind: DoctorErrorKind,
    pub backend: BackendKind,
    pub kernel_name: Option<String>,
    pub message: String,
    pub suggestion: String,
    pub artifacts: DoctorArtifacts,
}

// --- Environment Report ---

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum EnvStatus {
    Ok,
    Warning,
    Error,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnvironmentReport {
    pub timestamp: u64,
    pub status: EnvStatus,
    pub cuda_version: Option<String>,
    pub ptxas_version: Option<String>,
    pub rocm_version: Option<String>,
    pub metal_version: Option<String>,
    pub gpu_name: Option<String>,
    pub gpu_arch: Option<String>,
    pub driver_version: Option<String>,
    pub gpu_info: Option<String>, // Restored
    pub rocm_info: Option<String>, // Restored
    pub issues: Vec<String>,
    pub summary: String,
}

#[derive(Debug, Default)]
pub struct DoctorState {
    pub last_error: Option<DoctorErrorReport>,
    pub last_env: Option<EnvironmentReport>,
}

// --- Main Doctor Struct ---

#[derive(Debug)]
pub struct Doctor {
    pub config: DoctorConfig,
    pub state: Mutex<DoctorState>,
}

impl Doctor {
    pub fn new(config: DoctorConfig) -> Self {
        if config.enable_logging {
            if let Err(e) = fs::create_dir_all(&config.log_dir) {
                eprintln!("[Doctor] Failed to create log directory {:?}: {}", config.log_dir, e);
            }
        }
        Self {
            config,
            state: Mutex::new(DoctorState::default()),
        }
    }

    pub fn global() -> Arc<Self> {
        Arc::new(Self::new(DoctorConfig::default()))
    }

    pub fn last_error(&self) -> Option<DoctorErrorReport> {
        self.state.lock().unwrap().last_error.clone()
    }

    pub fn is_environment_healthy(&self) -> bool {
        let state = self.state.lock().unwrap();
        if let Some(env) = &state.last_env {
            matches!(env.status, EnvStatus::Ok | EnvStatus::Warning)
        } else {
            // Not diagnosed yet, assume OK
            true
        }
    }

    // --- Hooks ---

    pub fn on_jit_result(&self, info: JitResultInfo) {
        if !self.config.enable_logging { return; }

        let mut artifacts = DoctorArtifacts::default();

        if self.config.save_artifacts {
            let path = self.config.log_dir.join("last_source.cu");
            if fs::write(&path, &info.source).is_ok() {
                artifacts.source_path = Some(path);
            }
        }

        if info.return_code != 0 { // Failure
            let mut suggestion = String::new();
            if info.stderr.contains("identifier") && info.stderr.contains("undefined") {
                suggestion.push_str("Check for missing variable declarations or included headers.\n");
            }
            if info.stderr.contains("syntax error") {
                suggestion.push_str("Syntax error in generated code.\n");
            }

            let report = DoctorErrorReport {
                kind: DoctorErrorKind::JitFailure,
                backend: info.backend,
                kernel_name: Some(info.kernel_name.clone()),
                message: format!("JIT Compilation Failed. Code: {}", info.return_code),
                suggestion: if suggestion.is_empty() { "Check log for details".to_string() } else { suggestion },
                artifacts,
            };

            eprintln!("[Doctor] 🔴 JIT Failure: {}", report.message);
            self.state.lock().unwrap().last_error = Some(report);
        }
    }

    pub fn on_assembler_result(&self, info: AssemblerResultInfo) {
        if !self.config.enable_logging { return; }

        let mut artifacts = DoctorArtifacts::default();
        if self.config.save_artifacts {
             let ptx_path = self.config.log_dir.join("last.ptx");
             if fs::write(&ptx_path, &info.ptx_content).is_ok() { artifacts.ptx_path = Some(ptx_path); }
             
             let log_path = self.config.log_dir.join("last_asm.log");
             if fs::write(&log_path, &info.stderr).is_ok() { artifacts.asm_log_path = Some(log_path); }
        }

        if info.return_code != 0 {
            let mut suggest = String::new();
             if info.stderr.contains("Unresolved extern function") {
                suggest.push_str("You might be calling a device function that isn't linked.\n");
            }
            if info.stderr.contains("Register allocation failed") || info.stderr.contains("implementation limit") {
                suggest.push_str("Kernel uses too many registers. Reduce tile sizes or use #pragma unroll.\n");
            }

             let report = DoctorErrorReport {
                kind: DoctorErrorKind::AssemblerFailure,
                backend: info.backend,
                kernel_name: None, 
                message: format!("Assembler Failed. Arch: {}", info.arch),
                suggestion: if suggest.is_empty() { "Check asm log".to_string() } else { suggest },
                artifacts,
            };
            eprintln!("[Doctor] 🔴 Assembler Failure: {}", report.message);
            self.state.lock().unwrap().last_error = Some(report);
        }
    }

    pub fn on_module_load(&self, info: ModuleLoadInfo) {
        if info.return_code != 0 {
            let mut suggest = String::new();
            if let Some(err) = &info.error_msg {
                if err.contains("NO_BINARY_FOR_GPU") || err.contains("209") {
                    suggest.push_str("PTX/CUBIN architecture mismatch with GPU.\n");
                }
            }

            let report = DoctorErrorReport {
                kind: DoctorErrorKind::ModuleLoadFailure,
                backend: info.backend,
                kernel_name: Some(info.kernel_name),
                message: info.error_msg.unwrap_or_else(|| "Unknown Load Error".to_string()),
                suggestion: if suggest.is_empty() { "Verify GPU arch compatibility".to_string() } else { suggest },
                artifacts: DoctorArtifacts::default(),
            };
            eprintln!("[Doctor] 🔴 Module Load Failure: {}", report.message);
            self.state.lock().unwrap().last_error = Some(report);
        }
    }

    pub fn on_kernel_launch(&self, info: KernelLaunchInfo) {
        if self.config.enable_logging && self.config.save_artifacts {
             // Save launch snapshot
             let launch_data = serde_json::json!({
                 "kernel": info.kernel_name,
                 "grid": info.grid,
                 "block": info.block,
                 "smem": info.smem,
                 "timestamp": std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs()
             });
             let path = self.config.log_dir.join("last_launch.json");
             let _ = fs::write(path, launch_data.to_string());
        }

        if info.return_code != 0 {
             let mut suggest = String::new();
             if let Some(err) = &info.last_runtime_error {
                 if err.contains("INVALID_VALUE") {
                     suggest.push_str("Check grid/block dimensions, shared memory size, or kernel arguments alignment.\n");
                 }
                 if err.contains("LAUNCH_OUT_OF_RESOURCES") {
                     suggest.push_str("Too many threads or shared memory requested for this GPU.\n");
                 }
             }

             let report = DoctorErrorReport {
                kind: DoctorErrorKind::KernelLaunchFailure,
                backend: info.backend,
                kernel_name: Some(info.kernel_name),
                message: info.last_runtime_error.unwrap_or_else(|| "Launch Failed".to_string()),
                suggestion: if suggest.is_empty() { "Check kernel resource usage".to_string() } else { suggest },
                artifacts: DoctorArtifacts::default(),
             };
             eprintln!("[Doctor] 🔴 Kernel Launch Failure: {}", report.message);
             self.state.lock().unwrap().last_error = Some(report);
        }
    }

    pub fn diagnose_environment(&self) -> EnvironmentReport {
        let mut issues = Vec::new();
        let cuda_ver = self.run_cmd("nvcc", &["--version"]);
        let ptxas_ver = self.run_cmd("ptxas", &["--version"]);
        let gpu_info = self.run_cmd("nvidia-smi", &["-q", "-d", "MEMORY,COMPUTE"]);

        if cuda_ver.is_none() {
            issues.push("CUDA Toolkit (nvcc) not found.".to_string());
        }
        if ptxas_ver.is_none() {
             issues.push("ptxas not found.".to_string());
        }

        let mut status = EnvStatus::Ok;
        if !issues.is_empty() {
            status = EnvStatus::Warning;
        }
        // Strict checks could set Error

        let summary = if status == EnvStatus::Ok {
            "Environment Healthy".to_string()
        } else {
            format!("Found {} issues", issues.len())
        };

        let report = EnvironmentReport {
            timestamp: std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs(),
            status,
            cuda_version: cuda_ver,
            ptxas_version: ptxas_ver,
            rocm_version: None, // TODO
            metal_version: None,
            gpu_name: None,     // Parsing needed
            gpu_arch: None,
            driver_version: None, 
            gpu_info: gpu_info, // Populated
            rocm_info: None,
            issues,
            summary,
        };
        
        // Save to file
        if self.config.enable_logging {
             let path = self.config.log_dir.join("env_report.json");
             if let Ok(s) = serde_json::to_string_pretty(&report) {
                 let _ = fs::write(path, s);
             }
        }
        
        // Update state
        self.state.lock().unwrap().last_env = Some(report.clone());
        report
    }

     fn run_cmd(&self, cmd: &str, args: &[&str]) -> Option<String> {
        std::process::Command::new(cmd)
            .args(args)
            .output()
            .ok()
            .and_then(|output| {
                if output.status.success() {
                    Some(String::from_utf8_lossy(&output.stdout).trim().to_string())
                } else {
                    None
                }
            })
    }
}
