use super::capabilities::{TraceaCapabilities, BackendCapabilities};
use super::registry::BackendKind;
use cudarc::driver::{CudaDevice, sys};
use serde::{Serialize, Deserialize};
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

// Mock/Stub logic for backends not available in this environment
// In a real scenario, we'd use conditional compilation or dynamic loading (dlopen)

pub fn get_capabilities() -> TraceaCapabilities {
    let mut backends = Vec::new();

    // 1. CPU (Always present)
    backends.push(detect_cpu());

    // 2. CUDA
    #[cfg(not(target_os = "macos"))]
    if let Some(cuda_cap) = detect_cuda() {
        backends.push(cuda_cap);
    }

    // 3. ROCm
    if let Some(rocm_cap) = detect_rocm() {
        backends.push(rocm_cap);
    }

    // 4. Metal
    if let Some(metal_cap) = detect_metal() {
        backends.push(metal_cap);
    }

    // 5. Vulkan
    // 5. Vulkan
    #[cfg(feature = "vulkan")]
    if let Some(vulkan_cap) = detect_vulkan() {
        backends.push(vulkan_cap);
    }

    // Calculate Environment ID
    let mut env_id = [0u8; 32];
    calculate_env_id_from_backends(&backends, &mut env_id);

    TraceaCapabilities {
        env_id,
        backends,
    }
}

fn detect_cpu() -> BackendCapabilities {
    let core_count = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1) as u32;

    let simd_width_bits = {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
             if is_x86_feature_detected!("avx512f") { 512 }
             else if is_x86_feature_detected!("avx2") { 256 }
             else if is_x86_feature_detected!("sse2") { 128 }
             else { 0 }
        }
        #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
        {
            128
        }
    };

    // L2 cache approximation or lookup is hard without external crate. 
    // We'll use a conservative default or 0 (indicating look elsewhere/unbounded-ish).
    // Prompt says "L1/L2 cache size (approx max_shared_mem)".
    // Let's assume a standard 256KB L2 per core or similar, or just return 0 if unknown.
    // For "simulated" purpose, let's put 1MB (1024*1024).
    let l2_cache_bytes = 1024 * 1024;

    BackendCapabilities {
        backend: BackendKind::Cpu,
        max_shared_mem: l2_cache_bytes, 
        warp_or_wavefront: 1, // CPU threads are scalar from vector perspective (or 1 SIMD unit)
        has_tensor_core_like: false,
        arch_code: 0,
        driver_or_runtime_version: 0,
        simd_width_bits,
        core_count,
        current_occupancy: Some(0.0),
        register_pressure: Some(0),
    }
}

fn detect_cuda() -> Option<BackendCapabilities> {
    // Try via cudarc
    match CudaDevice::new(0) {
        Ok(dev) => {
            let major = dev.attribute(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR).unwrap_or(0) as u32;
            let minor = dev.attribute(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR).unwrap_or(0) as u32;
            let sm = major * 10 + minor;
            
            // max_shared_memory_per_block_optin is usually the max configurable
            let max_smem = dev.attribute(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN).unwrap_or(49152) as u32;
            let warp_size = dev.attribute(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_WARP_SIZE).unwrap_or(32) as u32;

             let mut driver_v: i32 = 0;
            unsafe { sys::lib().cuDriverGetVersion(&mut driver_v) };

            let has_tensor_core = major >= 7;

            Some(BackendCapabilities {
                backend: BackendKind::Cuda,
                max_shared_mem: max_smem,
                warp_or_wavefront: warp_size,
                has_tensor_core_like: has_tensor_core,
                arch_code: sm, // Store SM version as arch code
                driver_or_runtime_version: driver_v as u32,
                simd_width_bits: 0,
                core_count: 0,
                current_occupancy: Some(0.0),
                register_pressure: Some(0),
            })
        }
        Err(_) => None,
    }
}

fn detect_rocm() -> Option<BackendCapabilities> {
    // In a real impl, bind to libamdhip64.so / amdhip64.dll logic
    // Checking environment variable for "Simulation"
    if std::env::var("TRACEA_MOCK_ROCM").is_ok() {
        Some(BackendCapabilities {
            backend: BackendKind::Rocm,
            max_shared_mem: 65536, // 64KB LDS
            warp_or_wavefront: 64, // Default wavefront
            has_tensor_core_like: true, // Matrix Cores on CDNA
            arch_code: 900, // gfx900 or similar integer mapping
            driver_or_runtime_version: 50700, // ROCm 5.7
            simd_width_bits: 0,
            core_count: 0,
            current_occupancy: Some(0.0),
            register_pressure: Some(0),
        })
    } else {
        None
    }
}

fn detect_metal() -> Option<BackendCapabilities> {
    if std::env::var("TRACEA_MOCK_METAL").is_ok() {
        Some(BackendCapabilities {
            backend: BackendKind::Metal,
            max_shared_mem: 32768, 
            warp_or_wavefront: 32, 
            has_tensor_core_like: true, // simdgroup_matrix
            arch_code: 13, // Apple M1/M2 family placeholder
            driver_or_runtime_version: 300, 
            simd_width_bits: 0,
            core_count: 0,
            current_occupancy: Some(0.0),
            register_pressure: Some(0),
        })
    } else {
        None
    }
}

#[cfg(feature = "vulkan")]
fn detect_vulkan() -> Option<BackendCapabilities> {
    // Try via ash
    let entry = match unsafe { ash::Entry::load() } {
        Ok(e) => e,
        Err(_) => return None,
    };
    
    let app_info = ash::vk::ApplicationInfo::builder()
        .api_version(ash::vk::make_api_version(0, 1, 1, 0));
    let create_info = ash::vk::InstanceCreateInfo::builder().application_info(&app_info);
    
    let instance = match unsafe { entry.create_instance(&create_info, None) } {
        Ok(i) => i,
        Err(_) => return None,
    };
    
    let p_devices = unsafe { instance.enumerate_physical_devices().unwrap_or_default() };
    if p_devices.is_empty() {
        unsafe { instance.destroy_instance(None); }
        return None;
    }
    
    let p_device = p_devices[0];
    let props = unsafe { instance.get_physical_device_properties(p_device) };
    
    // Check for subgroup properties (requires Vulkan 1.1)
    let mut subgroup_props = ash::vk::PhysicalDeviceSubgroupProperties::default();
    let mut props2 = ash::vk::PhysicalDeviceProperties2::builder()
        .push_next(&mut subgroup_props);
    unsafe { instance.get_physical_device_properties2(p_device, &mut props2); }

    let res = Some(BackendCapabilities {
        backend: BackendKind::Vulkan,
        max_shared_mem: props.limits.max_compute_shared_memory_size as u32,
        warp_or_wavefront: subgroup_props.subgroup_size,
        has_tensor_core_like: false, // Could check for cooperative matrix extensions
        arch_code: props.device_id,
        driver_or_runtime_version: props.driver_version,
        simd_width_bits: 0,
        core_count: 0,
        current_occupancy: Some(0.0),
        register_pressure: Some(0),
    });
    
    unsafe { instance.destroy_instance(None); }
    res
}


fn calculate_env_id_from_backends(backends: &[BackendCapabilities], out_hash: &mut [u8; 32]) {
     // Deterministic hash of all backend info
    let json = serde_json::to_string(backends).unwrap_or_default();
    let bytes = json.as_bytes();
    for (i, &b) in bytes.iter().enumerate() {
        out_hash[i % 32] ^= b;
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ChromeTraceEvent {
    pub name: String,
    pub cat: String,
    pub ph: String, // "B" (Begin) or "E" (End)
    pub ts: u64,    // timestamp (micros)
    pub pid: u32,
    pub tid: u32,
}

pub struct TraceProfiler {
    events: Mutex<Vec<ChromeTraceEvent>>,
    start_time: Instant,
}

static PROFILER: OnceLock<TraceProfiler> = OnceLock::new();

impl TraceProfiler {
    pub fn get() -> &'static Self {
        PROFILER.get_or_init(|| Self {
            events: Mutex::new(Vec::new()),
            start_time: Instant::now(),
        })
    }

    pub fn record(&self, name: String, ph: &str) {
        let ts = self.start_time.elapsed().as_micros() as u64;
        let mut events = self.events.lock().unwrap();
        // Fallback for thread ID if as_u64 is not stable enough or different across platforms
        let tid = format!("{:?}", std::thread::current().id());
        // Simple hash of tid string to u32
        let tid_u32 = tid.chars().fold(0u32, |acc, c| acc.wrapping_add(c as u32));

        events.push(ChromeTraceEvent {
            name,
            cat: "execution".to_string(),
            ph: ph.to_string(),
            ts,
            pid: 1,
            tid: tid_u32,
        });
    }

    pub fn save(&self, path: &str) {
        let events = self.events.lock().unwrap();
        if let Ok(json) = serde_json::to_string(&*events) {
            let _ = std::fs::write(path, json);
            println!("[Doctor] 📊 Saved timeline trace to {}", path);
        }
    }
}
