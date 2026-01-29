use tracea::runtime::manager::{RuntimeManager, DeviceBackend, KernelArg};

fn main() -> Result<(), String> {
    println!("Direct NVRTC call...");
    let _ = cudarc::nvrtc::compile_ptx("extern \"C\" __global__ void test() {}");
    println!("Direct NVRTC success!");

    let runtime = RuntimeManager::init(Some(DeviceBackend::Cuda))?;
    let source = r#"
extern "C" __global__ void hello_kernel(float* out) {
    out[threadIdx.x] = (float)threadIdx.x;
}
"#;
    println!("Compiling hello_kernel...");
    let kernel_id = runtime.compile(source, "hello_kernel", DeviceBackend::Cuda)?;
    println!("Loading hello_kernel...");
    
    let buf = runtime.alloc_f32(10, DeviceBackend::Cuda)?;
    let grid = (1, 1, 1);
    let block = (10, 1, 1);
    
    runtime.launch(kernel_id, grid, block, 0, vec![KernelArg::Buffer(buf)])?;
    runtime.synchronize();
    
    let mut res = vec![0.0f32; 10];
    runtime.copy_from_device(buf, &mut res)?;
    println!("Result: {:?}", res);
    Ok(())
}
