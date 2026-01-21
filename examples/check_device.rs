fn main() {
    println!("Testing CudaDevice::new(0)...");
    let dev = cudarc::driver::CudaDevice::new(0);
    match dev {
        Ok(_) => println!("Success!"),
        Err(e) => println!("Error: {:?}", e),
    }
}
