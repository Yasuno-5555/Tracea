use crate::emitter::traits::{Emitter, UnifiedOpIR};
use crate::emitter::cuda::CUDAEmitter;
use crate::emitter::rocm::ROCMEmitter;
use crate::emitter::metal::MetalEmitter;
use crate::runtime::manager::DeviceBackend;

pub struct UniversalEmitter {
    pub backend: DeviceBackend,
}

impl UniversalEmitter {
    pub fn new(backend: DeviceBackend) -> Self {
        Self { backend }
    }

    pub fn generate(&self, ir: UnifiedOpIR) -> String {
        if let crate::emitter::traits::UnifiedOpType::Elementwise { .. } = ir.op_type {
            return crate::emitter::elementwise::generate_elementwise(&ir);
        }
        if let crate::emitter::traits::UnifiedOpType::Conv2d { .. } = ir.op_type {
            return crate::emitter::conv::generate_conv(&ir);
        }

        match self.backend {
            DeviceBackend::Cuda => {
                let emitter = CUDAEmitter::new();
                emitter.generate_from_ir(&ir)
            }
            DeviceBackend::Rocm => {
                let emitter = ROCMEmitter::detect();
                emitter.generate_from_ir(&ir)
            }
            DeviceBackend::Metal => {
                let emitter = MetalEmitter::detect();
                emitter.generate_from_ir(&ir)
            }
            DeviceBackend::Cpu => {
                "/* CPU implementation is static */".to_string()
            }
        }
    }
}
