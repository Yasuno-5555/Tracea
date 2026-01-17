use crate::semantic::transition::SyncRequirement;
use crate::semantic::fusion::EpilogueOp;

pub trait Emitter {
    fn emit_sync(&mut self, req: SyncRequirement) -> String;
    fn emit_epilogue(&self, _ops: &[EpilogueOp], _acc_name: &str) -> String {
        "".to_string()
    }
}
