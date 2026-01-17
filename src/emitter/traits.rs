use crate::semantic::transition::SyncRequirement;
use crate::semantic::fusion::EpilogueOp;
use crate::semantic::fragment::{FragmentOp, Fragment};

pub trait Emitter {
    fn emit_sync(&mut self, req: SyncRequirement) -> String;
    fn emit_epilogue(&self, _ops: &[EpilogueOp], _acc_name: &str, _global_n: &str) -> String {
        "".to_string()
    }
    fn emit_fragment_op(&self, _op: FragmentOp, _frags: &[Fragment]) -> String {
        "".to_string()
    }
}
