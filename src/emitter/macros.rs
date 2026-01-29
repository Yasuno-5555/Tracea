#[macro_export]
macro_rules! tracea_emitter {
    ($name:ident { 
        sync: $sync_fn:expr, 
        epilogue: $epilogue_fn:expr 
    }) => {
        impl Emitter for $name {
            fn emit_sync(&mut self, req: SyncRequirement) -> String {
                let f = $sync_fn;
                f(req)
            }
            fn emit_epilogue(&self, ops: &[EpilogueOp], acc: &str) -> String {
                let f = $epilogue_fn;
                f(ops, acc)
            }
            // Other defaults...
        }
    };
}
