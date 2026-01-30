use std::collections::HashMap;
use super::{RuntimeManager, BufferId, DeviceBackend};
use crate::policy::types::GraphTopology;
use crate::runtime::manager::cache::GraphCacheKey;

impl RuntimeManager {
    pub fn execute_graph(
        &self,
        graph: &GraphTopology,
        input_buffers: &HashMap<u64, BufferId>,
        backend: DeviceBackend,
    ) -> Result<HashMap<u64, BufferId>, String> {
        // 0. Cache Lookup
        // Note: GraphTopology must implement Hash/Eq for this to work.
        // Assuming Phase IX ensured this.
        let cache_key = GraphCacheKey {
            topology: graph.clone(),
            backend,
        };

        // Try Read Lock first
        // We use a scope to drop the read lock before potentially acquiring write lock
        let plan_opt = {
            let cache = self.graph_cache.read().map_err(|_| "Graph Cache Read Lock")?;
            cache.get(&cache_key).cloned()
        };

        let plan = if let Some(p) = plan_opt {
             // Hit
             println!("[Cache] ⚡️ Graph Cache HIT");
             p
        } else {
             // Miss -> Compile (Slow Path)
             {
                 println!("[Cache] 🐢 Graph Cache MISS - Compiling...");
                 let compiler = self.compiler.lock().map_err(|_| "Compiler Lock")?;
                 // Pass self as manager for JIT compilation inside compiler
                 let new_plan = compiler.compile(graph.clone(), self, backend)?;
                 
                 // Insert into Cache
                 let mut cache = self.graph_cache.write().map_err(|_| "Graph Cache Write Lock")?;
                 cache.insert(cache_key, new_plan.clone());
                 
                 new_plan
             }
        };

        // 1. Execute (Hot Path)
        self.executor.execute(&plan, input_buffers, self)
    }
}
