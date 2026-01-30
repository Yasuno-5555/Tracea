use std::collections::HashMap;
use crate::policy::types::GraphTopology;
use crate::runtime::manager::DeviceBackend;
use crate::runtime::plan::ExecutionPlan;

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct GraphCacheKey {
    pub topology: GraphTopology,
    pub backend: DeviceBackend,
}

#[derive(Debug)]
pub struct GraphCache {
    cache: std::collections::HashMap<GraphCacheKey, ExecutionPlan>,
}

impl GraphCache {
    pub fn new() -> Self {
        Self {
            cache: HashMap::new(),
        }
    }

    pub fn get(&self, key: &GraphCacheKey) -> Option<&ExecutionPlan> {
        self.cache.get(key)
    }

    pub fn insert(&mut self, key: GraphCacheKey, plan: ExecutionPlan) {
        self.cache.insert(key, plan);
    }
}
