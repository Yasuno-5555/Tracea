// src/optimizer/evolution.rs

use std::collections::HashMap;
use std::fs;
use serde::{Serialize, Deserialize};
use crate::core::manifold::ComputeAtom;
use crate::core::lattice::HardwareLattice;
use crate::core::mapper::{MappingStrategy, EvolutionEngine};
use crate::core::evolution::Mutatable;
use rand::prelude::*;

#[derive(Debug, Serialize, Deserialize)]
pub struct DnaDatabase {
    pub strategies: HashMap<String, MappingStrategy>,
}

impl DnaDatabase {
    pub fn load_or_create(path: &str) -> Self {
        if let Ok(content) = fs::read_to_string(path) {
            serde_json::from_str(&content).unwrap_or(Self { strategies: HashMap::new() })
        } else {
            Self { strategies: HashMap::new() }
        }
    }

    pub fn save(&self, path: &str) {
        let dir = std::path::Path::new(path).parent().unwrap();
        if !dir.exists() {
            let _ = fs::create_dir_all(dir);
        }
        if let Ok(content) = serde_json::to_string_pretty(self) {
            let _ = fs::write(path, content);
        }
    }
}

#[derive(Debug)]
pub struct EvolutionaryEngine {
    pub db_path: String,
}

impl EvolutionaryEngine {
    pub fn new(db_path: &str) -> Self {
        Self { db_path: db_path.to_string() }
    }

    /// Primary evolution entry point that runs N generations of mutation/selection
    pub fn search(
        &self,
        atom: &ComputeAtom,
        lattice: &HardwareLattice,
        generations: usize,
        evaluator: impl Fn(&MappingStrategy) -> f32
    ) -> MappingStrategy {
        let mut db = DnaDatabase::load_or_create(&self.db_path);
        let key = format!("{}:{}", atom.name, lattice.name);
        
        let mut best_strategy = db.strategies.get(&key).cloned().unwrap_or_else(|| {
             // Start from an informed baseline if possible, otherwise empty
             MappingStrategy {
                 tile_sizes: atom.domain.dimensions.iter().map(|d| (d.id.clone(), 32)).collect(),
                 loop_order: atom.domain.dimensions.iter().map(|d| d.id.clone()).collect(),
                 spatial_map: HashMap::new(),
             }
        });
        
        let mut best_score = evaluator(&best_strategy);
        let mut rng = StdRng::from_entropy();

        for gen in 0..generations {
            let candidate = best_strategy.mutate(&mut rng);
            let score = evaluator(&candidate);
            
            if score > best_score {
                eprintln!("[Evolution] Gen {}: Found new best strategy! Score: {:.2}", gen, score);
                best_score = score;
                best_strategy = candidate;
            }
        }

        // Persistence
        db.strategies.insert(key, best_strategy.clone());
        db.save(&self.db_path);

        best_strategy
    }
}

impl EvolutionEngine for EvolutionaryEngine {
    fn evolve(&self, atom: &ComputeAtom, lattice: &HardwareLattice) -> MappingStrategy {
        // Default evolution behavior with 5 gens and mock evaluator if not called via search
        self.search(atom, lattice, 5, |_| thread_rng().gen_range(0.0..10.0))
    }
}
