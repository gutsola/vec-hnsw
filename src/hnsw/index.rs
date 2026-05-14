use crate::{
    error::{Result, VecHnswError},
    metric::{DistanceMetric, L2Distance},
    query::{InsertOptions, SearchOptions, SearchResult},
    vector::{Vector, VectorId},
};

use super::{
    graph::HnswGraph,
    node::Node,
    search::{Candidate, greedy_search, search_layer},
};

#[derive(Debug, Clone)]
pub struct HnswConfig {
    pub m: usize,
    pub ef_construction: usize,
    pub ef_search: usize,
    pub max_level: usize,
    pub level_seed: u64,
}

impl Default for HnswConfig {
    fn default() -> Self {
        Self {
            m: 16,
            ef_construction: 64,
            ef_search: 32,
            max_level: 16,
            level_seed: 0x9e37_79b9_7f4a_7c15,
        }
    }
}

pub struct HnswIndex<M = L2Distance> {
    graph: HnswGraph,
    entry_point: Option<usize>,
    max_level: usize,
    dimension: Option<usize>,
    config: HnswConfig,
    metric: M,
    level_generator: LevelGenerator,
}

impl HnswIndex<L2Distance> {
    pub fn new(config: HnswConfig) -> Result<Self> {
        Self::with_metric(config, L2Distance)
    }
}

impl<M: DistanceMetric> HnswIndex<M> {
    pub fn with_metric(config: HnswConfig, metric: M) -> Result<Self> {
        validate_config(&config)?;
        let level_generator = LevelGenerator::new(config.level_seed);

        Ok(Self {
            graph: HnswGraph::default(),
            entry_point: None,
            max_level: 0,
            dimension: None,
            config,
            metric,
            level_generator,
        })
    }

    pub fn len(&self) -> usize {
        self.graph.len()
    }

    pub fn is_empty(&self) -> bool {
        self.graph.is_empty()
    }

    pub fn insert(&mut self, id: VectorId, vector: Vector) -> Result<()> {
        self.insert_with_options(id, vector, InsertOptions)
    }

    pub fn insert_with_options(
        &mut self,
        id: VectorId,
        vector: Vector,
        _options: InsertOptions,
    ) -> Result<()> {
        if self.graph.contains_id(id) {
            return Err(VecHnswError::DuplicateVectorId(id));
        }

        self.validate_vector(&vector)?;
        if self.dimension.is_none() {
            self.dimension = Some(vector.len());
        }

        let node_level = self.level_generator.random_level(&self.config);
        let new_index = self.graph.add_node(Node::new(id, vector, node_level));

        let Some(mut current_entry) = self.entry_point else {
            self.entry_point = Some(new_index);
            self.max_level = node_level;
            return Ok(());
        };

        let old_max_level = self.max_level;

        for level in ((node_level + 1)..=old_max_level).rev() {
            current_entry = greedy_search(
                &self.graph,
                &self.metric,
                self.graph.node(new_index).vector(),
                current_entry,
                level,
            );
        }

        for level in (0..=node_level.min(old_max_level)).rev() {
            let ef = self.config.ef_construction.max(self.config.m);
            let candidates = search_layer(
                &self.graph,
                &self.metric,
                self.graph.node(new_index).vector(),
                current_entry,
                ef,
                level,
            );
            let selected = self.select_neighbors(
                self.graph.node(new_index).vector(),
                candidates,
                self.config.m,
            );

            for neighbor in selected.iter().map(|candidate| candidate.index) {
                self.graph
                    .add_bidirectional_edge(new_index, neighbor, level);
                self.prune_neighbors(neighbor, level);
            }

            if let Some(candidate) = selected.first() {
                current_entry = candidate.index;
            }
        }

        if node_level > old_max_level {
            self.entry_point = Some(new_index);
            self.max_level = node_level;
        }

        Ok(())
    }

    pub fn delete(&mut self, id: VectorId) -> Result<()> {
        self.graph
            .mark_deleted(id)
            .map(|_| ())
            .ok_or(VecHnswError::VectorNotFound(id))
    }

    pub fn update(&mut self, id: VectorId, vector: Vector) -> Result<()> {
        if !self.graph.contains_id(id) {
            return Err(VecHnswError::VectorNotFound(id));
        }

        self.validate_vector(&vector)?;
        self.delete(id)?;
        self.insert(id, vector)
    }

    pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<SearchResult>> {
        self.search_with_options(query, SearchOptions::new(k))
    }

    pub fn search_with_options(
        &self,
        query: &[f32],
        options: SearchOptions,
    ) -> Result<Vec<SearchResult>> {
        let k = options.k;
        if k == 0 {
            return Ok(Vec::new());
        }

        self.validate_query(query)?;

        let Some(mut current_entry) = self.entry_point else {
            return Ok(Vec::new());
        };

        for level in (1..=self.max_level).rev() {
            current_entry = greedy_search(&self.graph, &self.metric, query, current_entry, level);
        }

        if self.graph.is_empty() {
            return Ok(Vec::new());
        }

        let ef = options.ef_search.unwrap_or(self.config.ef_search).max(k);
        let mut candidates = search_layer(&self.graph, &self.metric, query, current_entry, ef, 0);

        Ok(candidates
            .drain(..)
            .filter_map(|candidate| {
                let node = self.graph.node(candidate.index);
                (!node.is_deleted()).then_some(SearchResult {
                    id: node.id(),
                    distance: candidate.distance,
                })
            })
            .take(k)
            .collect())
    }

    fn validate_vector(&self, vector: &[f32]) -> Result<()> {
        if vector.is_empty() {
            return Err(VecHnswError::EmptyVector);
        }

        if let Some(expected) = self.dimension {
            if vector.len() != expected {
                return Err(VecHnswError::DimensionMismatch {
                    expected,
                    actual: vector.len(),
                });
            }
        }

        Ok(())
    }

    fn validate_query(&self, query: &[f32]) -> Result<()> {
        if query.is_empty() {
            return Err(VecHnswError::EmptyVector);
        }

        if let Some(expected) = self.dimension {
            if query.len() != expected {
                return Err(VecHnswError::DimensionMismatch {
                    expected,
                    actual: query.len(),
                });
            }
        }

        Ok(())
    }

    fn select_neighbors(
        &self,
        base_vector: &[f32],
        mut candidates: Vec<Candidate>,
        limit: usize,
    ) -> Vec<Candidate> {
        candidates.sort();
        let mut selected: Vec<Candidate> = Vec::with_capacity(limit);

        for candidate in candidates.iter().copied() {
            if selected.len() >= limit {
                break;
            }

            let candidate_vector = self.graph.node(candidate.index).vector();
            let is_diverse = selected.iter().all(|selected_candidate| {
                let selected_vector = self.graph.node(selected_candidate.index).vector();
                let distance_to_selected = self.metric.distance(candidate_vector, selected_vector);
                distance_to_selected > self.metric.distance(candidate_vector, base_vector)
            });

            if is_diverse {
                selected.push(candidate);
            }
        }

        if selected.len() < limit {
            for candidate in candidates {
                if selected
                    .iter()
                    .any(|selected| selected.index == candidate.index)
                {
                    continue;
                }

                selected.push(candidate);

                if selected.len() >= limit {
                    break;
                }
            }
        }

        selected
    }

    fn prune_neighbors(&mut self, index: usize, level: usize) {
        let base_vector = self.graph.node(index).vector().to_vec();
        let candidates = self
            .graph
            .node(index)
            .neighbors(level)
            .iter()
            .map(|&neighbor| Candidate {
                index: neighbor,
                distance: self
                    .metric
                    .distance(&base_vector, self.graph.node(neighbor).vector()),
            })
            .collect();
        let neighbors = self
            .select_neighbors(&base_vector, candidates, self.config.m)
            .into_iter()
            .map(|candidate| candidate.index)
            .collect();

        if let Some(slot) = self.graph.node_mut(index).neighbors_mut(level) {
            *slot = neighbors;
        }
    }
}

fn validate_config(config: &HnswConfig) -> Result<()> {
    if config.m == 0 {
        return Err(VecHnswError::InvalidConfig("m must be greater than 0"));
    }

    if config.ef_construction == 0 {
        return Err(VecHnswError::InvalidConfig(
            "ef_construction must be greater than 0",
        ));
    }

    if config.ef_search == 0 {
        return Err(VecHnswError::InvalidConfig(
            "ef_search must be greater than 0",
        ));
    }

    if config.max_level == 0 {
        return Err(VecHnswError::InvalidConfig(
            "max_level must be greater than 0",
        ));
    }

    Ok(())
}

#[derive(Debug, Clone)]
struct LevelGenerator {
    state: u64,
}

impl LevelGenerator {
    fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 { 1 } else { seed },
        }
    }

    fn random_level(&mut self, config: &HnswConfig) -> usize {
        let promotion_probability = 1.0 / config.m.max(2) as f32;
        let mut level = 0;

        while level < config.max_level && self.next_f32() < promotion_probability {
            level += 1;
        }

        level
    }

    fn next_f32(&mut self) -> f32 {
        let value = self.next_u64() >> 40;
        value as f32 / (1_u64 << 24) as f32
    }

    fn next_u64(&mut self) -> u64 {
        let mut value = self.state;
        value ^= value >> 12;
        value ^= value << 25;
        value ^= value >> 27;
        self.state = value;
        value.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metric::{CosineDistance, DotProductDistance};

    fn test_config() -> HnswConfig {
        HnswConfig {
            m: 4,
            ef_construction: 16,
            ef_search: 16,
            max_level: 8,
            level_seed: 42,
        }
    }

    #[test]
    fn empty_index_returns_no_results() {
        let index = HnswIndex::new(test_config()).unwrap();

        let results = index.search(&[1.0, 2.0], 3).unwrap();

        assert!(results.is_empty());
    }

    #[test]
    fn inserted_vector_can_be_found() {
        let mut index = HnswIndex::new(test_config()).unwrap();
        index.insert(7, vec![1.0, 2.0]).unwrap();

        let results = index.search(&[1.0, 2.0], 1).unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, 7);
        assert_eq!(results[0].distance, 0.0);
    }

    #[test]
    fn search_returns_nearest_vectors_first() {
        let mut index = HnswIndex::new(test_config()).unwrap();
        index.insert(1, vec![0.0, 0.0]).unwrap();
        index.insert(2, vec![10.0, 10.0]).unwrap();
        index.insert(3, vec![1.0, 1.0]).unwrap();

        let results = index.search(&[0.2, 0.2], 2).unwrap();

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].id, 1);
        assert_eq!(results[1].id, 3);
    }

    #[test]
    fn duplicate_vector_id_returns_error() {
        let mut index = HnswIndex::new(test_config()).unwrap();
        index.insert(1, vec![0.0, 0.0]).unwrap();

        let error = index.insert(1, vec![1.0, 1.0]).unwrap_err();

        assert_eq!(error, VecHnswError::DuplicateVectorId(1));
    }

    #[test]
    fn dimension_mismatch_returns_error() {
        let mut index = HnswIndex::new(test_config()).unwrap();
        index.insert(1, vec![0.0, 0.0]).unwrap();

        let error = index.search(&[0.0, 0.0, 0.0], 1).unwrap_err();

        assert_eq!(
            error,
            VecHnswError::DimensionMismatch {
                expected: 2,
                actual: 3
            }
        );
    }

    #[test]
    fn k_larger_than_len_returns_available_results() {
        let mut index = HnswIndex::new(test_config()).unwrap();
        index.insert(1, vec![0.0, 0.0]).unwrap();
        index.insert(2, vec![1.0, 1.0]).unwrap();

        let results = index.search(&[0.0, 0.0], 10).unwrap();

        assert_eq!(results.len(), 2);
    }

    #[test]
    fn search_options_can_override_ef_search() {
        let mut index = HnswIndex::new(test_config()).unwrap();
        index.insert(1, vec![0.0, 0.0]).unwrap();
        index.insert(2, vec![1.0, 1.0]).unwrap();

        let results = index
            .search_with_options(&[0.0, 0.0], SearchOptions::new(2).with_ef_search(1))
            .unwrap();

        assert_eq!(results.len(), 2);
    }

    #[test]
    fn m_one_config_is_supported() {
        let mut config = test_config();
        config.m = 1;
        config.ef_construction = 4;
        config.ef_search = 4;
        let mut index = HnswIndex::new(config).unwrap();

        index.insert(1, vec![0.0]).unwrap();
        index.insert(2, vec![1.0]).unwrap();

        let results = index.search(&[0.1], 1).unwrap();

        assert_eq!(results[0].id, 1);
    }

    #[test]
    fn delete_filters_vector_from_search_results() {
        let mut index = HnswIndex::new(test_config()).unwrap();
        index.insert(1, vec![0.0, 0.0]).unwrap();
        index.insert(2, vec![2.0, 2.0]).unwrap();

        index.delete(1).unwrap();
        let results = index.search(&[0.0, 0.0], 2).unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, 2);
        assert_eq!(index.len(), 1);
    }

    #[test]
    fn deleted_vector_id_can_be_inserted_again() {
        let mut index = HnswIndex::new(test_config()).unwrap();
        index.insert(1, vec![0.0, 0.0]).unwrap();

        index.delete(1).unwrap();
        index.insert(1, vec![2.0, 2.0]).unwrap();

        let results = index.search(&[2.0, 2.0], 1).unwrap();
        assert_eq!(results[0].id, 1);
    }

    #[test]
    fn update_replaces_vector_for_existing_id() {
        let mut index = HnswIndex::new(test_config()).unwrap();
        index.insert(1, vec![0.0, 0.0]).unwrap();
        index.insert(2, vec![10.0, 10.0]).unwrap();

        index.update(1, vec![9.0, 9.0]).unwrap();
        let results = index.search(&[9.0, 9.0], 1).unwrap();

        assert_eq!(results[0].id, 1);
    }

    #[test]
    fn deleting_unknown_id_returns_error() {
        let mut index = HnswIndex::new(test_config()).unwrap();

        let error = index.delete(404).unwrap_err();

        assert_eq!(error, VecHnswError::VectorNotFound(404));
    }

    #[test]
    fn cosine_distance_metric_can_be_used() {
        let mut index = HnswIndex::with_metric(test_config(), CosineDistance).unwrap();
        index.insert(1, vec![1.0, 0.0]).unwrap();
        index.insert(2, vec![0.0, 1.0]).unwrap();

        let results = index.search(&[2.0, 0.0], 1).unwrap();

        assert_eq!(results[0].id, 1);
    }

    #[test]
    fn dot_product_distance_metric_can_be_used() {
        let mut index = HnswIndex::with_metric(test_config(), DotProductDistance).unwrap();
        index.insert(1, vec![1.0, 0.0]).unwrap();
        index.insert(2, vec![3.0, 0.0]).unwrap();

        let results = index.search(&[1.0, 0.0], 1).unwrap();

        assert_eq!(results[0].id, 2);
    }

    #[test]
    fn indexes_with_same_seed_return_same_result_order() {
        let mut left = HnswIndex::new(test_config()).unwrap();
        let mut right = HnswIndex::new(test_config()).unwrap();

        for id in 0..20 {
            let vector = vec![id as f32, (id % 3) as f32];
            left.insert(id, vector.clone()).unwrap();
            right.insert(id, vector).unwrap();
        }

        let left_ids: Vec<_> = left
            .search(&[7.2, 1.0], 5)
            .unwrap()
            .into_iter()
            .map(|result| result.id)
            .collect();
        let right_ids: Vec<_> = right
            .search(&[7.2, 1.0], 5)
            .unwrap()
            .into_iter()
            .map(|result| result.id)
            .collect();

        assert_eq!(left_ids, right_ids);
    }

    #[test]
    fn small_linear_dataset_has_good_recall() {
        let mut config = test_config();
        config.m = 8;
        config.ef_construction = 32;
        config.ef_search = 32;
        let mut index = HnswIndex::new(config).unwrap();

        for id in 0..64 {
            index.insert(id, vec![id as f32]).unwrap();
        }

        let result_ids: Vec<_> = index
            .search(&[30.2], 5)
            .unwrap()
            .into_iter()
            .map(|result| result.id)
            .collect();

        assert!(result_ids.contains(&30));
        assert!(result_ids.contains(&31));
    }
}
