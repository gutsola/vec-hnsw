use std::cmp::{Ordering, Reverse};
use std::collections::{BinaryHeap, HashSet};

use crate::metric::DistanceMetric;

use super::graph::HnswGraph;

#[derive(Debug, Clone, Copy)]
pub(crate) struct Candidate {
    pub(crate) index: usize,
    pub(crate) distance: f32,
}

impl PartialEq for Candidate {
    fn eq(&self, other: &Self) -> bool {
        self.index == other.index && self.distance.total_cmp(&other.distance).is_eq()
    }
}

impl Eq for Candidate {}

impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> Ordering {
        self.distance
            .total_cmp(&other.distance)
            .then_with(|| self.index.cmp(&other.index))
    }
}

pub(crate) fn greedy_search<M: DistanceMetric>(
    graph: &HnswGraph,
    metric: &M,
    query: &[f32],
    entry_point: usize,
    level: usize,
) -> usize {
    let mut current = entry_point;
    let mut current_distance = metric.distance(query, graph.node(current).vector());

    loop {
        let mut improved = false;

        for &neighbor in graph.node(current).neighbors(level) {
            let distance = metric.distance(query, graph.node(neighbor).vector());
            if distance < current_distance {
                current = neighbor;
                current_distance = distance;
                improved = true;
            }
        }

        if !improved {
            return current;
        }
    }
}

pub(crate) fn search_layer<M: DistanceMetric>(
    graph: &HnswGraph,
    metric: &M,
    query: &[f32],
    entry_point: usize,
    ef: usize,
    level: usize,
) -> Vec<Candidate> {
    let entry_distance = metric.distance(query, graph.node(entry_point).vector());
    let entry = Candidate {
        index: entry_point,
        distance: entry_distance,
    };

    let mut visited = HashSet::from([entry_point]);
    let mut candidates = BinaryHeap::from([Reverse(entry)]);
    let mut nearest = BinaryHeap::from([entry]);

    while let Some(Reverse(candidate)) = candidates.pop() {
        let Some(worst_nearest) = nearest.peek() else {
            break;
        };

        if candidate.distance > worst_nearest.distance {
            break;
        }

        for &neighbor in graph.node(candidate.index).neighbors(level) {
            if !visited.insert(neighbor) {
                continue;
            }

            let distance = metric.distance(query, graph.node(neighbor).vector());
            let neighbor_candidate = Candidate {
                index: neighbor,
                distance,
            };

            if nearest.len() < ef || distance < nearest.peek().map_or(f32::INFINITY, |c| c.distance)
            {
                candidates.push(Reverse(neighbor_candidate));
                nearest.push(neighbor_candidate);

                if nearest.len() > ef {
                    nearest.pop();
                }
            }
        }
    }

    nearest.into_sorted_vec()
}
