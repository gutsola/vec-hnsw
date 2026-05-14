use crate::vector::{Vector, VectorId};

#[derive(Debug, Clone)]
pub(crate) struct Node {
    id: VectorId,
    vector: Vector,
    neighbors: Vec<Vec<usize>>,
    deleted: bool,
}

impl Node {
    pub(crate) fn new(id: VectorId, vector: Vector, level: usize) -> Self {
        Self {
            id,
            vector,
            neighbors: vec![Vec::new(); level + 1],
            deleted: false,
        }
    }

    pub(crate) fn id(&self) -> VectorId {
        self.id
    }

    pub(crate) fn vector(&self) -> &[f32] {
        &self.vector
    }

    pub(crate) fn neighbors(&self, level: usize) -> &[usize] {
        self.neighbors
            .get(level)
            .map(Vec::as_slice)
            .unwrap_or_default()
    }

    pub(crate) fn neighbors_mut(&mut self, level: usize) -> Option<&mut Vec<usize>> {
        self.neighbors.get_mut(level)
    }

    pub(crate) fn is_deleted(&self) -> bool {
        self.deleted
    }

    pub(crate) fn mark_deleted(&mut self) {
        self.deleted = true;
    }
}
