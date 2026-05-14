use crate::vector::VectorId;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SearchOptions {
    pub k: usize,
    pub ef_search: Option<usize>,
}

impl SearchOptions {
    pub fn new(k: usize) -> Self {
        Self { k, ef_search: None }
    }

    pub fn with_ef_search(mut self, ef_search: usize) -> Self {
        self.ef_search = Some(ef_search);
        self
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct InsertOptions;

#[derive(Debug, Clone, PartialEq)]
pub struct SearchResult {
    pub id: VectorId,
    pub distance: f32,
}
