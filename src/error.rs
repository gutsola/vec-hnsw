use std::fmt;

use crate::vector::VectorId;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VecHnswError {
    DuplicateVectorId(VectorId),
    EmptyVector,
    DimensionMismatch { expected: usize, actual: usize },
    InvalidConfig(&'static str),
    VectorNotFound(VectorId),
}

impl fmt::Display for VecHnswError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DuplicateVectorId(id) => write!(f, "vector id {id} already exists"),
            Self::EmptyVector => write!(f, "vector must not be empty"),
            Self::DimensionMismatch { expected, actual } => {
                write!(
                    f,
                    "vector dimension mismatch: expected {expected}, got {actual}"
                )
            }
            Self::InvalidConfig(message) => write!(f, "invalid HNSW config: {message}"),
            Self::VectorNotFound(id) => write!(f, "vector id {id} was not found"),
        }
    }
}

impl std::error::Error for VecHnswError {}

pub type Result<T> = std::result::Result<T, VecHnswError>;
