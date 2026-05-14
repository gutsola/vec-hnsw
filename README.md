# vec-hnsw

Rust library implementing an in-memory **HNSW** (Hierarchical Navigable Small World) index for approximate nearest neighbor search over dense `f32` vectors.

## Features

- **Insert**, **search** (`k` nearest neighbors), **delete** (logical delete), and **update** (delete + re-insert) by stable `VectorId` (`u64`).
- Pluggable **distance metrics**: L2 (squared Euclidean, default), cosine distance, dot-product distance (as a minimization objective).
- Configurable graph parameters: `m`, `ef_construction`, `ef_search`, `max_level`, and a reproducible `level_seed`.
- Optional **`ef_search` override** per query via `SearchOptions`.
- **`storage`** module is reserved for future persistence (WAL, segments); the index itself is not persisted yet.

## Requirements

- Rust toolchain with **Edition 2024** support (see `Cargo.toml`).

## Quick start

Add as a path dependency from another crate:

```toml
[dependencies]
vec-hnsw = { path = "../vec-hnsw" }
```

Example:

```rust
use vec_hnsw::{
    hnsw::{HnswConfig, HnswIndex},
    query::SearchOptions,
};

fn main() -> Result<(), vec_hnsw::error::VecHnswError> {
    let config = HnswConfig::default();
    let mut index = HnswIndex::new(config)?;

    index.insert(1, vec![0.0, 0.0])?;
    index.insert(2, vec![1.0, 0.0])?;

    let hits = index.search(&[0.1, 0.0], 1)?;
    assert_eq!(hits[0].id, 1);

    let hits = index.search_with_options(
        &[0.9, 0.0],
        SearchOptions::new(1).with_ef_search(64),
    )?;
    assert_eq!(hits[0].id, 2);

    Ok(())
}
```

## Development

Run unit tests:

```bash
cargo test
```

## License

This project does not specify a license in the repository; add one before publishing to [crates.io](https://crates.io).
