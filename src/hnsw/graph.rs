use std::collections::HashMap;

use crate::vector::VectorId;

use super::node::Node;

#[derive(Debug, Default)]
pub(crate) struct HnswGraph {
    nodes: Vec<Node>,
    id_to_index: HashMap<VectorId, usize>,
    active_len: usize,
}

impl HnswGraph {
    pub(crate) fn len(&self) -> usize {
        self.active_len
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.active_len == 0
    }

    pub(crate) fn contains_id(&self, id: VectorId) -> bool {
        self.id_to_index.contains_key(&id)
    }

    pub(crate) fn add_node(&mut self, node: Node) -> usize {
        let index = self.nodes.len();
        self.id_to_index.insert(node.id(), index);
        self.nodes.push(node);
        self.active_len += 1;
        index
    }

    pub(crate) fn node(&self, index: usize) -> &Node {
        &self.nodes[index]
    }

    pub(crate) fn node_mut(&mut self, index: usize) -> &mut Node {
        &mut self.nodes[index]
    }

    pub(crate) fn mark_deleted(&mut self, id: VectorId) -> Option<usize> {
        let index = self.id_to_index.remove(&id)?;
        if !self.nodes[index].is_deleted() {
            self.nodes[index].mark_deleted();
            self.active_len -= 1;
        }
        Some(index)
    }

    pub(crate) fn add_directed_edge(&mut self, from: usize, to: usize, level: usize) {
        if from == to {
            return;
        }

        if let Some(neighbors) = self.node_mut(from).neighbors_mut(level) {
            if !neighbors.contains(&to) {
                neighbors.push(to);
            }
        }
    }

    pub(crate) fn add_bidirectional_edge(&mut self, left: usize, right: usize, level: usize) {
        self.add_directed_edge(left, right, level);
        self.add_directed_edge(right, left, level);
    }
}
