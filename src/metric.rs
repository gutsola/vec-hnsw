/// Computes a distance where smaller values mean closer vectors.
pub trait DistanceMetric: Clone {
    fn distance(&self, left: &[f32], right: &[f32]) -> f32;
}

#[derive(Debug, Clone, Copy, Default)]
pub struct L2Distance;

impl DistanceMetric for L2Distance {
    fn distance(&self, left: &[f32], right: &[f32]) -> f32 {
        left.iter()
            .zip(right.iter())
            .map(|(left, right)| {
                let diff = left - right;
                diff * diff
            })
            .sum()
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct CosineDistance;

impl DistanceMetric for CosineDistance {
    fn distance(&self, left: &[f32], right: &[f32]) -> f32 {
        let dot = dot_product(left, right);
        let left_norm = squared_norm(left).sqrt();
        let right_norm = squared_norm(right).sqrt();

        if left_norm == 0.0 || right_norm == 0.0 {
            return 1.0;
        }

        1.0 - (dot / (left_norm * right_norm)).clamp(-1.0, 1.0)
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct DotProductDistance;

impl DistanceMetric for DotProductDistance {
    fn distance(&self, left: &[f32], right: &[f32]) -> f32 {
        -dot_product(left, right)
    }
}

fn dot_product(left: &[f32], right: &[f32]) -> f32 {
    left.iter()
        .zip(right.iter())
        .map(|(left, right)| left * right)
        .sum()
}

fn squared_norm(vector: &[f32]) -> f32 {
    vector.iter().map(|value| value * value).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn l2_distance_is_smaller_for_nearer_vectors() {
        let metric = L2Distance;

        assert!(
            metric.distance(&[0.0, 0.0], &[1.0, 1.0]) < metric.distance(&[0.0, 0.0], &[3.0, 3.0])
        );
    }

    #[test]
    fn cosine_distance_is_smaller_for_similar_direction() {
        let metric = CosineDistance;

        assert!(
            metric.distance(&[1.0, 0.0], &[1.0, 0.0]) < metric.distance(&[1.0, 0.0], &[0.0, 1.0])
        );
    }

    #[test]
    fn dot_product_distance_is_smaller_for_larger_dot_product() {
        let metric = DotProductDistance;

        assert!(
            metric.distance(&[1.0, 0.0], &[3.0, 0.0]) < metric.distance(&[1.0, 0.0], &[1.0, 0.0])
        );
    }
}
