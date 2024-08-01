// src/dtw.rs

use ndarray::{Array2, Array1, s};
use std::f64;

pub struct PathCell {
    pub i: usize,
    pub j: usize,
}

pub const MOVE0: usize = 0;
pub const MOVE1: usize = 1;
pub const MOVE2: usize = 2;

pub fn _nonnegative_difference(center_j: usize, half_delta: usize) -> usize {
    center_j.saturating_sub(half_delta)
}

pub fn _three_way_argmin(cost0: f64, cost1: f64, cost2: f64) -> usize {
    if (cost0 <= cost1) && (cost0 <= cost2) {
        MOVE0
    } else if cost1 <= cost2 {
        MOVE1
    } else {
        MOVE2
    }
}

pub fn _three_way_min(cost0: f64, cost1: f64, cost2: f64) -> f64 {
    if (cost0 <= cost1) && (cost0 <= cost2) {
        cost0
    } else if cost1 <= cost2 {
        cost1
    } else {
        cost2
    }
}

pub fn _compute_norm2(mfcc: &Array2<f64>) -> Array1<f64> {
    mfcc.map_axis(ndarray::Axis(0), |column| column.dot(&column).sqrt())
}

pub fn _compute_cost_matrix(
    mfcc1: &Array2<f64>,
    mfcc2: &Array2<f64>,
    delta: usize,
    cost_matrix: &mut Array2<f64>,
    centers: &mut Vec<usize>,
) -> Result<(), &'static str> {
    let norm2_1 = _compute_norm2(mfcc1);
    let norm2_2 = _compute_norm2(mfcc2);

    let n = mfcc1.shape()[1];
    let m = mfcc2.shape()[1];
    let l = mfcc1.shape()[0];

    for i in 0..n {
        let center_j = (m as f64 * (i as f64 / n as f64)).floor() as usize;
        let range_start = _nonnegative_difference(center_j, delta / 2);
        let range_end = (range_start + delta).min(m);
        centers[i] = range_start;

        for j in range_start..range_end {
            let mut sum = 0.0;
            for k in 0..l {
                sum += mfcc1[[k, i]] * mfcc2[[k, j]];
            }
            cost_matrix[[i, j - range_start]] = 1.0 - (sum / (norm2_1[i] * norm2_2[j]));
        }
    }

    Ok(())
}

pub fn _compute_accumulated_cost_matrix_in_place(
    cost_matrix: &mut Array2<f64>,
    centers: &Vec<usize>,
) -> Result<(), &'static str> {
    let n = cost_matrix.shape()[0];
    let delta = cost_matrix.shape()[1];
    let mut current_row = vec![0.0; delta];

    for j in 1..delta {
        cost_matrix[[0, j]] += cost_matrix[[0, j - 1]];
    }

    for i in 1..n {
        let offset = centers[i] - centers[i - 1];
        current_row.copy_from_slice(cost_matrix.slice(s![i, ..]).as_slice().unwrap());

        for j in 0..delta {
            let cost0 = if (j + offset) < delta {
                cost_matrix[[i - 1, j + offset]]
            } else {
                f64::INFINITY
            };

            let cost1 = if j > 0 {
                cost_matrix[[i, j - 1]]
            } else {
                f64::INFINITY
            };

            let cost2 = if j > 0 && (j + offset) >= 1 && (j + offset - 1) < delta {
                cost_matrix[[i - 1, j + offset - 1]]
            } else {
                f64::INFINITY
            };

            cost_matrix[[i, j]] = current_row[j] + _three_way_min(cost0, cost1, cost2);
        }
    }

    Ok(())
}

pub fn _compute_best_path(
    accumulated_cost_matrix: &Array2<f64>,
    centers: &Vec<usize>,
    n: usize,
    delta: usize,
) -> Result<Vec<PathCell>, &'static str> {
    let mut best_path = Vec::new();
    let mut i = n - 1;
    let mut j = centers[i] + delta - 1;

    while i > 0 || j > 0 {
        best_path.push(PathCell { i, j });
        if i == 0 {
            j -= 1;
        } else if j == 0 {
            i -= 1;
        } else {
            let offset = centers[i] - centers[i - 1];
            let r_j = j - centers[i];

            let cost0 = if (r_j + offset) < delta {
                accumulated_cost_matrix[[i - 1, r_j + offset]]
            } else {
                f64::INFINITY
            };

            let cost1 = if r_j > 0 {
                accumulated_cost_matrix[[i, r_j - 1]]
            } else {
                f64::INFINITY
            };

            let cost2 = if r_j > 0 && (r_j + offset) >= 1 && (r_j + offset - 1) < delta {
                accumulated_cost_matrix[[i - 1, r_j + offset - 1]]
            } else {
                f64::INFINITY
            };

            match _three_way_argmin(cost0, cost1, cost2) {
                MOVE0 => i -= 1,
                MOVE1 => j -= 1,
                MOVE2 => {
                    i -= 1;
                    j -= 1;
                }
                _ => {}
            }
        }
    }

    best_path.push(PathCell { i, j });
    best_path.reverse();

    Ok(best_path)
}
