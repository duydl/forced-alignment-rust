use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

#[pyfunction]
fn compute_best_path(mfcc1: Vec<Vec<f64>>, mfcc2: Vec<Vec<f64>>, delta: usize) -> PyResult<Vec<(usize, usize)>> {
    // Dummy implementation
    Ok(vec![(0, 0), (1, 1)])
}

#[pymodule]
fn rust_extension(py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(compute_best_path, m)?)?;
    Ok(())
}