use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use ndarray::{Array2, s};
use std::time::Instant;

mod dtw;
mod tts_engines;

use tts_engines::TTSEngine;
use tts_engines::espeak_ng::EspeakNg;

pub mod mfcc;
pub use mfcc::{compute_mfcc, MfccConfig};

#[pyfunction]
fn text_to_speech(engine: &str, text: String, output_path: String) -> PyResult<()> {
    let tts: Box<dyn TTSEngine> = match engine {
        "espeak-ng" => Box::new(EspeakNg::new()),
        _ => return Err(pyo3::exceptions::PyValueError::new_err("Unsupported TTS engine")),
    };

    tts.speak_to_file(&text, &output_path)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;

    Ok(())
}


#[pyfunction]
fn compute_best_path(mfcc1: Vec<Vec<f64>>, mfcc2: Vec<Vec<f64>>, delta: usize) -> PyResult<Vec<(usize, usize)>> {
    let start = Instant::now();
    let mfcc1_array = Array2::from_shape_vec((mfcc1.len(), mfcc1[0].len()), mfcc1.into_iter().flatten().collect())
        .map_err(|_| PyErr::new::<pyo3::exceptions::PyValueError, _>("Failed to create mfcc1 array"))?;
    let mfcc2_array = Array2::from_shape_vec((mfcc2.len(), mfcc2[0].len()), mfcc2.into_iter().flatten().collect())
        .map_err(|_| PyErr::new::<pyo3::exceptions::PyValueError, _>("Failed to create mfcc2 array"))?;
    let duration = start.elapsed();
    println!("Time 1: {:?}", duration);

    let start = Instant::now();
    let n = mfcc1_array.shape()[1];
    let m = mfcc2_array.shape()[1];

    let delta = delta.min(m);

    let mut cost_matrix = Array2::<f64>::zeros((n, delta));
    let mut centers = vec![0; n];
    let duration = start.elapsed();
    println!("Time 2: {:?}", duration);

    let start = Instant::now();
    dtw::_compute_cost_matrix(&mfcc1_array, &mfcc2_array, delta, &mut cost_matrix, &mut centers)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e))?;

    let duration = start.elapsed();
    println!("Time 3: {:?}", duration);

    let start = Instant::now();
    dtw::_compute_accumulated_cost_matrix_in_place(&mut cost_matrix, &centers)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e))?;
    let duration = start.elapsed();
    println!("Time 4: {:?}", duration);

    let start = Instant::now();
    let best_path = dtw::_compute_best_path(&cost_matrix, &centers, n, delta)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e))?;
    let duration = start.elapsed();
    println!("Time 5: {:?}", duration);

    let start = Instant::now();
    let result: Vec<(usize, usize)> = best_path.into_iter().map(|cell| (cell.i, cell.j)).collect();
    let duration = start.elapsed();
    println!("Time 6: {:?}", duration);

    Ok(result)
}

#[pymodule]
fn aeneas_rust(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(compute_best_path, m)?)?;
    m.add_function(wrap_pyfunction!(text_to_speech, m)?)?;
    Ok(())
}