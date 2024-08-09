use ndarray::Array1;
use rustfft::FftPlanner;
use rustfft::num_complex::Complex;

pub fn fft(frame: &Array1<f64>, fft_size: usize) -> Array1<f64> {
    let mut planner = FftPlanner::<f64>::new();
    let fft = planner.plan_fft_forward(fft_size);

    let mut input: Vec<Complex<f64>> = frame.iter().map(|&x| Complex::new(x, 0.0)).collect();
    input.resize(fft_size, Complex::new(0.0, 0.0));

    fft.process(&mut input);

    let power_spectrum: Array1<f64> = input.iter()
        .take(fft_size / 2 + 1)  // Only take positive frequencies
        .map(|c| c.norm_sqr())    // Square of the magnitude (|c|^2)
        .collect::<Array1<f64>>();
    power_spectrum
    // let fft_magnitude: Array1<f64> = Array1::from(input.iter().map(|c| c.norm()).collect::<Vec<f64>>());

    // fft_magnitude
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_fft() {
        let frame = array![1.0, 0.0, 0.0, 0.0];
        let result = fft(&frame, 4);
        assert_eq!(result.len(), 4 / 2 + 1);

        let expected = array![1.0, 1.0, 1.0];
        assert_eq!(result, expected);
    }
}
