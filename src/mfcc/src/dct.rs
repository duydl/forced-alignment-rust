use ndarray::Array1;
use ndarray::ArrayView1;

pub fn dct(mel_spectrum: &Array1<f64>, num_coefficients: usize) -> Array1<f64> {
    let mut result = Array1::zeros(num_coefficients);
    let len = mel_spectrum.len();

    for i in 0..num_coefficients {
        let sum: f64 = mel_spectrum
            .iter()
            .enumerate()
            .map(|(j, &value)| {
                value * (std::f64::consts::PI * i as f64 * (j as f64 + 0.5) / len as f64).cos()
            })
            .sum();
        result[i] = sum;
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_dct() {
        let mel_spectrum = array![1.0, 2.0, 3.0, 4.0];
        let result = dct(&mel_spectrum, 2);
        assert_eq!(result.len(), 2);
    }
}
