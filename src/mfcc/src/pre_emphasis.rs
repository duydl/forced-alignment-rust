use ndarray::Array1;

pub fn pre_emphasis(signal: &[f64], pre_emphasis_factor: f64) -> Vec<f64> {
    if signal.is_empty() {
        return vec![];
    }

    let mut emphasized_signal = Vec::with_capacity(signal.len());
    emphasized_signal.push(signal[0]); // The first sample is left unchanged

    for i in 1..signal.len() {
        emphasized_signal.push(signal[i] - pre_emphasis_factor * signal[i - 1]);
    }

    emphasized_signal
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pre_emphasis() {
        let signal = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let pre_emphasis_factor = 0.97;
        let expected = vec![
            1.0, 
            2.0 - 0.97 * 1.0,
            3.0 - 0.97 * 2.0,
            4.0 - 0.97 * 3.0,
            5.0 - 0.97 * 4.0,
        ];
        let result = pre_emphasis(&signal, pre_emphasis_factor);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_pre_emphasis_empty() {
        let signal: Vec<f64> = vec![];
        let pre_emphasis_factor = 0.97;
        let result = pre_emphasis(&signal, pre_emphasis_factor);
        assert_eq!(result, vec![]);
    }
}
