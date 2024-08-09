use ndarray::Array1;

pub fn apply_hamming_window(signal: &[f64], window_length: f64, window_shift: f64, sample_rate: u32) -> Vec<Array1<f64>> {
    let frame_length = (window_length * sample_rate as f64) as usize;
    let frame_shift = (window_shift * sample_rate as f64) as usize;
    
    // Calculate the total number of frames
    // let num_frames = ((signal.len() - frame_length + frame_shift) as f64 / frame_shift as f64).ceil() as usize;
    let num_frames = (signal.len() as f64 / frame_shift as f64).ceil() as usize;

    let mut frames = Vec::with_capacity(num_frames);
    let hamming_coefficients: Vec<f64> = (0..frame_length)
        .map(|i| 0.54 - 0.46 * (2.0 * std::f64::consts::PI * i as f64 / (frame_length as f64 - 1.0)).cos())
        .collect();

    for i in 0..num_frames {
        let start = i * frame_shift;
        let end = start + frame_length;

        // if start >= signal.len() {
        //     break;
        // }

        let frame_data = if end <= signal.len() {
            signal[start..end].to_vec()
        } else {
            let mut frame = signal[start..].to_vec();
            frame.resize(frame_length, 0.0); // Padding with zeros if the frame is incomplete
            frame
        };
        
        let mut frame: Array1<f64> = Array1::from(frame_data);
        for (j, value) in frame.iter_mut().enumerate() {
            *value *= hamming_coefficients[j];
        }
        frames.push(frame);
    }

    frames
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_apply_hamming_window() {
        let signal = vec![1.0; 16000];
        let frames = apply_hamming_window(&signal, 0.025, 0.01, 16000);
        assert_eq!(frames.len(), 100);
    }

    #[test]
    fn test_apply_hamming_window_with_partial_frame() {
        let signal = vec![1.0; 16010];
        let frames = apply_hamming_window(&signal, 0.025, 0.01, 16000);
        assert_eq!(frames.len(), 101);
    }
}
