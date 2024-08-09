use ndarray::Array1;

pub fn create_mel_filter_bank(sample_rate: u32, num_filters: usize, fft_size: usize, low_freq: f64, high_freq: f64) -> Vec<Array1<f64>> {
    let mel_min = hz_to_mel(low_freq);
    let mel_max = hz_to_mel(high_freq);
    let mel_points: Vec<f64> = (0..num_filters + 2)
        .map(|i| mel_min + (mel_max - mel_min) * i as f64 / (num_filters + 1) as f64)
        .collect();

    let hz_points: Vec<f64> = mel_points.iter().map(|&mel| mel_to_hz(mel)).collect();
    let bin_points: Vec<usize> = hz_points.iter().map(|&hz| (hz / (sample_rate as f64 / fft_size as f64)).floor() as usize).collect();

    let mut filter_bank = vec![Array1::zeros(fft_size / 2 + 1); num_filters];

    for i in 1..=num_filters {
        let start = bin_points[i - 1];
        let center = bin_points[i];
        let end = bin_points[i + 1];

        for j in start..center {
            filter_bank[i - 1][j] = (j - start) as f64 / (center - start) as f64;
        }
        for j in center..end {
            filter_bank[i - 1][j] = 1.0 - (j - center) as f64 / (end - center) as f64;
        }
    }

    filter_bank
}

fn hz_to_mel(hz: f64) -> f64 {
    2595.0 * (1.0 + hz / 700.0).log10()
}

fn mel_to_hz(mel: f64) -> f64 {
    700.0 * (10f64.powf(mel / 2595.0) - 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_mel_filter_bank() {
        let filters = create_mel_filter_bank(16000, 26, 512, 0.0, 8000.0);
        assert_eq!(filters.len(), 26);
    }
}
