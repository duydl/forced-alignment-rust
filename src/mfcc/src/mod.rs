extern crate ndarray;
extern crate rustfft;

pub mod fft;
pub mod window;
pub mod mel_filter;
pub mod dct;
pub mod pre_emphasis;

use fft::fft;
use window::apply_hamming_window;
use mel_filter::create_mel_filter_bank;
use dct::dct;
use pre_emphasis::pre_emphasis;
use ndarray::Array1;
use ndarray::ArrayView1;

pub struct MfccConfig {
    pub sample_rate: u32,
    pub num_filters: usize,
    pub num_coefficients: usize,
    pub fft_size: usize,
    pub low_freq: f64,
    pub high_freq: f64,
    pub pre_emphasis: f64,
    pub window_length: f64,
    pub window_shift: f64,
}

pub fn compute_mfcc(audio_data: &[f64], config: MfccConfig) -> Vec<Array1<f64>> {
    // Step 1: Pre-emphasis
    let emphasized_signal = pre_emphasis(audio_data, config.pre_emphasis);

    // Step 2: Framing and applying Hamming window
    let frames = apply_hamming_window(&emphasized_signal, config.window_length, config.window_shift, config.sample_rate);

    // Step 3: FFT and Power Spectrum
    let mut power_spectrum = Vec::new();
    for frame in frames.iter() {
        let fft_result = fft(frame, config.fft_size);
        power_spectrum.push(fft_result);
    }

    // Step 4: Apply Mel filter bank
    let mel_filters = create_mel_filter_bank(config.sample_rate, config.num_filters, config.fft_size, config.low_freq, config.high_freq);
    let mut mel_spectrum = Vec::new();
    for power in power_spectrum.iter() {
        let mut mel_energy = Array1::zeros(config.num_filters);
        for (i, filter) in mel_filters.iter().enumerate() {
            let filter_len = filter.len();
            let power_len = power.len();
            if filter_len != power_len {
                panic!("Length mismatch: Filter {} length = {}, Power length = {}", i, filter_len, power_len);
            }
            mel_energy[i] = ArrayView1::from(filter).dot(&power.view());
        }
        mel_spectrum.push(mel_energy.mapv(|x| x.log(10.0)));
    }

    // Step 5: DCT to get MFCCs
    let mut mfccs = Vec::new();
    for mel in mel_spectrum.iter() {
        mfccs.push(dct(mel, config.num_coefficients));
    }

    mfccs
}
