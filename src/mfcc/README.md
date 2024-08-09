# MFCC (Mel-Frequency Cepstral Coefficients) Module

This Rust module provides functionality to compute Mel-Frequency Cepstral Coefficients (MFCC), a key feature used in speech and audio signal processing. MFCCs are widely used in automatic speech recognition (ASR) and speaker recognition systems.

## Steps to Compute MFCC

### 1. Pre-Emphasis

**Objective**: Enhance the higher frequency components of the signal.

**Algorithm**:
For each sample `x[n]` in the input signal:
\[
y[n] = x[n] - \alpha \cdot x[n-1]
\]
where:
- `y[n]` is the output signal.
- `\alpha` is the pre-emphasis factor, typically between 0.95 and 0.97.

### 2. Framing

**Objective**: Divide the continuous audio signal into overlapping frames for short-time analysis.

**Algorithm**:
- Frame size: Typically 20-40 ms.
- Frame shift: Typically 10-20 ms.
- Overlapping is achieved by sliding a window over the signal.

### 3. Windowing

**Objective**: Apply a window function to each frame to reduce spectral leakage.

**Common Window Function**: Hamming window:
\[
w[n] = 0.54 - 0.46 \cdot \cos\left(\frac{2\pi n}{N-1}\right)
\]
where:
- `N` is the frame length.
- `n` is the sample index within the frame.

### 4. Fast Fourier Transform (FFT)

**Objective**: Convert each frame from the time domain to the frequency domain.

**Algorithm**:
- Compute the FFT of each frame, yielding the frequency spectrum.
- The FFT of a signal of length `N` is given by:
\[
X[k] = \sum_{n=0}^{N-1} x[n] \cdot e^{-j\frac{2\pi}{N}kn}
\]
where:
- `X[k]` is the `k`-th frequency component.
- `x[n]` is the time-domain sample.

### 5. Power Spectrum

**Objective**: Compute the power spectrum from the FFT output.

**Algorithm**:
- Power spectrum is computed as:
\[
P[k] = \frac{|X[k]|^2}{N}
\]
where `N` is the length of the FFT.

### 6. Mel Filter Bank

**Objective**: Apply a set of triangular filters to the power spectrum, spaced according to the Mel scale.

**Mel Scale**: Mimics human ear's perception of sound:
\[
\text{Mel}(f) = 2595 \cdot \log_{10}\left(1 + \frac{f}{700}\right)
\]

**Filter Bank**:
- Convert each FFT bin to the Mel scale.
- Apply triangular filters to the power spectrum to emphasize certain frequency bands.

### 7. Logarithm of Mel Spectrum

**Objective**: Compute the logarithm of the Mel spectrum.

**Algorithm**:
- Take the logarithm of each value in the Mel spectrum:
\[
\text{LogMel}[k] = \log(\text{MelSpectrum}[k])
\]
This step compresses the dynamic range of the values.

### 8. Discrete Cosine Transform (DCT)

**Objective**: Convert the Mel log power spectrum into the time domain to get the MFCCs.

**Algorithm**:
- Compute the DCT of the log Mel spectrum:
\[
\text{MFCC}[n] = \sum_{k=0}^{K-1} \text{LogMel}[k] \cdot \cos\left[\frac{\pi n (2k+1)}{2K}\right]
\]
where:
- `n` is the index of the MFCC coefficient.
- `K` is the number of Mel filters.

### 9. Delta and Delta-Delta Coefficients (Optional)

**Objective**: Compute the time derivatives of the MFCC coefficients to capture temporal dynamics.

**Algorithm**:
- Delta coefficients are the first-order differences of the MFCCs.
- Delta-Delta (or acceleration) coefficients are the second-order differences.

## Installation

To include this module in your Rust project, add the following to your `Cargo.toml`:

```toml
[dependencies]
mfcc = { path = "path_to_mfcc_module" }
```

## Usage

```rust
use mfcc::{MfccConfig, compute_mfcc};

fn main() {
    let audio_data = vec![0.0; 16000]; // Example audio data
    let config = MfccConfig {
        sample_rate: 16000,
        num_filters: 26,
        num_coefficients: 13,
        fft_size: 512,
        low_freq: 0.0,
        high_freq: 8000.0,
        pre_emphasis: 0.97,
        window_length: 0.025,
        window_shift: 0.01,
    };
    
    let mfccs = compute_mfcc(&audio_data, config);
    println!("{:?}", mfccs);
}
```

## Contributing

Contributions are welcome! Please submit a pull request or open an issue to discuss any changes or improvements.