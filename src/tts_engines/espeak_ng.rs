use super::TTSEngine;
use std::process::Command;

pub struct EspeakNg;

impl EspeakNg {
    pub fn new() -> Self {
        EspeakNg
    }
}

impl TTSEngine for EspeakNg {
    fn speak_to_file(&self, text: &str, output_path: &str) -> Result<(), String> {
        let status = Command::new("espeak-ng")
            .arg(text)
            .arg("-w")
            .arg(output_path)
            .status()
            .map_err(|e| format!("Failed to execute espeak-ng: {}", e))?;

        if status.success() {
            Ok(())
        } else {
            Err(format!("espeak-ng command failed with status: {}", status))
        }
    }
}
