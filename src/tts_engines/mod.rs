pub mod espeak_ng;

pub trait TTSEngine {
    fn speak_to_file(&self, text: &str, output_path: &str) -> Result<(), String>;
}