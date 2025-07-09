use candle_core::quantized::{ggml_file, gguf_file};
use candle_transformers::models::quantized_llama as model;
use model::ModelWeights;
use std::{fs::File, path::Path, sync::Arc};

pub struct Candle {
    model: Arc,
    n_ctx: u32,
}

impl Candle {
    fn get_gqa(model_name: &str) -> Option<usize> {
        let name = model_name.to_lowercase();

        if name.contains("llama-7b") || name.contains("llama2-7b") || name.contains("llama3-8b") {
            Some(1)
        } else if name.contains("llama-13b") || name.contains("llama2-13b") {
            Some(8)
        } else if name.contains("llama-30b") || name.contains("llama2-70b") {
            Some(8)
        } else if name.contains("mistral") {
            Some(8)
        } else if name.contains("mixtral") {
            Some(8)
        } else if name.contains("qwen-7b") {
            Some(1)
        } else if name.contains("qwen-14b") {
            Some(8)
        } else if name.contains("gemma")
            || name.contains("phi")
            || name.contains("falcon")
            || name.contains("deepseek")
        {
            Some(1)
        } else if name.contains("openhermes") || name.contains("capybara") {
            Some(8)
        } else {
            None
        }
    }

    fn new(model_filename: &str, _context_length: u32, use_gpu: bool) -> Self {
        let device = match use_gpu {
            false => candle_core::Device::Cpu,
            true => {
                if cfg!(target_os = "macos") {
                    candle_core::Device::new_metal(0).expect("failed to create metal device")
                } else {
                    candle_core::Device::new_cuda(0).expect("failed to create cuda device")
                }
            }
        };
        let extn = Path::new(model_filename)
            .extension()
            .expect("model name has to have extension")
            .to_str();
        let mut file = File::open(&model_filename).expect("could not open model file");
        let model = match extn {
            Some("gguf") => {
                let model = gguf_file::Content::read(&mut file)
                    .map_err(|e| e.with_path(model_filename))
                    .expect("failed to load");

                ModelWeights::from_gguf(model, &mut file, &device)
            }
            Some("ggml" | "bin") => {
                let gqa = Candle::get_gqa(model_filename).expect("could not get gpa for model");
                let model = ggml_file::Content::read(&mut file, &device)
                    .map_err(|e| e.with_path(model_filename))
                    .expect("failed to load");
                ModelWeights::from_ggml(model, gqa)
            }
            _ => panic!("the model filename specified does not have supported extension"),
        }
        .expect("could not load model weight to device");
        model.
    }
    
}
