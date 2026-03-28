use std::{num::NonZeroU32, path::PathBuf, pin::Pin, str::FromStr, sync::OnceLock};

use crate::dprintln;
use async_trait::async_trait;
use futures_util::{Stream, StreamExt};
use langchain_rust::{
    language_models::{llm::LLM, GenerateResult, LLMError, TokenUsage},
    schemas::{Message, StreamData},
};

use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::AddBos;
use llama_cpp_2::model::LlamaModel;
use llama_cpp_2::sampling::LlamaSampler;

use serde_json::Value;
use std::sync::Arc;
use tokio_stream::wrappers::ReceiverStream;

static BACKEND: OnceLock<LlamaBackend> = OnceLock::new();

fn llama_backend() -> &'static LlamaBackend {
    BACKEND.get_or_init(|| LlamaBackend::init().expect("failed to init llama backend"))
}

#[derive(Clone)]
pub struct Llama2 {
    model: Arc<LlamaModel>,
    n_ctx: u32,
    n_len: u32,
    temperature: f32,
    top_k: i32,
    top_p: f32,
    min_keep: usize,
}

impl Llama2 {
    pub fn new(model_filename: &str, use_gpu: bool) -> Self {
        let n_ctx: u32 = 1024;
        let n_len: u32 = 512;
        let temperature: f32 = 0.1;
        let top_k: i32 = 10;
        let top_p: f32 = 0.9;
        let min_keep: usize = 3;
        let model_params = LlamaModelParams::default()
            .with_n_gpu_layers(match use_gpu {
                true => 32,
                false => 0,
            })
            .with_use_mlock(true)
            .with_vocab_only(false);

        // params.split_mode = SplitMode::Row;

        // Create a model from anything that implements `AsRef<Path>`:

        let backend = llama_backend();
        let model_path_buf =
            PathBuf::from_str(model_filename).expect("failed to create model path buff");
        let model_path = model_path_buf.as_path();
        let model = Arc::new(
            LlamaModel::load_from_file(&backend, model_path, &model_params)
                .expect("Could not load model"),
        );

        Self {
            model,
            n_ctx,
            n_len,
            temperature,
            top_k,
            top_p,
            min_keep,
        }
    }

    pub fn with_ctx(mut self, n_ctx: u32) -> Self {
        self.n_ctx = n_ctx;
        self
    }

    pub fn with_len(mut self, n_len: u32) -> Self {
        self.n_len = n_len;
        self
    }

    pub fn with_temperature(mut self, temperature: f32) -> Self {
        self.temperature = temperature;
        self
    }

    pub fn with_top_k(mut self, top_k: i32) -> Self {
        self.top_k = top_k;
        self
    }

    pub fn with_top_p(mut self, top_p: f32) -> Self {
        self.top_p = top_p;
        self
    }

    pub fn with_min_keep(mut self, min_keep: usize) -> Self {
        self.min_keep = min_keep;
        self
    }
}

#[async_trait]
impl LLM for Llama2 {
    async fn generate(&self, messages: &[Message]) -> Result<GenerateResult, LLMError> {
        dprintln!("generate called !");
        let mut rx = self.stream(messages).await?;
        let mut buff = String::new();
        let mut token_usage = None;
        while let Some(stream_data) = rx.next().await {
            match stream_data {
                Ok(data) => {
                    buff.push_str(&data.content);
                    token_usage = data.tokens;
                }
                Err(err) => {
                    return Err(err);
                }
            }
        }
        Ok(GenerateResult {
            tokens: token_usage,
            generation: buff,
        })
    }

    async fn stream(
        &self,
        messages: &[Message],
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamData, LLMError>> + Send>>, LLMError> {
        dprintln!("stream called !");
        let chat: Vec<String> = messages
            .iter()
            .map(|p| format!("{}:{}", p.message_type.to_string(), p.content))
            .collect();
        let prompt = chat.join(". ");

        // tokio channel for streaming out tokens
        let (tx, rx) = tokio::sync::mpsc::channel(128);

        // clones we will move into the blocking closure (must be Send)
        let model = Arc::clone(&self.model);
        let backend = llama_backend();
        let n_ctx = self.n_ctx; // plain integer — Send
        let n_len = self.n_len;
        let prompt_clone = prompt.clone();
        let n_threads = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1) as i32;
        //sampling parameters
        let temperature: f32 = self.temperature;
        let top_k: i32 = self.top_k;
        let top_p: f32 = self.top_p;
        let min_keep: usize = self.min_keep;

        // Tokenize prompt on the async thread (safe)
        let tokens_list = model
            .str_to_token(&prompt_clone, AddBos::Always)
            .map_err(|e| LLMError::OtherError(format!("failed to tokenize prompt: {e}")))?;

        // move these into the blocking task
        tokio::task::spawn_blocking(move || {
            dprintln!("starting spawn_blocking stream source");

            // Create context inside the blocking thread — **important**:
            // context is created & used on the SAME thread so it doesn't cross threads.
            let ctx_params =
                LlamaContextParams::default().with_n_ctx(Some(NonZeroU32::new(n_ctx).unwrap()))
                .with_n_threads(n_threads)
                .with_n_threads_batch(n_threads*2);
                
            let mut ctx = match model.new_context(&backend, ctx_params) {
                Ok(c) => c,
                Err(err) => {
                    let _ = tx.blocking_send(Err(LLMError::OtherError(format!(
                        "failed to create llama context: {}",
                        err
                    ))));
                    return;
                }
            };

            // sanity checks that require ctx values (now that ctx exists)
            let n_ctx = ctx.n_ctx() as i32;
            let tokens_len = tokens_list.len() as i32;
            let n_len = n_len as i32;

            dprintln!(
                "n_ctx:{:?}, n_len:{:?}, tokens_len: {:?}",
                n_ctx,
                n_len,
                tokens_len
            );

            //this should be a warning
            if tokens_len >= n_len {
                let _ = tx.blocking_send(Err(LLMError::OtherError(
                    "the prompt is too long, it has more tokens than n_len".to_string(),
                )));
                return;
            }

            // build batch and feed prompt tokens
            let max_batch_len = ctx.n_batch() as usize;
            let mut batch = LlamaBatch::new(max_batch_len, 1);
            let sequences = tokens_list.chunks(max_batch_len);

            let mut pos = 0;
            for sequence in sequences {
                batch.clear();
                for token in sequence {
                    let is_last = pos == tokens_len - 1;
                    if let Err(err) = batch.add(*token, pos, &[0], is_last) {
                        let _ = tx.blocking_send(Err(LLMError::OtherError(format!(
                            "failed to add token to batch: {}",
                            err
                        ))));
                        return;
                    }
                    pos += 1;
                }

                if let Err(err) = ctx.decode(&mut batch) {
                    let _ = tx.blocking_send(Err(LLMError::OtherError(format!(
                        "llama decode failed for prompt: {}",
                        err
                    ))));
                    return;
                }
            }

            // generation loop
            let mut n_cur = tokens_len;
            let mut sample_pos = n_cur - 1;
            let mut decoder = encoding_rs::UTF_8.new_decoder();

            let seed: Option<u32> = None; // TODO: accept as input
            let mut sampler = LlamaSampler::chain_simple([
                LlamaSampler::top_k(top_k),
                LlamaSampler::top_p(top_p, min_keep),
                LlamaSampler::temp(temperature),
                LlamaSampler::dist(seed.unwrap_or(1234)),
            ]);

            let max_gen = n_len;
            let max_total = (tokens_len + max_gen).min(n_ctx);
            while n_cur <= max_total {
                // sample next token using ctx and batch state
                {
                    let token = sampler.sample(&ctx, sample_pos);
                    sampler.accept(token);
                    // end-of-generation
                    if model.is_eog_token(token) {
                        break;
                    }
                    // convert token to bytes -> to string
                    let output_bytes = model
                        .token_to_piece_bytes(token, 64, true, None)
                        .unwrap_or_default();
                    let mut output_string = String::with_capacity(32);
                    let _ = decoder.decode_to_string(&output_bytes, &mut output_string, false);

                    // send via the tokio sender from the blocking thread
                    let send_res = tx.blocking_send(Ok(StreamData::new(
                        Value::String(output_string.clone()),
                        Some(TokenUsage::new(tokens_len as u32, n_cur as u32)),
                        output_string.clone(),
                    )));

                    if send_res.is_err() {
                        // receiver was dropped — stop generation
                        break;
                    }
                    // prepare next token
                    batch.clear();
                    if let Err(err) = batch.add(token, n_cur, &[0], true) {
                        let _ = tx.blocking_send(Err(LLMError::OtherError(format!(
                            "failed to add generated token to batch: {}",
                            err
                        ))));
                        break;
                    }
                }
                if let Err(err) = ctx.decode(&mut batch) {
                    let _ = tx.blocking_send(Err(LLMError::OtherError(format!(
                        "failed to eval during generation: {}",
                        err
                    ))));
                    break;
                }
                n_cur += 1;
                sample_pos = batch.n_tokens() - 1;
            }

            // optionally send a final Ok or close by dropping tx
        });

        Ok(Box::pin(ReceiverStream::new(rx)))
    }
}
