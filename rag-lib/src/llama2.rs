use std::{fmt::Write, num::NonZeroU32, path::PathBuf, pin::Pin, str::FromStr, sync::OnceLock, time::Duration};

use crate::dprintln;
use async_trait::async_trait;
use futures_util::Stream;
use langchain_rust::{
    language_models::{llm::LLM, GenerateResult, LLMError, TokenUsage},
    schemas::{Message, StreamData},
};

use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::LlamaModel;
use llama_cpp_2::model::{AddBos, Special};
use llama_cpp_2::sampling::LlamaSampler;
use llama_cpp_2::{ggml_time_us};

use serde_json::Value;
use std::sync::Arc;
use tokio_stream::wrappers::ReceiverStream;


static BACKEND: OnceLock<LlamaBackend> = OnceLock::new();

fn llama_backend() -> &'static LlamaBackend {
    BACKEND.get_or_init(|| {
        LlamaBackend::init().expect("failed to init llama backend")
    })
}

#[derive(Clone)]
pub struct Llama2 {
    model: Arc<LlamaModel>,
    n_ctx: u32,
}


impl Llama2 {
    pub fn new(model_filename: &str, n_ctx: u32, use_gpu: bool) -> Self {
        let model_params = LlamaModelParams::default()
            .with_n_gpu_layers(match use_gpu {
                true => 99999,
                false => 0,
            })
            .with_use_mlock(true)
            .with_vocab_only(false);

        // params.split_mode = SplitMode::Row;

        // Create a model from anything that implements `AsRef<Path>`:

        let  backend = llama_backend();
        let model_path_buf =
            PathBuf::from_str(model_filename).expect("failed to create model path buff");
        let model_path = model_path_buf.as_path();
        let model = LlamaModel::load_from_file(&backend, model_path, &model_params)
            .expect("Could not load model");

        // let n_ctx = model.train_len() as u32;
        Self {
            model: Arc::new(model),
            n_ctx: n_ctx,
        }
    }
    async fn generate(&self, prompt: &str) -> Result<GenerateResult, LLMError> {
        //setting _context_length to train_len for now
        let ctx_params =
            LlamaContextParams::default().with_n_ctx(Some(NonZeroU32::new(self.n_ctx).unwrap()));

        let backend =llama_backend();

        let mut ctx = self
            .model
            .new_context(&backend, ctx_params)
            .expect("failed to create llama context");

        let prompt_tokens = prompt.split_ascii_whitespace().count() as u32;
        dprintln!("****************************");
        dprintln!("{}", prompt);
        dprintln!("****************************");

        //tokenize the prompt
        let tokens_list = self
            .model
            .str_to_token(prompt, AddBos::Always)
            .expect(&format!("failed to tokenize {prompt:}"));

        let n_len = 512; //TODO: accept this as input

        let n_cxt = ctx.n_ctx() as i32;
        let n_kv_req = tokens_list.len() as i32 + (n_len - tokens_list.len() as i32);
        dprintln!("n_len = {n_len}, n_ctx = {n_cxt}, k_kv_req = {n_kv_req}");

        // make sure the KV cache is big enough to hold all the prompt and generated tokens
        if n_kv_req > n_cxt {
            return Err(LLMError::OtherError(
                "n_kv_req > n_ctx, the required kv cache size is not big enough
    either reduce n_len or increase n_ctx"
                    .to_string(),
            ));
        }

        if tokens_list.len()
            >= usize::try_from(n_len).map_err(|err| LLMError::OtherError(err.to_string()))?
        {
            return Err(LLMError::OtherError(
                "the prompt is too long, it has more tokens than n_len".to_string(),
            ));
        }

        // create a llama_batch with size 512
        // we use this object to submit token data for decoding
        let mut batch = LlamaBatch::new(512, 1);

        let last_index: i32 = (tokens_list.len() - 1) as i32;
        for (i, token) in (0_i32..).zip(tokens_list.into_iter()) {
            // llama_decode will output logits only for the last token of the prompt
            let is_last = i == last_index;
            batch
                .add(token, i, &[0], is_last)
                .map_err(|err| LLMError::OtherError(err.to_string()))?;
        }

        ctx.decode(&mut batch).expect("llama_decode() failed");

        // main loop

        let mut n_cur = batch.n_tokens();
        let mut n_decode = 0;

        let t_main_start = ggml_time_us();

        // The `Decoder`
        let mut decoder = encoding_rs::UTF_8.new_decoder();

        let seed: Option<u32> = None; //TODO: accept this as an input
        let mut sampler = LlamaSampler::chain_simple([
            LlamaSampler::dist(seed.unwrap_or(1234)),
            LlamaSampler::greedy(),
        ]);
        let mut outpu_buffer = String::new();
        while n_cur <= n_len {
            // sample the next token
            {
                let token = sampler.sample(&ctx, batch.n_tokens() - 1);

                sampler.accept(token);

                // is it an end of stream?
                if self.model.is_eog_token(token) {
                    eprintln!();
                    break;
                }

                let output_bytes = self
                    .model
                    .token_to_bytes(token, Special::Tokenize)
                    .map_err(|err| LLMError::OtherError(err.to_string()))?;
                // use `Decoder.decode_to_string()` to avoid the intermediate buffer
                let mut output_string = String::with_capacity(32);
                let _decode_result =
                    decoder.decode_to_string(&output_bytes, &mut output_string, false);
                outpu_buffer.write_str(&output_string);

                batch.clear();
                batch
                    .add(token, n_cur, &[0], true)
                    .map_err(|err| LLMError::OtherError(err.to_string()))?;
            }

            n_cur += 1;

            ctx.decode(&mut batch).expect("failed to eval");

            n_decode += 1;
        }
        let t_main_end = ggml_time_us();
        let duration = Duration::from_micros((t_main_end - t_main_start) as u64);

        dprintln!(
            "decoded {} tokens in {:.2} s, speed {:.2} t/s\n",
            n_decode,
            duration.as_secs_f32(),
            n_decode as f32 / duration.as_secs_f32()
        );

        let text = outpu_buffer;
        let completion_tokens = n_cur as u32;

        Ok(GenerateResult {
            tokens: Some(TokenUsage::new(prompt_tokens, completion_tokens)),
            generation: text,
        })
    }
}

#[async_trait]
impl LLM for Llama2 {
    async fn generate(&self, messages: &[Message]) -> Result<GenerateResult, LLMError> {
        let chat: Vec<String> = messages
            .into_iter()
            .map(|p| {
                format!(
                    "{}:{}",
                    p.message_type.to_string().replace("Message", ""),
                    p.content
                )
            })
            .collect();
        let convo = chat.join(". ");
        self.generate(&convo).await
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
        let (tx, rx) = tokio::sync::mpsc::channel(10);

        // clones we will move into the blocking closure (must be Send)
        let model = Arc::clone(&self.model);
        let backend = llama_backend();
        let n_ctx = self.n_ctx; // plain integer — Send
        let prompt_clone = prompt.clone();

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
                LlamaContextParams::default().with_n_ctx(Some(NonZeroU32::new(n_ctx).unwrap()));

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
            let n_len: i32 = 1536; // TODO: accept as input
            let n_cxt = ctx.n_ctx() as i32;
            let tokens_len = tokens_list.len() as i32;
            let n_kv_req = tokens_len + (n_len - tokens_len);

            if n_kv_req > n_cxt {
                let _ = tx.blocking_send(Err(LLMError::OtherError(
                "n_kv_req > n_ctx, the required kv cache size is not big enough; either reduce n_len or increase n_ctx".to_string(),
            )));
                return;
            }

            if tokens_len >= n_len {
                let _ = tx.blocking_send(Err(LLMError::OtherError(
                    "the prompt is too long, it has more tokens than n_len".to_string(),
                )));
                return;
            }

            // build batch and feed prompt tokens
            let mut batch = LlamaBatch::new(tokens_len as usize, 1);
            let last_index: i32 = (tokens_len - 1) as i32;

            for (i, token) in (0_i32..).zip(tokens_list.into_iter()) {
                let is_last = i == last_index;
                if let Err(err) = batch.add(token, i, &[0], is_last) {
                    let _ = tx.blocking_send(Err(LLMError::OtherError(format!(
                        "failed to add token to batch: {}",
                        err
                    ))));
                    return;
                }
            }

            if let Err(err) = ctx.decode(&mut batch) {
                let _ = tx.blocking_send(Err(LLMError::OtherError(format!(
                    "llama decode failed for prompt: {}",
                    err
                ))));
                return;
            }

            // generation loop
            let mut n_cur = batch.n_tokens();
            let mut decoder = encoding_rs::UTF_8.new_decoder();

            let seed: Option<u32> = None; // TODO: accept as input
            let mut sampler = LlamaSampler::chain_simple([
                LlamaSampler::dist(seed.unwrap_or(1234)),
                LlamaSampler::greedy(),
            ]);

            while n_cur <= n_len {
                // sample next token using ctx and batch state
                let token = sampler.sample(&ctx, batch.n_tokens() - 1);
                sampler.accept(token);

                // end-of-generation
                if model.is_eog_token(token) {
                    break;
                }

                // convert token to bytes -> to string
                let output_bytes = model
                    .token_to_bytes(token, Special::Tokenize)
                    .unwrap_or_default();
                let mut output_string = String::with_capacity(32);
                let _ = decoder.decode_to_string(&output_bytes, &mut output_string, false);

                // send via the tokio sender from the blocking thread
                let send_res = tx.blocking_send(Ok(StreamData::new(
                    Value::String(output_string.clone()),
                    Some(TokenUsage::new(n_len as u32, n_cur as u32)),
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

                n_cur += 1;

                if let Err(err) = ctx.decode(&mut batch) {
                    let _ = tx.blocking_send(Err(LLMError::OtherError(format!(
                        "failed to eval during generation: {}",
                        err
                    ))));
                    break;
                }
            }

            // optionally send a final Ok or close by dropping tx
        });

        Ok(Box::pin(ReceiverStream::new(rx)))
    }
}
