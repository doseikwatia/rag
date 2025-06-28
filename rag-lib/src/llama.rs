use std::pin::Pin;

use crate::dprintln;
use async_trait::async_trait;
use futures_util::Stream;
use langchain_rust::{
    language_models::{llm::LLM, GenerateResult, LLMError, TokenUsage},
    schemas::{Message, StreamData},
};
use llama_cpp::{standard_sampler::StandardSampler, SplitMode};
use llama_cpp::{LlamaModel, LlamaParams, SessionParams};
use serde_json::Value;
use std::sync::Arc;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;

#[derive(Clone)]
pub struct Llama {
    model: Arc<LlamaModel>,
    n_ctx: u32,
}

impl Llama {
    pub fn new(model_filename: &str, context_length: u32, use_gpu: bool) -> Self {
        let mut params = LlamaParams::default();
        params.n_gpu_layers = match use_gpu {
            true => 9999,
            false => 0,
        };
        params.split_mode = SplitMode::Row;
        params.use_mlock = true;

        // Create a model from anything that implements `AsRef<Path>`:
        let model =
            LlamaModel::load_from_file(model_filename, params).expect("Could not load model");
        Llama {
            model: Arc::new(model),
            n_ctx: context_length,
        }
    }
    async fn generate(&self, prompt: &str) -> Result<GenerateResult, LLMError> {
        let mut ctx = self
            .model
            .create_session(SessionParams {
                n_ctx: self.n_ctx,
                ..Default::default()
            })
            .expect("failed to create session");

        let prompt_tokens = prompt.split_ascii_whitespace().count() as u32;
        dprintln!("****************************");
        dprintln!("{}", prompt);
        dprintln!("****************************");

        let advance_ctx_result = ctx.advance_context_async(prompt).await;
        if advance_ctx_result.is_err() {
            let err_msg = format!(
                "failed to advance context: {}",
                advance_ctx_result.unwrap_err()
            );
            return Err(LLMError::OtherError(err_msg));
        }
        // LLMs are typically used to predict the next word in a sequence. Let's generate some tokens!
        // let max_tokens = 1024;
        let tokens_result = ctx.start_completing_with(
            StandardSampler::default(),
            ctx.params().n_ctx as usize as usize - ctx.context_size(),
        );
        if tokens_result.is_err() {
            let err_msg = format!("failed to start completion");
            return Err(LLMError::OtherError(err_msg));
        }

        let tokens = tokens_result.unwrap();
        let text = tokens.into_string();
        let completion_tokens = text.len() as u32;

        Ok(GenerateResult {
            tokens: Some(TokenUsage::new(prompt_tokens, completion_tokens)),
            generation: text,
        })
    }
}

#[async_trait]
impl LLM for Llama {
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
        let convo = chat.join("\n");
        self.generate(&convo).await
    }

    async fn stream(
        &self,
        messages: &[Message],
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamData, LLMError>> + Send>>, LLMError> {
        dprintln!("stream called !");
        let chat: Vec<String> = messages
            .into_iter()
            .map(|p| format!("{}:{}", p.message_type.to_string(), p.content))
            .collect();
        let convo = chat.join("\n");

        let mut ctx = self
            .model
            .create_session(SessionParams {
                n_ctx: self.n_ctx,
                ..Default::default()
            })
            .map_err(|e| LLMError::OtherError(format!("Failed to create session: {}", e)))?;

        // let prompt_tokens = convo.split_ascii_whitespace().count() as u32;

        ctx.advance_context_async(&convo)
            .await
            .map_err(|e| LLMError::OtherError(format!("failed to advance context: {}", e)))?;

        dprintln!("****************************");
        dprintln!("{}", convo);
        dprintln!("****************************");

        let token_stream = ctx
            .start_completing(
                // StandardSampler::default(),
                // ctx.params().n_ctx as usize - ctx.context_size(),
            )
            .map_err(|e| LLMError::OtherError(format!("failed to start completion: {}", e)))?;

        let (tx, rx) = mpsc::channel(10);
        let model = Arc::clone(&self.model);

        tokio::spawn(async move {
            dprintln!("starting stream source");
            let mut full_text = String::new();
            // let mut completion_tokens = 0;
            let max_repeatitions = 10;
            let mut repititions = 0;
            let mut last_char = '\\';
            for token in token_stream {
                let s = model.token_to_piece(token);
                // completion_tokens += 1;
                full_text.push_str(&s);

                if let Some(curr_last_char) = s.chars().last() {
                    if last_char == curr_last_char {
                        repititions += 1;
                        if repititions == max_repeatitions {
                            break;
                        }
                    } else {
                        repititions = 0;
                        last_char = curr_last_char;
                    }
                }

                let _ = tx
                    .send(Ok(StreamData::new(
                        Value::String(s.clone()),
                        None, //Some(TokenUsage::new(prompt_tokens, completion_tokens))
                        s.clone(),
                    )))
                    .await;
            }
            dprintln!("exiting stream source");
        });
        Ok(Box::pin(ReceiverStream::new(rx)))
    }
}
