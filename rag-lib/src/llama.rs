use std::pin::Pin;

use async_trait::async_trait;
use futures_util::{stream, Stream};
use langchain_rust::{
    language_models::{llm::LLM, GenerateResult, LLMError, TokenUsage},
    schemas::{Message, StreamData},
};
use llama_cpp::standard_sampler::StandardSampler;
use llama_cpp::{LlamaModel, LlamaParams, SessionParams};
use serde_json::Value;

use crate::dprintln;


#[derive(Clone)]
pub struct Llama {
    model: LlamaModel,
    n_ctx : u32
}

impl Llama {
    pub fn new(model_filename: &str, context_length :u32) -> Self {
        let params = LlamaParams::default();

        // Create a model from anything that implements `AsRef<Path>`:
        let model =
            LlamaModel::load_from_file(model_filename, params).expect("Could not load model");
        Llama { model: model , n_ctx: context_length}
    }
    async fn generate(&self, prompt: &str) -> Result<GenerateResult, LLMError> {
        let mut ctx = self
            .model
            .create_session(SessionParams {
                n_ctx: self.n_ctx,
                ..Default::default()
            }).expect("failed to create session");

        let prompt_tokens = prompt.split_ascii_whitespace().count() as u32;
        dprintln!("****************************");
        dprintln!("{}", prompt);
        dprintln!("****************************");
        let advance_ctx_result = ctx.advance_context_async(prompt)
            .await;
        if advance_ctx_result.is_err() {
            let err_msg = format!("failed to advance context: {}",advance_ctx_result.unwrap_err());
            return Err(LLMError::OtherError(err_msg));
        }
        // LLMs are typically used to predict the next word in a sequence. Let's generate some tokens!
        // let max_tokens = 1024;
        let tokens_result = ctx
            .start_completing_with(
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
            .map(|p| format!("{}:{}", p.message_type.to_string().replace("Message", ""), p.content))
            .collect();
        let convo = chat.join("\n");
        self.generate(&convo).await
    }

    async fn stream(
        &self,
        messages: &[Message],
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamData, LLMError>> + Send>>, LLMError> {
        let chat: Vec<String> = messages
            .into_iter()
            .map(|p| format!("{}:{}", p.message_type.to_string(), p.content))
            .collect();
        let convo = chat.join("\n");
        let gen_result = self.generate(&convo).await?;
        let text = gen_result.generation;
        let stream_data = [Ok(StreamData::new(Value::String(text.clone()), text))];
        Ok(Box::pin(stream::iter(stream_data)))
    }
}
