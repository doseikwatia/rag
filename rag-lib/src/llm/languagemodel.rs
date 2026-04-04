use std::{pin::Pin, sync::Arc};

use async_trait::async_trait;
use futures_util::Stream;
use langchain_rust::{
    language_models::{llm::LLM, GenerateResult, LLMError},
    schemas::{Message, StreamData},
};

#[derive(Clone)]
pub struct RagLanguageModel {
    model: Arc<dyn LLM>,
}

impl RagLanguageModel {
    pub fn new<L: LLM+'static>(model:L)->Self{
        let model = Arc::new(model);
        Self{model}
    }
}

#[async_trait]
impl LLM for RagLanguageModel {
    async fn generate(&self, messages: &[Message]) -> Result<GenerateResult, LLMError> {
        self.model.generate(messages).await
    }
    async fn stream(
        &self,
        messages: &[Message],
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamData, LLMError>> + Send>>, LLMError> {
        self.model.stream(messages).await
    }
}
