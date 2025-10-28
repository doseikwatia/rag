use crate::dprintln;
use crate::helpers::{get_docs, get_llm};
use crate::utilities::errors::AiError;
use crate::utilities::shared_llm::SharedLLM;
use futures_util::future::join_all;
use futures_util::Stream;
use langchain_rust::chain::{
    self, Chain, ChainError, ConversationalRetrieverChain, ConversationalRetrieverChainBuilder,
    LLMChain, LLMChainBuilder,
};
use langchain_rust::language_models::llm::LLM;
use langchain_rust::schemas;
use langchain_rust::{
    fmt_message, fmt_template,
    memory::WindowBufferMemory,
    message_formatter,
    prompt::HumanMessagePromptTemplate,
    prompt_args,
    schemas::Document,
    schemas::Message,
    template_jinja2,
    vectorstore::{Retriever, VecStoreOptions, VectorStore},
};
use serde_json::Value;
use url::Url;

use std::collections::HashMap;
use std::ops::Deref;
use std::pin;
use std::rc::Rc;
use std::result::Result;
use std::sync::Arc;
pub struct RAGTrainer {
    store: Box<dyn VectorStore>,
    chunk_size: usize,
    chunk_overlap: usize,
}
pub struct RAGAssistant {
    retrieval_chain: ConversationalRetrieverChain,
    title_chain: LLMChain,
}

impl RAGTrainer {
    pub async fn new(
        store: Box<dyn VectorStore>,
        chunk_size: usize,
        chunk_overlap: usize,
        _use_gpu: bool,
    ) -> Self {
        RAGTrainer {
            store: store,
            chunk_size: chunk_size,
            chunk_overlap: chunk_overlap,
        }
    }

    pub async fn train(&self, sources: Vec<String>) -> Result<(), AiError> {
        let chunk_size = self.chunk_size;
        let chunk_overlap = self.chunk_overlap;

        let tasks = sources.into_iter().map(|src| {
            tokio::spawn(async move { get_docs(&src, chunk_size, chunk_overlap).await })
        });
        let (oks, errors): (Vec<_>, Vec<_>) = join_all(tasks)
            .await
            .into_iter()
            .flatten()
            .partition(Result::is_ok);

        if errors.len() > 0 {
            let error_msgs: Vec<String> = errors
                .into_iter()
                .map(|e| format!("{}", e.unwrap_err()))
                .collect();
            let error_msg = error_msgs.join("\n");
            let error = AiError::new(&error_msg);
            return Err(AiError::new(&format!("{:?}", error)));
        }

        // let docs: Vec<Document>
        let docs: Vec<Document> = oks
            .into_iter()
            .map(|i| i.unwrap()) // Convert from Vec<Vec<Document>> to Iterator<Vec<Document>>
            .flat_map(|doc_vec| doc_vec) // Flatten the Iterator<Vec<Document>> into Iterator<Document>
            .collect();

        let opt = &VecStoreOptions::default();
        for chunk_docs in docs.chunks(16) {
            let add_docs_result = self.store.add_documents(&chunk_docs, opt).await;
            if let Err(error) = add_docs_result {
                return Err(AiError::new(&format!("{:?}", error)));
            }
        }
        Ok(())
    }
}

impl RAGAssistant {
    pub async fn new(
        model_filename: &str,
        context_length: u32,
        retriev_store: Box<dyn VectorStore>,
        retrieve_doc_count: usize,
        use_gpu: bool,
        ollama_url: Option<Url>,
    ) -> Self {
        let llm = get_llm(model_filename, context_length, use_gpu, 0.1_f32, ollama_url);
        let title_prompt = message_formatter![
            fmt_message!(Message::new_system_message("You are an assistant that generates concise, descriptive titles for chat conversations, similar to how ChatGPT automatically titles chats.")),
            fmt_template!(HumanMessagePromptTemplate::new(template_jinja2!(r#"Given the first user message below, create a short and clear title (max 6 words) that summarizes what the conversation is likely about. Avoid using punctuation unless necessary. Capitalize main words like a title.

            Example:

            Message: "Explain how transformers work in NLP" → Title: "How Transformers Work in NLP"

            Message: "Give me a Python regex to extract IPs" → Title: "Regex for Extracting IP Addresses"
            
            User’s first message:
            {{message}}

            Title:
            "#,"message")))
        ];
        let retrieval_prompt = message_formatter![
            fmt_message!(Message::new_system_message(
                "You are a helpful assistant who always explains things clearly and concisely."
            )),
            fmt_template!(HumanMessagePromptTemplate::new(template_jinja2!(
                r#"You are an AI assistant helping answer questions based on retrieved document excerpts.
Use only the information contained in the context below. 
If the context does not contain enough information, say "I don't know based on the provided documents."

When answering:
- Write in a clear, conversational tone.
- Ground every factual claim in the provided context.
- Cite document filenames and page numbers in parentheses like (filename, p.X).
- If multiple sources agree, summarize them collectively.
- Do not speculate or use outside knowledge.
- Format the response in Markdown for readability.

Context:
{{context}}

=========
Question:
{{question}}

=========
Answer:
"#,
                "context",
                "question"
            )))
        ];

        let title_chain = LLMChainBuilder::new()
            .llm(llm.clone())
            .prompt(title_prompt)
            .build()
            .expect("Error building title chain");

        let memory = WindowBufferMemory::new(1);
        let retrieval_chain = ConversationalRetrieverChainBuilder::new()
            .llm(llm.clone())
            .rephrase_question(true)
            .memory(memory.into())
            .retriever(Retriever::new(retriev_store, retrieve_doc_count))
            .prompt(retrieval_prompt)
            .build()
            .expect("Error building ConversationalChain");
        RAGAssistant {
            retrieval_chain,
            title_chain,
        }
    }
    /**

    */
    pub async fn clear(&mut self) {
        let memory = self.retrieval_chain.memory.clone();
        memory.lock().await.clear();
    }

    pub async fn get_title(&self, message: &str) -> Result<String, ChainError> {
        dprintln!("Get title called");
        let input_variables = prompt_args! {
            "message" => message,
        };
        let result = self.title_chain.execute(input_variables).await?;
        let text = result
            .get("output")
            .unwrap()
            .to_string()
            .trim_matches(|c: char| !c.is_alphanumeric() && c != '_')
            .trim_start_matches(|c: char|!c.is_uppercase())
            .trim()
            .to_string()
            .trim()
            .to_string();
        Ok(text)
    }
    ///asks the rag chain question. The answer will be streamed to the standard output
    pub async fn ask(
        &self,
        question: &str,
    ) -> pin::Pin<Box<dyn Stream<Item = Result<schemas::StreamData, chain::ChainError>> + Send>>
    {
        dprintln!("Ask called");
        let input_variables = prompt_args! {
            "question" => question,
        };
        self.retrieval_chain
            .stream(input_variables)
            .await
            .expect("Failed to create stream")
    }
}
