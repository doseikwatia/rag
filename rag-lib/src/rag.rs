use crate::dprintln;
use crate::errors::AiError;
use crate::helpers::{create_sqlite_store, get_docs};
use crate::llama::Llama;
use crate::utilities::reranker_wrapper::RerankerWrapper;
use futures_util::future::join_all;
use futures_util::StreamExt;
use langchain_rust::chain::{
    Chain, ConversationalRetrieverChain, ConversationalRetrieverChainBuilder,
};
use langchain_rust::{
    fmt_message, fmt_template,
    memory::WindowBufferMemory,
    message_formatter,
    prompt::HumanMessagePromptTemplate,
    prompt_args,
    schemas::Document,
    schemas::Message,
    template_jinja2,
    vectorstore::{sqlite_vss::Store, Retriever, VecStoreOptions, VectorStore},
};

use std::error::Error;
use std::io::{self, Write};
use std::result::Result;

pub struct RAGTrainer
{
    store: RerankerWrapper<Store>,
    chunk_size: usize,
    chunk_overlap: usize,
}
pub struct RAGAssistant {
    chain: ConversationalRetrieverChain,
}

impl RAGTrainer
{
    pub async fn new(
        database: &str,
        table: &str,
        vector_dim: i32,
        chunk_size: usize,
        chunk_overlap: usize,
        use_gpu: bool,
    ) -> Self {
        let train_store = create_sqlite_store(database, table, vector_dim, use_gpu)
            .await
            .expect("failed to create train store");

        RAGTrainer {
            store: train_store,
            chunk_size: chunk_size,
            chunk_overlap: chunk_overlap,
        }
    }

    pub async fn train(&self, sources: Vec<String>) -> Result<(), Box<dyn Error>> {
        let (oks, errors): (Vec<_>, Vec<_>) = join_all(sources.iter().map(|path| async move {
            return get_docs(&path, self.chunk_size, self.chunk_overlap).await;
        }))
        .await
        .into_iter()
        .partition(Result::is_ok);

        if errors.len() > 0 {
            let error_msgs: Vec<String> = errors
                .into_iter()
                .map(|e| format!("{}", e.unwrap_err()))
                .collect();
            let error_msg = error_msgs.join("\n");
            let error = AiError::new(&error_msg);
            return Err(Box::new(error));
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
            if add_docs_result.is_err() {
                return Err(add_docs_result.unwrap_err());
            }
        }
        Ok(())
    }
}

impl RAGAssistant {
    pub async fn new(
        database: &str,
        table: &str,
        vector_dim: i32,
        model_filename: &str,
        context_length: u32,
        retrieve_doc_count: usize,
        use_gpu: bool,
    ) -> Self {
        let retriev_store = create_sqlite_store(database, table, vector_dim, use_gpu)
            .await
            .expect("failed to create retrieve store");
        // let llm = OpenAI::default().with_model(OpenAIModel::Gpt35.to_string());
        let llm = Llama::new(model_filename, context_length, use_gpu);
        let prompt = message_formatter![
            fmt_message!(Message::new_system_message(
                "You are a helpful assistant who always explains things clearly and concisely."
            )),
            fmt_template!(HumanMessagePromptTemplate::new(template_jinja2!(
                r#"
        Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.
        
        Context:{{context}}
        
        Question:{{question}}

        Helpful Answer: "#,
                "context",
                "question"
            )))
        ];

        let memory = WindowBufferMemory::new(3);
        let chain = ConversationalRetrieverChainBuilder::new()
            .llm(llm)
            .rephrase_question(true)
            .memory(memory.into())
            .retriever(Retriever::new(retriev_store, retrieve_doc_count))
            .prompt(prompt)
            .build()
            .expect("Error building ConversationalChain");
        RAGAssistant { chain: chain }
    }
    /**
    dsdfg
    */
    pub async fn clear(&mut self) {
        let memory = self.chain.memory.clone();
        memory.lock().await.clear();
    }

    ///asks the rag chain question. The answer will be streamed to the standard output
    pub async fn ask(&self, question: &str) {
        dprintln!("Ask called");
        let input_variables = prompt_args! {
            "question" => question,
        };
        let mut stream = self
            .chain
            .stream(input_variables)
            .await
            .expect("Failed to create stream");
        print!("\nAI\t> ");
        while let Some(result) = stream.next().await {
            match result {
                Ok(value) => {
                    let bytes = value.value.as_str().unwrap().as_bytes();
                    let _ = io::stdout().write(bytes);
                    let _ = io::stdout().flush();
                }
                Err(e) => panic!("Error invoking LLMChain: {e:}"),
            }
        }
    }
}
