use crate::docloader::get_docs;
use crate::dprintln;
use crate::llm::RagLanguageModel;
use crate::utilities::errors::RAGError;
use crate::vectorstore::RagVectorstore; //get_llm
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

use std::pin;

use std::result::Result;

pub struct RAG {
    retrieval_chain: ConversationalRetrieverChain,
    title_chain: LLMChain,
}

impl RAG {
    pub async fn new(
        llm: RagLanguageModel,
        store: RagVectorstore,
        retrieve_doc_count: usize,
        window_size: usize,
    ) -> Self {
        let (title_chain, retrieval_chain) =
            Self::build_chains(llm, store, retrieve_doc_count, window_size).await;
        RAG {
            title_chain: title_chain,
            retrieval_chain: retrieval_chain,
        }
    }

    pub async fn load_documents(
        store: RagVectorstore,
        sources: Vec<String>,
        chunk_size: usize,
        chunk_overlap: usize,
    ) -> Result<(), RAGError> {
        let tasks = sources.into_iter().map(|src| {
            tokio::task::spawn_blocking(move || get_docs(src, chunk_size, chunk_overlap))
        });

        let results = join_all(tasks).await;

        let mut oks = Vec::new();
        let mut errors = Vec::new();

        for res in results {
            match res {
                Ok(inner) => match inner.await {
                    Ok(val) => oks.push(val),
                    Err(e) => errors.push(e),
                },
                Err(join_err) => {
                    // handle task failure separately
                    eprintln!("Task failed: {:?}", join_err);
                }
            };
        }

        if errors.len() > 0 {
            let error_msgs: Vec<String> = errors.into_iter().map(|e| format!("{}", e)).collect();
            let error_msg = error_msgs.join("\n");
            let error = RAGError::DocLoader(error_msg);
            return Err(error);
        }

        // let docs: Vec<Document>
        let docs: Vec<Document> = oks
            .into_iter()
            .map(|i| i) // Convert from Vec<Vec<Document>> to Iterator<Vec<Document>>
            .flat_map(|doc_vec| doc_vec) // Flatten the Iterator<Vec<Document>> into Iterator<Document>
            .collect();

        let opt = &VecStoreOptions::default();
        for chunk_docs in docs.chunks(16) {
            let add_docs_result = store.add_documents(&chunk_docs, opt).await;
            if let Err(error) = add_docs_result {
                return Err(RAGError::DocLoader(error.to_string()));
            }
        }
        Ok(())
    }

    async fn build_chains<L: Clone + Into<Box<dyn LLM>>, V: Clone + Into<Box<dyn VectorStore>>>(
        llm: L,
        retriev_store: V,
        retrieve_doc_count: usize,
        window_size: usize,
    ) -> (LLMChain, ConversationalRetrieverChain) {
        let title_prompt = message_formatter![
            fmt_message!(Message::new_system_message(
                "Generate a short title for a message in 1 sentence. Stop when done"
            )),
            fmt_template!(HumanMessagePromptTemplate::new(template_jinja2!(
                r#"
        Message: "{{message}}"
        Title:"#,
                "message"
            )))
        ];
        let retrieval_prompt = message_formatter![
            fmt_message!(Message::new_system_message(
                r#"You are a question-answering assistant that uses only provided context.

Rules:
- Use only the given context. Do not use outside knowledge.
- If the answer is not in the context, say: "I don't know based on the provided documents."
- Do not speculate or infer beyond the text.
- Every factual statement must include a citation in the format (filename, p.X).
- Do not fabricate citations.

Style:
- Be clear, concise, and conversational.
- Use Markdown for readability.
- Avoid unnecessary verbosity."#
            )),
            fmt_template!(HumanMessagePromptTemplate::new(template_jinja2!(
                r#"Context:
{{context}}

Question:
{{question}}

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

        let memory = WindowBufferMemory::new(window_size);
        let retrieval_chain = ConversationalRetrieverChainBuilder::new()
            .llm(llm)
            .rephrase_question(false)
            .memory(memory.into())
            .retriever(Retriever::new(retriev_store, retrieve_doc_count))
            .prompt(retrieval_prompt)
            .build()
            .expect("Error building ConversationalChain");

        (title_chain, retrieval_chain)
    }

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
            .trim_start_matches(|c: char| !c.is_uppercase())
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
