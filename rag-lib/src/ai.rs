use crate::llama::Llama;
use core::fmt;
use futures_util::future::join_all;
use futures_util::StreamExt;
use langchain_rust::chain::{
    Chain, ConversationalRetrieverChain, ConversationalRetrieverChainBuilder,
};
use langchain_rust::document_loaders::{HtmlLoader, TextLoader};
use langchain_rust::embedding::{EmbeddingModel, InitOptions, TextEmbedding};
use langchain_rust::memory::SimpleMemory;
use langchain_rust::prompt::HumanMessagePromptTemplate;
use langchain_rust::schemas::Message;
use langchain_rust::text_splitter::SplitterOptions;
use langchain_rust::url::Url;
use langchain_rust::vectorstore::VecStoreOptions;
use langchain_rust::{
    document_loaders::{lo_loader::LoPdfLoader, InputFormat, Loader, PandocLoader},
    embedding::FastEmbed,
    schemas::Document,
    text_splitter::TokenSplitter,
    vectorstore::{
        sqlite_vss::{Store, StoreBuilder},
        Retriever, VectorStore,
    },
};
use langchain_rust::{fmt_message, fmt_template, message_formatter, prompt_args, template_jinja2};
use std::error::Error;
use std::path::Path;
use std::result::Result;
use std::{fs, future};

#[derive(Debug)]
struct AiError {
    details: String,
}
impl AiError {
    fn new(msg: &str) -> AiError {
        AiError {
            details: msg.to_string(),
        }
    }
}
impl fmt::Display for AiError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.details)
    }
}
impl Error for AiError {
    fn description(&self) -> &str {
        &self.details
    }
}

pub struct Trainer {
    store: Store,
    chunk_size: usize,
}
pub struct Assistant {
    chain: ConversationalRetrieverChain,
}
impl Trainer {
    pub async fn new(database: &str, table: &str, vector_dim: i32, chunk_size: usize) -> Self {
        let train_store = create_store(database, table, vector_dim)
            .await
            .expect("failed to create train store");

        Trainer {
            store: train_store,
            chunk_size: chunk_size,
        }
    }

    pub async fn train(&self, sources: Vec<String>) -> Result<(), Box<dyn Error>> {
        let (oks, errors): (Vec<_>, Vec<_>) = join_all(sources.iter().map(|path| async move {
            return get_docs(&path, self.chunk_size).await;
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

impl Assistant {
    pub async fn new(
        database: &str,
        table: &str,
        vector_dim: i32,
        model_filename: &str,
        context_length: u32,
        retrieve_doc_count:usize 
    ) -> Self {
        let retriev_store = create_store(database, table, vector_dim)
            .await
            .expect("failed to create retrieve store");
        // let llm = OpenAI::default().with_model(OpenAIModel::Gpt35.to_string());
        let llm = Llama::new(model_filename, context_length);
        let prompt= message_formatter![
                            fmt_message!(Message::new_system_message("You are a helpful assistant")),
                            fmt_template!(HumanMessagePromptTemplate::new(
                            template_jinja2!("
        Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.
        
        {{context}}
        
        Question:{{question}}

        Helpful Answer: ","context","question")))];

        let chain = ConversationalRetrieverChainBuilder::new()
            .llm(llm)
            .rephrase_question(true)
            .memory(SimpleMemory::new().into())
            .retriever(Retriever::new(retriev_store, retrieve_doc_count))
            .prompt(prompt)
            .build()
            .expect("Error building ConversationalChain");
        Assistant { chain: chain }
    }
    pub async fn ask(&self, question: &str) -> Result<String, Box<dyn Error>> {
        let input_variables = prompt_args! {
            "question" => question,
        };
        match self.chain.invoke(input_variables).await {
            Ok(r) => Ok(r),
            Err(err) => Err(Box::new(err)),
        }
    }
}

async fn create_store(
    database: &str,
    table: &str,
    vector_dim: i32,
) -> Result<Store, Box<dyn Error>> {
    let embedder = FastEmbed::from(TextEmbedding::try_new(InitOptions {
        model_name: EmbeddingModel::BGELargeENV15,
        show_download_progress: true,
        ..Default::default()
    })?);
    // if embedder.is_err() {
    //     return Err(Box::new(embedder.err().unwrap()));
    // }
    // let embedder = embedder.unwrap();
    let store = StoreBuilder::new()
        .connection_url(database)
        .embedder(embedder)
        .table(table)
        .vector_dimensions(vector_dim)
        .build()
        .await;
    if store.is_err() {
        return Err(store.err().unwrap());
    }
    let store = store.unwrap();
    let str_init = store.initialize().await;
    if str_init.is_err() {
        return Err(str_init.err().unwrap());
    }
    Ok(store)
}

pub async fn get_docs(path: &str, split_size: usize) -> Result<Vec<Document>, Box<dyn Error>> {
    let extension = Path::extension(Path::new(path));

    if extension.is_none() {
        return Err(Box::new(AiError::new("no extension specified")));
    }
    let extension = extension.unwrap_or_default().to_str().unwrap();
    let splitter_options = SplitterOptions::new()
        .with_chunk_size(split_size)
        .with_trim_chunks(false);
    let splitter = TokenSplitter::new(splitter_options);

    let docs = if extension == "html" {
        HtmlLoader::from_path(path, Url::parse(&format!("file:///{}", path)).unwrap())
            .expect("Failed to create html loader")
            .load_and_split(splitter)
            .await
            .unwrap()
            .map(|x| x.unwrap())
            .collect::<Vec<_>>()
            .await
    } else if extension == "pdf" {
        LoPdfLoader::from_path(path)
            .expect("failed to get the pdf loader")
            .load_and_split(splitter)
            .await
            .unwrap()
            .map(|d| d.unwrap())
            .collect::<Vec<_>>()
            .await
    } else if extension == "txt" {
        let text_content = fs::read_to_string(path)?;
        let splits = TextLoader::new(text_content)
            .load_and_split(splitter)
            .await?;

        splits
            .filter(|e| future::ready(e.is_ok()))
            .map(|d| d.unwrap())
            .collect::<Vec<_>>()
            .await
    } else {
        let format = match extension.to_lowercase().as_str() {
            "html" => InputFormat::Html,
            "epub" => InputFormat::Epub,
            "md" => InputFormat::Markdown,
            "docx" => InputFormat::Docx,
            "mediawiki" => InputFormat::MediaWiki,
            "typst" => InputFormat::Typst,
            _ => InputFormat::RichTextFormat,
        }
        .to_string();
        PandocLoader::from_path(format, path)
            .await
            .expect("failed to create PandocLoader")
            .load_and_split(splitter)
            .await
            .unwrap()
            .map(|d| d.unwrap())
            .collect::<Vec<_>>()
            .await
    };
    Ok(docs)
}
