use crate::dprintln;
use crate::llama::Llama;
use core::fmt;
use futures_util::future::join_all;
use futures_util::StreamExt;
use langchain_rust::chain::{
    Chain, ConversationalRetrieverChain, ConversationalRetrieverChainBuilder,
};
use langchain_rust::{
    document_loaders::{
        lo_loader::LoPdfLoader, HtmlLoader, InputFormat, Loader, PandocLoader, TextLoader,
    },
    embedding::{EmbeddingModel, FastEmbed, InitOptions, TextEmbedding},
    fmt_message, fmt_template,
    memory::WindowBufferMemory,
    message_formatter,
    prompt::HumanMessagePromptTemplate,
    prompt_args,
    schemas::Document,
    schemas::Message,
    template_jinja2,
    text_splitter::{SplitterOptions, TokenSplitter},
    url::Url,
    vectorstore::{
        sqlite_vss::{Store, StoreBuilder},
        Retriever, VecStoreOptions, VectorStore,
    },
};

use std::error::Error;
use std::io::{self, Write};
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
    chunk_overlap: usize,
}
pub struct Assistant {
    chain: ConversationalRetrieverChain,
}
impl Trainer {
    pub async fn new(
        database: &str,
        table: &str,
        vector_dim: i32,
        chunk_size: usize,
        chunk_overlap: usize,
        use_gpu: bool,
    ) -> Self {
        let train_store = create_store(database, table, vector_dim, use_gpu)
            .await
            .expect("failed to create train store");

        Trainer {
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

impl Assistant {
    pub async fn new(
        database: &str,
        table: &str,
        vector_dim: i32,
        model_filename: &str,
        context_length: u32,
        retrieve_doc_count: usize,
        use_gpu:bool
    ) -> Self {
        let retriev_store = create_store(database, table, vector_dim, use_gpu)
            .await
            .expect("failed to create retrieve store");
        // let llm = OpenAI::default().with_model(OpenAIModel::Gpt35.to_string());
        let llm = Llama::new(model_filename, context_length, use_gpu);
        let prompt= message_formatter![
                            fmt_message!(Message::new_system_message("You are a helpful assistant")),
                            fmt_template!(HumanMessagePromptTemplate::new(
                            template_jinja2!("
        Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.
        
        {{context}}
        
        Question:{{question}}

        Helpful Answer: ","context","question")))];

        let memory = WindowBufferMemory::new(3);
        let chain = ConversationalRetrieverChainBuilder::new()
            .llm(llm)
            .rephrase_question(false)
            .memory(memory.into())
            .retriever(Retriever::new(retriev_store, retrieve_doc_count))
            .prompt(prompt)
            .build()
            .expect("Error building ConversationalChain");
        Assistant { chain: chain }
    }
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

async fn create_store(
    database: &str,
    table: &str,
    vector_dim: i32,
    _use_gpu: bool,
) -> Result<Store, Box<dyn Error>> {
    dprintln!("{database:}");

    let init_options = InitOptions::new(EmbeddingModel::BGESmallENV15)
        .with_show_download_progress(true);
    let model = TextEmbedding::try_new(init_options)?;
    let embedder = FastEmbed::from(model);
    let store = StoreBuilder::new()
        .connection_url(database)
        .embedder(embedder)
        .table(table)
        .vector_dimensions(vector_dim)
        .build()
        .await?;
    store.initialize().await?;
    Ok(store)
}

pub async fn get_docs(
    path: &str,
    split_size: usize,
    chunk_overlap: usize,
) -> Result<Vec<Document>, Box<dyn Error>> {
    let extension = Path::extension(Path::new(path));

    if extension.is_none() {
        return Err(Box::new(AiError::new("no extension specified")));
    }
    let extension = extension.unwrap_or_default().to_str().unwrap();
    let splitter_options = SplitterOptions::new()
        .with_chunk_size(split_size)
        .with_chunk_overlap(chunk_overlap);

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
