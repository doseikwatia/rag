use langchain_rust::{
    document_loaders::{
        lo_loader::LoPdfLoader, HtmlLoader, InputFormat, Loader, PandocLoader, TextLoader,
    },
    embedding::{EmbeddingModel, FastEmbed, InitOptions, TextEmbedding},
    schemas::Document,
    text_splitter::{SplitterOptions, TokenSplitter},
    url::Url,
    vectorstore::{
        sqlite_vss::{Store, StoreBuilder},
        VectorStore,
    },
};

use crate::{
    configuration::{Config, EmbeddingModelCfg},
    dprintln,
    utilities::{
        configuration::StoreType, elasticsearchstore::ElasticsearchStore, errors::AiError,
        reranker_wrapper::RerankerWrapper,
    },
};
use futures_util::future;
use futures_util::StreamExt;
use std::error::Error;
use std::{fs, path::Path};

pub async fn create_sqlite_store(
    database: &str,
    table: &str,
    embedding_model:EmbeddingModelCfg,
    _use_gpu: bool,
) -> Result<RerankerWrapper<Store>, Box<dyn Error>> {
    let (vector_dim,_) = embedding_model.get_info();
    let embedding_model_name = embedding_model.into();
    let init_options =
        InitOptions::new(embedding_model_name).with_show_download_progress(true);
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
    let wrapped_store = RerankerWrapper::new(2, store);
    Ok(wrapped_store)
}

pub async fn create_elasticsearch_store(
    urls: &[&str],
    index_name: &str,
    api_id: &str,
    api_key: &str,
    embedding_model:EmbeddingModelCfg,
    _use_gpu: bool,
) -> Result<RerankerWrapper<ElasticsearchStore<FastEmbed>>, Box<dyn Error>> {
    let (vector_dim,_) = embedding_model.get_info();
    let embedding_model_name = embedding_model.into();
    let init_options =
        InitOptions::new(embedding_model_name).with_show_download_progress(true);
    let model = TextEmbedding::try_new(init_options)?;
    let embedder = FastEmbed::from(model);
    let store = ElasticsearchStore::new(urls, api_id, api_key, embedder, vector_dim, index_name);
    store.initialize().await?;
    let wrapped_store = RerankerWrapper::new(2, store);
    Ok(wrapped_store)
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

pub async fn get_store(config: &Config) -> Box<dyn VectorStore> {
    let store: Box<dyn VectorStore> = match config.store_type {
        StoreType::SQLITE => Box::new(
            create_sqlite_store(
                &config.sqlite.connection_string,
                &config.sqlite.table,
                config.embedding_model,
                config.use_gpu,
            )
            .await
            .expect("failed to create sqlite retrieve store"),
        ),
        
        StoreType::ELASTICSEARCH => {
            let urls:Vec<&str>= config.elasticsearch.urls.iter().map(|u|u.as_str()).collect();
            Box::new(
            create_elasticsearch_store(
                urls.as_slice(),
                &config.elasticsearch.index,
                &config.elasticsearch.api_id,
                &config.elasticsearch.api_key,
                config.embedding_model,
                config.use_gpu,
            )
            .await
            .expect("failed to create es retrieve store"),
        )},
    };
    store
}
