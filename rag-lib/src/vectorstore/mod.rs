mod  vectorstore;
use fastembed::{InitOptions, TextEmbedding};
use langchain_rust::{
    embedding::FastEmbed,
    vectorstore::sqlite_vss::{Store,  StoreBuilder},
};

use crate::{
    configuration::EmbeddingModelCfg,
    utilities::{errors::RAGError, reranker_wrapper::RerankerWrapper},
};

pub  use vectorstore::RagVectorstore;

pub async fn create_sqlite_store(
    database: &str,
    table: &str,
    embedding_model: EmbeddingModelCfg,
    _use_gpu: bool,
) -> Result<RerankerWrapper<Store>, RAGError> {
    let (vector_dim, _) = embedding_model.get_info();
    let embedding_model_name = embedding_model.into();
    let init_options = InitOptions::new(embedding_model_name).with_show_download_progress(true);
    let model = TextEmbedding::try_new(init_options)
        .map_err(|err| RAGError::VectorStore(err.to_string()))?;
    let embedder = FastEmbed::from(model);
    let store = StoreBuilder::new()
        .connection_url(database)
        .embedder(embedder)
        .table(table)
        .vector_dimensions(vector_dim)
        .build()
        .await
        .map_err(|err| RAGError::VectorStore(err.to_string()))?;
    store
        .initialize()
        .await
        .map_err(|err| RAGError::VectorStore(err.to_string()))?;
    let wrapped_store = RerankerWrapper::new(10, store);
   
    Ok(wrapped_store)
}
