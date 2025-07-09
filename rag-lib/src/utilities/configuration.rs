use crate::impl_enum_from;
use fastembed::EmbeddingModel;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, PartialEq, Debug, Clone,Copy)]
#[serde(rename_all = "lowercase")]
pub enum EmbeddingModelCfg {
    /// sentence-transformers/all-MiniLM-L6-v2
    AllMiniLML6V2,
    /// Quantized sentence-transformers/all-MiniLM-L6-v2
    AllMiniLML6V2Q,
    /// sentence-transformers/all-MiniLM-L12-v2
    AllMiniLML12V2,
    /// Quantized sentence-transformers/all-MiniLM-L12-v2
    AllMiniLML12V2Q,
    /// BAAI/bge-base-en-v1.5
    BGEBaseENV15,
    /// Quantized BAAI/bge-base-en-v1.5
    BGEBaseENV15Q,
    /// BAAI/bge-large-en-v1.5
    BGELargeENV15,
    /// Quantized BAAI/bge-large-en-v1.5
    BGELargeENV15Q,
    /// BAAI/bge-small-en-v1.5 - Default
    BGESmallENV15,
    /// Quantized BAAI/bge-small-en-v1.5
    BGESmallENV15Q,
    /// nomic-ai/nomic-embed-text-v1
    NomicEmbedTextV1,
    /// nomic-ai/nomic-embed-text-v1.5
    NomicEmbedTextV15,
    /// Quantized v1.5 nomic-ai/nomic-embed-text-v1.5
    NomicEmbedTextV15Q,
    /// sentence-transformers/paraphrase-MiniLM-L6-v2
    ParaphraseMLMiniLML12V2,
    /// Quantized sentence-transformers/paraphrase-MiniLM-L6-v2
    ParaphraseMLMiniLML12V2Q,
    /// sentence-transformers/paraphrase-mpnet-base-v2
    ParaphraseMLMpnetBaseV2,
    /// BAAI/bge-small-zh-v1.5
    BGESmallZHV15,
    /// BAAI/bge-large-zh-v1.5
    BGELargeZHV15,
    /// lightonai/modernbert-embed-large
    ModernBertEmbedLarge,
    /// intfloat/multilingual-e5-small
    MultilingualE5Small,
    /// intfloat/multilingual-e5-base
    MultilingualE5Base,
    /// intfloat/multilingual-e5-large
    MultilingualE5Large,
    /// mixedbread-ai/mxbai-embed-large-v1
    MxbaiEmbedLargeV1,
    /// Quantized mixedbread-ai/mxbai-embed-large-v1
    MxbaiEmbedLargeV1Q,
    /// Alibaba-NLP/gte-base-en-v1.5
    GTEBaseENV15,
    /// Quantized Alibaba-NLP/gte-base-en-v1.5
    GTEBaseENV15Q,
    /// Alibaba-NLP/gte-large-en-v1.5
    GTELargeENV15,
    /// Quantized Alibaba-NLP/gte-large-en-v1.5
    GTELargeENV15Q,
    /// Qdrant/clip-ViT-B-32-text
    ClipVitB32,
    /// jinaai/jina-embeddings-v2-base-code
    JinaEmbeddingsV2BaseCode,
}

impl_enum_from!(EmbeddingModelCfg=>EmbeddingModel{
    AllMiniLML6V2,
    AllMiniLML6V2Q,
    AllMiniLML12V2,
    AllMiniLML12V2Q,
    BGEBaseENV15,
    BGEBaseENV15Q,
    BGELargeENV15,
    BGELargeENV15Q,
    BGESmallENV15,
    BGESmallENV15Q,
    NomicEmbedTextV1,
    NomicEmbedTextV15,
    NomicEmbedTextV15Q,
    ParaphraseMLMiniLML12V2,
    ParaphraseMLMiniLML12V2Q,
    ParaphraseMLMpnetBaseV2,
    BGESmallZHV15,
    BGELargeZHV15,
    ModernBertEmbedLarge,
    MultilingualE5Small,
    MultilingualE5Base,
    MultilingualE5Large,
    MxbaiEmbedLargeV1,
    MxbaiEmbedLargeV1Q,
    GTEBaseENV15,
    GTEBaseENV15Q,
    GTELargeENV15,
    GTELargeENV15Q,
    ClipVitB32,
    JinaEmbeddingsV2BaseCode,
});

impl EmbeddingModelCfg {
    pub fn get_info(&self) -> (i32, usize) {
        match self {
            Self::AllMiniLML6V2 | Self::AllMiniLML6V2Q => (384, 256),
            Self::AllMiniLML12V2 | Self::AllMiniLML12V2Q => (384, 512),
            Self::BGEBaseENV15 | Self::BGEBaseENV15Q => (768, 512),
            Self::BGELargeENV15 | Self::BGELargeENV15Q => (1024, 512),
            Self::BGESmallENV15 | Self::BGESmallENV15Q => (384, 512),
            Self::BGESmallZHV15 => (384, 512),
            Self::BGELargeZHV15 => (1024, 512),
            Self::NomicEmbedTextV1 => (768, 8192),
            Self::NomicEmbedTextV15 | Self::NomicEmbedTextV15Q => (768, 8192),
            Self::ParaphraseMLMiniLML12V2 | Self::ParaphraseMLMiniLML12V2Q => (384, 512),
            Self::ParaphraseMLMpnetBaseV2 => (768, 512),
            Self::ModernBertEmbedLarge => (1024, 512),
            Self::MultilingualE5Small => (384, 512),
            Self::MultilingualE5Base => (768, 512),
            Self::MultilingualE5Large => (1024, 512),
            Self::MxbaiEmbedLargeV1 | Self::MxbaiEmbedLargeV1Q => (1024, 8192),
            Self::GTEBaseENV15 | Self::GTEBaseENV15Q => (768, 512),
            Self::GTELargeENV15 | Self::GTELargeENV15Q => (1024, 512),
            Self::ClipVitB32 => (512, 77),
            Self::JinaEmbeddingsV2BaseCode => (768, 8192),
        }
    }
}

#[derive(Serialize, Deserialize, PartialEq, Debug)]
#[serde(rename_all = "lowercase")]
pub enum StoreType {
    SQLITE,
    ELASTICSEARCH,
}

#[derive(Serialize, Deserialize, PartialEq, Debug)]
pub struct Elasticsearch {
    pub api_id: String,
    pub api_key: String,
    pub index: String,
    pub urls: Vec<String>,
}

#[derive(Serialize, Deserialize, PartialEq, Debug)]
pub struct SQLite {
    pub connection_string: String,
    pub table: String,
}

fn default_store_type() -> StoreType {
    StoreType::SQLITE
}
fn default_chunk_size() -> usize {
    256
}
fn default_chunk_overlap() -> usize {
    32
}
fn default_context_size() -> u32 {
    8192
}
fn default_use_gpu() -> bool {
    false
}
fn default_sqlite_connection_string() -> String {
    "data.db".to_string()
}
fn default_retrieve_doc_count() -> usize {
    5
}

fn default_embedding_model() -> EmbeddingModelCfg {
    EmbeddingModelCfg::BGESmallENV15
}

#[derive(Serialize, Deserialize, PartialEq, Debug)]
pub struct Config {
    #[serde(default = "default_store_type")]
    pub store_type: StoreType,
    pub sqlite: SQLite,
    pub elasticsearch: Elasticsearch,
    #[serde(default = "default_sqlite_connection_string")]
    pub sqlite_connection_string: String,
    #[serde(default = "default_use_gpu")]
    pub use_gpu: bool,
    #[serde(default = "default_chunk_size")]
    pub chunk_size: usize,
    #[serde(default = "default_chunk_overlap")]
    pub chunk_overlap: usize,
    #[serde(default = "default_context_size")]
    pub context_size: u32,
    #[serde(default = "default_retrieve_doc_count")]
    pub retrieve_doc_count: usize,
    pub llm_model: String,
    pub ollama_url: String,
    #[serde(default = "default_embedding_model")]
    pub embedding_model: EmbeddingModelCfg,
}
