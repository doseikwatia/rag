use serde::{Deserialize, Serialize};

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
fn default_vector_dim()->i32 {
    384
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
    #[serde(default = "default_vector_dim")]
    pub vector_dim: i32,
    #[serde(default = "default_chunk_size")]
    pub chunk_size: usize,
    #[serde(default = "default_chunk_overlap")]
    pub chunk_overlap: usize,
    #[serde(default = "default_context_size")]
    pub context_size: u32,
    #[serde(default = "default_retrieve_doc_count")]
    pub retrieve_doc_count: usize,
    pub llm_model:String,
    pub embedding_model: String
}
