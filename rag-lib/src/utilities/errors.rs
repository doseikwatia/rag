use std::{error::Error, fmt};

#[derive(Debug)]
pub enum RAGError {
    DocLoader(String),
    VectorStore(String),
    LLM(String),
}

impl fmt::Display for RAGError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RAGError::DocLoader(msg) => write!(f, "DocLoader: {}", msg),
            RAGError::VectorStore(msg) => write!(f, "VectorStore: {}", msg),
            RAGError::LLM(msg) => write!(f, "LLM: {}", msg),
        }
    }
}
impl Error for RAGError {}
