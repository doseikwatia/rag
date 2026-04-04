use std::{error::Error, sync::Arc};

use async_trait::async_trait;
use langchain_rust::{
    schemas::Document,
    vectorstore::{VecStoreOptions, VectorStore},
};

#[derive(Clone)]
pub struct RagVectorstore {
    store: Arc<dyn VectorStore>,
}
impl RagVectorstore {
    pub fn new<V: VectorStore + 'static>(store: V) -> Self {
        let store = Arc::new(store);
        Self { store }
    }
}
#[async_trait]
impl VectorStore for RagVectorstore {
    async fn add_documents(
        &self,
        docs: &[Document],
        opt: &VecStoreOptions,
    ) -> Result<Vec<String>, Box<dyn Error>> {
        self.store.add_documents(docs, opt).await
    }

    async fn similarity_search(
        &self,
        query: &str,
        limit: usize,
        opt: &VecStoreOptions,
    ) -> Result<Vec<Document>, Box<dyn Error>> {
        self.store.similarity_search(query, limit, opt).await
    }
}
