use async_trait::async_trait;
use fastembed::{RerankInitOptions, RerankerModel, TextRerank};
use langchain_rust::schemas::Document;
use langchain_rust::vectorstore::{VecStoreOptions, VectorStore};
use std::error::Error;


pub struct RerankerWrapper<V>
where
    V: VectorStore,
{
    store: V,
    model: TextRerank,
    limit_factor:usize
}

impl<V> RerankerWrapper<V>
where
    V: VectorStore,
{
    pub fn new(limit_factor:usize,store: V,) -> Self {
        let init_opt = RerankInitOptions::new(RerankerModel::BGERerankerBase)
            .with_show_download_progress(true);
        let model = TextRerank::try_new(init_opt).unwrap();
        RerankerWrapper { store, model,limit_factor}
    }
}
#[async_trait]
impl<V> VectorStore for RerankerWrapper<V>
where
    V: VectorStore,
{
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
        k: usize,
        opt: &VecStoreOptions,
    ) -> Result<Vec<Document>, Box<dyn Error>> {
        let docs = self.store.similarity_search(query, k*self.limit_factor, opt).await?;
        let doc_content = docs.iter().map(|d| d.page_content.as_str()).collect();
        // Optionally rerank with embedder/model
        let reranked: Vec<Document> = self
            .model
            .rerank(query, doc_content, true, Some(k))?
            .iter()
            .map(|r| {
                Document::new((r.document).clone().unwrap()).with_score(r.score as f64)
            })
            .collect();
        let head: Vec<Document> = reranked[0..k].to_vec();
        Ok(head)
    }
}
