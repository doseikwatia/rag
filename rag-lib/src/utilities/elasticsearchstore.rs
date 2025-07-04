use async_trait::async_trait;
use elasticsearch::auth::Credentials;
use elasticsearch::http::request::JsonBody;
use elasticsearch::http::response::Response;
use elasticsearch::http::transport::Transport;
use elasticsearch::BulkParts;
use elasticsearch::Elasticsearch;
use elasticsearch::Error as ESerror;
use elasticsearch::IndexParts;
use elasticsearch::SearchParts;
use langchain_rust::embedding::Embedder;
use langchain_rust::schemas::Document;
use langchain_rust::vectorstore::VecStoreOptions;
use langchain_rust::vectorstore::VectorStore;
use serde_json::{json, Value};
use std::error::Error;

use crate::utilities::errors::AiError;

pub struct ElasticsearchStore<E>
where
    E: Embedder,
{
    client: Elasticsearch,
    embedder: E,
    vector_dim: i32,
    index: String,
}

impl<E> ElasticsearchStore<E>
where
    E: Embedder,
{
    pub fn new(
        urls: &[&str],
        api_id: &str,
        api_key: &str,
        embedder: E,
        vector_dim: i32,
        index_name: &str,
    ) -> Self {
        let transport =
            Transport::static_node_list(urls.into()).expect("could not create transport");
        let credentials = Credentials::ApiKey(api_id.to_string(), api_key.to_string());
        transport.set_auth(credentials);
        let client = Elasticsearch::new(transport);
        let index = index_name.to_string();
        ElasticsearchStore {
            client,
            embedder,
            vector_dim,
            index,
        }
    }
    pub async fn initialize(&self) -> Result<Response, ESerror> {
        self.client
            .index(IndexParts::Index(&self.index))
            .body(json!({
            "mappings": {
                "properties": {
                    "page_content": {
                    "type": "text"
                    },
                    "embedding": {
                    "type": "dense_vector",
                    "dims": self.vector_dim,
                    "index": true,
                    "similarity": "cosine"
                    },
                    "metadata": {
                        "type": "object",
                        "enabled": true
                    }
                }
                }
              }))
            .send()
            .await
    }
}

#[async_trait]
impl<E> VectorStore for ElasticsearchStore<E>
where
    E: Embedder,
{
    async fn add_documents(
        &self,
        docs: &[Document],
        _opt: &VecStoreOptions,
    ) -> Result<Vec<String>, Box<dyn Error>> {
        let page_contents: Vec<String> = docs.iter().map(|d| d.page_content.clone()).collect();
        let embeddings = self
            .embedder
            .embed_documents(&page_contents)
            .await
            .expect("failed to generate page_contents embeddings");

        let body: Vec<JsonBody<_>> = docs
            .iter()
            .zip(embeddings)
            .map(|(doc, emb)| {
                json!(
                    {
                        "page_content": doc.page_content,
                        "metadata": doc.metadata,
                        "embedding": emb
                    }
                )
                .into()
            })
            .collect();

        let response = self
            .client
            .bulk(BulkParts::Index(&self.index))
            .body(body)
            .send()
            .await?;
        let response_body = response.json::<Value>().await?;
        match response_body["errors"].as_bool().unwrap() {
            true => Ok(page_contents.to_vec()),
            false => Err(Box::new(AiError::new(&response_body.to_string()))),
        }
    }

    async fn similarity_search(
        &self,
        query: &str,
        limit: usize,
        _opt: &VecStoreOptions,
    ) -> Result<Vec<Document>, Box<dyn Error>> {
        let query_emb = self.embedder.embed_query(query).await?;
        let response = self
            .client
            .search(SearchParts::Index(&[self.index.as_str()]))
            .size(limit as i64)
            .body(json!({
                "query":{
                    "match_all": {}
                },
                "script": {
                    "source": "cosineSimilarity(params.query_emb, 'embedding') + 1.0",
                    "params": {
                        "query_emb": query_emb
                    }
                }
            }))
            .send()
            .await?;
        let response_body = response.json::<Value>().await?;
        let result: Vec<Document> = response_body["hits"]["hits"]
            .as_array()
            .unwrap()
            .iter()
            .map(|j| {
                let content = j["_source"]["page_content"]
                    .as_str()
                    .expect("could not extract page_content as string")
                    .to_string();
                Document::new(content)
            })
            .collect();
        Ok(result)
    }
}
