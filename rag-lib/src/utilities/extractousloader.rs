use std::{io::Read, path::Path, pin::Pin};

use async_trait::async_trait;
use extractous::Extractor;
use futures_util::{stream, Stream, StreamExt};

use langchain_rust::{
    document_loaders::{Loader, LoaderError},
    schemas::Document,
    text_splitter::TextSplitter,
};
use serde_json::Value;

#[derive(Debug, Clone)]
pub struct ExtractousLoader {
    path: String,
}

impl ExtractousLoader {
    pub fn new(path: &str) -> Self {
        Self {
            path: path.to_string(),
        }
    }
    pub fn from_path(path: &Path)->Self{
        Self {path:path.to_str().unwrap().to_string()}
    }
}

#[async_trait]
impl Loader for ExtractousLoader {
    async fn load(
        mut self,
    ) -> Result<
        Pin<Box<dyn Stream<Item = Result<Document, LoaderError>> + Send + 'static>>,
        LoaderError,
    > {
        let extractor = Extractor::new();
        let (mut content_stream, metadata) = extractor
            .extract_file(&self.path)
            .map_err(|err| LoaderError::LoadDocumentError(err.to_string()))?;
        let mut buf = String::new();
        content_stream.read_to_string(&mut buf);
        let doc_metadata = metadata
            .into_iter()
            .map(|(key, value)| {
                (
                    key,
                    Value::Array(value.iter().map(|v| Value::String(v.clone())).collect()),
                )
            })
            .collect();
        let doc = Document::new(buf).with_metadata(doc_metadata);
        let stream = stream::iter(vec![Ok(doc)]);
        Ok(Box::pin(stream))
    }

    async fn load_and_split<TS: TextSplitter + 'static>(
        mut self,
        splitter: TS,
    ) -> Result<
        Pin<Box<dyn Stream<Item = Result<Document, LoaderError>> + Send + 'static>>,
        LoaderError,
    > {
        let doc_stream = self.load().await?;

        let stream = process_doc_stream(doc_stream, splitter).await;
        Ok(Box::pin(stream))
    }
}

pub(crate) async fn process_doc_stream<TS: TextSplitter + 'static>(
    doc_stream: Pin<Box<dyn Stream<Item = Result<Document, LoaderError>> + Send>>,
    splitter: TS,
) -> impl Stream<Item = Result<Document, LoaderError>> {
    async_stream::stream! {
        futures_util::pin_mut!(doc_stream);
        while let Some(doc_result) = doc_stream.next().await {
            match doc_result {
                Ok(doc) => {
                    match splitter.split_documents(&[doc]).await {
                        Ok(docs) => {
                            for doc in docs {
                                yield Ok(doc);
                            }
                        },
                        Err(e) => yield Err(LoaderError::TextSplitterError(e)),
                    }
                }
                Err(e) => yield Err(e),
            }
        }
    }
}
