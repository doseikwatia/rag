mod extractousloader;
use std::path::Path;

use futures_util::StreamExt;
use langchain_rust::{document_loaders::Loader, schemas::Document, text_splitter::{SplitterOptions, TokenSplitter}};

use crate::{docloader::extractousloader::ExtractousLoader, utilities::errors::RAGError};


pub async fn get_docs(
    docpath: String,
    split_size: usize,
    chunk_overlap: usize,
) -> Result<Vec<Document>, RAGError> {
    let path = Path::new(&docpath);
    let extension = Path::extension(path);

    if extension.is_none() {
        return Err(RAGError::DocLoader("no extension specified".into()));
    }

    let splitter_options = SplitterOptions::new()
        .with_chunk_size(split_size)
        .with_chunk_overlap(chunk_overlap);
        

    let splitter = TokenSplitter::new(splitter_options);

    let filename_value = Path::file_name(path)
        .expect("failed to extract filename from the path")
        .to_str()
        .unwrap();
    let filename_key = "filename".to_string();

    let docs = ExtractousLoader::new(&docpath)
        .load_and_split(splitter)
        .await
        .unwrap()
        .map(|x| {
            let mut doc = x.expect("unable to get the document");
            doc.metadata
                .insert(filename_key.clone(), filename_value.into());
            doc
        })
        .collect::<Vec<_>>()
        .await;
    Ok(docs)
}
