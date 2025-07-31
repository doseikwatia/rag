use std::error::Error;

use async_trait::async_trait;
use langchain_rust::tools::Tool;
use reqwest::{header::HeaderValue, Client};
use serde_json::{json, Value};

const API_URL: &str = "https://threatfox-api.abuse.ch/api/v1/";

pub struct ThreatFoxTool {
    auth_key: String,
}

impl ThreatFoxTool {
    pub fn new(api_key: &str) -> Self {
        let auth_key = String::from(api_key);
        ThreatFoxTool { auth_key }
    }

    pub async fn search(
        &self,
        search_term: &str,
    ) -> Result<String, reqwest::Error> {
        let auth_key = HeaderValue::from_str(&self.auth_key).unwrap();
        let client = Client::new();
        let body = json!({
            "query": "search_ioc",
            "search_term": search_term,
            "exact_match": false }
        );
        client
            .post(API_URL)
            .json(&body)
            .header("Auth-Key", auth_key)
            .send()
            .await
            .expect("could send request")
            .text()
            .await
    }
}

#[async_trait]
impl Tool for ThreatFoxTool {
    #[doc = "Returns the name of the tool."]
    fn name(&self) -> String {
        String::from("ThreatFox")
    }

    #[doc = " Provides a description of what the tool does and when to use it."]
    fn description(&self) -> String {
        String::from(
            r#" Use this tool to retrieve threat intelligence about known malicious IP addresses, domains, URLs, and file hashes. Data is sourced from the Abuse.ch ThreatFox database and includes IOC type, threat confidence, malware family, and timestamps. Ideal for detecting known indicators of compromise (IOCs) and classifying cyber threats.
        "#,
        )
    }

    #[doc = " Executes the core functionality of the tool."]
    #[doc = ""]
    #[doc = " Example implementation:"]
    #[doc = " ```rust,ignore"]
    #[doc = " async fn run(&self, input: Value) -> Result<String, Box<dyn Error>> {"]
    #[doc = "     let input_str = input.as_str().ok_or(\"Input should be a string\")?;"]
    #[doc = "     self.simple_search(input_str).await"]
    #[doc = " }"]
    #[doc = " ```"]
    #[must_use]
    #[allow(clippy::type_complexity, clippy::type_repetition_in_bounds)]
    async fn run(&self, input: Value) -> Result<String, Box<dyn Error>> {
        println!("input {:?}",input);
        let search_term = input
            .as_str()
            .expect("unable to get search term");
        let response = self.search(search_term).await?;
        Ok(response)
    }

    fn parameters(&self) -> Value {
        let prompt = r#"A wrapper around Threatfox Indicator of Compromise (IOC) search.
            Useful for when you need to answer questions about Indicator of Compromise (IOC) or determine whether an IP, domain, hash or socket is a threat or not
            Input should be a search query. Output is a JSON array of the query results."#;

        json!({
            "description": prompt,
            "type": "object",
            "properties": {
                "search_term": {
                    "type": "string",
                    "description": "The value to search for — can be an IP address, domain name, URL, or file hash (MD5, SHA1, SHA256)."
                }
            },
            "required": ["search_term"]
        })
    }
}




#[cfg(test)]
mod tests {
    use std::env;

    use super::ThreatFoxTool;
    use dotenv::dotenv;

    #[tokio::test]
    #[ignore]
    async fn threatfox_tool() {
        dotenv().ok();
        let api_key = env::var("API_KEY").unwrap();
        let tool = ThreatFoxTool::new(&api_key);
        let s = tool.search("196.251.83.29").await.expect("could not search for domain");
        println!("{}", s);
    }
}