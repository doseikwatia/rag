use std::sync::Arc;

use langchain_rust::{
    agent::{AgentExecutor, ConversationalAgent, ConversationalAgentBuilder}, chain::{Chain, ChainError, options::ChainCallOptions}, language_models::llm::LLM, memory::SimpleMemory, prompt_args, tools::Tool
};
use url::Url;

use crate::{dprintln, utilities::threatfox_tool::ThreatFoxTool};

pub struct RagAgent {
    executor: AgentExecutor<ConversationalAgent>,
}

impl RagAgent {
    pub fn new<L:Into<Box<dyn LLM>>>(
        llm:L,
        threatfox_api_key: &str
    ) -> Self {
        let memory = SimpleMemory::new();
        let threatfox_tool =  ThreatFoxTool::new(&threatfox_api_key);
        
        let tools: &[Arc<dyn Tool>] = &[Arc::new(threatfox_tool)];
        let agent = ConversationalAgentBuilder::new()
            .tools(tools)
            .options(ChainCallOptions::new().with_max_tokens(1024))
            .build(llm)
            .expect("unable to create conversational agent");
        let executor = AgentExecutor::from_agent(agent).with_memory(memory.into());
        Self { executor }
    }

    pub async fn invoke(&self, input: &str) -> Result<String, ChainError> {
        dprintln!("Invoke called");
        let input_variables = prompt_args! {
            "input" => input,
        };
        self.executor.invoke(input_variables).await
    }
}
