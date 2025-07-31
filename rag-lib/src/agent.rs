use std::sync::Arc;

use langchain_rust::{
    agent::{AgentExecutor, ConversationalAgent, ConversationalAgentBuilder},
    chain::{options::ChainCallOptions, Chain, ChainError},
    memory::SimpleMemory,
    prompt_args,
    tools::{CommandExecutor, Tool},
};
use url::Url;

use crate::{dprintln, helpers::get_llm, utilities::threatfox_tool::ThreatFoxTool};

pub struct RagAgent {
    executor: AgentExecutor<ConversationalAgent>,
}

impl RagAgent {
    pub fn new(
        model_filename: &str,
        context_length: u32,
        use_gpu: bool,
        ollama_url: Option<Url>,
        threatfox_api_key: &str
    ) -> Self {
        let memory = SimpleMemory::new();
        let llm = get_llm(model_filename, context_length, use_gpu, 0.4_f32, ollama_url);
        let command_executor = CommandExecutor::default();
        let threatfox_tool =  ThreatFoxTool::new(&threatfox_api_key);
        
        let tools: &[Arc<dyn Tool>] = &[Arc::new(command_executor), Arc::new(threatfox_tool)];
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
