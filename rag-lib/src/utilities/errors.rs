use std::{error::Error, fmt};

#[derive(Debug)]
pub struct AiError {
    details: String,
}
impl AiError {
    pub fn new(msg: &str) -> AiError {
        AiError {
            details: msg.to_string(),
        }
    }
}
impl fmt::Display for AiError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.details)
    }
}
impl Error for AiError {
    fn description(&self) -> &str {
        &self.details
    }
}
