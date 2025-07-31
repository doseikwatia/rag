use std::path::PathBuf;

use clap::{arg, value_parser, Command};

pub const CMD_TRAIN: &str = "train";
pub const CMD_CONSOLE: &str = "rag";
pub const CMD_AGENT: &str = "agent";

pub fn cli() -> Command {
    let config_arg = arg!(-c --config <CONFIG> "specifies configuration file to be loaded")
        .num_args(1)
        .value_parser(value_parser!(PathBuf))
        .default_value("config.yaml");
    Command::new("rag")
        .about("implementation of a console based RAG system")
        .subcommand_required(true)
        .subcommand(
            Command::new(CMD_TRAIN).arg(&config_arg).arg(
                arg!(-s --sources <SOURCES> ... "specifies sources to be loaded").num_args(1..),
            ),
        )
        .subcommand(Command::new(CMD_CONSOLE).arg(&config_arg))
        .subcommand(Command::new(CMD_AGENT).arg(&config_arg))
}
