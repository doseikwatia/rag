use std::path::PathBuf;

use clap::{arg, value_parser, Command};

pub const CMD_TRAIN: &str = "train";
pub const CMD_CONSOLE: &str = "console";


pub fn cli() -> Command {
    Command::new("rag")
        .about("implementation of a console based RAG system")
        .subcommand_required(true)
        .subcommand(
            Command::new(CMD_TRAIN)
                .arg(
                    arg!(-c --config <CONFIG> "specifies configuration file to be loadedf")
                        .num_args(1)
                        .value_parser(value_parser!(PathBuf))
                        .default_value("config.yaml"),
                )
                .arg(
                    arg!(-s --sources <SOURCES> ... "specifies sources to be loaded").num_args(1..),
                ),
        )
        .subcommand(
            Command::new(CMD_CONSOLE).arg(
                arg!(-c --config <CONFIG> "specifies configuration file to be loadedf")
                    .num_args(1)
                    .value_parser(value_parser!(PathBuf))
                    .default_value("config.yaml"),
            ),
        )
}
