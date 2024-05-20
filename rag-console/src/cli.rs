use clap::{arg, value_parser, Command};

const DEFAULT_DB: &str = "data.db";
pub const CMD_TRAIN: &str = "train";
pub const CMD_CONSOLE: &str = "console";
const DEFAULT_VECTOR_SIZE:&str = "384";
const DEFAULT_TABLE_NAME: &str = "document";
const DEFAULT_CHUNK_SIZE: &str = "512";
const DEFAULT_CONTEXT_SIZE : &str = "4096";
const DEFAULT_MINTYPEDELAY: &str = "50";
const DEFAULT_MAXTYPEDELAY: &str = "100";
const DEFAULT_RETRIEVEDOCCOUNT: &str = "5";

pub fn cli() -> Command {
    Command::new("rag")
        .about("implementation of a console based RAG system")
        .subcommand_required(true)
        .subcommand(
            Command::new(CMD_TRAIN)
                .arg(
                    arg!(-d --database <DATABASE> "specifies the database file to use")
                        .num_args(1)
                        .default_value(DEFAULT_DB)
                        .required(false),
                )
                .arg(
                    arg!(-s --sources <SOURCES> ... "specifies sources to be loaded").num_args(1..),
                )
                .arg(
                    arg!(-v --vectorsize <VECTORSIZE>)
                        .num_args(1)
                        .value_parser(value_parser!(i32))
                        .default_value(DEFAULT_VECTOR_SIZE)
                        .required(false),
                )
                .arg(
                    arg!(-t --tablename <TABLENAME>)
                        .num_args(1)
                        .default_value(DEFAULT_TABLE_NAME)
                        .required(false),
                )
                .arg(
                    arg!(-c --chunksize <CHUNKSIZE>)
                        .num_args(1)
                        .value_parser(value_parser!(usize))
                        .default_value(DEFAULT_CHUNK_SIZE)
                        .required(false),
                )
                .arg(
                    arg!(-z --contextsize <CONTEXTSIZE>)
                        .num_args(1)
                        .value_parser(value_parser!(u32))
                        .default_value(DEFAULT_CONTEXT_SIZE)
                        .required(false),
                ),
        )
        .subcommand(
            Command::new(CMD_CONSOLE)
                .arg(
                    arg!(-d --database <DATABASE> "specifies the database file to use")
                        .num_args(1)
                        .default_value(DEFAULT_DB)
                        .required(false),
                )
                .arg(
                    arg!(-m --model <MODEL> "specifies the model to be used")
                        .num_args(1)
                        .default_value("llama-2-7b-chat.gguf")
                        .required(false),
                )
                .arg(
                    arg!(-v --vectorsize <VECTORSIZE>)
                        .num_args(1)
                        .value_parser(value_parser!(i32))
                        .default_value(DEFAULT_VECTOR_SIZE)
                        .required(false),
                )
                .arg(
                    arg!(-t --tablename <TABLENAME>)
                        .num_args(1)
                        .default_value(DEFAULT_TABLE_NAME)
                        .required(false),
                )
                .arg(
                    arg!(-c --chunksize <CHUNKSIZE>)
                        .num_args(1)
                        .value_parser(value_parser!(usize))
                        .default_value(DEFAULT_CHUNK_SIZE)
                        .required(false),
                )
                .arg(
                    arg!(-z --contextsize <CONTEXTSIZE>)
                        .num_args(1)
                        .value_parser(value_parser!(u32))
                        .default_value(DEFAULT_CONTEXT_SIZE)
                        .required(false),
                )
                .arg(
                    arg!(--min_type_delay <MINTYPEDELAY>)
                        .num_args(1)
                        .value_parser(value_parser!(u64))
                        .default_value(DEFAULT_MINTYPEDELAY)
                        .required(false),
                )
                .arg(
                    arg!(--max_type_delay <MAXTYPEDELAY>)
                        .num_args(1)
                        .value_parser(value_parser!(u64))
                        .default_value(DEFAULT_MAXTYPEDELAY)
                        .required(false),
                )
                .arg(
                    arg!(-r --retrieve_doc_count <RETRIEVEDOCCOUNT>)
                    .num_args(1)
                    .value_parser(value_parser!(usize))
                    .default_value(DEFAULT_RETRIEVEDOCCOUNT)
                    .required(false),
                ),
        )
}
