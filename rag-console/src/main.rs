mod cli;
use crate::cli::CMD_TRAIN;
use clap::ArgMatches;
use cli::{cli, CMD_CONSOLE};
use rag_lib::{
    dprintln,
    rag::{RAGAssistant, RAGTrainer},
};
use std::{
    io::{stdin, stdout, Write},
    sync::{Arc, Mutex},
    thread::{self, JoinHandle},
    time,
};
use tracing_subscriber::fmt::Subscriber;

#[tokio::main]
async fn main() {
    Subscriber::builder()
        .with_max_level(tracing::Level::WARN)
        .init();
    let cli_matches = cli().get_matches();
    match cli_matches.subcommand() {
        Some((sub_command, sub_matches)) => {
            let database = sub_matches
                .get_one::<String>("database")
                .expect("failed to obtain database");

            let table_name = sub_matches
                .get_one::<String>("tablename")
                .expect("failed to get tablename");
            let vector_dim = *sub_matches
                .get_one::<i32>("vectorsize")
                .expect("failed to get vectorsize");
            let use_gpu = sub_matches.get_flag("gpu");

            if sub_command == CMD_TRAIN {
                start_training(sub_matches, &database, &table_name, vector_dim, use_gpu).await
            } else if sub_command == CMD_CONSOLE {
                start_console(sub_matches, &database, &table_name, vector_dim, use_gpu).await;
            }
        }
        None => {}
    }
}

fn show_processing_animation(exit_processing: Arc<Mutex<bool>>) -> JoinHandle<()> {
    thread::spawn(move || {
        while !*exit_processing.lock().unwrap() {
            for bar in ["|", "/", "-", "\\"] {
                print!("\r{}", bar);
                stdout().flush().expect("unable to flush standard output");
                thread::sleep(time::Duration::from_millis(100));
            }
        }
        print!("\x1b[A");
    })
}

async fn start_training(
    sub_matches: &ArgMatches,
    database: &str,
    table_name: &str,
    vector_dim: i32,
    use_gpu: bool,
) {
    let main_is_processing = Arc::new(Mutex::new(false));
    let doc_chunk_size = *sub_matches
        .get_one::<usize>("chunksize")
        .expect("failed to get chunksize");
    let doc_overlap = *sub_matches
        .get_one::<usize>("overlap")
        .expect("failed to get overlap");
    let trainer = RAGTrainer::new(
        &database,
        table_name,
        vector_dim,
        doc_chunk_size,
        doc_overlap,
        use_gpu,
    )
    .await;
    let sources = sub_matches
        .get_many::<String>("sources")
        .unwrap()
        .map(|s| s.to_string())
        .collect();

    //progress indicator setup
    let anim_is_processing = Arc::clone(&main_is_processing);
    let processing_anim_handler = show_processing_animation(anim_is_processing);

    //do work
    trainer.train(sources).await.unwrap();

    //clean up
    handle_process_anim(&main_is_processing, processing_anim_handler);
}

async fn start_console(
    sub_matches: &ArgMatches,
    database: &str,
    table_name: &str,
    vector_dim: i32,
    use_gpu: bool,
) {
    let model_filename = sub_matches.get_one::<String>("model").unwrap();
    let retrieve_doc_count = *sub_matches
        .get_one::<usize>("retrieve_doc_count")
        .expect("failed to get retrieve_doc_count");
    let contex_szie = *sub_matches
        .get_one::<u32>("contextsize")
        .expect("failed to get contextsize");
    let mut ai_assistant = RAGAssistant::new(
        &database,
        &table_name,
        vector_dim,
        model_filename,
        contex_szie,
        retrieve_doc_count,
        use_gpu,
    )
    .await;
    let mut usr_input = String::new();
    loop {
        usr_input.clear();
        print!("\nHuman\t> ");
        stdout().flush().expect("unable to flush standard output");
        stdin().read_line(&mut usr_input).unwrap();
        let cleaned_input = usr_input.trim();

        match cleaned_input {
            ":x" => {
                print!("\nSystem\t> ");
                println!("Exiting");
                break;
            }
            ":c" => {
                ai_assistant.clear().await;
                print!("\nSystem\t> ");
                println!("Context cleared");
                continue;
            }
            "^[[A" => print!("up arrow"),
            _ => {
                dprintln!("calling ask");
                ai_assistant.ask(&usr_input.trim()).await;
            }
        }
        println!("");
    }
}

fn handle_process_anim(
    main_is_processing: &Arc<Mutex<bool>>,
    processing_anim_handler: JoinHandle<()>,
) {
    *main_is_processing.lock().unwrap() = true;
    processing_anim_handler.join().unwrap();
    *main_is_processing.lock().unwrap() = false;
}
