mod cli;
use std::{
    io::{stdin, stdout, Write},
    sync::{Arc, Mutex},
    thread::{self, JoinHandle},
    time,
};

use crate::cli::CMD_TRAIN;
use clap::ArgMatches;
use cli::{cli, CMD_CONSOLE};
use colored::Colorize;
use rag_lib::ai::{Assistant, Trainer};
use rand::Rng;

#[tokio::main]
async fn main() {
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

            if sub_command == CMD_TRAIN {
                start_training(sub_matches, &database, &table_name, vector_dim).await
            } else if sub_command == CMD_CONSOLE {
                start_console(sub_matches, &database, &table_name, vector_dim).await;
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
) {
    let main_is_processing = Arc::new(Mutex::new(false));
    let doc_chunk_size = *sub_matches
        .get_one::<usize>("chunksize")
        .expect("failed to get chunksize");
    let trainer = Trainer::new(&database, table_name, vector_dim, doc_chunk_size).await;
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
) {
    let main_is_processing = Arc::new(Mutex::new(false));
    let model_filename = sub_matches.get_one::<String>("model").unwrap();
    let retrieve_doc_count = *sub_matches
        .get_one::<usize>("retrieve_doc_count")
        .expect("failed to get retrieve_doc_count");
    let contex_szie = *sub_matches
        .get_one::<u32>("contextsize")
        .expect("failed to get contextsize");
    let min_type_delay = *sub_matches
        .get_one::<u64>("min_type_delay")
        .expect("failed to get min_type_delay");
    let max_type_delay = *sub_matches
        .get_one::<u64>("max_type_delay")
        .expect("failed to get max_type_delay");
    let mut rng = rand::thread_rng();
    let ai_assistant = Assistant::new(
        &database,
        &table_name,
        vector_dim,
        model_filename,
        contex_szie,
        retrieve_doc_count,
    )
    .await;
    let mut usr_input = String::new();
    loop {
        usr_input.clear();
        print!("\nHuman\t> ");
        stdout().flush().expect("unable to flush standard output");
        stdin().read_line(&mut usr_input).unwrap();
        let cleaned_input = usr_input.trim();
        if cleaned_input == ":x" {
            break;
        }else if cleaned_input == "^[[A" {
            print!("up arrow")
        }
         else if cleaned_input.len() == 0 {
            continue;
        }
        //set up progress indicator
        let anim_is_processing = Arc::clone(&main_is_processing);
        let processing_anim_handler = show_processing_animation(anim_is_processing);

        //do work
        let answer_result = ai_assistant.ask(&usr_input.trim()).await;

        if answer_result.is_err() {
            println!("{}", "something went wrong  :(".red());
            handle_process_anim(&main_is_processing, processing_anim_handler);
            continue;
        }

        //clean up
        handle_process_anim(&main_is_processing, processing_anim_handler);

        //display ai answer
        format!("\nAI\t> {}", answer_result.unwrap().trim())
            .chars()
            .for_each(|c| {
                print!("{}", c.to_string().blue());
                stdout().flush().expect("unable to flush standard output");
                let delay = rng.gen_range(min_type_delay..max_type_delay);
                thread::sleep(time::Duration::from_millis(delay));
            });
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
