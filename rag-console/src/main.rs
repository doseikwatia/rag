mod cli;
mod screen;
use crate::{cli::CMD_TRAIN, screen::Screen};
use cli::{cli, CMD_CONSOLE};
use futures_util::StreamExt;
use rag_lib::{
    configuration::Config,
    dprintln, get_store,
    rag::{RAGAssistant, RAGTrainer},
};
use std::fs::File;
use std::{
    io::{stdout, Write},
    path::PathBuf,
    sync::{Arc, Mutex},
    thread::{self, JoinHandle},
    time,
};
use tracing_subscriber::fmt::Subscriber;
use url::Url;

#[tokio::main]
async fn main() {
    Subscriber::builder()
        .with_max_level(tracing::Level::WARN)
        .init();
    let cli_matches = cli().get_matches();
    match cli_matches.subcommand() {
        Some((sub_command, sub_matches)) => {
            let config_path = sub_matches
                .get_one::<PathBuf>("config")
                .expect("failed to obtail the configuration path");
            let file = File::open(config_path).expect("could not open configuration file");
            let config: Config =
                serde_yaml::from_reader(file).expect("could not parse config yaml");
            dprintln!("{config:?}",);

            if sub_command == CMD_TRAIN {
                let sources = sub_matches
                    .get_many::<String>("sources")
                    .unwrap()
                    .map(|s| s.to_string())
                    .collect();
                start_training(&config, sources).await
            } else if sub_command == CMD_CONSOLE {
                start_console(&config).await;
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

async fn start_training(config: &Config, sources: Vec<String>) {
    let (_, max_chunk_size) = config.embedding_model.get_info();
    assert!(
        max_chunk_size >= config.chunk_size,
        "chunk_size {} is greater than max chunk_size, {} supported by embedding model, {:?}.",
        config.chunk_size,
        max_chunk_size,
        config.embedding_model
    );
    let store = get_store(config).await;
    let main_is_processing = Arc::new(Mutex::new(false));
    let trainer = RAGTrainer::new(
        store,
        config.chunk_size,
        config.chunk_overlap,
        config.use_gpu,
    )
    .await;

    //progress indicator setup
    let anim_is_processing = Arc::clone(&main_is_processing);
    let processing_anim_handler = show_processing_animation(anim_is_processing);

    //do work
    trainer.train(sources).await.unwrap();

    //clean up
    handle_process_anim(&main_is_processing, processing_anim_handler);
}

async fn start_console(config: &Config) {
    let main_is_processing = Arc::new(Mutex::new(false));
    let store = get_store(config).await;
    let ollama_url = Url::parse(&config.ollama_url).ok();
    let mut ai_assistant = RAGAssistant::new(
        &config.llm_model,
        config.context_size,
        store,
        config.retrieve_doc_count,
        config.use_gpu,
        ollama_url,
    )
    .await;
    let mut screen = Screen::new(None);
    loop {
        let cleaned_input = screen.read_human();

        match cleaned_input.as_str() {
            ":x" => {
                let _ = screen.write_system("Exiting\n");
                break;
            }
            ":c" => {
                ai_assistant.clear().await;
                screen.clear();
                let _ = screen.write_system("Context cleared\n");
                continue;
            }
            _ => {
                if cleaned_input.is_empty(){
                    continue;
                }
                dprintln!("calling ask");
                //progress indicator setup
                let anim_is_processing = Arc::clone(&main_is_processing);
                let processing_anim_handler = show_processing_animation(anim_is_processing);
                let mut stream = ai_assistant.ask(&cleaned_input).await;
                handle_process_anim(&main_is_processing, processing_anim_handler);

                let _ = screen.write_str("AI\t> ");
                while let Some(result) = stream.next().await {
                    match result {
                        Ok(data) => {
                            let _ = screen.write_str(&data.content);
                        }
                        Err(e) => panic!("Error invoking LLMChain: {e:}"),
                    }
                }
                let _ = screen.write_str("\n");
            }
        }
        // println!("");
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
