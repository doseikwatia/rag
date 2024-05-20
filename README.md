#  Introduction
This is an implemenation of a retrieval augmented generation (RAG) system built to run locally on systems. It consists of a `console` application and a `library`. This implementation using `sqlite` to store data. 

# Building
1. `cd rag-console`
2. `cargo build --release`

# Running
1. You may need the following libraries
- `vector0` and `vss0` downloaded from [here](https://github.com/asg017/sqlite-vss/releases)
2. `llama-2-7b-chat` is used by default however you may also need any llm model in `*.gguf` format in the list here. [here](https://github.com/ggerganov/llama.cpp)

## Loading knowledge
See help information for loading information into the database.
```
$rag-console train --help
Usage: rag-console train [OPTIONS]

Options:
  -d, --database <DATABASE>        specifies the database file to use [default: data.db]
  -s, --sources <SOURCES>...       specifies sources to be loaded
  -v, --vectorsize <VECTORSIZE>    [default: 384]
  -t, --tablename <TABLENAME>      [default: document]
  -c, --chunksize <CHUNKSIZE>      [default: 512]
  -z, --contextsize <CONTEXTSIZE>  [default: 4096]
  -h, --help                       Print help
```
## Asking questions
See help information for interracting with model.
```
rag-console console --help
Usage: rag-console console [OPTIONS]

Options:
  -d, --database <DATABASE>
          specifies the database file to use [default: data.db]
  -m, --model <MODEL>
          specifies the model to be used [default: llama-2-7b-chat.gguf]
  -v, --vectorsize <VECTORSIZE>
          [default: 384]
  -t, --tablename <TABLENAME>
          [default: document]
  -c, --chunksize <CHUNKSIZE>
          [default: 512]
  -z, --contextsize <CONTEXTSIZE>
          [default: 4096]
      --min_type_delay <MINTYPEDELAY>
          [default: 50]
      --max_type_delay <MAXTYPEDELAY>
          [default: 100]
  -r, --retrieve_doc_count <RETRIEVEDOCCOUNT>
          [default: 5]
  -h, --help
          Print help
```