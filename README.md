#  Introduction
This is an implemenation of a retrieval augmented generation (RAG) system built to run locally on systems. It consists of a `console` application and a `library`. This implementation using `sqlite` to store data. 

# Building
1. `cd rag-console`
2. `cargo build --release`

# Running
1. You may need the following libraries when you use sqlite
- `vector0` and `vss0` downloaded from [here](https://github.com/asg017/sqlite-vss/releases)

##
See sample configuration file below.
```
---
store_type: sqlite
elasticsearch:
  api_id: id
  api_key: key
  index: documents
  urls:
    - https://uri1:9200
    - https://uri2:9200
sqlite:
  connection_string: "data.db"
  table: documents
use_gpu: true
chunk_size: 256
chunk_overlap: 32
context_size: 8192
retrieve_doc_count: 5
llm_model: "models/deepseek-llm-7b-chat.Q4_K_M.gguf"
embedding_model: "bgesmallenv15"
```
## Loading knowledge
See help information for loading information into the database.
```
$rag-console train --help
Usage: rag-console train [OPTIONS]

Options:
  -c, --config <CONFIG>       specifies configuration file to be loadedf [default: config.yaml]
  -s, --sources <SOURCES>...  specifies sources to be loaded
  -h, --help                  Print help
```
## Asking questions
See help information for interracting with model.
```
$rag-console console --help
Usage: rag-console console [OPTIONS]

Options:
  -c, --config <CONFIG>  specifies configuration file to be loadedf [default: config.yaml]
  -h, --help             Print help
```
