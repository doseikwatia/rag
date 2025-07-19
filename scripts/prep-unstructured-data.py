import yaml
import argparse
from unstructured.partition.auto import partition 
from unstructured.chunking.title import chunk_by_title
from tqdm import tqdm,trange
from fastembed import TextEmbedding
import re
from elasticsearch import Elasticsearch,helpers
import base64
import urllib3
import warnings

warnings.filterwarnings("ignore")

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def encode_api_key(api_key_id: str, api_key_secret: str) -> str:
    """
    Encodes the API key ID and secret into a base64 string for Elasticsearch authentication.
    
    Args:
        api_key_id (str): The API key ID.
        api_key_secret (str): The API key secret.

    Returns:
        str: A base64-encoded string in the format `base64(id:api_key)`.
    """
    combined = f"{api_key_id}:{api_key_secret}"
    encoded_bytes = base64.b64encode(combined.encode("utf-8"))
    return encoded_bytes.decode("utf-8")


def normalize_text(text: str, lowercase: bool = True) -> str:
    """
    Normalize text before embedding.
    
    Args:
        text (str): Raw input text chunk.
        lowercase (bool): Whether to convert to lowercase.
    
    Returns:
        str: Cleaned, normalized text.
    """
    # 1. Strip leading/trailing whitespace
    text = text.strip()

    # 2. Remove non-content patterns (tweak as needed)
    # e.g., remove page numbers like "Page 3", headers, footers
    text = re.sub(r'\bpage\s*\d+\b', '', text, flags=re.IGNORECASE)
    
    # 3. Remove excessive line breaks and whitespace
    text = re.sub(r'\s+', ' ', text)

    # 4. Optional: Lowercase the text
    if lowercase:
        text = text.lower()

    return text

def generate_actions(docs, index):
    for doc in docs:
        yield {
            "_op_type": "index",
            "_index": index,
            "_source": doc
        }

def get_fastembed_model_name(alias: str) -> str:
    """
    Converts a simplified alias into the official fastembed model name.
    
    Args:
        alias (str): User-friendly model name (e.g., "nomicembedtextv15").
        
    Returns:
        str: FastEmbed-compatible model name.
    """
    
    # Define mappings
    model_map = {
        "allminilml6v2": "sentence-transformers/all-MiniLM-L6-v2",
        "allminilml6v2q": "Quantized sentence-transformers/all-MiniLM-L6-v2",
        "allminilml12v2": "sentence-transformers/all-MiniLM-L12-v2",
        "allminilml12v2q": "Quantized sentence-transformers/all-MiniLM-L12-v2",
        "bgebaseenv15": "BAAI/bge-base-en-v1.5",
        "bgebaseenv15q": "Quantized BAAI/bge-base-en-v1.5",
        "bgelargeenv15": "BAAI/bge-large-en-v1.5",
        "bgelargeenv15q": "Quantized BAAI/bge-large-en-v1.5",
        "bgesmallenv15": "BAAI/bge-small-en-v1.5",
        "bgesmallenv15q": "Quantized BAAI/bge-small-en-v1.5",
        "nomicembedtextv1": "nomic-ai/nomic-embed-text-v1",
        "nomicembedtextv15": "nomic-ai/nomic-embed-text-v1.5",
        "nomicembedtextv15q": "Quantized v1.5 nomic-ai/nomic-embed-text-v1.5",
        "paraphrase mlminilml12v2": "sentence-transformers/paraphrase-MiniLM-L6-v2",
        "paraphrase mlminilml12v2q": "Quantized sentence-transformers/paraphrase-MiniLM-L6-v2",
        "paraphrase mlmpnetbasev2": "sentence-transformers/paraphrase-mpnet-base-v2",
        "bgesmallzhv15": "BAAI/bge-small-zh-v1.5",
        "bgelargezhv15": "BAAI/bge-large-zh-v1.5",
        "modernbertembedlarge": "lightonai/modernbert-embed-large",
        "multilinguale5small": "intfloat/multilingual-e5-small",
        "multilinguale5base": "intfloat/multilingual-e5-base",
        "multilinguale5large": "intfloat/multilingual-e5-large",
        "mxbaiembedlargev1": "mixedbread-ai/mxbai-embed-large-v1",
        "mxbaiembedlargev1q": "Quantized mixedbread-ai/mxbai-embed-large-v1",
        "gtebaseenv15": "Alibaba-NLP/gte-base-en-v1.5",
        "gtebaseenv15q": "Quantized Alibaba-NLP/gte-base-en-v1.5",
        "gtelargeenv15": "Alibaba-NLP/gte-large-en-v1.5",
        "gtelargeenv15q": "Quantized Alibaba-NLP/gte-large-en-v1.5",
        "clipvitb32": "Qdrant/clip-ViT-B-32-text",
        "jinaembeddingsv2basecode": "jinaai/jina-embeddings-v2-base-code"
    }
    
    return model_map.get(alias, None)


def main():
    arg_parser = argparse.ArgumentParser(prog='prep-unstructured-data')
    arg_parser.add_argument('-c','--config',default='config.yaml')
    arg_parser.add_argument('-i','--index',required= True)
    arg_parser.add_argument('-g','--gpu', default=False, action='store_true' )
    arg_parser.add_argument('-j','--jobs',default=5,required=False,type=int)
    arg_parser.add_argument('files', nargs="+")
    args = arg_parser.parse_args()
    
    #loading configuration
    with open(args.config) as cfg:
        config = yaml.load(cfg,yaml.SafeLoader)
    
    embedding_model = config['embedding_model']
    chunk_overlap = config['chunk_overlap']
    chunk_size = config['chunk_size']
    es_api_id = config['elasticsearch']['api_id']
    es_api_key = config['elasticsearch']['api_key']
    es_host = config['elasticsearch']['urls']
    n_jobs = args.jobs
    index_name = args.index
    

    es_client = Elasticsearch(hosts=es_host,api_key=(es_api_id,es_api_key),verify_certs=False)
    
    model_name = get_fastembed_model_name(embedding_model)
    embedder = TextEmbedding(model_name=model_name, cuda=args.gpu, threads=n_jobs)
    
    index_config = {
            "mappings": {
                "properties": {
                    "page_content": {
                    "type": "text"
                    },
                    "embedding": {
                    "type": "dense_vector",
                    "dims": embedder.embedding_size,
                    "index": True,
                    "similarity": "cosine"
                    },
                    "metadata": {
                        "type": "object",
                        "enabled": True
                    }
                }
            }
        }
    if es_client.ping():
        print('ping worked')
    else:
        print('ping failed')
        return

    
    if not es_client.indices.exists(index=index_name):
        es_client.indices.create(index=index_name, body=index_config)
        print(f"Index '{index_name}' created.")
    else:
        print(f"Index '{index_name}' already exists.")
    
    for file in tqdm(args.files):
        elements= partition(file,strategy="hi_res")
        chunks  = chunk_by_title(elements, overlap=chunk_overlap, max_characters=chunk_size)
        normalized_txt = map(lambda c: normalize_text(c.text,True),chunks)
        embeddings = list(embedder.passage_embed(normalized_txt))
        num_chunks = len(chunks)
        
        for i in trange(0,num_chunks,1):
            chunks[i] = chunks[i].to_dict()
            chunks[i]["embedding"] = embeddings[i].tolist()
            chunks[i]["page_content"]=chunks[i]["text"]
            del(chunks[i]["metadata"]["file_directory"])
            del(chunks[i]["text"])
            
        helpers.bulk(es_client,generate_actions(chunks,index_name))
        
if __name__=='__main__':
    main()