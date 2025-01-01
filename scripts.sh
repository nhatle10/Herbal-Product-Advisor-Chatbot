python modules/chunker.py --mode create --tokenizer "uitnlp/visobert" --embedding_model "text-embedding-3-large"
python modules/vector_store.py --chunk_mode load
python modules/vector_store.py --chunk_mode load --use_openai_embedding
python modules/vector_store.py --chunk_mode load --use_openai_embedding --embedding_model "text-embedding-3-large"