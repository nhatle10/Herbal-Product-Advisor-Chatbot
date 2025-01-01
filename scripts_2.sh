python modules/chunker.py --mode create --output "data/chunks_data_2" --embedding_model "text-embedding-3-large"
python modules/vector_store.py --chunk_mode load --chunks_output "data/chunks_data_2" --vectorstore_output "data/vector_store_2"
python modules/vector_store.py --chunk_mode load --use_openai_embedding --chunks_output "data/chunks_data_2" --vectorstore_output "data/vector_store_2"
python modules/vector_store.py --chunk_mode load --use_openai_embedding --embedding_model "text-embedding-3-large" --chunks_output "data/chunks_data_2" --vectorstore_output "data/vector_store_2"
