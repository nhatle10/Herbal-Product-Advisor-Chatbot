# 🛍️ Herbal Product Advisor Chatbot with RAG

This project implements a Retrieval-Augmented Generation (RAG) chatbot designed to provide expert advice and recommendations on herbal products from **Trà Toàn Thắng**. The chatbot leverages advanced Natural Language Processing (NLP) techniques to understand user queries and provide informative responses, enhancing the customer experience.

## 📝 Project Overview

The chatbot utilizes a combination of techniques:

*   **Information Retrieval:**
    *   **BM25:** A traditional keyword-based ranking algorithm for initial retrieval.
    *   **Semantic Search:** FAISS vector store with `text-embedding-3-large` or `hiieu/halong_embedding` embeddings for capturing semantic meaning.
*   **Reranking:** `BAAI/bge-reranker-v2-m3` to refine the retrieved results based on contextual relevance.
*   **Response Generation:** LangChain's prompt template with `bge-large-vi` or `vinallama/vinallama-7b-chat-v1.0` to generate informative and coherent answers.
*   **Function Calling:** Tools like `ContactShop`, `RefuseToAnswer`, and `Retrieve` to handle specific user requests.
*   **Refined Query:** Improves the user's query by using the conversation history and the `Reflection` module.
*   **Streamlit:** Provides an interactive web-based user interface.
  
## 📑 Members

| Name           | Email                    | GitHub                                                                               |
| -------------- | ------------------------ | ------------------------------------------------------------------------------------ |
| Tăng Nhất      | faigar1004@gmail.com     | [![GadGadGad](https://img.shields.io/badge/GadGadGad-%2324292f.svg?style=flat-square&logo=github)](https://github.com/GadGadGad) |
| Lê Cảnh Nhật   | canhnhat922017@gmail.com | [![nhatle10](https://img.shields.io/badge/nhatle10-%2324292f.svg?style=flat-square&logo=github)](https://github.com/nhatle10)    |
| Thái Ngọc Quân | 22521189@gm.uit.edu.vn   | [![QuanThaiX](https://img.shields.io/badge/QuanThaiX-%2324292f.svg?style=flat-square&logo=github)](https://github.com/QuanThaiX) |

## Chatbot System Architecture:
![alt text](images/chatbot_pipeline.jpg)

## 🚀 Getting Started

### Prerequisites

1. **Python:** Ensure you have Python 3.10 installed.
2. **Conda (Recommended):** Install Anaconda or Miniconda to manage the project environment.
3. **ngrok Account (Optional):** If you want to deploy the chatbot with a public URL, sign up for a free ngrok account.
4. **OpenAI API Key (Optional):** Needed if you are using `text-embedding-3-large` for embeddings. Set this as an environment variable or put in colab secret.

### Installation

1. **Clone the repository:**

    ```bash
    git clone https://github.com/nhatle10/Herbal-Product-Advisor-Chatbot.git
    cd Herbal-Product-Advisor-Chatbot
    ```

2. **Create and activate a Conda environment (Recommended):**

    ```bash
    conda create --name chatbot-env --file requirements.txt
    conda activate chatbot-env
    ```

    Or, **Install dependencies using pip:**

    ```bash
    pip install -r requirements.txt
    ```

3. **Download VnCoreNLP Model (Required for Vietnamese tokenization):**

    ```bash
    python -c "import py_vncorenlp; py_vncorenlp.download_model(save_dir='/usr/local/lib/python3.10/dist-packages/py_vncorenlp')"
    ```
4. **Download Embedding Models and Reranker (Optional):**
    If you want to use `hiieu/halong_embedding` or `BAAI/bge-reranker-v2-m3`, ensure you have the Hugging Face `transformers` library installed and these models downloaded. They will download automatically the first time they are used, but pre-downloading can save time during first execution.
    ```python
    from sentence_transformers import SentenceTransformer
    from FlagEmbedding import FlagReranker

    # Download embedding model
    sentence_transformer = SentenceTransformer("hiieu/halong_embedding")

    # Download reranker model
    reranker = FlagReranker('BAAI/bge-reranker-v2-m3', use_fp16=True)
    ```

### Data Setup

1. **Web Crawling:** The `crawler.py` script is used to scrape product data from `tratoanthang.com`. Run it as follows:

    ```bash
    python crawler.py --output_links product_links.txt --output_json product_data.json --headless
    ```
    This will generate `product_links.txt` (containing product URLs) and `product_data.json` (containing product details).

2. **Data Preprocessing:**
    * You need to run `data_loader.py` first to load data into `documents.txt`.
    *   `data/documents.txt`: This file is created by extracting relevant text content from `product_data.json`.
    *   `data/chunks_data/chunks.json`: This file contains pre-chunked data using a specific text splitter (you might need to adjust chunking parameters based on your needs).
    *   `data/vector_store`: This directory will store the FAISS index and metadata files. The `app.py` script will create and save them here when you run it for the first time.

### Running the Chatbot
1. **Start the Flask API:**
    ```bash
    python app.py
    ```
    This will start the Flask backend API and expose it via ngrok (if configured correctly). The ngrok URL will be printed in the console and saved to `pages/ngrok_url.txt`.

2. **Start the Streamlit Frontend:**
    *   Open the `chatbot.py` file.
    *   Make sure the `ngrok_url` is correctly read from `ngrok_url.txt`.
    *   Run the Streamlit app:

        ```bash
        streamlit run chatbot.py
        ```

    This will open the chatbot interface in your web browser.

## ⚙️ Configuration

The `config/config.py` file contains various configuration options:

*   **Model Choices:**
    *   `MODEL_NAME`: The language model for response generation.
    *   `USE_OPENAI_EMBEDDINGS`: Whether to use OpenAI embeddings or Sentence Transformers.
    *   `OPENAI_EMBEDDING_MODEL`: The OpenAI embedding model to use.
    *   `OPENAI_EMBEDDING_DIMENSIONS`: Dimensions for OpenAI embeddings.
    *   `SENTENCE_TRANSFORMER_MODEL`: The Sentence Transformer model to use (if not using OpenAI embeddings).
    *   `RERANKER_MODEL`: The reranker model.
*   **Temperature:**  `TEMPERATURE` controls the randomness of the language model's output.
*   **Other Parameters:** Chunking parameters, vector store settings, etc.

## 📂 Project Structure

*   `.gitignore`: Specifies files and directories to be ignored by Git.
*   `app.py`: The main Flask application that handles API requests and the chatbot logic.
*   `chatbot.py`: The Streamlit application for the chatbot user interface.
*   `crawler.py`: Web scraper for collecting data from tratoanthang.com.
*   `data/`: Stores every data related.
*   `demo.ipynb`: Jupyter Notebook demonstrating the chatbot's functionality.
*   `Evaluation.ipynb`: Jupyter Notebook for evaluating the performance of different retrieval methods.
*   `modules/`: Contains Python modules for different components of the chatbot:
    *   `chunker.py`: Handles text chunking.
    *   `data_loader.py`: Loads data from files.
    *   `llm_generator.py`: Initializes the language model chain.
    *   `reflection.py`: Implements query refinement logic.
    *   `retriever.py`: Implements the hybrid retrieval logic.
    *   `tools.py`: Defines the tools used by the chatbot (e.g., `ContactShop`).
    *   `vector_store.py`: Handles loading and saving the vector store.
*   `chatbot.py`: The main chatbot.
*   `ngrok_url.txt`: Stores the ngrok URL.
*   `requirements.txt`: Lists the required Python packages.

## 🤝 Contributions
Contributions to this project are welcome. Please feel free to fork the repository, make changes, and submit a pull request.

## 📄 License
This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgements
- Trà Toàn Thắng for providing the data used in this project.
- Hugging Face for their excellent Transformers library.
- LangChain for providing the framework for building the RAG pipeline.
- Streamlit for making it easy to create interactive web applications.

---
