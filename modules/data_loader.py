import json
import re
import argparse
from typing import List, Dict
from langchain_core.documents import Document
import pandas as pd

def process_data(df: pd.DataFrame) -> pd.DataFrame:
    """Processes the product data DataFrame.

    Args:
        df: The input DataFrame.

    Returns:
        The processed DataFrame.
    """
    df['info'] = df['info'].str.replace('Tình Trạng: ', '')
    df = df.replace(' ', ' ')
    df.drop("options", axis=1, inplace=True)
    df = df.replace(r'\n', '-', regex=True)
    df['price'] = df['price'].fillna(df['sale-price'])
    df = df[:-1]

    def categorize_title(title):
        if 'COMBO' in title:
            return 'Sản phẩm Combo'
        elif any(keyword in title for keyword in ['VIÊN', 'Viên']):
            return 'Sản phẩm dạng Viên'
        elif any(keyword in title for keyword in ['CAO', 'Cao']):
            return 'Sản phẩm dạng Cao'
        elif any(keyword in title for keyword in ['BỘT', 'Bột']):
            return 'Sản phẩm dạng Bột'
        elif any(keyword in title for keyword in ['Trà', 'Túi trà', 'TRÀ', 'TÚI TRÀ']):
            return 'Sản phẩm Trà'
        return 'Sản phẩm khác'

    df['category'] = df['title'].apply(categorize_title)
    return df

def load_product_data(file_path: str = 'data/raw_product_data.json',
                      output_file_path: str = 'data/processed_product_data.json') -> List[Dict]:
    """Loads product data from a JSON file, processes it, and removes special characters from details."""
    df = pd.read_json(file_path)
    df = process_data(df)
    df.to_json(output_file_path, orient='records', indent=4, force_ascii=False)
    data = df.to_dict('records')
    for item in data:
        item['details'] = re.sub(r'[^\S ]+', ' ', item['details'])
    return data

def create_documents_from_data(data: List[Dict]) -> List[Document]:
    """Converts product data into a list of LangChain Document objects."""
    list_of_documents = []
    for item in data:
        tag_strings = ", ".join(str(tag) for tag in item['tags'])
        content = 'Tên sản phẩm: ' + item['title'] + " - Giá gốc: " + item['price'] + " - Giá khuyến mãi: " + item['sale-price'] + " - Danh mục sản phẩm: " + item['category']  + ' - Mô tả: ' + item['description'] + ' - Chi tiết: ' + item['details'] 
        list_of_documents.append(Document(page_content=content, metadata={"source": item['url'],
                                                                "image_urls":item['image_url'],
                                                                "info": item['info'],
                                                                "tags": tag_strings,
                                                                }))
    return list_of_documents

def save_documents_to_file(documents: List[Document], file_path: str = 'documents.txt') -> None:
    """Saves a list of LangChain Document objects to a text file."""
    with open(file_path, 'w', encoding='utf-8') as f:
        for doc in documents:
            f.write(f"Content: {doc.page_content}\n")
            f.write(f"Metadata: {doc.metadata}\n")
            f.write("----\n")
    print(f"Saved {len(documents)} documents to {file_path}")

def load_documents_from_file(file_path: str = 'documents.txt') -> List[Document]:
    """Loads a list of LangChain Document objects from a text file."""
    documents = []
    with open(file_path, 'r', encoding='utf-8') as f:
        content = ""
        metadata = {}
        for line in f:
            if line.strip() == "----":
                documents.append(Document(page_content=content.strip(), metadata=metadata))
                content = ""
                metadata = {}
            elif line.startswith("Content: "):
                content = line[len("Content: "):]
            elif line.startswith("Metadata: "):
                metadata_str = line[len("Metadata: "):].strip()
                try:
                    metadata = json.loads(metadata_str.replace("'", "\""))
                except json.JSONDecodeError:
                    print(f"Warning: Could not decode metadata: {metadata_str}")
                    metadata = {}
            else:
                content += line

        if content:
             documents.append(Document(page_content=content.strip(), metadata=metadata))

    print(f"Loaded {len(documents)} documents from {file_path}")
    return documents

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process product data and create/load LangChain documents.")
    parser.add_argument("--input", type=str, default="data/raw_product_data.json", help="Path to the input JSON file.")
    parser.add_argument("--output-json", type=str, default="data/product_data.json", help="Path to the out JSON file.")
    parser.add_argument("--output", type=str, default="data/documents.txt", help="Path to the output documents file.")
    parser.add_argument("--mode", choices=["create", "load"], default="create", help="Whether to create new documents or load from file.")

    args = parser.parse_args()

    if args.mode == "create":
        data = load_product_data(args.input, args.output_json)
        print(f"Loaded {len(data)} product items from {args.input}.")
        if len(data) > 0:
            print(data[0]['details']) 

        list_of_documents = create_documents_from_data(data)
        print(f"Created {len(list_of_documents)} documents.")
        if len(list_of_documents) > 0:
            print(list_of_documents[0].page_content)

        save_documents_to_file(list_of_documents, args.output)
    elif args.mode == "load":
        list_of_documents = load_documents_from_file(args.output)
        print(f"Loaded documents from {args.output}")
        if len(list_of_documents) > 0:
            print(list_of_documents[0].page_content)
    else:
        print("Invalid mode specified. Use 'create' or 'load'.")