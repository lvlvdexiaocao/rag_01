from pathlib import Path
from typing import List

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

# =========配置项=========
FILE_PATH="../data"
EMBEDDING_MODEL="quentinz/bge-small-zh-v1.5:f16"
DB_PERSIST_DIRECTORY="../db/base_db"
COLLECTION_NAME="base01"



# =========离线操作=========
# 加载文档
# 遍历文件中的多pdf，每页pdf
# 为metadata添加数据，source文件名，category所属文件夹名
def load_document(data_path)-> List[Document]:
    """加载文档数据"""
    data_path_obj = Path(data_path)
    print(f"从{data_path_obj}加载")
    all_docs = []
    for pdf_file in data_path_obj.rglob("*.pdf"):
        print(f"从{pdf_file} 加载文档")
        loader = PyMuPDFLoader(str(pdf_file))
        content = loader.load()
        for doc in content:
            doc.metadata.update({
                "source": pdf_file.name,
                "category": pdf_file.parent.name
            })
            all_docs.append(doc)
    print(f"共加载{len(all_docs)}页pdf")
    return all_docs

# 分块
def split_documents(docs):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    chunks = text_splitter.split_documents(docs)
    print(f"已经分{len(chunks)}块")
    return chunks


# 灌库
def create_vectorstore():
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=DB_PERSIST_DIRECTORY,
    )
    return vector_store

if __name__ == '__main__':
    # 加载文档
    documents = load_document(FILE_PATH)
    # 分块
    all_chunk = split_documents(documents)
    # 将分块的数据灌入向量数据库
    vectorstore = create_vectorstore().add_documents(all_chunk)

