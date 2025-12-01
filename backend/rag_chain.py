from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate,ChatPromptTemplate
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv
load_dotenv()
import config

# =========配置项=========
EMBEDDING_MODEL = config.EMBEDDING_MODEL
DB_PERSIST_DIRECTORY = config.DB_PERSIST_DIRECTORY
COLLECTION_NAME = config.COLLECTION_NAME
RETRIEVER_NUM = config.RETRIEVER_NUM


def get_response_from_rag(query, provider, llm_id):
    if provider == "Ollama":
        llm = ChatOllama(model=llm_id)
    elif provider == "GLM":
        llm = ChatOpenAI(model=llm_id,
                     api_key=os.environ.get("ZHIPUAI_API_KEY"),
                     base_url=os.environ.get("ZHIPUAI_BASE_URL"),
                        temperature=0,
                        )
    elif provider == "DEEPSEEK":
        llm = ChatOpenAI(model=llm_id)


    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=DB_PERSIST_DIRECTORY,
        embedding_function=embeddings,
    )
    retriever = vector_store.as_retriever(search_kwargs={"k": RETRIEVER_NUM})
    # ✅ 步骤 1: 手动检索
    retrieved_docs = retriever.invoke(query)
    # ✅ 步骤 2: 构造 context 字符串（供 LLM 使用）
    context_text = "\n\n".join([doc.page_content for doc in retrieved_docs])

    prompt = PromptTemplate.from_template(
        """你是一个保险顾问，请根据以下保险条款回答用户问题。如果条款未提及，请回答“未找到相关信息”。

    条款内容：
    {context}

    用户问题：
    {question}

    回答："""
    )

    chain = prompt| llm| StrOutputParser()

    response = chain.invoke({"context": context_text, "question": query})


    # ✅ 步骤 4: 返回回答 + 检索到的原始文档（含元数据）
    return response, retrieved_docs

if __name__ == '__main__':
    query = "介绍你自己"
    answer_text, ref_docs = get_response_from_rag(query=query, provider="Ollama", llm_id="qwen2.5:3b")

    print("回答：", answer_text)
    print("\n参考文档：")
    for i, doc in enumerate(ref_docs):
        print(f"\n文档{i + 1}：")
        print("内容：", doc.page_content[:50] + "...")  # 截断显示
        print("来源：", doc.metadata["source"])