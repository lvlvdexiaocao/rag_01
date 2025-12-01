import streamlit as st

from rag_chain import get_response_from_rag


# 页面标题
st.title("RAG ChatBot")

# 初始化会话状态
if "messages" not in st.session_state:
    st.session_state.messages = []

# # 显示历史消息
# for message in st.session_state.messages:
#     with st.chat_message(message["role"]):
#         st.markdown(message["content"])

MODEL_NAMES_OLLAMA = ["qwen2.5:3b"]
MODEL_NAMES_GLM = ["glm-4.6", "glm-4.5-flash"]
MODEL_NAMES_DEEPSEEK = ["deepseek-chat"]

with st.sidebar:
    st.header("YIYIYIYIY")
    with st.popover("Settings",use_container_width=True):
        # allow_web_search = st.checkbox("Allow Web Search")
        model = st.selectbox("LLM to use", options=["Ollama", "GLM", "DEEPSEEK"])
        if model == "Ollama":
            selected_model = st.selectbox("Select Groq Model:", MODEL_NAMES_OLLAMA)
        elif model == "GLM":
            selected_model = st.selectbox("Select OpenAI Model:", MODEL_NAMES_GLM)
        elif model == "DEEPSEEK":
            selected_model = st.selectbox("Select OpenAI Model:", MODEL_NAMES_DEEPSEEK)

# 显示历史消息（包含可展开的检索依据）
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        # 如果是助手回复且有 sources，显示可展开的检索内容
        if msg["role"] == "assistant" and "sources" in msg:
            with st.expander("🔍 查看检索到的条款依据"):
                sources = msg["sources"]
                if sources:
                    for i, doc in enumerate(sources):
                        source_name = doc.metadata.get("source", "未知来源")
                        with st.expander(f"📄 片段 {i+1} | 来源：{source_name}"):
                            st.text(doc.page_content)
                else:
                    st.write("❌ 未检索到相关条款。")


# 用户输入
if prompt := st.chat_input("请输入您的保险问题，例如：“等待期内确诊乳腺原位癌能赔吗？”"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)


    # 调用 RAG，获取完整结果
    answer_text, ref_docs = get_response_from_rag(query=prompt, provider=model, llm_id=selected_model)


    # 保存助手消息（含 sources）
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer_text,
        "sources": ref_docs
    })
    # 保存并显示回答
    # 显示当前回答
    with st.chat_message("assistant"):
        st.markdown(answer_text)

    # 显示当前回答的检索依据（折叠）
    with st.expander("🔍 查看检索到的条款依据"):
        if ref_docs:
            for i, doc in enumerate(ref_docs):
                source_name = doc.metadata.get("source", "未知来源")
                with st.expander(f"📄 片段 {i+1} | 来源：{source_name}"):
                    st.text(doc.page_content)
        else:
            st.write("❌ 未检索到相关条款。")
