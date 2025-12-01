# app/main.py
import streamlit as st

st.set_page_config(
    page_title="RAG 评估系统",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️  RAG 评估系统")
st.markdown("""
欢迎使用RAG 问答与评估系统。

- **RAG ChatBot**：输入问题，获取依据与答案  
- **评估中心**：上传测试集，运行 RAGAS 客观评估
""")