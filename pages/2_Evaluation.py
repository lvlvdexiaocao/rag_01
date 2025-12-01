import streamlit as st
import pandas as pd
# from ragas.embeddings import BaseRagasEmbeddings
# from ragas.llms import BaseRagasLLM
# from zhipuai import ZhipuAI


st.set_page_config(
    page_title="RAGAS 评估结果可视化应用",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ====== 侧边栏 ======
with st.sidebar:
    st.header("🔑 API 与模型选择")

    api_key = st.text_input("请输入 OpenAI API Key", type="password")

    model = st.selectbox(
        "选择模型",
        ["glm-4.6"],
        index=0
    )
    st.markdown("---")

    st.subheader("上传评估结果文件 (Excel/CSV)")
    uploaded_file = st.file_uploader(
        "Drag and drop file here\nLimit 200MB per file • XLSX, CSV",
        type=["xlsx", "csv"],
        accept_multiple_files=False
    )

    if uploaded_file:
        st.info(f"📁 {uploaded_file.name} ({uploaded_file.size // 1024} KB)")

# ====== 主页面 ======
st.title("📊 RAGAS 评估结果可视化应用 (带授权)")

# 读取数据（仅用于展示，不处理）
df = None
row_count = 0
if uploaded_file:
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        row_count = len(df)
    except Exception as e:
        st.error(f"❌ 文件读取失败：{e}")

st.write(f"已读取数据行数: **{row_count}**")

# ====== 原始数据表 ======
st.subheader("📁 原始数据表")

# 列说明（和视频一致）
with st.expander("查看说明"):
    col_names = ["question", "contexts", "answer", "ground_truth"]
    chinese_names = ["原始问题", "上下文", "生成回答", "标准答案"]
    metrics_desc = [
        "Answer Relevance; Context Precision",
        "Faithfulness; Context Precision; Context Recall",
        "Faithfulness; Answer Relevance; Answer Semantic Similarity; Answer Correctness",
        "Context Recall; Answer Semantic Similarity; Answer Correctness"
    ]

    cols = st.columns(len(col_names))
    for i, col in enumerate(cols):
        with col:
            st.markdown(f"**{col_names[i]}**")
            st.caption(chinese_names[i])
            st.caption(metrics_desc[i])

with st.expander("查看传入文件"):
    if df is not None:
        # 只显示前 5 行（可调整）
        display_rows = min(5, len(df))

        st.subheader(f"📄 预览前 {display_rows} 条数据")

        for idx in range(display_rows):
            row = df.iloc[idx]

            # 创建四列
            cols = st.columns(4)

            # question
            with cols[0]:
                st.markdown("**❓ 问题**")
                st.markdown(f"<div style='font-size: 0.9em; line-height: 1.6;'>{row.get('question', '')}</div>", unsafe_allow_html=True)

            # contexts
            with cols[1]:
                st.markdown("**📚 上下文**")
                ctx_text = row.get('contexts', '')
                if isinstance(ctx_text, list):
                    ctx_text = "\n\n".join(ctx_text)
                st.markdown(f"<div style='font-size: 0.9em; line-height: 1.6;'>{ctx_text}</div>", unsafe_allow_html=True)

            # answer
            with cols[2]:
                st.markdown("**💬 回答**")
                ans_text = row.get('answer', '')
                st.markdown(f"<div style='font-size: 0.9em; line-height: 1.6;'>{ans_text}</div>", unsafe_allow_html=True)

            # ground_truth
            with cols[3]:
                st.markdown("**✅ 标准答案**")
                gt_text = row.get('ground_truth', '')
                st.markdown(f"<div style='font-size: 0.9em; line-height: 1.6;'>{gt_text}</div>", unsafe_allow_html=True)

            # 分隔线
            st.markdown("---")

    else:
        st.info("请上传 CSV 或 Excel 文件以预览数据。")


# ====== RAGAS 自动评估区域 ======
st.subheader("🤖 RAGAS 自动评估")

start_eval = st.button("🚀 开始RAGAS评估", type="primary", key="start_eval_btn")
gen_report = st.checkbox("📄 生成LLM文本报告", key="gen_report_checkbox")

if "ragas_result" not in st.session_state:
    st.session_state.ragas_result = None

if start_eval:
    if df is None:
        st.error("❌ 请先上传有效的测试数据文件。")
    elif not api_key:
        st.error("❌ 请在左侧边栏输入 OpenAI API Key。")
    else:
        try:
            with st.spinner("⏳ 正在运行 RAGAS 评估（可能需要 1-2 分钟）..."):
                # 调用你提供的函数
                result_df = run_ragas_evaluation(df, model_name=model, api_key=api_key)
                st.session_state.ragas_result = result_df
        except Exception as e:
            st.error(f"❌ 评估出错：{e}")
            st.session_state.ragas_result = None

# 显示评估结果（如果存在）
if st.session_state.ragas_result is not None:
    result_df = st.session_state.ragas_result

    # 显示平均分
    metric_cols = [col for col in result_df.columns if col not in ['question', 'answer', 'contexts', 'ground_truth']]
    if metric_cols:
        avg_scores = result_df[metric_cols].mean()
        st.subheader("📈 平均指标得分")
        cols = st.columns(len(metric_cols))
        for i, col in enumerate(metric_cols):
            cols[i].metric(col, f"{avg_scores[col]:.3f}")

    # 显示完整结果表格
    st.subheader("📋 详细评估结果")
    st.dataframe(result_df, width="stretch", height=500)
    from datetime import datetime

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = f"ragas_result_{timestamp}.csv"
    # ====== 新增：下载按钮 ======
    st.subheader("💾 导出评估结果")
    csv = result_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="⬇️ 下载 CSV 文件",
        data=csv,
        file_name=file_name,
        mime="text/csv",
        key="download-csv"
    )
    # 可选：生成报告（留接口）
    if gen_report:
        st.info("📝 LLM 文本报告功能待实现（可调用 LLM 总结指标）")
else:
    st.info("请点击上方按钮开始RAGAS评估。")


