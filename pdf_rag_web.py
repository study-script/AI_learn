import os
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from pypdf import PdfReader

load_dotenv()

# 网页标题
st.set_page_config(page_title="我的智能文档助手", page_icon="📄")
st.title("📄 我的智能文档问答助手")
st.caption("上传一份 PDF 或 TXT，让 AI 帮你读书！")

# ================= 初始化 Session State =================
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "current_file" not in st.session_state:
    st.session_state.current_file = None

# ================= 侧边栏：上传文档 =================
with st.sidebar:
    st.header("📂 文档上传区")
    uploaded_file = st.file_uploader("请上传 PDF 或 TXT 文件", type=["pdf", "txt"])
    
    # 如果上传了新文件，清空之前的记忆
    if uploaded_file:
        if st.session_state.current_file != uploaded_file.name:
            st.session_state.vectorstore = None
            st.session_state.messages = []
            st.session_state.current_file = uploaded_file.name
            
        if st.session_state.vectorstore is None:
            with st.spinner("正在努力阅读文档并建立索引..."):
                text = ""
                # 1. 提取文本
                if uploaded_file.type == "application/pdf":
                    reader = PdfReader(uploaded_file)
                    for page in reader.pages:
                        text += page.extract_text() + "\n"
                else:
                    text = uploaded_file.read().decode("utf-8", errors="ignore")

                # 2. 切片
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=500, chunk_overlap=50,
                    separators=["\n\n", "\n", "。", "！", "？", "，", " ", ""]
                )
                chunks = text_splitter.split_text(text)
                docs = [Document(page_content=chunk) for chunk in chunks]

                # 3. 存入向量数据库
                embeddings = OpenAIEmbeddings(
                    api_key=os.getenv("SILICONFLOW_API_KEY"),
                    base_url="https://api.siliconflow.cn/v1",
                    model="BAAI/bge-m3"
                )
                st.session_state.vectorstore = Chroma.from_documents(documents=docs, embedding=embeddings)
                st.success(f"✅ 文档解析完成！共切分为 {len(docs)} 个片段。")

# ================= 核心对话逻辑 =================
# 显示历史聊天记录
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 底部输入框
if prompt := st.chat_input("请基于上传的文档提问..."):
    if st.session_state.vectorstore is None:
        st.warning("⚠️ 请先在左侧上传文档！")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("AI 正在查阅资料思考中..."):
                retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 5})
                retrieved_docs = retriever.invoke(prompt)
                context = "\n\n".join([doc.page_content for doc in retrieved_docs])

                llm = ChatOpenAI(
                    api_key=os.getenv("DEEPSEEK_API_KEY"),
                    base_url="https://api.deepseek.com",
                    model="deepseek-chat"
                )
                
                sys_prompt = f"""
                你是一个专业的文档分析助手。请严格基于下面提供的参考资料回答用户的问题。
                如果参考资料中没有相关信息，请直接回答“文档中未提及”，不要编造。
                
                【参考资料】：
                {context}
                
                【用户问题】：{prompt}
                """
                
                response = llm.invoke(sys_prompt)
                st.markdown(response.content)
                st.session_state.messages.append({"role": "assistant", "content": response.content})