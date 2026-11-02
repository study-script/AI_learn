import os
import streamlit as st
import json
from dotenv import load_dotenv
from openai import OpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from pypdf import PdfReader

load_dotenv()

st.set_page_config(page_title="全能 AI 助手", page_icon="🧠")
st.title("🧠 全能 AI 助手 (聊天 + 文档阅读)")
st.caption("直接聊天，或者上传文档让我帮你读书！")

# ================= 初始化状态 =================
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "current_file" not in st.session_state:
    st.session_state.current_file = None

# ================= 侧边栏：文档上传 =================
with st.sidebar:
    st.header("📂 文档上传区")
    uploaded_file = st.file_uploader("上传后，AI 将优先基于文档回答", type=["pdf", "txt"])
    
    if uploaded_file:
        if st.session_state.current_file != uploaded_file.name:
            st.session_state.vectorstore = None
            st.session_state.messages = [] # 换文档了，清空旧聊天
            st.session_state.current_file = uploaded_file.name
            
        if st.session_state.vectorstore is None:
            with st.spinner("正在解析文档..."):
                text = ""
                if uploaded_file.type == "application/pdf":
                    reader = PdfReader(uploaded_file)
                    for page in reader.pages:
                        text += page.extract_text() + "\n"
                else:
                    text = uploaded_file.read().decode("utf-8", errors="ignore")

                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=500, chunk_overlap=50,
                    separators=["\n\n", "\n", "。", "！", "？", "，", " ", ""]
                )
                chunks = text_splitter.split_text(text)
                docs = [Document(page_content=chunk) for chunk in chunks]

                embeddings = OpenAIEmbeddings(
                    api_key=os.getenv("SILICONFLOW_API_KEY"),
                    base_url="https://api.siliconflow.cn/v1",
                    model="BAAI/bge-m3"
                )
                st.session_state.vectorstore = Chroma.from_documents(documents=docs, embedding=embeddings)
                st.success(f"✅ 解析完成！共 {len(docs)} 个片段。")
    
    # 清除文档按钮
    if st.button("🗑️ 移除当前文档 (变回普通聊天)"):
        st.session_state.vectorstore = None
        st.session_state.current_file = None
        st.session_state.messages = []
        st.rerun()

# ================= 核心对话逻辑 =================
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("请问点什么？可以直接聊天，也可以问文档..."):
    # 1. 记录用户问题
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. AI 思考并回答
    with st.chat_message("assistant"):
        with st.spinner("AI 正在思考..."):
            
            client = OpenAI(
                api_key=os.getenv("DEEPSEEK_API_KEY"), 
                base_url="https://api.deepseek.com"
            )

            # 模式 A：有文档，走 RAG 模式
            if st.session_state.vectorstore is not None:
                retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 3})
                docs = retriever.invoke(prompt)
                context = "\n\n".join([d.page_content for d in docs])
                
                sys_prompt = f"""
                你是一个专业的文档分析助手。
                【参考资料】：{context}
                【用户问题】：{prompt}
                请严格根据参考资料回答。如果资料中确实没有答案，请说“文档中未提及，但根据我的常识...”，然后结合你自己的知识回答。
                """
                response = client.chat.completions.create(
                    model="deepseek-chat",
                    messages=[{"role": "user", "content": sys_prompt}]
                )
                reply = response.choices[0].message.content

            # 模式 B：没有文档，走普通聊天模式（甚至可以带上之前的计算器工具）
            else:
                # 这里为了简单，我们直接做纯聊天，去掉了工具，保证体验流畅
                sys_prompt = "你是一个幽默风趣的智能助手。"
                # 把我们网页上记录的历史记录带上，保证记忆
                messages = [{"role": "system", "content": sys_prompt}] + st.session_state.messages
                
                response = client.chat.completions.create(
                    model="deepseek-chat",
                    messages=messages
                )
                reply = response.choices[0].message.content

            # 输出结果并记录
            st.markdown(reply)
            st.session_state.messages.append({"role": "assistant", "content": reply})