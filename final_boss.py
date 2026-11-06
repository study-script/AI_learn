import os
import json
import requests
import streamlit as st
import math
from dotenv import load_dotenv
from openai import OpenAI
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from pypdf import PdfReader

load_dotenv()

# ================= 页面设置 =================
st.set_page_config(page_title="全能超级助理", page_icon="🤖")
st.title("🤖 全能超级助理")
st.caption("支持：聊天、计算、联网搜索、文档阅读")

# ================= 工具定义 =================
def calculate(expression):
    try:
        # 定义一个安全的字典，不让 AI 乱搞
        safe_dict = {"math": math, "__builtins__": None}
        result = eval(expression, safe_dict)
        return str(result)
    except Exception as e:
        return f"计算错误: 表达式写错了。错误信息: {e}"

def search_web(query):
    try:
        api_key = os.getenv("BOCHA_API_KEY")
        url = "https://api.bochaai.com/v1/web-search"
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        payload = {"query": query, "summary": True, "count": 3}
        response = requests.post(url, headers=headers, json=payload, timeout=10)
        data = response.json()
        results = data.get("data", {}).get("webPages", {}).get("value", [])
        formatted = ""
        for r in results:
            formatted += f"标题: {r.get('name')}\n摘要: {r.get('summary')}\n\n"
        return formatted if formatted else "未找到相关结果。"
    except Exception as e:
        return f"搜索失败: {e}"

# ================= 状态管理 =================
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "current_file" not in st.session_state:
    st.session_state.current_file = None

# ================= 侧边栏：文档上传 =================
with st.sidebar:
    st.header("📂 个人知识库")
    uploaded_file = st.file_uploader("上传后，助理将优先基于文档回答", type=["pdf", "txt"])
    
    if uploaded_file:
        if st.session_state.current_file != uploaded_file.name:
            st.session_state.vectorstore = None
            st.session_state.messages = [] 
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

                # 🔥 优化版切片：增大 overlap，防止上下文断裂
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=400, chunk_overlap=150,
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
    
    if st.button("🗑️ 移除当前文档"):
        st.session_state.vectorstore = None
        st.session_state.current_file = None
        st.session_state.messages = []
        st.rerun()

# ================= 展示历史聊天 =================
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ================= 核心大脑（Agent Loop） =================
def agent_run(user_input):
    # 准备工具说明书
    tools = [
        {"type": "function", "function": {"name": "calculate", "description": "数学计算", "parameters": {"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]}}},
        {"type": "function", "function": {"name": "search_web", "description": "实时联网查询", "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
        {"type": "function", "function": {"name": "search_local_docs", "description": "查询本地上传的文档", "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}}
    ]
    
    # 加上系统提示词和本次输入
    messages = [{"role": "system", "content": "你是一个全能私人助理。"}] + st.session_state.messages
    
    client = OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com")

    # 开始最多 5 轮思考
    for step in range(5):
        
        # ⚠️ 注意看下面这行，system_prompt 的缩进要和 for 循环对齐
        system_prompt = """
        你是一个全能私人助理。
        如果用户的问题涉及任何“故事”、“文档”、“小说”，你必须优先调用 `search_local_docs` 工具！
        
        【计算原则】：
        1. 遇到非常复杂的纯数字公式（如房贷、复利），请直接写成 Python 语法传给 `calculate` 工具。
        2. 遇到带有单位（如斤、厘米、公里）的换算，或者简单的常识性数学题（如计算 BMI、消耗多少卡路里），请你自己直接在脑海里算出结果，不要调用 `calculate` 工具，以免报错！
        """
        
        # 把提示词插到最前面
        current_messages = [{"role": "system", "content": system_prompt}] + messages
        
        with st.spinner(f"AI 思考中 (第 {step+1} 轮)..."):
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=current_messages,
                tools=tools
            )
            message = response.choices[0].message
            
            # 1. 如果不需要工具，直接返回最终回答并结束函数
            if not message.tool_calls:
                return message.content
                
            # 2. 如果需要调工具，把 AI 的思考加入消息记录
            messages.append(message.model_dump())
            
            # 3. 遍历并执行所有工具
            for tool_call in message.tool_calls:
                func_name = tool_call.function.name
                args = json.loads(tool_call.function.arguments)
                
                # 在网页上显示工具调用状态
                st.info(f"🛠️ 正在使用工具: {func_name}...")
                
                if func_name == "calculate":
                    result = calculate(args["expression"])
                elif func_name == "search_web":
                    result = search_web(args["query"])
                elif func_name == "search_local_docs":
                    if st.session_state.vectorstore:
                        retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 4})
                        docs = retriever.invoke(args["query"])
                        result = "\n\n".join([d.page_content for d in docs])
                    else:
                        result = "未上传本地文档。"
                
                # 将工具执行结果喂回给大模型
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result
                })
    
    # 4. 兜底提示
    return "抱歉，这个任务太复杂了，我尝试了 5 次都没成功。请尝试把问题拆分成几个小步骤。"

# ================= 聊天输入框 =================
if prompt := st.chat_input("请问点什么？(聊天、算数、查天气、问文档都可以)"):
    # 记录并显示用户输入
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 调用大脑
    with st.chat_message("assistant"):
        reply = agent_run(prompt)
        st.markdown(reply)
        st.session_state.messages.append({"role": "assistant", "content": reply})