import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

load_dotenv()

# ================= 1. 配置两大核心引擎 =================
embeddings = OpenAIEmbeddings(
    api_key=os.getenv("SILICONFLOW_API_KEY"),
    base_url="https://api.siliconflow.cn/v1",
    model="BAAI/bge-m3"
)

llm = ChatOpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
    model="deepseek-chat"
)

# ================= 2. 读取文档并切分 =================
print("📚 正在阅读你的文档...")
with open("knowledge.txt", "r", encoding="utf-8") as f:
    text = f.read()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,     
    chunk_overlap=50,
    separators=["\n\n", "\n", "。", "！", "？", "，", " ", ""]
)
chunks = text_splitter.split_text(text)
docs = [Document(page_content=chunk) for chunk in chunks]

# 班主任加的透视眼：看看文档被切成了什么鬼样子！
print("\n=== 切分出来的所有片段 ===")
for i, doc in enumerate(docs):
    print(f"片段[{i}]: {doc.page_content}")
print("============================\n")

# ================= 3. 存入向量数据库 =================
print(f"🧠 正在将 {len(docs)} 个片段转化为高维向量并存入数据库...")
vectorstore = Chroma.from_documents(documents=docs, embedding=embeddings)
print("✅ 数据库构建完成！\n")

# ================= 4. 开始对话 =================
print("🤖 超级 RAG 智能体已启动！(输入 '退出' 结束)")

while True:
    query = input("\n你: ")
    if query == "退出":
        break
        
    print("🔍 正在进行语义搜索...")
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    retrieved_docs = retriever.invoke(query)
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])
    
    # 看看后台到底翻出了什么资料！
       # ... 前面的代码不变 ...
    
    print("🔍 正在把整本资料直接塞给 AI...")
    # 不要 retriever 了，直接把之前读到的完整 text 塞进去
    context = text 
    
    prompt: str = f"""
    你是一个极其严谨的图书管理员。
    请【逐字阅读】下面提供的参考资料。
    
    【参考资料】：
    {context}

    【用户问题】：{query}
    
    【回答规则】：
    1. 必须严格在参考资料里寻找答案。
    2. 如果资料中提到了某个人名被明确标注为“主角”，你必须直接说出这个名字。
    3. 如果资料里真的没写，你才可以说“我不知道”。
    """
    
    print("🤖 AI 正在思考...")
    response = llm.invoke(prompt)
    print("AI:", response.content)