import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

class RAGAgent:
    def __init__(self, text_file_path):
        print("🤖 正在组装带读书功能的智能体...")
        api_key = os.getenv("DEEPSEEK_API_KEY")
        self.client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
        
        # 1. 读书：把文本文件读到内存里，并按“换行”切分成一段一段的
        with open(text_file_path, "r", encoding="utf-8") as f:
            full_text = f.read()
            
        # 用两行换行符切分，变成一个个段落（这就相当于书店把书拆成了一页一页）
        self.chunks = [p for p in full_text.split("\n\n") if p.strip() != ""]
        print(f"📚 已经将你的文档切分成了 {len(self.chunks)} 个片段。")

    def retrieve(self, question):
        # 2. 检索：最原始的暴力查找。看看哪个段落包含了用户问题里的字
        # 真实的 RAG 这里用的是“语义相似度（向量）”，我们第一节课先用最简单的“关键词”匹配
        relevant_chunks = []
        for chunk in self.chunks:
            # 如果用户问的词在这个段落里出现了（这里为了简单，直接硬匹配）
            if any(word in chunk for word in question.split()):
                relevant_chunks.append(chunk)
                
        # 就算没匹配到，也把整本书的前两段给它看看（防止它什么都不知道）
        if not relevant_chunks:
            relevant_chunks = self.chunks[:2] 
            
        return "\n---\n".join(relevant_chunks)

    def chat(self, question):
        # 3. 增强与生成：先把资料找出来，然后塞给大模型
        context = self.retrieve(question)
        
        # 给 AI 定个规矩，让它只看我们提供的资料回答
        prompt = f"""你是一个知识渊博的助手。请严格根据下面提供的参考资料，回答用户的问题。
如果参考资料中找不到答案，请直接说“根据提供的资料，我不知道”，千万不要自己瞎编。

【参考资料】：
{context}

【用户问题】：{question}
"""
        response = self.client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content

# ================= 测试运行 =================
# 确保你有一个 knowledge.txt 并且里面有内容
agent = RAGAgent("knowledge.txt")
print("对话开始！(输入 '退出' 结束)")

while True:
    user_input = input("\n你: ")
    if user_input == "退出":
        break
    print("AI 正在查阅资料并思考...")
    reply = agent.chat(user_input)
    print("AI:", reply)