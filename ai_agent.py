import os
from dotenv import load_dotenv
from openai import OpenAI

# 1. 打开保险箱
load_dotenv()

# 2. 造机器人的模具
class ChatAgent:
    # 注意：括号里只有 self 了，不要 api_key 了
    def __init__(self):
        print("🤖 正在组装智能体...")
        
        # 机器人自己去保险箱里拿钥匙
        api_key = os.getenv("DEEPSEEK_API_KEY")
        
        # 如果钥匙拿错了或者没找到，直接报警
        if not api_key:
            raise ValueError("🚨 找不到 API Key！请检查你的 .env 文件配置。")
            
        # 把大脑装上
        self.client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
        self.history = []

    def chat(self, user_text):
        self.history.append({"role": "user", "content": user_text})
        response = self.client.chat.completions.create(
            model="deepseek-chat",
            messages=self.history
        )
        ai_reply = response.choices[0].message.content
        self.history.append({"role": "assistant", "content": ai_reply})
        return ai_reply

# ================= 测试运行 =================

# 3. 造一个机器人（括号是空的，因为它自己会找钥匙）
my_robot = ChatAgent()

print("对话开始！(输入 '退出' 结束)")

# 4. 唤醒机器人干活
while True:
    user_input = input("\n你: ")
    if user_input == "退出":
        break
    print("AI 正在思考...")
    reply = my_robot.chat(user_input) 
    print("AI:", reply)