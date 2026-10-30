from openai import OpenAI

client = OpenAI(
    api_key="sk-df06aa3fdff4428f8a5fb5e2519f0b0d",
    base_url="https://api.deepseek.com"
)

print("🤖 AI 聊天机器人已启动！(输入 '退出' 结束对话)")

# 1. 加上一个永远为真的循环
while True:
    # 2. 让用户输入问题
    user_input = input("\n你: ")
    
    # 3. 如果用户输入“退出”，就打破循环结束程序
    if user_input == "退出":
        print("再见！")
        break
    
    # 4. 把用户输入的话发给 AI
    print("AI 正在思考...")
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "user", "content": user_input} # 这里填刚才接收到的变量
        ]
    )
    
    # 5. 打印 AI 的回答
    print("AI:", response.choices[0].message.content)