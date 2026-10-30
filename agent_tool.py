import os
from dotenv import load_dotenv
from openai import OpenAI
import json

load_dotenv()

# 1. 这是一个真实的 Python 函数（工具）
def calculate(expression):
    try:
        # 危险操作：eval可以计算字符串的数学公式，比如 "998 * 776"
        return str(eval(expression))
    except:
        return "计算错误"

class ToolAgent:
    def __init__(self):
        api_key = os.getenv("DEEPSEEK_API_KEY")
        self.client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
        
        # 给 AI 的“工具说明书”
        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "calculate",
                    "description": "当遇到数学计算题时，使用此工具",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "expression": {
                                "type": "string",
                                "description": "数学表达式，例如 '2 + 2' 或 '998 * 776'"
                            }
                        },
                        "required": ["expression"]
                    }
                }
            }
        ]

    def chat(self, user_text):
        print("🤔 AI 正在思考，是否需要调用工具...")
        # 发给 AI 问题，并附上工具说明书
        response = self.client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": user_text}],
            tools=self.tools
        )
        
        message = response.choices[0].message
        
        # 检查 AI 是不是想调用工具
        if message.tool_calls:
            tool_call = message.tool_calls[0]
            function_name = tool_call.function.name
            arguments = json.loads(tool_call.function.arguments)
            
            print(f"🛠️ AI 决定调用工具: {function_name}, 参数: {arguments}")
            
            # 如果 AI 要计算，我们就替它算
            if function_name == "calculate":
                result = calculate(arguments["expression"])
                print(f"👀 工具执行结果: {result}")
                
                # 把结果喂回给 AI，让它总结
                second_response = self.client.chat.completions.create(
                    model="deepseek-chat",
                    messages=[
                        {"role": "user", "content": user_text},
                        message,
                        {"role": "tool", "tool_call_id": tool_call.id, "content": result}
                    ]
                )
                return second_response.choices[0].message.content
        else:
            # AI 不需要用工具，直接回答
            return message.content

# 测试
agent = ToolAgent()
# ================= 测试运行 =================
agent = ToolAgent()

print("🤖 智能体已启动！(输入 '退出' 结束对话)")

while True:
    # 让程序停下来，听你说话
    user_input = input("\n你: ")
    
    if user_input == "退出":
        break
        
    # 把你说的话传给智能体
    reply = agent.chat(user_input)
    print("Agent:", reply)