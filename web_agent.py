import os
from dotenv import load_dotenv
from openai import OpenAI
import json
import streamlit as st

load_dotenv()

def calculate(expression):
    try:
        return str(eval(expression))
    except:
        return "计算错误"

class ToolAgent:
    def __init__(self):
        api_key = os.getenv("DEEPSEEK_API_KEY")
        self.client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
        
        self.system_prompt = {"role": "system", "content": "你是一个幽默风趣的智能助手，喜欢用简短的话回答问题。"}
        
        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "calculate",
                    "description": "当遇到数学计算题时，使用此工具",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "expression": {"type": "string", "description": "数学表达式"}
                        },
                        "required": ["expression"]
                    }
                }
            }
        ]

    def chat(self, messages):
        # 1. 把系统人设加上，再把网页上的历史记录拼在后面
        full_messages = [self.system_prompt] + messages
        
        response = self.client.chat.completions.create(
            model="deepseek-chat",
            messages=full_messages,
            tools=self.tools
        )
        message = response.choices[0].message
        
        if message.tool_calls:
            tool_call = message.tool_calls[0]
            function_name = tool_call.function.name
            arguments = json.loads(tool_call.function.arguments)
            
            if function_name == "calculate":
                result = calculate(arguments["expression"])
                
                # 🔥 核心修复：使用 model_dump() 把大模型对象转成纯字典格式
                full_messages.append(message.model_dump()) 
                full_messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": result})
                
                second_response = self.client.chat.completions.create(
                    model="deepseek-chat",
                    messages=full_messages
                )
                return second_response.choices[0].message.content
        else:
            return message.content

# ================= 网页界面部分 =================
st.title("🤖 我的专属 AI 助手")

# 2. 为了防止之前的错误记忆卡在网页里，加一个“清空记忆”的按钮
if st.button("🗑️ 清空聊天记录"):
    st.session_state.messages = []
    st.rerun() # 刷新页面

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("请问点什么？比如：算一下 99 * 88"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    @st.cache_resource
    def get_agent():
        return ToolAgent()
    agent = get_agent()

    with st.chat_message("assistant"):
        with st.spinner("AI 正在思考中..."):
            reply = agent.chat(st.session_state.messages)
            st.markdown(reply)
    
    st.session_state.messages.append({"role": "assistant", "content": reply})