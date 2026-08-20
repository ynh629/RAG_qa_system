import os
from dotenv import load_dotenv
from openai import OpenAI
load_dotenv()
client=OpenAI(
    api_key=os.getenv("qwen_api_key"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
user_input=input("你：")
messages=[{"role":"system","content":"你是友好的客服助手"},
         { "role":"user","content":user_input}]
stream=client.chat.completions.create(
    model="qwen-plus",
    messages=messages,
    stream=True
)
for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content,end="",flush=True)