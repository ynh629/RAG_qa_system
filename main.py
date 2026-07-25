from dotenv import load_dotenv 
from openai import OpenAI
import os
from token_utils import trim_message
load_dotenv()
client=OpenAI(
    api_key=os.getenv("qwen_api_key"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
messages=[{"role":"system","content":"你是个客服助手"}]
while True:
    userinput=input("you:")
    if userinput.lower()=="exit":
        break
    messages.append({"role":"user","content":userinput})
    messages=trim_message(messages,max_tokens=3000,model="qwen-plus")
    stream=client.chat.completions.create(
        model="qwen-plus",
        messages=messages,
        stream=True
    )
    full_reply=""
    for chunk in stream:
        if chunk.choices[0].delta.content:
            text=chunk.choices[0].delta.content
            print(text,end="",flush=True)
            full_reply+=text
    print()
    messages.append({"role":"assistant","content":full_reply})
