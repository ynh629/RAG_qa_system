from dotenv import load_dotenv 
from openai import OpenAI
import os
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
    response=client.chat.completions.create(model="qwen-plus",messages=messages)
    assistant_reply=response.choices[0].message.content
    print(assistant_reply)
    messages.append({"role":"assistant","content":assistant_reply})
