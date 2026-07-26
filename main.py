from dotenv import load_dotenv 
from openai import OpenAI
import os
from token_utils import trim_message
from conversation import Conversation
load_dotenv()
client=OpenAI(
    api_key=os.getenv("qwen_api_key"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
conv=Conversation(
    system_prompt="你是个客服助手",
    max_tokens=3000,
    model="qwen-plus",
    persist_path="chat_history.json"
)
while True:
    userinput=input("you:")
    if userinput.lower()=="exit":
        break
    if userinput.lower()=="/reset":
        conv.reset()
        print("已重置对话")
        continue
    conv.add_user_message(userinput)
    messages=conv.get_messages()
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
    conv.add_assistant_message(full_reply)
    conv.save()
