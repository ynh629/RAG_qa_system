import os
import json
from openai import OpenAI
from dotenv import load_dotenv
import datetime
load_dotenv()
client=OpenAI(
    api_key=os.getenv("qwen_api_key"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
tools=[{
    "type":"function",
    "function":{
        "name":"get_time",
        "description":"当用户明确想要获取时间时才使用",
        "parameters":{
            "type":"object",
            "properties":{},
            "required":[]
        }
    }
}]
def get_time():
        now=datetime.datetime.now()
        return now.strftime("%Y-%m-%d %H:%M:%S")
result=get_time()
def ask_with_tools(user_message,model="qwen_plus",temperature=0.1):
  response=client.chat.completions.create(
    model="qwen-plus",
    messages=[
        {"role":"user","content":user_message},
    ],
    tools=tools,
    temperature=0.1,
    tool_choice="auto"
   )
  print(response.choices[0].message.tool_calls)
  msg=response.choices[0].message
  if msg.tool_calls:
    tool_call=msg.tool_calls[0]
    if isinstance(tool_call,dict):
        func_name=tool_call['function']['name']
        arge=json.loads(tool_call['function']['arguments'])
    else:
        func_name = tool_call.function.name
        args = json.loads(tool_call.function.arguments)
  else:
    return msg.content
  print(f"模型调用函数{func_name},参数{args}")
  print(f"函数执行结果{result}")
  messages = [
        {"role": "user", "content": user_message},
        msg, 
        {
            "role": "tool",
            "tool_call_id": tool_call.id if hasattr(tool_call, 'id') else tool_call.get('id'),
            "content": result
        }
    ]
  final_response = client.chat.completions.create(
        model="qwen-plus",
        messages=messages,
        temperature=0.1
    )
  return final_response.choices[0].message.content
if __name__=="__main__":
    answer=ask_with_tools("现在北京时间")
    print(answer)


