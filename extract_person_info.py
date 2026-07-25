#信息处理
import os
from dotenv import load_dotenv
from openai import OpenAI
load_dotenv()
import instructor
from pydantic import BaseModel,Field
from typing import List
client=instructor.from_openai(
    OpenAI(
        api_key=os.getenv("qwen_api_key"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    ),
    mode=instructor.Mode.JSON
)
class person(BaseModel):
    name: str = Field(..., description="姓名")
    age: int = Field(..., description="年龄，整数")
    city: str = Field(..., description="城市")
def extract_person_info(text: str) -> person:
    response = client.chat.completions.create(
        model="qwen-plus",
        response_model=person,
        messages=[
            {"role":"system","content":"You are a helpful assistant that extracts personal information from text."},
            {"role": "user", "content": text}
        ]
    )
    return response
if __name__ == "__main__":
    text = "我叫小明，25岁，来自北京。"
    person = extract_person_info(text)
    print(f"姓名: {person.name}, 年龄: {person.age}, 城市: {person.city}")