#企业级工具
import os
import instructor
from pydantic import BaseModel,Field
import sqlite3
from dotenv import load_dotenv
from openai import OpenAI
from typing import List, Optional
load_dotenv()
base_client = OpenAI(
    api_key=os.getenv("qwen_api_key"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)
instructor_client = instructor.from_openai(base_client,mode=instructor.Mode.JSON)
class SentimentResult(BaseModel):
    sentiment: str = Field( description="情感倾向:正面，负面，中性")
    confidence: float = Field( description="情感倾向的置信度，0~1")
#------------------工具函数---------------------
def summarize_text(text:str,max_length:int=150) -> str:
    system_prompt = f"你是一个文本摘要专家，请将输入的文本进行简明扼要的总结，输出字符不超过{max_length}，确保保留核心信息。"
    response=base_client.chat.completions.create(
        model="qwen-plus",
        messages=[
            {"role":"system","content":system_prompt},
            {"role": "user", "content": text}
        ],
        temperature=0.2,
    )
    result = response.choices[0].message.content
    return result
def classify_sentiment(text: str) -> SentimentResult:
    system_prompt = "你是一个情感分析专家，请判断输入文本的情感倾向，并给出置信度评分。"
    result = instructor_client.chat.completions.create(
        model="qwen-plus",
        response_model=SentimentResult,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text}
        ],
        temperature=0,
    )
    return result
def polish_text(text: str,style:str="正式") -> str:
    system_prompt = f"你是一个文本润色专家，请对输入文本进行润色，使其符合{style}风格。"
    response = base_client.chat.completions.create(
        model="qwen-plus",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text}
        ],
        temperature=0.4,
    )
    result = response.choices[0].message.content
    return result
def text_to_sql(query: str, schema: str) -> str:
    conn = sqlite3.connect(":memory:")
    conn.executescript(schema)
    system_prompt = f"你是一个 SQL 生成器。只输出一条完整的 SQL 语句，不要包含任何解释、注释或 Markdown 标记。"
    max_retries = 3
    messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"数据库 Schema:\n{schema}\n\n问题：{query}\n请输出 SQL。"}
            ]
    for attempt in range(max_retries):
        response = base_client.chat.completions.create(
            model="qwen-plus",
            messages=messages,
            temperature=0,
        )
        sql_candidate = response.choices[0].message.content.strip()
        if sql_candidate.lower().startswith("```sql"):
            sql_candidate = sql_candidate[6:]
        if sql_candidate.lower().endswith("```"):
            sql_candidate = sql_candidate[:-3]
        sql_candidate = sql_candidate.strip(";") + ";"
        try:
            conn.execute(sql_candidate)
            return sql_candidate
        except Exception as e:
            error_msg = str(e)
            print(f"SQL尝试 {attempt + 1}失败 failed with error: {error_msg}.")
            messages.append({"role": "assistant", "content": sql_candidate})
            messages.append({"role": "user", "content": f"生成的 SQL 语句执行失败，错误信息: {error_msg}。请根据错误信息修改 SQL 语句并重新生成。"})
        raise Exception("SQL执行失败")
#---------------------测试用例--------------
if __name__ == "__main__":
    print("=" * 50)
    print("测试1：文本摘要")
    long_text = """
    人工智能（AI）是计算机科学的一个分支，它企图了解智能的实质，并生产出一种新的能以人类智能相似的方式做出反应的智能机器。
    该领域的研究包括机器人、语言识别、图像识别、自然语言处理和专家系统等。
    人工智能从诞生以来，理论和技术日益成熟，应用领域也不断扩大。
    可以设想，未来人工智能带来的科技产品，将会是人类智慧的“容器”。
    """
    print("原始文本:", long_text[:50] + "...")
    summary = summarize_text(long_text, max_length=80)
    print("摘要:", summary)
    print()

    print("=" * 50)
    print("测试2：情感分类")
    test_text = "虽然配送慢了点，但商品质量是真的好！"
    sentiment_result = classify_sentiment(test_text)
    print(f"文本: {test_text}")
    print(f"情感: {sentiment_result.sentiment}, 置信度: {sentiment_result.confidence:.2f}")
    print()

    print("=" * 50)
    print("测试3：文本润色")
    raw_text = "老板，明天我有事想请个假，行不？"
    polished = polish_text(raw_text, style="正式")
    print("原文:", raw_text)
    print("润色后:", polished)
    print()
    print("=" * 50)
    print("测试5：Text-to-SQL")
    # 变量：模拟的电商数据库 Schema
    test_schema = """
    CREATE TABLE products (
        id INTEGER PRIMARY KEY,
        name TEXT,
        price REAL,
        stock INTEGER
    );
    CREATE TABLE orders (
        id INTEGER PRIMARY KEY,
        product_id INTEGER,
        quantity INTEGER,
        order_date TEXT
    );
    """
     # 变量：自然语言查询
    test_query = "找出库存低于10件的商品名称和库存量"
    try:
        sql = text_to_sql(test_query, test_schema)
        print(f"问题: {test_query}")
        print(f"生成的 SQL: {sql}")
    except Exception as e:
        print("最终失败:", e)
    print()

    print("全部测试完成 ✅")
  