import json
import requests
import os
from dotenv import load_dotenv
import csv
import pandas as pd
import time
import json
load_dotenv()
api_key=os.getenv("api_key")
if not api_key:
    raise ValueError("请在 .env 文件中设置api_key")
def ask_deepseek(prompt,model="deepseek-v4-flash",temperature=0.7):
    url="https://api.deepseek.com/v1/chat/completions"
    headers= {
        "Content-Type":"application/json",
        "Authorization": f"Bearer {api_key}"
    }
    payload={
        "model":model,
        "messages":[{"role":"user","content": prompt}],
        "temperature":temperature
    }
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]
    except requests.exceptions.RequestException as e:
        print(f"请求失败: {e}")
        return None
    except (KeyError, IndexError, json.JSONDecodeError) as e:
        print(f"解析响应出错: {e}")
        return None
if __name__=="__main__":
    question = [
        ("编程", "一句话说明用 Python 读取 CSV 文件的代码示例"),
        ("编程", "一句话说明介绍 Python 中的列表推导式"),
        ("生活", "一句话说明如何保持健康的作息时间？"),
        ("生活", "一句话推荐一道简单易做的家常菜"),
        ("哲学", "一句话说明人生的意义是什么？"),
        ("哲学", "一句话说明为什么说‘存在即合理’？"),
    ]
    records=[]
    for category,q in question:
        print(f"正在询问[{category}]:{q}")
        answer=ask_deepseek(q)
        records.append({
             "类别": category,
             "问题": q,
             "AI回答": answer if answer else "获取失败",
             "回答长度": len(answer) if answer else 0,
             "模型": "deepseek-chat"
        })
        time.sleep(20)
csv_file="deepseek_responses.csv"
with open(csv_file,"w",newline="",encoding="utf-8-sig")as f:
    writer=csv.writer(f)
    writer.writerow(["类别", "问题", "AI回答", "回答长度", "模型"])
    for r in records:
        writer.writerow([r["类别"], r["问题"], r["AI回答"], r["回答长度"], r["模型"]])
print(f"数据已保存至{csv_file}")
df = pd.read_csv(csv_file, encoding="utf-8-sig")
print("\n===== 数据预览 =====")
print(df.head())
print("\n===== 统计摘要 =====")
avg_len = df["回答长度"].mean()
print(f"平均回答长度（字符数）: {avg_len:.1f}")
max_row = df.loc[df["回答长度"].idxmax()]
print(f"最长回答: [{max_row['类别']}] {max_row['问题']} ({max_row['回答长度']} 字符)")
min_row = df.loc[df["回答长度"].idxmin()]
print(f"最短回答: [{min_row['类别']}] {min_row['问题']} ({min_row['回答长度']} 字符)")
print("\n各类别问题数量:")
print(df["类别"].value_counts().to_string())
print("\n各类别平均回答长度:")
print(df.groupby("类别")["回答长度"].mean().to_string())
python_mentions = df[df["AI回答"].str.contains("Python", na=False)].shape[0]
print(f"\n回答中提到 'Python' 的次数: {python_mentions}")

        