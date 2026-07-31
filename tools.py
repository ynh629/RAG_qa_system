from enterprise_tools import summarize_text, classify_sentiment, text_to_sql

# 测试摘要
print(summarize_text("人工智能是研究...", max_length=50))

# 测试情感分类
sent = classify_sentiment("这个产品很棒")
print(sent.sentiment, sent.confidence)

# 测试 SQL 生成
schema = """
CREATE TABLE students (id INTEGER PRIMARY KEY, name TEXT, score REAL);
"""
sql = text_to_sql("查询所有学生的姓名和分数", schema)
print(sql)
  