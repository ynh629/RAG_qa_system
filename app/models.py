# models.py
from sqlalchemy import Column, Integer, String, Text, Float, DateTime, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func
from datetime import datetime

# 创建 Base 类，所有模型都要继承它
Base = declarative_base()

class ChatHistory(Base):
    """
    对话历史表：保存用户每次提问和系统回答。
    """
    __tablename__ = "chat_history"

    # 主键，自增整数
    id = Column(Integer, primary_key=True, autoincrement=True)
    # 用户标识（可以是用户名、UUID 或匿名）
    user_id = Column(String(255), nullable=False, default="anonymous")
    # 会话 ID（用于区分不同对话轮次，支持多轮上下文）
    session_id = Column(String(255), nullable=False)
    # 用户原始问题
    query = Column(Text, nullable=False)
    # 系统最终答案
    answer = Column(Text, nullable=False)
    # 检索到的上下文（JSON 字符串或文本，便于后续分析）
    retrieved_contexts = Column(Text, nullable=True)
    # 消耗的 token 数（可选）
    total_tokens = Column(Integer, nullable=True)
    # 创建时间
    created_at = Column(DateTime, server_default=func.now())

class UserFeedback(Base):
    """
    用户反馈表：记录用户对某个回答的赞/踩或评分。
    """
    __tablename__ = "user_feedback"

    # 主键
    id = Column(Integer, primary_key=True, autoincrement=True)
    # 关联对话历史表的主键（外键）
    chat_id = Column(Integer, nullable=False)
    # 用户标识
    user_id = Column(String(255), nullable=False)
    # 反馈类型：'up' 表示赞，'down' 表示踩
    feedback_type = Column(String(10), nullable=False)
    # 可选的文字评论
    comment = Column(Text, nullable=True)
    # 评分（1~5，可选）
    rating = Column(Integer, nullable=True)
    # 反馈时间
    created_at = Column(DateTime, server_default=func.now())