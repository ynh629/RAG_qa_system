# schemas.py
"""API 请求/响应 Pydantic 模型。"""
from typing import Optional, List, Literal

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """聊天请求体"""
    query: str = Field(..., min_length=1, max_length=1000)
    top_k: int = Field(5, ge=1, le=10)
    include_sources: bool = True
    user_id: Optional[str] = None       # 用户标识（可选）
    session_id: Optional[str] = None    # 会话 ID（可选，用于多轮上下文）


class Source(BaseModel):
    title_path: str
    text_snippet: str
    rerank_score: float


class ChatResponse(BaseModel):
    answer: str
    sources: Optional[List[Source]] = None
    retrieved_contexts: Optional[List[str]] = None
    chat_id: Optional[int] = None       # 保存历史后返回数据库ID


class FeedbackRequest(BaseModel):
    """反馈请求体"""
    chat_id: int
    feedback_type: Literal["up", "down"]  # 只允许赞 / 踩
    comment: Optional[str] = Field(None, max_length=2000)
    rating: Optional[int] = Field(None, ge=1, le=5)  # 1~5 星，可选
