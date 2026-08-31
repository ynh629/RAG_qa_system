import os
import sys

# 确保项目根目录在 sys.path，使直接运行 python app/api.py 也能正常导入
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from contextlib import asynccontextmanager
from typing import Dict, List

from fastapi import FastAPI, HTTPException, Depends
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

# 导入 RAG 系统初始化函数
from app.qa_system import initialize_qa_system, QASystem
# 导入数据库模块
from app.database import init_db, get_db
from app.models import ChatHistory, UserFeedback
# Pydantic 请求/响应模型
from app.schemas import ChatRequest, ChatResponse, FeedbackRequest, Source
# 统一配置中心
from app.config import settings

# 全局变量
qa_system_instance = None

# 路径配置（统一由配置中心管理）
DATA_JSON = settings.DATA_JSON


# ---------- lifespan：初始化数据库和 RAG ----------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global qa_system_instance
    # 初始化数据库表
    print("初始化数据库...")
    init_db()
    # 初始化 RAG 系统
    try:
        print("正在初始化 RAG 系统...")
        qa_system_instance = initialize_qa_system(DATA_JSON)
        print("RAG 系统初始化完成。")
    except Exception as e:
        print(f"系统初始化失败：{e}")
        qa_system_instance = None
    yield
    # 可以在这里添加关闭时的清理逻辑

app = FastAPI(
    lifespan=lifespan,
    title="年报智能问答 API",
    description="基于手写 RAG 系统的问答服务（混合检索 + BGE 重排序 + LLM 生成）",
    version="1.0.0"
)


# ---------- 异常处理器 ----------
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail}
    )


# ---------- 路由 ----------
def _load_history(db: Session, session_id: str, max_turns: int = 5) -> List[Dict]:
    """按 session_id 从 ChatHistory 加载最近几轮问答，构造多轮上下文消息。"""
    rows = (
        db.query(ChatHistory)
        .filter(ChatHistory.session_id == session_id)
        .order_by(ChatHistory.id.desc())
        .limit(max_turns)
        .all()
    )
    rows.reverse()  # 时间正序
    messages = []
    for r in rows:
        messages.append({"role": "user", "content": r.query})
        messages.append({"role": "assistant", "content": r.answer})
    return messages


@app.get("/health", tags=["系统"])
async def health_check():
    if qa_system_instance is None:
        return {"status": "not_ready", "message": "系统尚未初始化完成"}
    return {"status": "ok", "message": "RAG 系统可用"}


@app.post("/chat", response_model=ChatResponse, tags=["问答"])
async def chat_endpoint(request: ChatRequest, db: Session = Depends(get_db)):
    """问答接口：RAG 回答 + 保存对话历史"""
    if qa_system_instance is None:
        raise HTTPException(status_code=503, detail="系统尚未初始化完成，请稍后重试")

    try:
        # 多轮上下文：按 session_id 取最近 5 轮历史（无 session_id 时为空 → 单轮行为不变）
        history = _load_history(db, request.session_id) if request.session_id else []
        # 调用 RAG 系统
        result = qa_system_instance.answer(
            query=request.query,
            top_k=request.top_k,
            include_sources=request.include_sources,
            history=history,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"问答过程中发生错误：{str(e)}")

    # 保存对话历史
    history = ChatHistory(
        user_id=request.user_id or "anonymous",
        session_id=request.session_id or "default",
        query=request.query,
        answer=result["answer"],
        retrieved_contexts="\n\n".join(result.get("retrieved_contexts", [])),
        total_tokens=result.get("total_tokens")
    )
    db.add(history)
    db.commit()
    db.refresh(history)

    # 构建响应，添加 chat_id
    response = ChatResponse(
        answer=result["answer"],
        sources=result.get("sources"),
        retrieved_contexts=result.get("retrieved_contexts"),
        chat_id=history.id
    )
    return response


@app.post("/feedback", tags=["反馈"])
async def submit_feedback(request: FeedbackRequest, db: Session = Depends(get_db)):
    """保存用户反馈"""
    feedback = UserFeedback(
        chat_id=request.chat_id,
        user_id="anonymous",  # 可从请求中获取真实 user_id
        feedback_type=request.feedback_type,
        comment=request.comment,
        rating=request.rating
    )
    db.add(feedback)
    db.commit()
    return {"status": "ok", "feedback_id": feedback.id}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.api:app", host=settings.API_HOST, port=settings.API_PORT, reload=True)