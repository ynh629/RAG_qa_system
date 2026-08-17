# app.py
import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from fastapi.responses import JSONResponse
from typing import Optional, List
from contextlib import asynccontextmanager

# 加载环境变量（确保 qwen_api_key 等存在）
load_dotenv()

# 导入你的 RAG 系统
from qa_system import QASystem, initialize_qa_system

# 全局变量：保存启动时构建好的 QA 系统实例
qa_system_instance = None

# 数据文件路径（根据你的项目结构调整）
# 这里假设 app.py 与 qa_system.py 在同一目录，并且数据文件位于 ../data/structured_segments.json
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_JSON = os.path.join(BASE_DIR, "..", "data", "structured_segments.json")


# ---------- Pydantic 请求/响应模型 ----------
class ChatRequest(BaseModel):
    """聊天请求体"""
    query: str = Field(..., description="用户问题", min_length=1, max_length=1000)
    top_k: int = Field(5, description="重排序后保留的文档数", ge=1, le=10)
    include_sources: bool = Field(True, description="是否在响应中包含引用来源")


class Source(BaseModel):
    """引用来源结构"""
    title_path: str
    text_snippet: str
    rerank_score: float


class ChatResponse(BaseModel):
    """聊天响应体"""
    answer: str
    sources: Optional[List[Source]] = None
    retrieved_contexts: Optional[List[str]] = None


# ---------- 启动事件：lifespan构建 RAG 系统 ----------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global qa_system_instance
    try:
        print("正在初始化 RAG 系统...")
        qa_system_instance = initialize_qa_system(DATA_JSON)
        print("RAG 系统初始化完成。")
    except Exception as e:
        print(f"系统初始化失败：{e}")
        qa_system_instance = None
    yield  # 这里分隔启动和关闭

#创建app，传入lifespan和文档信息
app = FastAPI(
    lifespan=lifespan,
    title="年报智能问答 API",
    description="基于手写 RAG 系统的问答服务（混合检索 + BGE 重排序 + LLM 生成）",
    version="1.0.0"
)

# ---------- 异常处理器 ----------
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """统一 HTTP 异常响应格式"""
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail}
    )


# ---------- 路由 ----------
@app.get("/health", tags=["系统"])
async def health_check():
    """健康检查：检测 QA 系统是否就绪"""
    if qa_system_instance is None:
        return {"status": "not_ready", "message": "系统尚未初始化完成"}
    return {"status": "ok", "message": "RAG 系统可用"}


@app.post("/chat", response_model=ChatResponse, tags=["问答"])
async def chat_endpoint(request: ChatRequest):
    """
    问答接口：接收用户问题，返回基于年报的答案和引用来源。
    """
    if qa_system_instance is None:
        raise HTTPException(status_code=503, detail="系统尚未初始化完成，请稍后重试")

    try:
        # 调用 QASystem.answer 方法
        result = qa_system_instance.answer(
            query=request.query,
            top_k=request.top_k,
            include_sources=request.include_sources,
        )
        return result
    except Exception as e:
        # 捕获未知异常，返回 500
        raise HTTPException(status_code=500, detail=f"问答过程中发生错误：{str(e)}")


# 如果直接运行该文件，则启动 uvicorn 服务器（开发用）
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000,reload=True)