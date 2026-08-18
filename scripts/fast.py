# main.py
from fastapi import FastAPI, Depends, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional

# 创建 FastAPI 应用实例
app = FastAPI(
    title="AI 应用 API",
    description="FastAPI 基础示例",
    version="0.1.0"
)

# ---------- 1. Pydantic 请求模型 ----------
class ChatRequest(BaseModel):
    """聊天请求体"""
    message: str = Field(..., description="用户输入的消息", min_length=1, max_length=500)
    user_id: Optional[str] = Field(None, description="用户ID，可选")

# ---------- 2. Pydantic 响应模型 ----------
class ChatResponse(BaseModel):
    """聊天响应体"""
    reply: str = Field(..., description="AI 回复内容")
    model: str = Field(..., description="使用的模型名")
    tokens: int = Field(..., description="消耗的 token 数量")

# ---------- 3. 自定义异常 ----------
class APIError(Exception):
    """业务逻辑异常基类"""
    def __init__(self, message: str, status_code: int = 400):
        self.message = message
        self.status_code = status_code

# ---------- 4. 全局异常处理器 ----------
@app.exception_handler(APIError)
async def api_error_handler(request: Request, exc: APIError):
    """处理自定义 APIError 异常，返回统一 JSON 格式"""
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.message, "status_code": exc.status_code}
    )

@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError):
    """处理请求参数校验失败（Pydantic 自动抛出）"""
    return JSONResponse(
        status_code=422,
        content={"error": "请求参数错误", "details": exc.errors()}
    )

# ---------- 5. 依赖注入 ----------
def verify_api_key(api_key: str = "test-key"):
    """
    模拟鉴权依赖：校验 API Key。
    实际项目中可替换为从数据库/环境变量读取并验证。
    """
    def _verify(api_key: str = api_key):
        if api_key != "valid-key":
            raise HTTPException(status_code=401, detail="无效的 API Key")
        return api_key
    return _verify

# 直接使用一个简单的依赖函数（推荐写法）
def get_current_user(user_id: Optional[str] = None) -> str:
    """根据 user_id 获取当前用户（这里简单返回）"""
    return user_id or "anonymous"

# ---------- 6. 路由 ----------
@app.get("/health", tags=["系统"])
async def health_check():
    """健康检查接口，用于监控服务是否存活"""
    return {"status": "ok", "service": "ai-api"}

@app.post("/chat", response_model=ChatResponse, tags=["对话"])
async def chat(
    request: ChatRequest,
    current_user: str = Depends(get_current_user),
    api_key: str = Depends(verify_api_key())
):
    """
    聊天接口：接收用户消息，返回模拟 AI 回复。
    依赖注入：
        - get_current_user：获取当前用户
        - verify_api_key：校验 API Key
    """
    if request.message == "错误":
        # 手动抛出自定义异常，测试异常处理器
        raise APIError("触发了错误测试", status_code=400)

    # 模拟调用 LLM 生成回复（实际项目应替换为你的 RAG 系统）
    reply_text = f"你说的是：{request.message}"
    return ChatResponse(
        reply=reply_text,
        model="mock-model",
        tokens=len(request.message) // 2  # 简单估算
    )

# 如果直接运行该文件，则启动服务器（仅开发环境用）
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)