# main.py
"""统一启动入口。

用法：
    python -m app.main            # 开发模式（reload=True）
    uvicorn app.api:app --host 0.0.0.0 --port 8000   # 生产模式
"""
import uvicorn

from app.config import settings

if __name__ == "__main__":
    uvicorn.run(
        "app.api:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=True,
    )
