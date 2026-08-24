# main.py
"""统一启动入口。

用法：
    python -m app.main            # 开发模式（reload=True）
    uvicorn app.api:app --host 0.0.0.0 --port 8000   # 生产模式
"""
import uvicorn

import sys

# 确保仓库根目录在 sys.path，支持直接运行（python app/main.py）与任意工作目录
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from app.config import settings

if __name__ == "__main__":
    uvicorn.run(
        "app.api:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=True,
    )
