# database.py
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session

from app.models import Base
from app.config import settings

# 从配置中心读取数据库 URL（默认 SQLite，存储于 data/chat_app.db）
DATABASE_URL = settings.DATABASE_URL

# 创建引擎：SQLite 需要特殊参数 connect_args
if DATABASE_URL.startswith("sqlite"):
    engine = create_engine(
        DATABASE_URL,
        connect_args={"check_same_thread": False},  # SQLite 允许多线程访问
        echo=False  # 设为 True 可打印 SQL 日志
    )
else:
    # PostgreSQL 等不需要特殊参数
    engine = create_engine(DATABASE_URL, echo=False)

# 创建会话工厂
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# 自动建表（开发期使用；生产环境建议使用 Alembic 迁移）
def init_db():
    Base.metadata.create_all(bind=engine)

# FastAPI 依赖：获取数据库会话
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()