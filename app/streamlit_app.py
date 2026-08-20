# streamlit_app.py
"""Streamlit 前端：聊天界面（流式）+ 文件上传 + 历史记录展示。

用法：
    streamlit run app/streamlit_app.py
"""
import asyncio
import os
import sys
import uuid

import streamlit as st

# 确保仓库根目录在 sys.path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from sqlalchemy import func

from app.config import settings
from app.database import init_db, SessionLocal
from app.models import ChatHistory, UserFeedback
from app.qa_system import initialize_qa_system, build_qa_system_from_segments

# ---------- 页面配置 ----------
st.set_page_config(page_title="年报智能问答", page_icon=":material/chat:", layout="wide")

# 初始化数据库表
init_db()

# ---------- 会话状态 ----------
if "session_id" not in st.session_state:
    st.session_state.session_id = uuid.uuid4().hex[:12]
if "messages" not in st.session_state:
    st.session_state.messages = []
if "qa" not in st.session_state:
    st.session_state.qa = None
if "current_doc" not in st.session_state:
    st.session_state.current_doc = "默认（年报.pdf）"


# ---------- 异步生成器适配（供 st.write_stream 逐字流式渲染） ----------
def _iter_async(agen):
    """把异步生成器适配为同步生成器，避免依赖 Streamlit 对 async 的原生支持。"""
    loop = asyncio.new_event_loop()
    try:
        it = agen.__aiter__()
        while True:
            try:
                yield loop.run_until_complete(it.__anext__())
            except StopAsyncIteration:
                break
    finally:
        loop.close()

# ---------- 数据库辅助 ----------
def _list_sessions():
    """按会话分组统计历史（时间倒序）。"""
    db = SessionLocal()
    try:
        rows = (
            db.query(
                ChatHistory.session_id,
                func.max(ChatHistory.created_at).label("last_time"),
                func.count(ChatHistory.id).label("n"),
            )
            .group_by(ChatHistory.session_id)
            .order_by(func.max(ChatHistory.created_at).desc())
            .all()
        )
        return [(r.session_id, r.n, r.last_time) for r in rows]
    finally:
        db.close()


def _load_session_messages(session_id):
    """加载某个会话的全部问答消息。"""
    db = SessionLocal()
    try:
        rows = (
            db.query(ChatHistory)
            .filter(ChatHistory.session_id == session_id)
            .order_by(ChatHistory.id.asc())
            .all()
        )
        msgs = []
        for r in rows:
            msgs.append({"role": "user", "content": r.query})
            msgs.append({
                "role": "assistant",
                "content": r.answer,
                "chat_id": r.id,
                "sources": None,
            })
        return msgs
    finally:
        db.close()


def _save_chat(query, answer, contexts):
    """保存一轮问答到 ChatHistory。"""
    db = SessionLocal()
    try:
        h = ChatHistory(
            user_id="anonymous",
            session_id=st.session_state.session_id,
            query=query,
            answer=answer,
            retrieved_contexts="\n\n".join(contexts or []),
        )
        db.add(h)
        db.commit()
        db.refresh(h)
        return h.id
    finally:
        db.close()


def _save_feedback(chat_id, feedback_type):
    """记录 有用/没用 反馈到 UserFeedback。"""
    db = SessionLocal()
    try:
        db.add(UserFeedback(
            chat_id=chat_id, user_id="anonymous", feedback_type=feedback_type
        ))
        db.commit()
    finally:
        db.close()


def _render_sources_and_feedback(sources, chat_id):
    """渲染引用来源与 有用/没用 反馈按钮。"""
    if sources:
        with st.expander("引用来源"):
            for i, src in enumerate(sources, 1):
                st.markdown(f"**{i}. {src.get('title_path', '无标题')}**")
                st.caption(f"相关度：{src.get('rerank_score', 0):.4f}")
                st.markdown(src.get("text_snippet", ""))
    if chat_id:
        c1, c2 = st.columns(2)
        if c1.button("有用", key=f"up_{chat_id}"):
            _save_feedback(chat_id, "up")
            st.toast("感谢反馈！")
        if c2.button("没用", key=f"down_{chat_id}"):
            _save_feedback(chat_id, "down")
            st.toast("感谢反馈！")


def _parse_uploaded_file(file_path, original_name):
    """根据扩展名解析上传文件，返回结构化片段列表。"""
    ext = os.path.splitext(original_name)[1].lower()
    if ext == ".pdf":
        from rag_system.parsing.parse_pdf import parse_pdf_to_segments
        return parse_pdf_to_segments(file_path, mode="leaf")
    if ext in (".txt", ".md"):
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        return [{"title_path": [original_name], "content": text, "level": 0, "page": None}]
    raise ValueError(f"不支持的文件类型: {ext}（支持 .pdf / .txt / .md）")

# ---------- 侧边栏：上传 + 历史 ----------
with st.sidebar:
    st.title("年报智能问答")
    st.caption("上传文档 → 智能问答")

    st.subheader("文档上传")
    uploaded = st.file_uploader("上传 PDF / TXT / MD", type=["pdf", "txt", "md"])
    if uploaded is not None:
        uploads_dir = os.path.join(settings.DATA_DIR, "uploads")
        os.makedirs(uploads_dir, exist_ok=True)
        save_path = os.path.join(uploads_dir, uploaded.name)
        with open(save_path, "wb") as f:
            f.write(uploaded.getbuffer())
        with st.spinner("正在解析文档并构建索引（首次运行需下载模型，较慢）..."):
            try:
                segments = _parse_uploaded_file(save_path, uploaded.name)
                st.session_state.qa = build_qa_system_from_segments(segments)
                st.session_state.current_doc = uploaded.name
                st.success(f"已加载：{uploaded.name}（{len(segments)} 个片段）")
            except Exception as e:
                st.error(f"解析失败：{e}")

    st.divider()

    st.subheader("历史会话")
    if st.button("新建会话", use_container_width=True):
        st.session_state.session_id = uuid.uuid4().hex[:12]
        st.session_state.messages = []
        st.rerun()

    sessions = _list_sessions()
    if sessions:
        st.caption("点击会话可查看历史")
        for sid, n, _t in sessions:
            if st.button(f"会话 {sid}（{n} 条）", key=f"sess_{sid}", use_container_width=True):
                st.session_state.session_id = sid
                st.session_state.messages = _load_session_messages(sid)
                st.session_state.current_doc = "历史会话"
                st.rerun()
    else:
        st.caption("暂无历史会话")


# ---------- 主区域 ----------
st.title("年报智能问答")
st.caption(f"当前文档：{st.session_state.current_doc} ｜ 会话 ID：{st.session_state.session_id}")

# 初始化知识库
if st.session_state.qa is None:
    if os.path.exists(settings.DATA_JSON):
        with st.spinner("正在初始化知识库（首次运行需下载模型，请耐心等待）..."):
            try:
                st.session_state.qa = initialize_qa_system(settings.DATA_JSON)
                st.session_state.current_doc = "默认（年报.pdf）"
            except Exception as e:
                st.error(f"知识库初始化失败：{e}")
                st.info("请在 .env 配置 qwen_api_key，或上传文档后重试。")
    else:
        st.info("未找到默认知识库，请在左侧上传文档开始。")


# ---------- 渲染历史消息 ----------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            _render_sources_and_feedback(msg.get("sources"), msg.get("chat_id"))


# ---------- 输入与流式回答 ----------
if prompt := st.chat_input("请输入你的问题..."):
    if st.session_state.qa is None:
        st.error("知识库尚未就绪，请先上传文档或检查配置。")
    else:
        # 记录用户消息
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # 流式生成回答
        with st.chat_message("assistant"):
            try:
                answer = st.write_stream(
                    _iter_async(st.session_state.qa.stream_answer(prompt, top_k=5))
                )
                sources = st.session_state.qa.last_sources
                _render_sources_and_feedback(sources, None)
            except Exception as e:
                answer = f"抱歉，生成回答时出错：{e}"
                st.markdown(answer)
                sources = []

        # 保存到数据库
        chat_id = _save_chat(prompt, answer, st.session_state.qa.last_retrieved_contexts)
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources": sources,
            "chat_id": chat_id,
        })