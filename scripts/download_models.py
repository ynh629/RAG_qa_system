# download_models.py
"""从 ModelScope 预下载 BGE 模型到本地缓存目录（国内 ECS 专用，绕开 HuggingFace）。

用法（容器内执行，模型落入 hf_cache 共享卷，之后离线加载）：
    pip install -q modelscope -i https://mirrors.aliyun.com/pypi/simple/
    python scripts/download_models.py

下载位置：~/.cache/huggingface/local/<model_id>/
代码侧配合 rag_system/retrieval/model_paths.py 的 resolve_model() 离线加载。
"""
import os
import shutil

from modelscope import snapshot_download

DEST = os.path.expanduser("~/.cache/huggingface/local")

MODELS = [
    "BAAI/bge-small-zh-v1.5",   # 嵌入模型
    "BAAI/bge-reranker-base",   # 重排序模型
]

for repo in MODELS:
    print("DOWNLOAD:", repo)
    src = snapshot_download(repo)
    dst = os.path.join(DEST, repo)
    if os.path.exists(dst):
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    print("SAVED:", dst)

print("=== ALL MODELS READY ===")
