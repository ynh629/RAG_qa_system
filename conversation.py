#消息维护（存档）
import json
import os
from typing import List, Optional
from token_utils import trim_message, count_tokens
class Conversation:
    def __init__(self, system_prompt: str, max_tokens: int=3000, model: str="qwen-plus", persist_path: Optional[str]=None):
        self.system_prompt = system_prompt
        self.max_tokens = max_tokens
        self.model = model
        self.persist_path = persist_path
        self.messages = [{"role": "system", "content": system_prompt}]
        if persist_path and os.path.exists(persist_path):
            self.load()
    def add_message(self, role: str, content: str):
        self.messages.append({"role": role, "content": content})
        self._trim()
    def add_user_message(self, content: str):
        self.add_message("user", content)
    def add_assistant_message(self, content: str):
        self.add_message("assistant", content)
    def _trim(self):
        self.messages = trim_message(self.messages, max_tokens=self.max_tokens, model=self.model)

    def get_messages(self) -> List[dict]:
        return self.messages
    def save(self, path: Optional[str] = None):
        """保存当前对话到 JSON 文件。"""
        save_path = path or self.persist_path
        if not save_path:
            raise ValueError("未指定持久化路径")
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(self.messages, f, ensure_ascii=False, indent=2)

    def load(self, path: Optional[str] = None):
        load_path = path or self.persist_path
        if not load_path:
            raise ValueError("未指定加载路径")
        with open(load_path, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        if loaded:
            self.messages = loaded
            self._trim()
    def reset(self):
        self.messages = [{"role": "system", "content": self.system_prompt}]

    def __len__(self):
        return len(self.messages) - 1
    def __repr__(self):
        return f"<Conversation:{len(self)} exchange,{count_tokens(self.messages)} tokens>"
