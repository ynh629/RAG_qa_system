import os
import re
import sys
from typing import List, Dict, Optional

# 确保可以导入上级目录的公共模块（日志、异常）
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from 系统日志.config import get_logger

logger = get_logger(__name__)


class MarkdownHeadingSplitter:       # Markdown 结构切片器：按标题层级切分，保留上下文
    def __init__(self, text: str):
        self.text = text or ""
        self.lines = self.text.splitlines(keepends=True)  # 保留换行符
        self.headings = self._parse_headings()

    # -----------------------------文档分析----------------------------------
    def _parse_headings(self) -> List[Dict]:
        """
        解析所有标题行，返回列表，每个元素包含：
            - level: int    标题级别 (1~6)
            - title: str    标题文字（去除#号和空格）
            - line_index: int  标题在原文本行列表中的索引
            - start_pos: int   标题在全文中的字符起始位置（基于行的累积长度）
        """
        # 变量：标题列表
        headings = []
        # 变量：累积字符数，用于定位文本位置
        char_count = 0
        for idx, line in enumerate(self.lines):
            # 检查是否为标题行：以1~6个#开头，后跟空格
            match = re.match(r'^(#{1,6})\s+(.*)$', line.strip())
            if match:
                level = len(match.group(1))          # 变量：标题级别
                title = match.group(2).strip()       # 变量：标题文字
                headings.append({
                    "level": level,
                    "title": title,
                    "line_index": idx,
                    "start_pos": char_count
                })
            char_count += len(line)
        logger.info("解析到 %d 个标题", len(headings))
        return headings

    # ----------------------------切分---------------------------------------
    def split_by_headings(self, mode: str = "leaf", target_level: Optional[int] = None) -> List[Dict]:
        """
        按标题层级切分 Markdown，返回结构化片段列表。
        参数：
            mode: 切分模式
                - "leaf": 叶节点模式，返回最底层的标题区域作为块。
                - "level": 层级模式，按 target_level 切分（如 target_level=2 时按 ## 切）。
            target_level: 仅在 mode="level" 时有效，指定切割的标题级别
        返回：片段列表，每个片段包含：
            - title_path: List[str]   标题路径（父标题链）
            - content: str            Markdown 内容
            - level: int              当前标题级别
            - page: None              预留页码
        """
        # 空文本保护
        if not self.text or not self.text.strip():
            logger.warning("输入文本为空，返回空片段列表")
            return []

        if not self.headings:
            logger.warning("文档无任何标题，返回整篇作为单个片段")
            return [{"title_path": [], "content": self.text.strip(), "level": 0, "page": None}]

        # 变量：结果片段列表
        fragments = []

        # 辅助函数：根据层级计算父标题路径
        def get_title_path(index):
            """根据标题索引，回溯构建父标题链。"""
            path = []
            current_level = self.headings[index]["level"]
            # 从当前标题向前找所有严格小于当前级别的标题，直到最外层
            for i in range(index, -1, -1):
                if self.headings[i]["level"] < current_level:
                    path.insert(0, self.headings[i]["title"])
                    current_level = self.headings[i]["level"]
            path.append(self.headings[index]["title"])
            return path

        # 根据模式选择要输出的标题索引
        if mode == "leaf":
            # 叶节点：找出所有“没有更低级标题”的标题
            leaf_indices = []
            for i, h in enumerate(self.headings):
                if i == len(self.headings) - 1:
                    leaf_indices.append(i)
                else:
                    next_level = self.headings[i+1]["level"]
                    if next_level > h["level"]:
                        # 后面有更低级的标题，当前不是叶节点，跳过
                        continue
                    else:
                        # 后面标题的级别 <= 当前级别，说明当前标题是它自己范围内的最底层
                        leaf_indices.append(i)
            # 对每个叶标题，提取其内容，并记录标题路径
            for idx in leaf_indices:
                content_start = self.headings[idx]["line_index"] + 1
                # 内容结束：下一个标题的起始行（不包括标题行），或文本末尾
                if idx < len(self.headings) - 1:
                    content_end = self.headings[idx+1]["line_index"]
                else:
                    content_end = len(self.lines)
                # 变量：切片内容的行列表，不包含标题行本身
                content_lines = self.lines[content_start:content_end]
                content = "".join(content_lines).strip()

                # 过滤空内容片段
                if not content:
                    continue

                fragments.append({
                    "title_path": get_title_path(idx),
                    "content": content,
                    "level": self.headings[idx]["level"],
                    "page": None
                })

        elif mode == "level":
            if target_level is None:
                raise ValueError("target_level must be specified in 'level' mode")
            # 层级模式：找出所有级别等于 target_level 的标题，每个标题加上其所属的所有子内容
            for i, h in enumerate(self.headings):
                if h["level"] == target_level:
                    # 变量：当前标题的内容起始行（标题行的下一行）
                    content_start = h["line_index"] + 1
                    # 内容结束：下一个同级别或更高级别标题的行，或文本末尾
                    content_end = len(self.lines)
                    for j in range(i + 1, len(self.headings)):
                        if self.headings[j]["level"] <= target_level:
                            content_end = self.headings[j]["line_index"]
                            break
                    content_lines = self.lines[content_start:content_end]
                    content = "".join(content_lines).strip()

                    # 过滤空内容片段
                    if not content:
                        continue

                    fragments.append({
                        "title_path": get_title_path(i),
                        "content": content,
                        "level": target_level,
                        "page": None
                    })
        else:
            raise ValueError(f"Unsupported mode: {mode}")

        logger.info("切分完成（mode=%s），共 %d 个片段", mode, len(fragments))
        return fragments
