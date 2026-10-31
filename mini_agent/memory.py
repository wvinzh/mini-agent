"""记忆：短期（对话历史 + 窗口截断）与长期（JSON 文件持久化）。

短期记忆 ConversationMemory
    保存 user / assistant / tool 消息（system prompt 由 Agent 单独管理，永远不会被截断）。
    上下文窗口有限，所以需要截断策略。这里用最简单可靠的"滑动窗口"：
        - 最多保留 max_messages 条
        - 总字符数不超过 max_chars（粗略代替 token 数，中文约 1~2 字/token）
    截断时有一个容易踩的坑：OpenAI 要求 role=tool 的消息前面必须有
    对应的 assistant(tool_calls) 消息。所以截断后我们会一直丢到第一条 user 消息为止，
    保证窗口总是从一个完整的"回合"开始。

长期记忆 JSONStore
    一个极简的键值笔记本，存在 JSON 文件里；可以通过 as_tools() 暴露成
    remember / recall 两个工具，让模型自己决定记什么、查什么。
"""
import json
import os
import threading
from typing import Any, Callable, Dict, List, Optional

from .tools import Tool, tool

Message = Dict[str, Any]


def _msg_chars(m: Message) -> int:
    n = len(m.get("content") or "")
    for tc in m.get("tool_calls") or []:
        n += len(tc.get("function", {}).get("arguments", ""))
    return n


class ConversationMemory:
    """对话历史 + 滑动窗口截断；传入 path 时自动持久化到 JSON 文件。

    想自定义策略？继承本类并重写 `window()`（如何"看"历史）或
    `maybe_compress()`（如何"改写"历史）。LLM 摘要压缩见 compress.py 的 SummarizingMemory。
    """

    def __init__(self, max_messages: int = 50, max_chars: int = 24000,
                 path: Optional[str] = None):
        self.max_messages = max_messages
        self.max_chars = max_chars
        self.path = path
        self.messages: List[Message] = []
        if path and os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                self._load(json.load(f))

    # ------------------------------------------------------------------ 扩展点（Agent 会调用）
    def attach(self, llm: Any = None, emit: Optional[Callable[..., None]] = None) -> None:
        """Agent 初始化时调用，把自己的 LLM 和钩子分发函数交给记忆。基础版不需要，空实现。"""

    def maybe_compress(self) -> bool:
        """Agent 每次调用 LLM 前都会调用。基础版不压缩，返回 False。"""
        return False

    def add(self, message: Message) -> None:
        self.messages.append(message)
        self._save()

    def extend(self, messages: List[Message]) -> None:
        self.messages.extend(messages)
        self._save()

    def clear(self) -> None:
        self.messages = []
        self._save()

    def window(self) -> List[Message]:
        """返回本次要发给模型的历史（不修改原始记录）。"""
        msgs = self.messages[-self.max_messages:] if self.max_messages else list(self.messages)
        total = sum(_msg_chars(m) for m in msgs)
        while msgs and total > self.max_chars:
            total -= _msg_chars(msgs.pop(0))
        # 保证从 user 消息开始，避免出现"孤儿" tool 消息导致 API 报 400
        while msgs and msgs[0].get("role") != "user":
            msgs.pop(0)
        if not msgs and self.messages:
            # 极端情况：当前这一轮本身就超长。至少保留"最后一条 user 消息及其之后"的内容，
            # 否则模型会丢掉刚拿到的工具结果，陷入反复调用工具的死循环。
            for i in range(len(self.messages) - 1, -1, -1):
                if self.messages[i].get("role") == "user":
                    return self.messages[i:]
        return msgs

    def _load(self, data: Any) -> None:
        # 兼容两种文件格式：纯消息列表，或 {"summary": ..., "messages": [...]}（压缩记忆）
        self.messages = data["messages"] if isinstance(data, dict) else data

    def _dump(self) -> Any:
        return self.messages

    def _save(self) -> None:
        if not self.path:
            return
        # 先写临时文件再替换，避免写一半崩溃导致文件损坏
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self._dump(), f, ensure_ascii=False, indent=1)
        os.replace(tmp, self.path)


class JSONStore:
    """极简长期记忆：key -> 文本，存 JSON 文件。线程安全（多工具并发时也不怕）。"""

    def __init__(self, path: str = "memory.json"):
        self.path = path
        self._lock = threading.Lock()
        self.data: Dict[str, str] = {}
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                self.data = json.load(f)

    def set(self, key: str, value: str) -> None:
        with self._lock:
            self.data[key] = value
            tmp = self.path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(self.data, f, ensure_ascii=False, indent=1)
            os.replace(tmp, self.path)

    def get(self, key: str) -> Optional[str]:
        return self.data.get(key)

    def search(self, query: str) -> Dict[str, str]:
        """朴素的子串匹配；想要语义检索可以替换成向量库。"""
        q = query.lower()
        return {k: v for k, v in self.data.items() if q in k.lower() or q in v.lower()}

    def as_tools(self) -> List[Tool]:
        store = self

        @tool
        def remember(key: str, value: str) -> str:
            """把一条信息写入长期记忆，之后的对话也能查到。

            Args:
                key: 简短的主题/键，如 "用户名字"
                value: 要记住的内容
            """
            store.set(key, value)
            return "已记住: %s" % key

        @tool
        def recall(query: str) -> str:
            """在长期记忆中搜索信息。

            Args:
                query: 关键词；传空字符串返回全部记忆
            """
            hits = store.search(query) if query else store.data
            return json.dumps(hits, ensure_ascii=False) if hits else "没有找到相关记忆"

        return [remember, recall]
