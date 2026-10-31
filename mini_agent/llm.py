"""LLM 客户端：OpenAI 兼容的 Chat Completions + 可离线运行的 MockLLM。

结构：
    BaseLLM            抽象基类，负责"重试 + 指数退避"这类通用逻辑
      ├─ OpenAIClient  真实 HTTP 调用（urllib），支持工具调用和流式输出
      └─ MockLLM       按剧本返回结果的假模型，用于示例和单元测试

为什么把重试放在基类？
    重试策略与"怎么发请求"无关。放在基类后，MockLLM 也能复用，
    于是我们可以在离线测试里模拟 "先 503 两次、再成功" 的场景。
"""
import json
import logging
import os
import random
import socket
import time
import http.client
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Union

from .errors import (
    LLMAuthError, LLMBadRequestError, LLMConnectionError, LLMError,
    LLMRateLimitError, LLMResponseError, LLMServerError, LLMTimeoutError,
)

logger = logging.getLogger("mini_agent.llm")

Message = Dict[str, Any]  # 消息直接用 OpenAI 格式的 dict，零学习成本


# ====================================================================== 数据结构
@dataclass
class ToolCall:
    """模型发起的一次工具调用。arguments 是模型给的原始 JSON 字符串。"""

    id: str
    name: str
    arguments: str = "{}"

    def to_dict(self) -> Dict[str, Any]:
        """转换回 OpenAI 消息格式（写入 assistant 消息的 tool_calls 字段）。"""
        return {"id": self.id, "type": "function",
                "function": {"name": self.name, "arguments": self.arguments}}


@dataclass
class LLMResponse:
    """一次模型调用的结果（已从各家原始格式中归一化）。"""

    content: Optional[str] = None
    tool_calls: List[ToolCall] = field(default_factory=list)
    finish_reason: Optional[str] = None
    usage: Dict[str, Any] = field(default_factory=dict)
    raw: Any = None  # 原始返回，调试用

    def to_message(self) -> Message:
        """转成可以追加到对话历史里的 assistant 消息。"""
        msg: Message = {"role": "assistant", "content": self.content}
        if self.tool_calls:
            msg["tool_calls"] = [tc.to_dict() for tc in self.tool_calls]
        return msg


# ====================================================================== 基类
class BaseLLM:
    """所有 LLM 实现的基类：子类只需实现 `_chat_once`，重试由这里统一处理。

    重试规则（只针对 err.retryable == True 的错误）：
        第 n 次重试前等待  min(max_backoff, backoff_base * 2**(n-1)) * 随机抖动(0.5~1.0)
        若服务端给了 Retry-After（429 常见），则优先使用它。
    抖动(jitter)的作用：避免大量客户端在同一时刻一起重试，把服务端再次打挂。
    """

    def __init__(self, max_retries: int = 3, backoff_base: float = 1.0,
                 max_backoff: float = 30.0, sleep: Callable[[float], None] = time.sleep):
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self.max_backoff = max_backoff
        self._sleep = sleep  # 可注入，测试时传入假的 sleep 就不用真等待

    def chat(self, messages: List[Message], tools: Optional[List[Dict]] = None,
             on_token: Optional[Callable[[str], None]] = None) -> LLMResponse:
        """发送对话并返回结果；on_token 不为空时启用流式输出（逐段回调文本）。"""
        attempt = 0
        while True:
            emitted = [False]  # 记录是否已经向外输出过 token

            def _on_token(t: str) -> None:
                emitted[0] = True
                on_token(t)  # type: ignore[misc]

            try:
                return self._chat_once(messages, tools, _on_token if on_token else None)
            except LLMError as e:
                # 流式输出已经吐出部分内容时不能重试，否则用户会看到重复文字
                if not e.retryable or attempt >= self.max_retries or emitted[0]:
                    raise
                attempt += 1
                delay = self._backoff(attempt, e)
                logger.warning("LLM 调用失败(%s)，%.1fs 后第 %d/%d 次重试",
                               e, delay, attempt, self.max_retries)
                self._sleep(delay)

    def _backoff(self, attempt: int, err: LLMError) -> float:
        retry_after = getattr(err, "retry_after", None)
        if retry_after:
            return min(float(retry_after), self.max_backoff)
        delay = min(self.max_backoff, self.backoff_base * (2 ** (attempt - 1)))
        return delay * random.uniform(0.5, 1.0)

    def _chat_once(self, messages: List[Message], tools: Optional[List[Dict]],
                   on_token: Optional[Callable[[str], None]]) -> LLMResponse:
        raise NotImplementedError


# ====================================================================== 真实客户端
class OpenAIClient(BaseLLM):
    """OpenAI 兼容接口客户端（OpenAI / DeepSeek / Qwen / Moonshot / 各类中转站 / Ollama 等）。

    配置优先级：构造参数 > 环境变量 > 默认值
        OPENAI_BASE_URL  例如 https://api.openai.com/v1
        OPENAI_API_KEY
        MODEL            例如 gpt-4o-mini
    """

    def __init__(self, model: Optional[str] = None, api_key: Optional[str] = None,
                 base_url: Optional[str] = None, timeout: float = 60.0,
                 temperature: Optional[float] = None, extra_body: Optional[Dict] = None,
                 **retry_kwargs: Any):
        super().__init__(**retry_kwargs)
        self.model = model or os.getenv("MODEL") or os.getenv("OPENAI_MODEL") or "gpt-4o-mini"
        self.api_key = api_key if api_key is not None else os.getenv("OPENAI_API_KEY", "")
        self.base_url = (base_url or os.getenv("OPENAI_BASE_URL")
                         or "https://api.openai.com/v1").rstrip("/")
        self.timeout = timeout
        self.temperature = temperature
        self.extra_body = extra_body or {}

    # ------------------------------------------------------------------ 发请求
    def _chat_once(self, messages, tools, on_token):
        payload: Dict[str, Any] = {"model": self.model, "messages": messages}
        if tools:
            payload["tools"] = tools
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        if on_token:
            payload["stream"] = True
        payload.update(self.extra_body)

        req = urllib.request.Request(
            self.base_url + "/chat/completions",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json",
                     "Authorization": "Bearer " + self.api_key},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                if on_token:
                    return self._read_stream(resp, on_token)
                text = resp.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as e:  # 必须在 URLError 之前捕获（它是子类）
            raise self._http_error(e) from e
        except urllib.error.URLError as e:
            if isinstance(e.reason, (socket.timeout, TimeoutError)):
                raise LLMTimeoutError("请求超时(%ss)" % self.timeout) from e
            raise LLMConnectionError("网络连接失败: %s" % e.reason) from e
        except (socket.timeout, TimeoutError) as e:  # 读取 body 阶段超时
            raise LLMTimeoutError("读取响应超时(%ss)" % self.timeout) from e
        except (ConnectionError, http.client.HTTPException, OSError) as e:
            # 连接被重置 / 服务器提前断开 / IncompleteRead 等
            raise LLMConnectionError("连接异常: %r" % e) from e
        return self._parse(text)

    # ------------------------------------------------------------------ 错误归类
    @staticmethod
    def _http_error(e: urllib.error.HTTPError) -> LLMError:
        """把 HTTP 状态码映射成语义化异常，这是"重试是否有意义"的关键判断。"""
        try:
            body = e.read().decode("utf-8", errors="replace")
        except Exception:
            body = ""
        code = e.code
        if code in (401, 403):
            return LLMAuthError("鉴权失败，请检查 OPENAI_API_KEY / 权限", code, body)
        if code == 429:
            ra = e.headers.get("Retry-After") if e.headers else None
            try:
                retry_after = float(ra) if ra else None
            except ValueError:
                retry_after = None
            return LLMRateLimitError("触发限流(429)", code, body, retry_after)
        if code == 408:
            return LLMTimeoutError("服务端请求超时(408)", code, body)
        if code >= 500:
            return LLMServerError("服务端错误", code, body)
        return LLMBadRequestError("请求被拒绝，检查模型名/参数", code, body)

    # ------------------------------------------------------------------ 解析
    def _parse(self, text: str) -> LLMResponse:
        try:
            data = json.loads(text)
        except ValueError:
            raise LLMResponseError("返回不是合法 JSON（可能是网关错误页）", 200, text)
        # 部分中转站会以 HTTP 200 返回 {"error": {...}}
        if isinstance(data, dict) and data.get("error"):
            err = data["error"]
            msg = err.get("message", str(err)) if isinstance(err, dict) else str(err)
            if "rate" in msg.lower() or "limit" in msg.lower():
                raise LLMRateLimitError("限流: " + msg, 200, text)
            raise LLMServerError("服务端返回错误: " + msg, 200, text)
        try:
            choice = data["choices"][0]
            msg = choice["message"]
        except (KeyError, IndexError, TypeError):
            raise LLMResponseError("返回缺少 choices[0].message", 200, text)
        calls = [ToolCall(id=tc.get("id") or "call_%d" % i,
                          name=tc["function"]["name"],
                          arguments=tc["function"].get("arguments") or "{}")
                 for i, tc in enumerate(msg.get("tool_calls") or [])]
        return LLMResponse(content=msg.get("content"), tool_calls=calls,
                           finish_reason=choice.get("finish_reason"),
                           usage=data.get("usage") or {}, raw=data)

    def _read_stream(self, resp, on_token) -> LLMResponse:
        """解析 SSE 流：每行形如 `data: {...}`，以 `data: [DONE]` 结束。

        工具调用在流式模式下是"分片"到达的：同一个 index 的 arguments
        会被拆成很多段，需要按 index 拼接。
        """
        content_parts: List[str] = []
        calls: Dict[int, Dict[str, str]] = {}
        finish = None
        for raw_line in resp:
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                chunk = json.loads(data)
            except ValueError:
                continue  # 忽略无法解析的心跳/杂行
            if chunk.get("error"):
                raise LLMServerError("流式返回错误: %s" % chunk["error"], 200, data)
            for choice in chunk.get("choices") or []:
                delta = choice.get("delta") or {}
                if delta.get("content"):
                    content_parts.append(delta["content"])
                    on_token(delta["content"])
                for tc in delta.get("tool_calls") or []:
                    slot = calls.setdefault(tc.get("index", 0), {"id": "", "name": "", "arguments": ""})
                    slot["id"] = tc.get("id") or slot["id"]
                    fn = tc.get("function") or {}
                    slot["name"] += fn.get("name") or ""
                    slot["arguments"] += fn.get("arguments") or ""
                finish = choice.get("finish_reason") or finish
        tool_calls = [ToolCall(id=c["id"] or "call_%d" % i, name=c["name"],
                               arguments=c["arguments"] or "{}")
                      for i, c in sorted(calls.items())]
        return LLMResponse(content="".join(content_parts) or None,
                           tool_calls=tool_calls, finish_reason=finish)


# ====================================================================== 假模型
ScriptItem = Union[str, LLMResponse, Exception, Callable[[List[Message], Optional[List[Dict]]], LLMResponse]]


class MockLLM(BaseLLM):
    """按"剧本"依次返回结果的假模型，让示例和测试完全离线运行。

    剧本里的每一项可以是：
        str            —— 直接作为最终回答
        LLMResponse    —— 原样返回（可包含 tool_calls）
        Exception      —— 抛出（用来测试重试/错误处理）
        callable       —— fn(messages, tools) -> LLMResponse，可根据上下文动态决定

    例:
        llm = MockLLM([MockLLM.call("add", a=1, b=2), "结果是 3"])
    """

    def __init__(self, script: List[ScriptItem], **retry_kwargs: Any):
        retry_kwargs.setdefault("sleep", lambda s: None)  # 测试不真的等待
        super().__init__(**retry_kwargs)
        self.script = list(script)
        self.calls: List[List[Message]] = []  # 记录每次收到的消息，便于断言

    @staticmethod
    def call(name: str, /, **arguments: Any) -> LLMResponse:
        """快捷构造一个"调用某工具"的响应。

        name 是仅限位置参数（`/`），所以工具参数本身也可以叫 name，如 call("load_skill", name="x")。
        """
        return LLMResponse(tool_calls=[ToolCall(id="call_%s_%d" % (name, random.randint(0, 99999)),
                                                name=name,
                                                arguments=json.dumps(arguments, ensure_ascii=False))])

    def _chat_once(self, messages, tools, on_token):
        self.calls.append([dict(m) for m in messages])
        if not self.script:
            return LLMResponse(content="（MockLLM 剧本已用完）", finish_reason="stop")
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        if callable(item) and not isinstance(item, LLMResponse):
            item = item(messages, tools)
        if isinstance(item, str):
            item = LLMResponse(content=item, finish_reason="stop")
        if on_token and item.content:  # 模拟流式：逐字回调
            for ch in item.content:
                on_token(ch)
        return item
