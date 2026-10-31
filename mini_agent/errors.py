"""统一的异常体系。

为什么要单独定义异常？
    第三方 LLM API 经常"抽风"：网络超时、限流(429)、网关错误(502/503)、
    返回半截 JSON、甚至 HTTP 200 但 body 里带 error 字段。
    如果全部抛成一个 Exception，上层根本不知道"该重试"还是"该放弃"。

设计原则：
    - 所有框架异常都继承 AgentError，方便一把 catch。
    - LLMError 下再细分；用 `retryable` 属性标记"重试是否可能成功"，
      重试逻辑只看这个属性，不需要到处写 isinstance 判断。
"""
from typing import Optional


class AgentError(Exception):
    """框架内所有异常的基类。"""


# ---------------------------------------------------------------- LLM 相关
class LLMError(AgentError):
    """调用大模型失败的基类。

    属性:
        status: HTTP 状态码（网络层错误时为 None）
        body:   服务端原始返回（截断后），便于排查
        retryable: 是否值得重试
    """

    retryable = False

    def __init__(self, message: str, status: Optional[int] = None, body: str = ""):
        super().__init__(message)
        self.status = status
        self.body = body[:2000] if body else ""

    def __str__(self) -> str:
        base = super().__str__()
        if self.status is not None:
            base = "[HTTP %s] %s" % (self.status, base)
        if self.body:
            base += " | body: " + self.body[:300]
        return base


class LLMConnectionError(LLMError):
    """网络层错误：DNS 失败、连接被拒、连接被重置、超时等。通常重试有用。"""

    retryable = True


class LLMTimeoutError(LLMConnectionError):
    """请求超时（是连接错误的一种）。"""


class LLMRateLimitError(LLMError):
    """HTTP 429 限流。可以重试，并尽量遵守服务端的 Retry-After。"""

    retryable = True

    def __init__(self, message: str, status: Optional[int] = 429, body: str = "",
                 retry_after: Optional[float] = None):
        super().__init__(message, status, body)
        self.retry_after = retry_after


class LLMServerError(LLMError):
    """HTTP 5xx：服务端/网关临时故障。通常重试有用。"""

    retryable = True


class LLMAuthError(LLMError):
    """HTTP 401/403：API Key 错误或无权限。重试没有意义，必须改配置。"""


class LLMBadRequestError(LLMError):
    """HTTP 400/404/422 等：请求参数有问题（模型名错、消息格式错）。不重试。"""


class LLMResponseError(LLMError):
    """服务端返回了无法解析/结构不对的内容（非 JSON、缺 choices 等）。

    默认标记为可重试：很多"中转站"偶尔会返回 HTML 错误页或空 body，
    重试一次往往就好了。
    """

    retryable = True


# ---------------------------------------------------------------- 工具 / Agent
class ToolError(AgentError):
    """工具定义或执行相关错误（注意：工具运行时异常默认不会抛出，而是回传给模型）。"""


class MaxStepsExceeded(AgentError):
    """Agent 循环超过 max_steps 仍未得到最终回答。"""
