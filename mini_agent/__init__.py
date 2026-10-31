"""mini_agent —— 纯标准库实现的最小 LLM Agent 框架。

最常用的东西都可以直接从这里导入:
    from mini_agent import Agent, OpenAIClient, MockLLM, tool
"""
from .agent import Agent, DEFAULT_SYSTEM_PROMPT, pretty_messages
from .compress import SummarizingMemory, estimate_tokens
from .confirm import ConsoleConfirmer, Decision, auto_approve, auto_deny
from .errors import (
    AgentError, LLMAuthError, LLMBadRequestError, LLMConnectionError, LLMError,
    LLMRateLimitError, LLMResponseError, LLMServerError, LLMTimeoutError,
    MaxStepsExceeded, ToolError,
)
from .hooks import Callback, ConsolePrinter, LoggingCallback, TraceRecorder
from .llm import BaseLLM, LLMResponse, MockLLM, OpenAIClient, ToolCall
from .memory import ConversationMemory, JSONStore
from .skills import Skill, SkillRegistry, parse_frontmatter
from .tools import Tool, ToolRegistry, tool

__version__ = "0.1.0"
__all__ = [
    "Agent", "DEFAULT_SYSTEM_PROMPT", "pretty_messages",
    "BaseLLM", "OpenAIClient", "MockLLM", "LLMResponse", "ToolCall",
    "Tool", "ToolRegistry", "tool",
    "ConversationMemory", "JSONStore",
    "Skill", "SkillRegistry", "parse_frontmatter",
    "SummarizingMemory", "estimate_tokens",
    "Decision", "ConsoleConfirmer", "auto_approve", "auto_deny",
    "Callback", "ConsolePrinter", "LoggingCallback", "TraceRecorder",
    "AgentError", "LLMError", "LLMConnectionError", "LLMTimeoutError", "LLMRateLimitError",
    "LLMServerError", "LLMAuthError", "LLMBadRequestError", "LLMResponseError",
    "ToolError", "MaxStepsExceeded",
]
