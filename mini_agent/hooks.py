"""回调 / 钩子：让你"看见" Agent 内部发生了什么（可观测性）。

用法：继承 Callback，只重写关心的方法，然后 Agent(callbacks=[MyCallback()])。

事件时间线（一次 run）：
    on_agent_start
      ┌ on_step_start(step)
      │   (on_compress / on_compress_error)  （使用 SummarizingMemory 且超过阈值时）
      │   on_llm_start → (on_token...) → on_llm_end / on_llm_error
      │   on_tool_start → (on_confirm) → (on_skill_load) → on_tool_end
      │       （每个工具调用一次；需要人工确认的工具多一个 on_confirm；
      │         调用 load_skill 成功时多一个 on_skill_load）
      └ ...重复...
    on_agent_end
"""
import json
import logging
import time
from typing import Any, Dict, List

logger = logging.getLogger("mini_agent.hooks")


class Callback:
    """所有钩子的空实现，按需覆盖。"""

    def on_agent_start(self, agent: Any, user_input: str) -> None: ...
    def on_step_start(self, agent: Any, step: int) -> None: ...
    def on_llm_start(self, agent: Any, messages: List[Dict]) -> None: ...
    def on_token(self, agent: Any, token: str) -> None: ...
    def on_llm_end(self, agent: Any, response: Any) -> None: ...
    def on_llm_error(self, agent: Any, error: Exception) -> None: ...
    def on_tool_start(self, agent: Any, name: str, arguments: str) -> None: ...
    def on_tool_end(self, agent: Any, name: str, result: str, seconds: float) -> None: ...
    def on_skill_load(self, agent: Any, name: str) -> None: ...
    def on_confirm(self, agent: Any, name: str, arguments: Dict, decision: Any) -> None: ...
    def on_compress(self, agent: Any, info: Dict[str, Any]) -> None: ...
    def on_compress_error(self, agent: Any, error: Exception) -> None: ...
    def on_agent_end(self, agent: Any, answer: str) -> None: ...


class CallbackManager:
    """把一个事件分发给所有回调；回调自身出错只记日志，绝不影响主流程。"""

    def __init__(self, callbacks: List[Callback]):
        self.callbacks = list(callbacks)

    def emit(self, event: str, *args: Any) -> None:
        for cb in self.callbacks:
            try:
                getattr(cb, event)(*args)
            except Exception:
                logger.exception("回调 %s.%s 出错（已忽略）", type(cb).__name__, event)


class LoggingCallback(Callback):
    """把所有事件写到 logging，适合线上排查。"""

    def __init__(self, level: int = logging.INFO):
        self.level = level
        self.log = logging.getLogger("mini_agent.trace")

    def on_agent_start(self, agent, user_input):
        self.log.log(self.level, "[%s] 开始: %s", agent.name, user_input[:200])

    def on_llm_start(self, agent, messages):
        self.log.log(self.level, "[%s] 调用 LLM，消息数=%d", agent.name, len(messages))

    def on_llm_end(self, agent, response):
        self.log.log(self.level, "[%s] LLM 返回: content=%r tool_calls=%s usage=%s", agent.name,
                     (response.content or "")[:100], [t.name for t in response.tool_calls],
                     response.usage)

    def on_llm_error(self, agent, error):
        self.log.error("[%s] LLM 出错: %s", agent.name, error)

    def on_tool_start(self, agent, name, arguments):
        self.log.log(self.level, "[%s] 调用工具 %s(%s)", agent.name, name, arguments[:200])

    def on_tool_end(self, agent, name, result, seconds):
        self.log.log(self.level, "[%s] 工具 %s 完成(%.2fs): %s", agent.name, name, seconds, result[:200])

    def on_skill_load(self, agent, name):
        self.log.log(self.level, "[%s] 加载技能 %s", agent.name, name)

    def on_confirm(self, agent, name, arguments, decision):
        self.log.log(self.level, "[%s] 人工确认 %s -> %s %s", agent.name, name, decision.action, decision.reason)

    def on_compress(self, agent, info):
        self.log.log(self.level, "[%s] 上下文压缩：%d 条消息 -> 摘要，token %d -> %d", agent.name,
                     info["compressed_messages"], info["tokens_before"], info["tokens_after"])

    def on_compress_error(self, agent, error):
        self.log.warning("[%s] 上下文压缩失败，退回窗口截断: %s", agent.name, error)

    def on_agent_end(self, agent, answer):
        self.log.log(self.level, "[%s] 结束: %s", agent.name, answer[:200])


class ConsolePrinter(Callback):
    """在终端里友好地打印思考/工具调用过程，CLI 和示例使用。"""

    def __init__(self, stream_tokens: bool = False):
        self.stream_tokens = stream_tokens

    def on_token(self, agent, token):
        if self.stream_tokens:
            print(token, end="", flush=True)

    def on_llm_end(self, agent, response):
        if self.stream_tokens and response.content:
            print()
        elif response.content and response.tool_calls:
            print("  💭 %s" % response.content)  # 模型在调用工具前的"思考"

    def on_tool_start(self, agent, name, arguments):
        args = arguments if len(arguments) < 200 else arguments[:200] + "..."
        print("  🔧 [%s] %s(%s)" % (agent.name, name, args))

    def on_tool_end(self, agent, name, result, seconds):
        short = result if len(result) < 300 else result[:300] + "..."
        print("  📎 %s -> %s  (%.2fs)" % (name, short, seconds))

    def on_skill_load(self, agent, name):
        print("  📚 [%s] 已加载技能: %s" % (agent.name, name))

    def on_confirm(self, agent, name, arguments, decision):
        icon = {"approve": "✅ 批准", "reject": "⛔ 拒绝", "edit": "✏️ 修改参数"}.get(decision.action, decision.action)
        print("  🙋 人工确认 %s: %s%s" % (name, icon, ("（%s）" % decision.reason) if decision.reason else ""))

    def on_compress(self, agent, info):
        print("  🗜️  上下文压缩（第 %d 次）：%d 条旧消息 → 摘要，估算 token %d → %d" % (
            info["count"], info["compressed_messages"], info["tokens_before"], info["tokens_after"]))

    def on_compress_error(self, agent, error):
        print("  ⚠️  上下文压缩失败，本轮退回窗口截断：%s" % error)


class TraceRecorder(Callback):
    """把事件按顺序记录在内存列表里：测试断言、导出轨迹、统计 token 都很方便。"""

    def __init__(self):
        self.events: List[Dict[str, Any]] = []
        self._t0 = time.time()

    def _rec(self, event: str, **data: Any) -> None:
        self.events.append({"event": event, "t": round(time.time() - self._t0, 3), **data})

    def on_agent_start(self, agent, user_input): self._rec("agent_start", input=user_input)
    def on_llm_start(self, agent, messages): self._rec("llm_start", n_messages=len(messages))
    def on_llm_end(self, agent, response): self._rec("llm_end", content=response.content,
                                                      tool_calls=[t.name for t in response.tool_calls],
                                                      usage=response.usage)
    def on_llm_error(self, agent, error): self._rec("llm_error", error=str(error))
    def on_tool_start(self, agent, name, arguments): self._rec("tool_start", name=name, arguments=arguments)
    def on_tool_end(self, agent, name, result, seconds): self._rec("tool_end", name=name, result=result)
    def on_skill_load(self, agent, name): self._rec("skill_load", name=name)
    def on_confirm(self, agent, name, arguments, decision): self._rec(
        "confirm", name=name, arguments=arguments, action=decision.action, reason=decision.reason)
    def on_compress(self, agent, info): self._rec("compress", **info)
    def on_compress_error(self, agent, error): self._rec("compress_error", error=repr(error))
    def on_agent_end(self, agent, answer): self._rec("agent_end", answer=answer)

    def names(self) -> List[str]:
        return [e["event"] for e in self.events]

    def dump(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.events, f, ensure_ascii=False, indent=1, default=str)
