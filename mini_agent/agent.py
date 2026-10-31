"""Agent 主循环：ReAct（Reason + Act）风格。

一句话概括：
    while 没超过 max_steps:
        回复 = LLM(系统提示 + 历史, 可用工具)
        if 回复里没有工具调用:  return 回复.content          # 停止条件①
        for 每个工具调用: 执行 -> 把结果作为 role=tool 消息加入历史
        if 某个工具 return_direct: return 工具结果              # 停止条件②
    超过 max_steps -> 不带工具再问一次模型要最终答案 / 或抛异常  # 停止条件③

所谓 "Think → Act → Observe"：
    Think   = 模型输出的 content（可能为空，有些模型直接调用工具）
    Act     = 模型输出的 tool_calls
    Observe = 我们把工具执行结果追加回对话，下一轮模型就能"看到"
"""
import json
import time
from typing import Any, Dict, List, Optional

from .confirm import ConfirmHandler, Decision, deny_without_handler, normalize, rejection_message
from .errors import LLMError, MaxStepsExceeded
from .hooks import Callback, CallbackManager
from .llm import BaseLLM, LLMResponse
from .memory import ConversationMemory
from .skills import SkillRegistry
from .tools import Tool, ToolRegistry

DEFAULT_SYSTEM_PROMPT = (
    "你是一个能使用工具的智能助手。\n"
    "工作方式：先思考需要什么信息；需要时调用工具；根据工具结果继续推理；"
    "信息足够时直接给出简洁、准确的最终回答。\n"
    "如果工具返回 Error，请分析原因并修正参数后重试，或换一种方法。"
    "不要编造工具没有返回的信息。"
)

FORCE_ANSWER_PROMPT = "已达到最大推理步数。请不要再调用工具，基于以上已有信息直接给出最终回答。"


class Agent:
    """一个最小但完整的 Agent。

    参数:
        llm:           任意 BaseLLM（OpenAIClient / MockLLM / 你自己的实现）
        tools:         Tool 或普通函数的列表（普通函数会被自动包装）
        system_prompt: 系统提示词
        memory:        对话记忆，默认 ConversationMemory()
        max_steps:     最多调用 LLM 的轮数，防止无限循环和烧钱
        callbacks:     钩子列表，用于日志/打印/追踪
        stream:        是否流式输出（token 通过 on_token 回调发出）
        on_max_steps:  "force_answer"（默认，再问一次要答案）或 "raise"
        name:          Agent 名字，多 Agent 时便于区分日志
        skills:        SkillRegistry（可选）。会把技能目录追加到 system prompt，
                       并自动注册 load_skill / read_skill_file 两个工具
        confirm:       人工确认处理器 (tool_name, args) -> Decision|bool，用于 require_confirm 的工具。
                       为 None 时这类工具一律拒绝执行（安全默认值），见 confirm.py
    """

    def __init__(self, llm: BaseLLM, tools: Optional[List[Any]] = None,
                 system_prompt: str = DEFAULT_SYSTEM_PROMPT,
                 memory: Optional[ConversationMemory] = None, max_steps: int = 10,
                 callbacks: Optional[List[Callback]] = None, stream: bool = False,
                 on_max_steps: str = "force_answer", name: str = "agent",
                 skills: Optional[SkillRegistry] = None,
                 confirm: Optional[ConfirmHandler] = None):
        self.llm = llm
        self.tools = tools if isinstance(tools, ToolRegistry) else ToolRegistry(tools)
        self.system_prompt = system_prompt
        self.memory = memory if memory is not None else ConversationMemory()
        self.max_steps = max_steps
        self.hooks = CallbackManager(callbacks or [])
        self.stream = stream
        self.on_max_steps = on_max_steps
        self.name = name
        self.last_steps = 0  # 上一次 run 实际用了几步，便于调试/测试
        self.confirm = confirm
        # 把 LLM 和钩子交给记忆：SummarizingMemory 需要用 LLM 做摘要、发出 on_compress 事件
        if hasattr(self.memory, "attach"):
            self.memory.attach(llm=self.llm, emit=lambda event, *a: self.hooks.emit(event, self, *a))
        # 技能：第 1 层（目录）进 system prompt；第 2/3 层通过工具按需加载
        self.skills = skills
        self.loaded_skills: List[str] = []  # 本会话已加载过的技能（按加载顺序）
        if skills is not None and len(skills):
            self.system_prompt = system_prompt + "\n" + skills.prompt_section()
            for t in skills.as_tools(on_load=self._on_skill_load):
                self.tools.register(t)

    # ------------------------------------------------------------------ 主入口
    def run(self, user_input: str) -> str:
        """处理一条用户输入，返回最终回答。多次调用会共享 memory（即多轮对话）。"""
        self.hooks.emit("on_agent_start", self, user_input)
        self.memory.add({"role": "user", "content": user_input})

        for step in range(1, self.max_steps + 1):
            self.last_steps = step
            self.hooks.emit("on_step_start", self, step)
            resp = self._call_llm(self._build_messages(), use_tools=True)
            self.memory.add(resp.to_message())

            if not resp.tool_calls:  # 停止条件①：模型给出了最终回答
                return self._finish(resp.content or "")

            direct_answer = None
            for tc in resp.tool_calls:
                result = self._run_tool(tc.name, tc.arguments)
                # 每个 tool_call 都必须有对应的 tool 消息，否则下一次请求会 400
                self.memory.add({"role": "tool", "tool_call_id": tc.id, "content": result})
                t = self.tools.get(tc.name)
                if t is not None and t.return_direct and not result.startswith("Error:"):
                    direct_answer = result
            if direct_answer is not None:  # 停止条件②：工具结果即答案
                self.memory.add({"role": "assistant", "content": direct_answer})
                return self._finish(direct_answer)

        # 停止条件③：步数用尽
        if self.on_max_steps == "raise":
            raise MaxStepsExceeded("超过 max_steps=%d 仍未得到最终回答" % self.max_steps)
        messages = self._build_messages() + [{"role": "user", "content": FORCE_ANSWER_PROMPT}]
        resp = self._call_llm(messages, use_tools=False)
        answer = resp.content or "（未能在限定步数内得到答案）"
        self.memory.add({"role": "assistant", "content": answer})
        return self._finish(answer)

    __call__ = run  # agent("你好") 等价于 agent.run("你好")

    def reset(self) -> None:
        """清空对话历史，开始新会话。"""
        self.memory.clear()
        self.loaded_skills = []

    # ------------------------------------------------------------------ 内部步骤
    def _build_messages(self) -> List[Dict[str, Any]]:
        # 每次调用 LLM 前给记忆一个"整理"的机会（基础记忆什么都不做；SummarizingMemory 会按需压缩）
        if hasattr(self.memory, "maybe_compress"):
            self.memory.maybe_compress()
        return [{"role": "system", "content": self.system_prompt}] + self.memory.window()

    def _call_llm(self, messages: List[Dict[str, Any]], use_tools: bool) -> LLMResponse:
        self.hooks.emit("on_llm_start", self, messages)
        on_token = (lambda t: self.hooks.emit("on_token", self, t)) if self.stream else None
        tools = self.tools.schemas() if (use_tools and len(self.tools)) else None
        try:
            resp = self.llm.chat(messages, tools=tools, on_token=on_token)
        except LLMError as e:
            # LLM 错误（已经过重试仍失败）向上抛出：调用方最清楚该怎么处理（提示用户/降级/换模型）
            self.hooks.emit("on_llm_error", self, e)
            raise
        self.hooks.emit("on_llm_end", self, resp)
        return resp

    def _run_tool(self, name: str, arguments: str) -> str:
        self.hooks.emit("on_tool_start", self, name, arguments)
        t0 = time.time()
        decision = self._check_confirm(name, arguments)
        if decision is not None and decision.action == "reject":
            # 不执行，但仍返回一条 tool 消息 —— tool_call 和 tool 结果必须成对出现
            result = rejection_message(name, decision)
        else:
            if decision is not None and decision.action == "edit":
                arguments = json.dumps(decision.arguments, ensure_ascii=False)
            result = self.tools.execute(name, arguments)  # 永不抛异常
            if decision is not None and decision.action == "edit":
                # 历史里的 tool_call 仍是模型原来的参数，必须告诉模型实际执行的是什么
                result = "（注意：用户修改了参数，实际执行参数为 %s）\n%s" % (arguments, result)
        self.hooks.emit("on_tool_end", self, name, result, time.time() - t0)
        return result

    def _check_confirm(self, name: str, arguments: str) -> Optional[Decision]:
        """需要确认时返回 Decision；不需要确认（或参数无法解析，交给 execute 报错）时返回 None。"""
        t = self.tools.get(name)
        if t is None or not t.require_confirm:
            return None
        try:
            args = json.loads(arguments) if arguments.strip() else {}
        except ValueError:
            return None
        if not isinstance(args, dict):
            return None
        if any(p not in args for p in t.parameters.get("required", [])):
            return None  # 缺必填参数：execute 会直接报错让模型修正，没必要打扰用户
        if not t.needs_confirm(args):
            return None
        handler = self.confirm or deny_without_handler
        try:
            decision = normalize(handler(name, args))
        except Exception as e:  # 确认流程本身出错 → 拒绝（安全优先），而不是让 Agent 崩溃
            decision = Decision.reject("确认过程出错: %r" % e)
        self.hooks.emit("on_confirm", self, name, args, decision)
        return decision

    def _on_skill_load(self, name: str) -> None:
        # 注意：即使重复加载也会返回完整正文 —— 早先加载的内容可能已被记忆窗口截断掉了
        if name not in self.loaded_skills:
            self.loaded_skills.append(name)
        self.hooks.emit("on_skill_load", self, name)

    def _finish(self, answer: str) -> str:
        self.hooks.emit("on_agent_end", self, answer)
        return answer

    # ------------------------------------------------------------------ 多 Agent
    def as_tool(self, name: Optional[str] = None, description: str = "") -> Tool:
        """把整个 Agent 包装成一个工具，供"上级" Agent 调用（子 Agent / 多 Agent 协作）。

        每次调用都使用全新的对话记忆，避免不同任务互相污染。
        """
        agent = self

        def delegate(task: str) -> str:
            saved = agent.memory
            agent.memory = ConversationMemory(saved.max_messages, saved.max_chars)
            try:
                return agent.run(task)
            finally:
                agent.memory = saved

        return Tool(
            name=name or self.name,
            description=description or ("把子任务委派给助手 %s，它会独立完成并返回结果" % self.name),
            parameters={"type": "object",
                        "properties": {"task": {"type": "string",
                                                "description": "完整、自包含的任务描述"}},
                        "required": ["task"]},
            func=delegate,
        )


def pretty_messages(messages: List[Dict[str, Any]]) -> str:
    """把消息列表格式化成易读文本（调试用）。"""
    lines = []
    for m in messages:
        body = m.get("content") or ""
        if m.get("tool_calls"):
            body += " " + json.dumps([tc["function"] for tc in m["tool_calls"]], ensure_ascii=False)
        lines.append("%-9s| %s" % (m["role"], body))
    return "\n".join(lines)
