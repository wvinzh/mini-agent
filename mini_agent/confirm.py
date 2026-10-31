"""人工确认（Human-in-the-loop）：危险工具执行前先问人。

用法：
    @tool(require_confirm=True)                       # 每次都要确认
    def delete_file(path: str) -> str: ...

    @tool(require_confirm=lambda args: args["amount"] > 100)   # 按参数决定：大额才确认
    def transfer(to: str, amount: float) -> str: ...

    agent = Agent(llm, tools=[...], confirm=ConsoleConfirmer())   # 终端里问 y/n
    agent = Agent(llm, tools=[...], confirm=auto_approve)          # 全部放行（测试/可信环境）

确认处理器（confirm handler）就是一个函数：
    handler(tool_name: str, arguments: dict) -> Decision | bool
返回 True/False 也可以，会被自动转换成 approve/reject。

决策（Decision）三种：
    approve            照常执行
    reject(reason)     不执行，给模型返回 "Error: 用户拒绝执行…原因…"，模型可据此调整
    edit(new_args)     用修改后的参数执行（例如把写入路径改到安全目录）

安全默认值：没有配置 confirm 处理器时，需要确认的工具一律【拒绝】。
    —— "忘了配置"不应该等于"全部放行"。CLI 默认使用终端确认，--auto-approve 才全部放行。
"""
import json
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Union

APPROVE, REJECT, EDIT = "approve", "reject", "edit"


@dataclass
class Decision:
    action: str = APPROVE                  # approve / reject / edit
    reason: str = ""                       # 拒绝理由（会转告模型）
    arguments: Optional[Dict[str, Any]] = field(default=None)  # edit 时的新参数

    @classmethod
    def approve(cls) -> "Decision":
        return cls(APPROVE)

    @classmethod
    def reject(cls, reason: str = "") -> "Decision":
        return cls(REJECT, reason=reason)

    @classmethod
    def edit(cls, arguments: Dict[str, Any], reason: str = "") -> "Decision":
        return cls(EDIT, reason=reason, arguments=arguments)


ConfirmHandler = Callable[[str, Dict[str, Any]], Union[Decision, bool]]


def normalize(result: Any) -> Decision:
    """把处理器的返回值统一成 Decision；无法识别的返回值按"拒绝"处理（安全优先）。"""
    if isinstance(result, Decision):
        if result.action == EDIT and not isinstance(result.arguments, dict):
            return Decision.reject("确认处理器返回了 edit 但没有给出合法的新参数")
        return result if result.action in (APPROVE, REJECT, EDIT) else Decision.reject("未知的确认决策")
    if result is True:
        return Decision.approve()
    if result is False:
        return Decision.reject()
    return Decision.reject("确认处理器返回了无法识别的结果: %r" % (result,))


def auto_approve(name: str, arguments: Dict[str, Any]) -> Decision:
    """全部放行（测试、或完全可信的环境）。"""
    return Decision.approve()


def auto_deny(name: str, arguments: Dict[str, Any]) -> Decision:
    """全部拒绝（只读模式、演示"被拒绝后模型如何应对"）。"""
    return Decision.reject("当前为只读模式，不允许执行此操作")


def deny_without_handler(name: str, arguments: Dict[str, Any]) -> Decision:
    """Agent 未配置 confirm 时的默认行为。"""
    return Decision.reject("未配置人工确认处理器，出于安全考虑默认拒绝（可在 Agent(confirm=...) 中配置）")


class ConsoleConfirmer:
    """在终端里询问用户。输入：
        y / yes / 是      批准
        n / 回车          拒绝（回车默认拒绝 —— 安全优先）
        e                 修改参数（再输入一行 JSON）
        其他任意文字      视为拒绝理由，转告模型
    """

    def __init__(self, input_fn: Callable[[str], str] = input, print_fn: Callable[..., None] = print):
        self.input = input_fn  # 可注入，便于测试
        self.print = print_fn

    def __call__(self, name: str, arguments: Dict[str, Any]) -> Decision:
        shown = json.dumps(arguments, ensure_ascii=False, indent=2)
        if len(shown) > 1500:
            shown = shown[:1500] + "\n…（参数过长，已截断显示）"
        self.print("\n⚠️  模型请求执行需要确认的工具: %s\n%s" % (name, shown))
        try:
            ans = self.input("允许执行吗？[y]允许 [n/回车]拒绝 [e]修改参数 或直接输入拒绝理由: ").strip()
        except (EOFError, KeyboardInterrupt):
            return Decision.reject("用户中断了确认")
        if ans.lower() in ("y", "yes", "是", "好", "ok"):
            return Decision.approve()
        if ans.lower() == "e":
            try:
                new_args = json.loads(self.input("请输入新的参数 JSON: "))
            except (ValueError, EOFError, KeyboardInterrupt) as e:
                return Decision.reject("用户修改参数失败: %s" % e)
            if not isinstance(new_args, dict):
                return Decision.reject("修改后的参数必须是 JSON 对象")
            return Decision.edit(new_args, reason="用户修改了参数")
        if ans.lower() in ("", "n", "no", "否", "不"):
            return Decision.reject()
        return Decision.reject(ans)


def rejection_message(name: str, decision: Decision) -> str:
    """拒绝时回给模型的 tool 消息。以 "Error:" 开头，和其他工具错误保持一致。"""
    reason = decision.reason or "用户未说明原因"
    return ("Error: 用户拒绝执行工具 %s。原因：%s。请不要用相同参数重试；"
            "可以根据原因调整方案，或向用户说明情况/询问下一步。" % (name, reason))
