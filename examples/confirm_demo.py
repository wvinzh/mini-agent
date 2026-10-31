"""人工确认（Human-in-the-loop）演示。

运行:
    python examples/confirm_demo.py                # 离线：用"脚本化的人"自动输入 y/n/e
    python examples/confirm_demo.py --interactive  # 离线模型，但由你亲自在终端输入确认
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from mini_agent import Agent, ConsolePrinter, ConsoleConfirmer, MockLLM, pretty_messages, tool
from mini_agent.builtin_tools import write_file

INTERACTIVE = "--interactive" in sys.argv


def scripted_human(answers):
    """模拟一个坐在终端前的人：依次"输入"预设的回答，并把输入回显出来。"""
    it = iter(answers)

    def fake_input(prompt):
        ans = next(it)
        print(prompt + ans)
        return ans
    return input if INTERACTIVE else fake_input


@tool(require_confirm=lambda args: args.get("amount", 0) > 100)  # 按参数决定：超过 100 元才需要确认
def transfer(to: str, amount: float) -> str:
    """转账（演示用，不会真的转）。

    Args:
        to: 收款人
        amount: 金额（元）
    """
    return "已向 %s 转账 %.2f 元" % (to, amount)


def title(s):
    print("\n" + "=" * 70 + "\n" + s + "\n" + "=" * 70)


def demo_default_deny():
    title("① 没有配置确认处理器：需要确认的工具默认拒绝（安全默认值）")
    llm = MockLLM([MockLLM.call("write_file", path="notes.txt", content="hi"),
                   "抱歉，写文件的操作被拒绝了：当前没有开启人工确认。请配置后再试。"])
    agent = Agent(llm, tools=[write_file], callbacks=[ConsolePrinter()])
    print("助手> " + agent.run("把 hi 写进 notes.txt"))


def demo_console():
    title("② 终端确认：拒绝(附理由) → 模型换方案 → 批准 → 修改参数")
    llm = MockLLM([
        MockLLM.call("write_file", path="config/prod.yaml", content="debug: true"),
        # 模型读到了拒绝理由（"改 dev 的"），换了一个路径重试
        MockLLM.call("write_file", path="config/dev.yaml", content="debug: true"),
        MockLLM.call("write_file", path="notes/todo.md", content="- 上线前检查"),
        "完成：已写入开发配置；待办写入了你指定的 notes/2026-todo.md。",
    ])
    confirmer = ConsoleConfirmer(input_fn=scripted_human([
        "生产配置不能动，改 dev 的",                                           # 拒绝并说明理由
        "y",                                                                   # 批准
        "e", '{"path": "notes/2026-todo.md", "content": "- 上线前检查"}',      # 修改参数后执行
    ]))
    agent = Agent(llm, tools=[write_file], confirm=confirmer, callbacks=[ConsolePrinter()])
    print("助手> " + agent.run("打开调试开关，再记一条待办"))
    print("\n📜 消息序列（每个 tool_call 都有对应的 tool 结果，被拒绝的也不例外）：")
    print(pretty_messages(agent.memory.messages))
    root = os.environ["AGENT_WORKDIR"]
    print("\n📁 实际写入的文件:", sorted(os.path.relpath(os.path.join(r, f), root)
                                   for r, _, fs in os.walk(root) for f in fs))


def demo_predicate():
    title("③ 按参数决定是否确认：小额直接执行，大额才问人")
    llm = MockLLM([MockLLM.call("transfer", to="小王", amount=50),
                   MockLLM.call("transfer", to="小李", amount=5000),
                   "给小王的 50 元已转；给小李的 5000 元被你拒绝了。"])
    agent = Agent(llm, tools=[transfer], callbacks=[ConsolePrinter()],
                  confirm=ConsoleConfirmer(input_fn=scripted_human(["n"])))
    print("助手> " + agent.run("给小王转 50，给小李转 5000"))


if __name__ == "__main__":
    os.environ["AGENT_WORKDIR"] = tempfile.mkdtemp(prefix="confirm_demo_")
    demo_default_deny()
    demo_console()
    demo_predicate()
