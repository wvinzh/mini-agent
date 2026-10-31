"""多 Agent 演示：①子 Agent 作为工具（委派）  ②先规划再执行（Plan-and-Execute）。

运行:
    python examples/multi_agent_demo.py          # 离线 Mock
    python examples/multi_agent_demo.py --real   # 使用真实 API（需设置环境变量）
"""
import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from mini_agent import Agent, ConsolePrinter, MockLLM, OpenAIClient
from mini_agent.builtin_tools import calculator

REAL = "--real" in sys.argv


def llm_or_mock(script):
    """真实模式下所有 Agent 共用一个客户端；Mock 模式下每个 Agent 各有自己的剧本。"""
    return OpenAIClient() if REAL else MockLLM(script)


# ====================================================================== ① 子 Agent 作为工具
def demo_delegation():
    print("=" * 60 + "\n① 主管 Agent 把子任务委派给专家 Agent\n" + "=" * 60)
    math_expert = Agent(
        llm_or_mock([MockLLM.call("calculator", expression="1200 * 1.05 ** 3"), "约 1389.15 元"]),
        tools=[calculator], name="math_expert",
        system_prompt="你是数学专家，必须用 calculator 工具计算，回答只给出结果。",
        callbacks=[ConsolePrinter()])
    writer = Agent(
        llm_or_mock(["存 1200 元、年化 5%，三年后约 1389 元——复利让钱自己长大。"]),
        name="writer", system_prompt="你是文案专家，把给定内容改写成一句生动的中文。",
        callbacks=[ConsolePrinter()])

    boss = Agent(
        llm_or_mock([
            MockLLM.call("math_expert", task="计算 1200 元本金，年利率 5% 复利 3 年后的金额"),
            lambda msgs, tools: MockLLM.call("writer", task="把这个结果写成一句话：" + msgs[-1]["content"]),
            lambda msgs, tools: msgs[-1]["content"],
        ]),
        tools=[math_expert.as_tool(description="数学专家：负责所有计算"),
               writer.as_tool(description="文案专家：负责润色表达")],
        name="boss", system_prompt="你是项目主管，先找数学专家算，再找文案专家润色，最后输出文案。",
        callbacks=[ConsolePrinter()])
    print("最终回答:", boss.run("1200 元存 3 年，年化 5% 复利，最后有多少？写成一句话"))


# ====================================================================== ② 规划 + 执行
PLAN_PROMPT = ("把用户任务拆成 2~5 个可以独立执行的步骤。"
               "只输出 JSON 数组，每个元素是一个字符串，不要输出其他内容。")


def plan_and_execute(planner_llm, executor: Agent, task: str) -> str:
    """最简单的 Plan-and-Execute：
        1) 让模型输出步骤列表（JSON）
        2) 逐步交给执行 Agent（它共享记忆，所以后一步能看到前一步结果）
        3) 最后让执行 Agent 汇总
    """
    resp = planner_llm.chat([{"role": "system", "content": PLAN_PROMPT},
                             {"role": "user", "content": task}])
    text = resp.content or "[]"
    m = re.search(r"\[.*\]", text, re.S)  # 模型常在 JSON 外面包 ```json，这里宽松提取
    try:
        steps = json.loads(m.group()) if m else []
    except ValueError:
        steps = []
    steps = steps or [task]  # 规划失败就退化为直接执行
    print("📋 计划:", steps)
    for i, step in enumerate(steps, 1):
        print("▶ 步骤 %d: %s" % (i, step))
        print("  结果:", executor.run(step))
    return executor.run("请基于以上所有步骤的结果，给出对原始任务「%s」的最终回答。" % task)


def demo_plan_and_execute():
    print("\n" + "=" * 60 + "\n② 先规划再执行\n" + "=" * 60)
    planner = llm_or_mock(['["计算 3 个苹果的价格(单价 4.5)", "计算 2 瓶牛奶的价格(单价 6.8)", "求总价"]'])
    executor = Agent(llm_or_mock([
        MockLLM.call("calculator", expression="3 * 4.5"), "13.5 元",
        MockLLM.call("calculator", expression="2 * 6.8"), "13.6 元",
        MockLLM.call("calculator", expression="13.5 + 13.6"), "27.1 元",
        "3 个苹果 13.5 元 + 2 瓶牛奶 13.6 元，总价 27.1 元。",
    ]), tools=[calculator], name="executor", callbacks=[ConsolePrinter()])
    print("最终回答:", plan_and_execute(planner, executor, "买 3 个苹果(4.5元/个)和 2 瓶牛奶(6.8元/瓶)一共多少钱？"))


if __name__ == "__main__":
    demo_delegation()
    demo_plan_and_execute()
