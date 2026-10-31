"""上下文压缩演示：旧对话被 LLM 总结成摘要，最近几轮保留原文。

运行:
    python examples/compression_demo.py          # 离线 MockLLM
    python examples/compression_demo.py --real   # 真实 API（需 OPENAI_API_KEY 等环境变量）

看点：
    1. 压缩前/后的消息列表对比（摘要以"一问一答"的形式出现在最前面）
    2. 第二次压缩时，摘要模型收到的提示里包含了【已有摘要】（增量更新）
    3. 摘要模型挂掉时，Agent 照常回答（退回窗口截断 + on_compress_error 事件）
"""
import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from mini_agent import (Agent, Callback, ConsolePrinter, LLMResponse, LLMServerError, MockLLM,
                        OpenAIClient, SummarizingMemory)
from mini_agent.builtin_tools import calculator

REAL = "--real" in sys.argv
logging.getLogger("mini_agent").setLevel(logging.ERROR)  # 压缩失败的警告由下面的回调打印，这里不重复


def show(title, messages):
    print("    ┌─ %s（%d 条）" % (title, len(messages)))
    for m in messages:
        text = (m.get("content") or "").replace("\n", "⏎ ")
        if m.get("tool_calls"):
            text += " 🔧" + ",".join(tc["function"]["name"] for tc in m["tool_calls"])
        print("    │ %-9s| %s" % (m["role"], text[:90]))
    print("    └─")


class CompressionViewer(Callback):
    """在每一步开始时拍快照；发生压缩时打印 压缩前 / 压缩后 / 摘要。"""

    def on_step_start(self, agent, step):
        self.before = list(agent.memory.messages)

    def on_compress(self, agent, info):
        print("\n  🗜️  ====== 第 %d 次压缩 ======" % info["count"])
        show("压缩前 memory.messages", self.before)
        show("压缩后 发给模型的 window()", agent.memory.window())
        print("    📝 当前摘要：%s" % info["summary"])
        print("    📊 %d 条旧消息被压缩，保留 %d 条原文；估算 token %d → %d\n" % (
            info["compressed_messages"], info["kept_messages"], info["tokens_before"], info["tokens_after"]))

    def on_compress_error(self, agent, error):
        print("  ⚠️  压缩失败：%r → 本轮退回窗口截断，Agent 继续工作" % error)


# ---------------------------------------------------------------- Mock 模型
def summarizer_mock(prompts_seen):
    """假的摘要模型：记录收到的提示，并"总结"出固定内容。"""
    def first(messages, tools):
        prompts_seen.append(messages[-1]["content"])
        return LLMResponse(content="用户叫小明，正在做一个 AI 短剧平台；助手已算出 120 集×90 秒=180 分钟。")

    def second(messages, tools):
        prompts_seen.append(messages[-1]["content"])
        return LLMResponse(content="用户叫小明，正在做 AI 短剧平台；120 集×90 秒=180 分钟；平台月预算 3 万元。")
    return MockLLM([first, second])


def agent_mock():
    def recall(messages, tools):
        # 此时 "我叫小明""预算 3 万" 的原文早已被压缩，只能从摘要里读到
        summary = next((m["content"] for m in messages if "【历史对话摘要】" in (m.get("content") or "")), "")
        return LLMResponse(content="根据摘要：%s" % summary.split("\n", 1)[-1])
    return MockLLM([
        "你好小明！AI 短剧平台很有前景。",
        MockLLM.call("calculator", expression="120*90/60"), "一共 180 分钟。",
        "好的，记下了：月预算 3 万元。",
        "建议先做 10 集试水。",
        "逆袭、甜宠题材在短剧里最稳。",
        recall,
    ])


TURNS = ["我叫小明，在做一个 AI 短剧平台", "120 集、每集 90 秒，一共多少分钟？",
         "平台每月预算 3 万元", "第一季应该做多少集？", "什么题材比较稳？", "总结一下：我叫什么、做什么、预算多少？"]


def demo_compress():
    print("=" * 70 + "\n① 正常压缩（trigger_messages=6, keep_turns=1）\n" + "=" * 70)
    prompts_seen = []
    memory = SummarizingMemory(llm=OpenAIClient() if REAL else summarizer_mock(prompts_seen),
                               trigger_messages=6, keep_turns=1)
    agent = Agent(OpenAIClient() if REAL else agent_mock(), tools=[calculator], memory=memory,
                  callbacks=[ConsolePrinter(), CompressionViewer()])
    for text in TURNS:
        print("你> " + text)
        print("助手> " + agent.run(text))
    if prompts_seen:
        print("\n🔎 第 2 次压缩时，摘要模型收到的提示（注意【已有摘要】部分）：")
        print("    " + prompts_seen[-1].replace("\n", "\n    "))


def demo_failure():
    print("\n" + "=" * 70 + "\n② 摘要模型故障：不影响回答\n" + "=" * 70)
    broken = MockLLM([LLMServerError("summarizer down", 503)] * 3, max_retries=0)
    memory = SummarizingMemory(llm=broken, trigger_messages=2, keep_turns=1, max_messages=3)
    agent = Agent(MockLLM(["第一轮回答", "第二轮回答（照常工作）"]), memory=memory,
                  callbacks=[CompressionViewer()])
    for text in ["你好", "继续"]:
        print("你> " + text)
        print("助手> " + agent.run(text))
    show("退回窗口截断后发给模型的 window()", memory.window())


if __name__ == "__main__":
    demo_compress()
    if not REAL:
        demo_failure()
