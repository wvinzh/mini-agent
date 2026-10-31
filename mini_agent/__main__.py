"""命令行聊天入口：python -m mini_agent  [--mock] [--stream] [-v] ...

内置命令:  /tools 查看工具  /skills 查看技能  /history 查看历史  /reset 清空对话  /exit 退出
"""
import argparse
import logging
import os
import re
import sys

from . import (Agent, ConsoleConfirmer, ConsolePrinter, ConversationMemory, JSONStore, LLMError, MockLLM,
               OpenAIClient, SkillRegistry, SummarizingMemory, auto_approve)
from .agent import pretty_messages
from .builtin_tools import BUILTIN_TOOLS
from .llm import LLMResponse


def rule_based_mock() -> MockLLM:
    """没有 API Key 时使用的"规则版假模型"：识别算式/时间问题并调用工具，便于体验流程。"""

    def brain(messages, tools):
        last = messages[-1]
        if last["role"] == "tool":  # 已经拿到工具结果 -> 给出回答
            return LLMResponse(content="（Mock）工具返回：%s" % last["content"])
        text = last.get("content") or ""
        expr = re.search(r"[\d\.\s\(\)\+\-\*/%^]{3,}", text)
        if "时间" in text or "几点" in text or "time" in text.lower():
            return MockLLM.call("current_time")
        if expr and re.search(r"\d\s*[\+\-\*/%^]\s*\d", expr.group()):
            return MockLLM.call("calculator", expression=expr.group().strip())
        return LLMResponse(content="（Mock）你说的是：%s。试试问我 “1+2*3 等于多少” 或 “现在几点”。" % text)

    return MockLLM([brain] * 10000)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="mini_agent", description="mini_agent 命令行聊天")
    p.add_argument("--mock", action="store_true", help="使用离线假模型（无需 API Key）")
    p.add_argument("--model", help="模型名，默认读环境变量 MODEL")
    p.add_argument("--base-url", help="接口地址，默认读环境变量 OPENAI_BASE_URL")
    p.add_argument("--stream", action="store_true", help="流式输出")
    p.add_argument("--max-steps", type=int, default=10)
    p.add_argument("--timeout", type=float, default=60)
    p.add_argument("--history-file", help="把对话历史持久化到此 JSON 文件")
    p.add_argument("--memory-file", help="启用长期记忆(remember/recall 工具)，存到此 JSON 文件")
    p.add_argument("--skills-dir", action="append",
                   help="技能目录（含若干 <技能名>/SKILL.md），可重复指定多个")
    p.add_argument("--compress", action="store_true",
                   help="启用上下文压缩：历史过长时用 LLM 把旧对话总结成摘要")
    p.add_argument("--compress-tokens", type=int, default=3000, help="触发压缩的估算 token 数")
    p.add_argument("--auto-approve", action="store_true",
                   help="需要人工确认的工具（如 write_file）全部自动批准；默认在终端询问")
    p.add_argument("-v", "--verbose", action="store_true", help="打印调试日志（含重试信息）")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.WARNING,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    if args.mock or not os.getenv("OPENAI_API_KEY"):
        if not args.mock:
            print("⚠️  未设置 OPENAI_API_KEY，自动使用离线 Mock 模型（加 --mock 可隐藏此提示）。")
        llm = rule_based_mock()
    else:
        llm = OpenAIClient(model=args.model, base_url=args.base_url, timeout=args.timeout)
        print("模型: %s @ %s" % (llm.model, llm.base_url))

    tools = list(BUILTIN_TOOLS)
    if args.memory_file:
        tools += JSONStore(args.memory_file).as_tools()
    skills = SkillRegistry(args.skills_dir) if args.skills_dir else None
    if args.compress:
        memory = SummarizingMemory(trigger_tokens=args.compress_tokens, path=args.history_file)
    else:
        memory = ConversationMemory(path=args.history_file)
    agent = Agent(llm, tools=tools, max_steps=args.max_steps, stream=args.stream, skills=skills,
                  memory=memory, confirm=auto_approve if args.auto_approve else ConsoleConfirmer(),
                  callbacks=[ConsolePrinter(stream_tokens=args.stream)])
    print("mini_agent 已就绪，工具: %s" % ", ".join(agent.tools.names()))
    if skills:
        print("技能: %s" % (", ".join(skills.names()) or "（目录中没有发现技能）"))
    print("输入 /exit 退出，/reset 清空，/history 查看历史，/tools 查看工具，/skills 查看技能")

    while True:
        try:
            text = input("\n你> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not text:
            continue
        if text in ("/exit", "/quit"):
            return 0
        if text == "/reset":
            agent.reset()
            print("（已清空对话）")
            continue
        if text == "/history":
            if getattr(agent.memory, "summary", ""):
                print("【摘要】" + agent.memory.summary)
            print(pretty_messages(agent.memory.messages) or "（空）")
            continue
        if text == "/skills":
            print(skills.catalog() if skills else "（未启用技能，使用 --skills-dir 指定目录）")
            print("已加载:", agent.loaded_skills)
            continue
        if text == "/tools":
            for t in agent.tools.schemas():
                print("- %s: %s" % (t["function"]["name"], t["function"]["description"]))
            continue
        try:
            if args.stream:
                print("助手> ", end="", flush=True)
                agent.run(text)
            else:
                print("助手> " + agent.run(text))
        except LLMError as e:
            # 已经重试过仍失败：告诉用户原因，但不退出程序
            print("\n❌ 调用模型失败: %s: %s" % (type(e).__name__, e), file=sys.stderr)
        except KeyboardInterrupt:
            print("\n（已中断本轮）")


if __name__ == "__main__":
    sys.exit(main())
