"""真实 API 演示：连接任意 OpenAI 兼容服务。

先设置环境变量（示例）:
    export OPENAI_BASE_URL=https://api.openai.com/v1      # 或 DeepSeek / 通义 / 中转站 / http://localhost:11434/v1 (Ollama)
    export OPENAI_API_KEY=sk-xxx
    export MODEL=gpt-4o-mini
运行:
    python examples/real_api_demo.py
"""
import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from mini_agent import (Agent, ConsolePrinter, LLMAuthError, LLMBadRequestError, LLMError,
                        OpenAIClient, TraceRecorder)
from mini_agent.builtin_tools import BUILTIN_TOOLS

# WARNING 级别即可看到"第 n 次重试"的日志；想看更多细节改成 DEBUG
logging.basicConfig(level=logging.WARNING, format="[%(levelname)s] %(name)s: %(message)s")


def main():
    if not os.getenv("OPENAI_API_KEY"):
        print("请先设置 OPENAI_API_KEY（以及 OPENAI_BASE_URL、MODEL）。没有 Key 可以先跑 examples/offline_demo.py")
        return 1

    llm = OpenAIClient(
        timeout=60,          # 单次请求超时（秒）
        max_retries=4,       # 网络错误 / 429 / 5xx / 坏响应 最多重试 4 次
        backoff_base=1.0,    # 等待 1s, 2s, 4s, 8s（带随机抖动）
        max_backoff=20.0,
    )
    trace = TraceRecorder()
    agent = Agent(llm, tools=BUILTIN_TOOLS, stream=True, max_steps=8,
                  callbacks=[ConsolePrinter(stream_tokens=True), trace])
    print("模型: %s @ %s\n" % (llm.model, llm.base_url))

    question = sys.argv[1] if len(sys.argv) > 1 else "现在几点？另外帮我算一下 (2**10 + sqrt(144)) / 4"
    print("你> " + question)
    try:
        agent.run(question)
    except LLMAuthError as e:
        print("\n🔑 鉴权失败，检查 API Key：", e)
    except LLMBadRequestError as e:
        print("\n📝 请求参数有误（常见：模型名写错 / 该模型不支持工具调用）：", e)
    except LLMError as e:
        print("\n🌐 重试后仍失败（%s）：%s" % (type(e).__name__, e))
    else:
        trace.dump("trace.json")
        print("\n完整事件轨迹已保存到 trace.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
