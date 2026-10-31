"""离线演示：不需要 API Key，用 MockLLM 把框架的每个能力都走一遍。

运行:  python examples/offline_demo.py
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from mini_agent import (Agent, ConsolePrinter, ConversationMemory, JSONStore, LLMServerError,
                        MockLLM, TraceRecorder, auto_approve, tool)
from mini_agent.builtin_tools import calculator, current_time, read_file, write_file


def title(s):
    print("\n" + "=" * 60 + "\n" + s + "\n" + "=" * 60)


# ---------------------------------------------------------------- 1. 自定义工具
@tool
def get_weather(city: str, unit: str = "celsius") -> dict:
    """查询城市天气（演示用，返回假数据）。

    Args:
        city: 城市名，如 "北京"
        unit: 温度单位 celsius 或 fahrenheit
    """
    fake = {"北京": 12, "上海": 18}
    if city not in fake:
        raise ValueError("暂不支持城市: %s" % city)
    return {"city": city, "temp": fake[city], "unit": unit, "weather": "晴"}


def demo_tool_schema():
    title("1. @tool 自动生成的 JSON Schema")
    import json
    print(json.dumps(get_weather.schema(), ensure_ascii=False, indent=2))


def demo_react_loop():
    title("2. ReAct 循环：多步工具调用 + 工具出错后自我修正")
    llm = MockLLM([
        MockLLM.call("get_weather", city="杭州"),            # 第1步：工具会报错
        MockLLM.call("get_weather", city="上海"),            # 第2步：模型"修正"了参数
        MockLLM.call("calculator", expression="18 * 9 / 5 + 32"),  # 第3步：换算华氏度
        "上海今天晴，18°C（约 64.4°F）。杭州暂不支持查询。",   # 第4步：最终回答
    ])
    trace = TraceRecorder()
    agent = Agent(llm, tools=[get_weather, calculator], callbacks=[ConsolePrinter(), trace])
    print("最终回答:", agent.run("杭州和上海天气怎么样？顺便换算成华氏度"))
    print("用了 %d 步；事件序列: %s" % (agent.last_steps, " → ".join(trace.names())))


def demo_retry():
    title("3. 网络抖动：两次 503 后自动重试成功（Mock 中不真正 sleep）")
    import logging
    logging.basicConfig(level=logging.WARNING, format="  [log] %(message)s")
    llm = MockLLM([LLMServerError("Bad Gateway", 503), LLMServerError("Bad Gateway", 503),
                   "重试成功后的回答 ✅"], max_retries=3)
    print("最终回答:", Agent(llm).run("你好"))


def demo_memory_and_stream():
    title("4. 多轮记忆 + 流式输出 + 历史持久化")
    tmp = tempfile.mkdtemp()
    path = os.path.join(tmp, "history.json")
    llm = MockLLM(["你好小明，很高兴认识你！",
                   # callable 剧本：可以读取模型实际收到的消息，证明历史确实被带上了
                   lambda msgs, tools: "你叫小明呀（我收到了 %d 条历史消息，第一条是：%s）"
                   % (len(msgs) - 1, msgs[1]["content"])])
    agent = Agent(llm, tools=[current_time], stream=True,
                  memory=ConversationMemory(path=path), callbacks=[ConsolePrinter(stream_tokens=True)])
    print("你> 我叫小明");     agent.run("我叫小明")
    print("你> 我叫什么？");   agent.run("我叫什么？")
    print("历史已保存到 %s，共 %d 条消息" % (path, len(ConversationMemory(path=path).messages)))


def demo_long_term_memory_and_files():
    title("5. 长期记忆(JSONStore) + 沙箱文件工具")
    tmp = tempfile.mkdtemp()
    os.environ["AGENT_WORKDIR"] = tmp
    store = JSONStore(os.path.join(tmp, "notes.json"))
    llm = MockLLM([
        MockLLM.call("remember", key="喜欢的语言", value="Python"),
        MockLLM.call("write_file", path="hello.txt", content="Hello from agent"),
        MockLLM.call("read_file", path="../../etc/passwd"),    # 越权访问会被拒绝
        MockLLM.call("read_file", path="hello.txt"),
        "已记住你喜欢 Python，并创建了 hello.txt。",
    ])
    # write_file 需要人工确认；演示里自动批准（人工确认的完整演示见 examples/confirm_demo.py）
    agent = Agent(llm, tools=[write_file, read_file] + store.as_tools(), callbacks=[ConsolePrinter()],
                  confirm=auto_approve)
    print("最终回答:", agent.run("记住我喜欢 Python，再建个 hello.txt"))
    print("长期记忆内容:", store.data)


if __name__ == "__main__":
    demo_tool_schema()
    demo_react_loop()
    demo_retry()
    demo_memory_and_stream()
    demo_long_term_memory_and_files()
    print("\n🎉 离线演示全部完成")
