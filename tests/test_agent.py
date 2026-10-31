"""Agent 主循环测试（全部离线，使用 MockLLM）。"""
import unittest

from mini_agent import (Agent, Callback, LLMAuthError, MaxStepsExceeded, MockLLM, TraceRecorder, tool)


@tool
def add(a: int, b: int) -> int:
    """两数相加"""
    return a + b


@tool(return_direct=True)
def final_report(text: str) -> str:
    """直接输出报告"""
    return "报告:" + text


class TestAgentLoop(unittest.TestCase):
    def test_tool_then_answer(self):
        llm = MockLLM([MockLLM.call("add", a=2, b=3), "答案是 5"])
        trace = TraceRecorder()
        agent = Agent(llm, tools=[add], callbacks=[trace])
        self.assertEqual(agent.run("2+3?"), "答案是 5")
        self.assertEqual(agent.last_steps, 2)
        # 第二次调用 LLM 时，模型应能看到工具结果
        second = llm.calls[1]
        self.assertEqual(second[0]["role"], "system")
        self.assertEqual(second[-1], {"role": "tool", "tool_call_id": second[-2]["tool_calls"][0]["id"],
                                      "content": "5"})
        self.assertEqual(trace.names(), ["agent_start", "llm_start", "llm_end", "tool_start", "tool_end",
                                         "llm_start", "llm_end", "agent_end"])

    def test_multi_turn_memory(self):
        llm = MockLLM(["记住了", lambda m, t: "你说过: " + m[1]["content"]])
        agent = Agent(llm)
        agent.run("我叫小红")
        self.assertEqual(agent.run("我叫什么"), "你说过: 我叫小红")
        agent.reset()
        self.assertEqual(agent.memory.messages, [])

    def test_tool_error_fed_back(self):
        llm = MockLLM([MockLLM.call("add", a=1), "修正后完成"])
        agent = Agent(llm, tools=[add])
        self.assertEqual(agent.run("x"), "修正后完成")
        self.assertIn("缺少必填参数", llm.calls[1][-1]["content"])

    def test_max_steps_force_answer(self):
        llm = MockLLM([MockLLM.call("add", a=1, b=1)] * 3 + ["被迫总结"])
        agent = Agent(llm, tools=[add], max_steps=3)
        self.assertEqual(agent.run("loop"), "被迫总结")
        self.assertEqual(len(llm.calls), 4)
        self.assertIn("最大推理步数", llm.calls[-1][-1]["content"])

    def test_max_steps_raise(self):
        llm = MockLLM([MockLLM.call("add", a=1, b=1)] * 5)
        with self.assertRaises(MaxStepsExceeded):
            Agent(llm, tools=[add], max_steps=2, on_max_steps="raise").run("loop")

    def test_return_direct(self):
        llm = MockLLM([MockLLM.call("final_report", text="ok")])
        agent = Agent(llm, tools=[final_report])
        self.assertEqual(agent.run("x"), "报告:ok")
        self.assertEqual(len(llm.calls), 1)

    def test_llm_error_propagates_and_hook_fires(self):
        trace = TraceRecorder()
        agent = Agent(MockLLM([LLMAuthError("bad key", 401)]), callbacks=[trace])
        with self.assertRaises(LLMAuthError):
            agent.run("x")
        self.assertEqual(trace.names()[-1], "llm_error")

    def test_broken_callback_does_not_break_agent(self):
        class Broken(Callback):
            def on_llm_start(self, agent, messages):
                raise RuntimeError("回调炸了")
        self.assertEqual(Agent(MockLLM(["fine"]), callbacks=[Broken()]).run("x"), "fine")

    def test_streaming_tokens(self):
        got = []

        class Collect(Callback):
            def on_token(self, agent, token):
                got.append(token)
        Agent(MockLLM(["你好"]), stream=True, callbacks=[Collect()]).run("x")
        self.assertEqual("".join(got), "你好")

    def test_sub_agent_as_tool(self):
        child = Agent(MockLLM([MockLLM.call("add", a=1, b=2), "子结果3"]), tools=[add], name="child")
        parent = Agent(MockLLM([MockLLM.call("child", task="算 1+2"), lambda m, t: "父: " + m[-1]["content"]]),
                       tools=[child.as_tool(description="会算数")])
        self.assertEqual(parent.run("go"), "父: 子结果3")
        self.assertEqual(child.memory.messages, [])  # 子 Agent 每次使用独立记忆


class TestCLI(unittest.TestCase):
    def test_cli_mock_session(self):
        import io
        from contextlib import redirect_stdout
        from unittest import mock
        from mini_agent.__main__ import main
        inputs = iter(["1+2*3 等于多少", "/history", "/exit"])
        buf = io.StringIO()
        with mock.patch("builtins.input", lambda _: next(inputs)), redirect_stdout(buf):
            self.assertEqual(main(["--mock"]), 0)
        self.assertIn("工具返回：7", buf.getvalue())

    def test_cli_compress_and_auto_approve_flags(self):
        import io
        from contextlib import redirect_stdout
        from unittest import mock
        from mini_agent import SummarizingMemory, auto_approve
        from mini_agent.__main__ import main
        created = {}
        real_init = __import__("mini_agent").Agent.__init__

        def spy(agent, *a, **kw):
            real_init(agent, *a, **kw)
            created["agent"] = agent
        inputs = iter(["/exit"])
        with mock.patch("builtins.input", lambda _: next(inputs)), redirect_stdout(io.StringIO()), \
                mock.patch("mini_agent.agent.Agent.__init__", spy):
            self.assertEqual(main(["--mock", "--compress", "--auto-approve"]), 0)
        self.assertIsInstance(created["agent"].memory, SummarizingMemory)
        self.assertIs(created["agent"].confirm, auto_approve)


if __name__ == "__main__":
    unittest.main()
