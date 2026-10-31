"""人工确认测试：批准 / 拒绝 / 修改参数 / 默认拒绝 / 按参数确认 / 消息配对。"""
import json
import unittest

from mini_agent import (Agent, ConsoleConfirmer, Decision, MockLLM, TraceRecorder, auto_approve, auto_deny,
                        tool)
from mini_agent.builtin_tools import write_file
from mini_agent.confirm import normalize

LOG = []


@tool(require_confirm=True)
def danger(path: str) -> str:
    """危险操作"""
    LOG.append(path)
    return "done:" + path


@tool(require_confirm=lambda a: a.get("amount", 0) > 100)
def pay(amount: int) -> str:
    """付款"""
    LOG.append(amount)
    return "paid %d" % amount


def run(confirm, calls=None, tools=(danger,)):
    calls = calls or [MockLLM.call("danger", path="/x")]
    llm = MockLLM(list(calls) + ["final"])
    trace = TraceRecorder()
    agent = Agent(llm, tools=list(tools), confirm=confirm, callbacks=[trace])
    agent.run("go")
    return agent, trace


def assert_paired(test, messages):
    called = [tc["id"] for m in messages for tc in m.get("tool_calls") or []]
    answered = [m["tool_call_id"] for m in messages if m["role"] == "tool"]
    test.assertEqual(called, answered)


class TestConfirm(unittest.TestCase):
    def setUp(self):
        LOG.clear()

    def test_approve(self):
        agent, trace = run(auto_approve)
        self.assertEqual(LOG, ["/x"])
        self.assertEqual(agent.memory.messages[2]["content"], "done:/x")
        self.assertEqual([e["action"] for e in trace.events if e["event"] == "confirm"], ["approve"])

    def test_reject_with_reason_keeps_pairing(self):
        agent, trace = run(lambda n, a: Decision.reject("太危险"))
        self.assertEqual(LOG, [])
        tool_msg = agent.memory.messages[2]
        self.assertTrue(tool_msg["content"].startswith("Error: 用户拒绝执行工具 danger"))
        self.assertIn("太危险", tool_msg["content"])
        assert_paired(self, agent.memory.messages)

    def test_edit_args(self):
        agent, _ = run(lambda n, a: Decision.edit({"path": "/safe"}))
        self.assertEqual(LOG, ["/safe"])
        self.assertIn('"/safe"', agent.memory.messages[2]["content"])  # 告知模型实际参数
        self.assertIn("done:/safe", agent.memory.messages[2]["content"])

    def test_default_without_handler_is_deny(self):
        agent, _ = run(None)
        self.assertEqual(LOG, [])
        self.assertIn("未配置人工确认处理器", agent.memory.messages[2]["content"])

    def test_bool_and_bad_results_and_handler_crash(self):
        run(lambda n, a: True)
        self.assertEqual(LOG, ["/x"])
        LOG.clear()
        for handler in (lambda n, a: False, lambda n, a: "yes?", auto_deny,
                        lambda n, a: 1 / 0, lambda n, a: Decision("edit")):
            agent, _ = run(handler)
            self.assertTrue(agent.memory.messages[2]["content"].startswith("Error: 用户拒绝"))
        self.assertEqual(LOG, [])

    def test_predicate(self):
        asked = []
        agent, _ = run(lambda n, a: asked.append(a) or False,
                       calls=[MockLLM.call("pay", amount=50), MockLLM.call("pay", amount=500)], tools=(pay,))
        self.assertEqual(LOG, [50])
        self.assertEqual(asked, [{"amount": 500}])
        assert_paired(self, agent.memory.messages)

    def test_no_confirm_tools_untouched_and_bad_json_skips_confirm(self):
        called = []
        llm = MockLLM([MockLLM.call("danger"), "final"])  # 缺参数：交给 execute 报错，不打扰用户
        Agent(llm, tools=[danger], confirm=lambda n, a: called.append(1) or True).run("go")
        self.assertEqual(called, [])

    def test_parallel_calls_mixed(self):
        from mini_agent import LLMResponse, ToolCall
        resp = LLMResponse(tool_calls=[ToolCall("a", "danger", '{"path": "/1"}'),
                                       ToolCall("b", "danger", '{"path": "/2"}')])
        agent = Agent(MockLLM([resp, "final"]), tools=[danger],
                      confirm=lambda n, a: a["path"] == "/2")
        agent.run("go")
        self.assertEqual(LOG, ["/2"])
        assert_paired(self, agent.memory.messages)

    def test_builtin_write_file_requires_confirm(self):
        self.assertTrue(write_file.require_confirm)


class TestConsoleConfirmer(unittest.TestCase):
    def make(self, answers):
        it = iter(answers)
        return ConsoleConfirmer(input_fn=lambda p: next(it), print_fn=lambda *a: None)

    def test_inputs(self):
        self.assertEqual(self.make(["y"])("t", {}).action, "approve")
        self.assertEqual(self.make([""])("t", {}).action, "reject")       # 回车默认拒绝
        d = self.make(["路径不对"])("t", {})
        self.assertEqual((d.action, d.reason), ("reject", "路径不对"))
        d = self.make(["e", json.dumps({"path": "/ok"})])("t", {})
        self.assertEqual((d.action, d.arguments), ("edit", {"path": "/ok"}))
        self.assertEqual(self.make(["e", "not json"])("t", {}).action, "reject")

    def test_eof_rejects(self):
        def boom(prompt):
            raise EOFError
        self.assertEqual(ConsoleConfirmer(input_fn=boom, print_fn=lambda *a: None)("t", {}).action, "reject")

    def test_normalize(self):
        self.assertEqual(normalize(Decision.approve()).action, "approve")
        self.assertEqual(normalize(None).action, "reject")


if __name__ == "__main__":
    unittest.main()
