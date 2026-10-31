"""上下文压缩测试：阈值触发、切分边界安全、增量摘要、失败兜底、持久化。"""
import os
import tempfile
import unittest

from mini_agent import (Agent, LLMResponse, LLMServerError, MockLLM, SummarizingMemory, TraceRecorder,
                        estimate_tokens)


def turn(i, with_tool=False):
    msgs = [{"role": "user", "content": "问题%d" % i}]
    if with_tool:
        msgs += [{"role": "assistant", "content": None, "tool_calls": [
            {"id": "c%d" % i, "type": "function", "function": {"name": "f", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "c%d" % i, "content": "结果%d" % i}]
    msgs.append({"role": "assistant", "content": "回答%d" % i})
    return msgs


class Summarizer(MockLLM):
    """记录收到的提示，返回编号摘要。"""

    def __init__(self):
        self.prompts = []
        super().__init__([])

    def _chat_once(self, messages, tools, on_token):
        assert tools is None  # 摘要请求不应携带工具
        self.prompts.append(messages[-1]["content"])
        return LLMResponse(content="摘要v%d" % len(self.prompts))


class TestSummarizingMemory(unittest.TestCase):
    def test_estimate_tokens(self):
        self.assertEqual(estimate_tokens("你好世界"), 4)
        self.assertEqual(estimate_tokens("abcdefgh"), 2)

    def test_below_threshold_no_compress(self):
        m = SummarizingMemory(llm=Summarizer(), trigger_messages=10)
        m.extend(turn(1))
        self.assertFalse(m.maybe_compress())
        self.assertEqual(m.summary, "")

    def test_compress_by_messages_keeps_recent_turns(self):
        s = Summarizer()
        m = SummarizingMemory(llm=s, trigger_messages=5, keep_turns=2)
        for i in range(3):
            m.extend(turn(i, with_tool=True))   # 每轮 4 条，共 12 条
        self.assertTrue(m.maybe_compress())
        self.assertEqual(m.summary, "摘要v1")
        self.assertEqual(m.messages[0]["content"], "问题1")  # 保留最近 2 轮原文
        self.assertEqual(len(m.messages), 8)
        self.assertIn("问题0", s.prompts[0])
        self.assertNotIn("问题1", s.prompts[0])

    def test_compress_by_tokens(self):
        m = SummarizingMemory(llm=Summarizer(), trigger_tokens=50, trigger_messages=1000, keep_turns=1)
        m.extend([{"role": "user", "content": "长" * 100}, {"role": "assistant", "content": "好"}])
        m.add({"role": "user", "content": "新问题"})
        self.assertTrue(m.maybe_compress())
        self.assertEqual([x["content"] for x in m.messages], ["新问题"])

    def test_boundary_never_splits_tool_pairs(self):
        m = SummarizingMemory(llm=Summarizer(), trigger_messages=3, keep_turns=1)
        for i in range(4):
            m.extend(turn(i, with_tool=True))
        m.maybe_compress()
        self.assertEqual(m.messages[0]["role"], "user")
        ids_called = {tc["id"] for x in m.messages for tc in x.get("tool_calls") or []}
        ids_answered = {x["tool_call_id"] for x in m.messages if x["role"] == "tool"}
        self.assertEqual(ids_called, ids_answered)

    def test_single_turn_never_compressed(self):
        m = SummarizingMemory(llm=Summarizer(), trigger_messages=1, keep_turns=1)
        m.extend(turn(0, with_tool=True))  # 只有当前这一轮：无可压缩内容
        self.assertFalse(m.maybe_compress())

    def test_recompression_includes_previous_summary(self):
        s = Summarizer()
        m = SummarizingMemory(llm=s, trigger_messages=3, keep_turns=1)
        m.extend(turn(0) + turn(1))
        m.maybe_compress()
        m.extend(turn(2) + turn(3))
        self.assertTrue(m.maybe_compress())
        self.assertIn("【已有摘要】\n摘要v1", s.prompts[1])
        self.assertIn("【已有摘要】\n（无）", s.prompts[0])
        self.assertEqual(m.summary, "摘要v2")
        self.assertEqual(m.compress_count, 2)

    def test_window_injects_summary_pair(self):
        m = SummarizingMemory(llm=Summarizer(), trigger_messages=3, keep_turns=1)
        m.extend(turn(0) + turn(1))
        m.maybe_compress()
        w = m.window()
        self.assertEqual([x["role"] for x in w[:3]], ["user", "assistant", "user"])
        self.assertIn("【历史对话摘要】", w[0]["content"])
        self.assertIn("摘要v1", w[0]["content"])

    def test_failure_falls_back_and_does_not_retry_same_turn(self):
        broken = MockLLM([LLMServerError("down", 503)] * 10, max_retries=0)
        events = []
        m = SummarizingMemory(llm=broken, trigger_messages=3, keep_turns=1, max_messages=2)
        m.attach(emit=lambda ev, *a: events.append(ev))
        m.extend(turn(0) + turn(1))
        self.assertFalse(m.maybe_compress())
        self.assertFalse(m.maybe_compress())          # 同一轮不再重复打 LLM
        self.assertEqual(len(broken.calls), 1)
        self.assertEqual(events, ["on_compress_error"])
        self.assertEqual(len(m.messages), 4)          # 历史原样保留
        self.assertEqual(m.window()[0]["content"], "问题1")  # 退回窗口截断

    def test_empty_summary_is_failure(self):
        m = SummarizingMemory(llm=MockLLM([LLMResponse(content="  ")]), trigger_messages=1, keep_turns=1)
        m.extend(turn(0) + turn(1))
        self.assertFalse(m.maybe_compress())
        self.assertEqual(m.summary, "")

    def test_persistence_with_summary(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "h.json")
            m = SummarizingMemory(llm=Summarizer(), trigger_messages=3, keep_turns=1, path=p)
            m.extend(turn(0) + turn(1))
            m.maybe_compress()
            m2 = SummarizingMemory(path=p)
            self.assertEqual(m2.summary, "摘要v1")
            self.assertEqual(m2.messages, m.messages)
            m2.clear()
            self.assertEqual(SummarizingMemory(path=p).summary, "")


class TestAgentWithCompression(unittest.TestCase):
    def test_agent_compresses_and_model_sees_summary(self):
        llm = MockLLM(["a0", "a1", "a2"])
        mem = SummarizingMemory(llm=Summarizer(), trigger_messages=3, keep_turns=1)
        trace = TraceRecorder()
        agent = Agent(llm, memory=mem, callbacks=[trace])
        for q in ("q0", "q1", "q2"):
            agent.run(q)
        self.assertIn("compress", trace.names())
        last_prompt = llm.calls[-1]
        self.assertEqual(last_prompt[0]["role"], "system")
        self.assertIn("【历史对话摘要】", last_prompt[1]["content"])
        self.assertEqual(last_prompt[-1]["content"], "q2")

    def test_agent_uses_own_llm_when_none_given(self):
        llm = MockLLM(["a0", "a1", "这是摘要", "a2"])
        mem = SummarizingMemory(trigger_messages=3, keep_turns=1)
        agent = Agent(llm, memory=mem)
        for q in ("q0", "q1", "q2"):
            agent.run(q)
        self.assertEqual(mem.summary, "这是摘要")

    def test_agent_survives_summarizer_failure(self):
        broken = MockLLM([LLMServerError("down", 503)] * 5, max_retries=0)
        trace = TraceRecorder()
        agent = Agent(MockLLM(["a0", "a1"]), callbacks=[trace],
                      memory=SummarizingMemory(llm=broken, trigger_messages=2, keep_turns=1))
        agent.run("q0")
        self.assertEqual(agent.run("q1"), "a1")
        self.assertIn("compress_error", trace.names())


if __name__ == "__main__":
    unittest.main()
