"""记忆测试：窗口截断、不产生孤儿 tool 消息、持久化。"""
import os
import tempfile
import unittest

from mini_agent import ConversationMemory, JSONStore


def turn(i):
    return [{"role": "user", "content": "q%d" % i},
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "c%d" % i, "type": "function", "function": {"name": "f", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "c%d" % i, "content": "r%d" % i},
            {"role": "assistant", "content": "a%d" % i}]


class TestConversationMemory(unittest.TestCase):
    def test_window_by_count_starts_with_user(self):
        m = ConversationMemory(max_messages=6)
        for i in range(3):
            m.extend(turn(i))
        w = m.window()
        self.assertEqual(w[0]["role"], "user")      # 不会从孤儿 tool 消息开始
        self.assertEqual(w[0]["content"], "q2")
        self.assertEqual(len(m.messages), 12)       # 原始记录不被修改

    def test_window_by_chars(self):
        m = ConversationMemory(max_messages=0, max_chars=10)
        m.add({"role": "user", "content": "x" * 50})
        m.add({"role": "assistant", "content": "y" * 50})
        m.add({"role": "user", "content": "short"})
        self.assertEqual(m.window(), [{"role": "user", "content": "short"}])

    def test_oversized_current_turn_kept(self):
        m = ConversationMemory(max_chars=5)
        m.extend(turn(0))
        m.add({"role": "tool", "tool_call_id": "x", "content": "很长" * 10})
        w = m.window()
        self.assertEqual(w[0]["content"], "q0")     # 当前回合整体保留
        self.assertEqual(len(w), 5)

    def test_persistence(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "h.json")
            m = ConversationMemory(path=p)
            m.add({"role": "user", "content": "你好"})
            self.assertEqual(ConversationMemory(path=p).messages, [{"role": "user", "content": "你好"}])
            m.clear()
            self.assertEqual(ConversationMemory(path=p).messages, [])


class TestJSONStore(unittest.TestCase):
    def test_store_and_tools(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "n.json")
            remember, recall = JSONStore(p).as_tools()
            remember(key="城市", value="杭州")
            self.assertEqual(JSONStore(p).get("城市"), "杭州")   # 重新加载仍在
            self.assertIn("杭州", recall(query="城"))
            self.assertEqual(recall(query="不存在"), "没有找到相关记忆")


if __name__ == "__main__":
    unittest.main()
