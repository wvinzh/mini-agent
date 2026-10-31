"""工具系统测试：schema 生成、参数校验、异常捕获、内置工具。"""
import json
import os
import tempfile
import unittest
from typing import List, Literal, Optional

from mini_agent import Tool, ToolRegistry, tool
from mini_agent.builtin_tools import calculator, list_dir, read_file, write_file


@tool
def search(query: str, top_k: int = 3, tags: Optional[List[str]] = None,
           mode: Literal["fast", "slow"] = "fast") -> str:
    """搜索文档。

    Args:
        query: 关键词
        top_k: 返回条数
        tags: 标签过滤
        mode: 速度模式
    """
    return "%s|%d|%s|%s" % (query, top_k, tags, mode)


class TestSchema(unittest.TestCase):
    def test_schema_from_signature_and_docstring(self):
        fn = search.schema()["function"]
        self.assertEqual(fn["name"], "search")
        self.assertEqual(fn["description"], "搜索文档。")
        props = fn["parameters"]["properties"]
        self.assertEqual(props["query"], {"type": "string", "description": "关键词"})
        self.assertEqual(props["top_k"]["type"], "integer")
        self.assertEqual(props["top_k"]["default"], 3)
        self.assertEqual(props["tags"]["type"], "array")
        self.assertEqual(props["tags"]["items"], {"type": "string"})
        self.assertEqual(props["mode"]["enum"], ["fast", "slow"])
        self.assertEqual(fn["parameters"]["required"], ["query"])

    def test_decorator_with_options_and_still_callable(self):
        @tool(name="add_numbers", description="两数相加")
        def add(a: int, b: int) -> int:
            return a + b
        self.assertIsInstance(add, Tool)
        self.assertEqual(add.name, "add_numbers")
        self.assertEqual(add.description, "两数相加")
        self.assertEqual(add(2, 3), 5)

    def test_register_plain_function(self):
        def hello(name: str) -> str:
            """打招呼"""
            return "hi " + name
        reg = ToolRegistry([hello])
        self.assertEqual(reg.names(), ["hello"])
        self.assertEqual(reg.execute("hello", '{"name": "Bob"}'), "hi Bob")


class TestSafeExecute(unittest.TestCase):
    def setUp(self):
        @tool
        def boom(x: int) -> int:
            """总是出错"""
            raise RuntimeError("坏了")

        @tool
        def big() -> str:
            """返回很长的结果"""
            return "x" * 100
        self.reg = ToolRegistry([search, boom, big], max_output_chars=10)

    def test_unknown_tool(self):
        self.assertIn("未知工具", self.reg.execute("nope", "{}"))

    def test_bad_json(self):
        self.assertIn("不是合法 JSON", self.reg.execute("search", "{bad"))

    def test_missing_required(self):
        self.assertIn("缺少必填参数", self.reg.execute("search", "{}"))

    def test_exception_becomes_error_text(self):
        out = self.reg.execute("boom", {"x": 1})
        self.assertTrue(out.startswith("Error:"))
        self.assertIn("RuntimeError", out)

    def test_extra_args_ignored_and_truncation(self):
        self.assertTrue(self.reg.execute("big", '{"unused": 1}').startswith("xxxxxxxxxx\n...[输出过长"))

    def test_non_string_result_json(self):
        reg = ToolRegistry([tool(lambda: {"a": 1}, name="d", description="d")])
        self.assertEqual(json.loads(reg.execute("d", "")), {"a": 1})


class TestBuiltinTools(unittest.TestCase):
    def test_calculator(self):
        self.assertEqual(calculator("(3 + 5) * sqrt(16)"), "32.0")
        self.assertEqual(calculator("2^10"), "1024")
        with self.assertRaises(ValueError):
            calculator("__import__('os').system('ls')")  # 不允许任意代码

    def test_file_sandbox(self):
        with tempfile.TemporaryDirectory() as d:
            os.environ["AGENT_WORKDIR"] = d
            try:
                self.assertIn("已写入", write_file("a/b.txt", "hello"))
                self.assertEqual(read_file("a/b.txt"), "hello")
                self.assertEqual(list_dir("."), "a/")
                with self.assertRaises(PermissionError):
                    read_file("../../etc/passwd")
            finally:
                del os.environ["AGENT_WORKDIR"]


if __name__ == "__main__":
    unittest.main()
