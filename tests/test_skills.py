"""技能机制测试：frontmatter 解析、目录、加载、未知技能、路径穿越、Agent 集成。"""
import os
import tempfile
import unittest

from mini_agent import Agent, MockLLM, SkillRegistry, TraceRecorder, parse_frontmatter

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")


def write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


class TestFrontmatter(unittest.TestCase):
    def test_basic(self):
        meta, body = parse_frontmatter("---\nname: a\ndescription: \"做 A: 很有用\"\n---\n# 正文\n内容")
        self.assertEqual(meta, {"name": "a", "description": "做 A: 很有用"})  # 值里的冒号保留
        self.assertEqual(body, "# 正文\n内容")

    def test_bom_crlf_comment_and_multiline(self):
        text = "\ufeff---\r\n# 注释\r\nname: b\r\ndescription: >\r\n  第一行\r\n  第二行\r\n---\r\nbody"
        meta, body = parse_frontmatter(text)
        self.assertEqual(meta["description"], "第一行 第二行")
        self.assertEqual(body, "body")

    def test_no_or_unclosed_frontmatter(self):
        self.assertEqual(parse_frontmatter("# 只有正文"), ({}, "# 只有正文"))
        self.assertEqual(parse_frontmatter("---\nname: x\n没有结尾")[0], {})


class TestSkillRegistry(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        d = self.tmp.name
        write(os.path.join(d, "alpha", "SKILL.md"), "---\nname: alpha\ndescription: 处理 alpha 任务\n---\n步骤：先读模板")
        write(os.path.join(d, "alpha", "tpl", "t.md"), "模板内容")
        write(os.path.join(d, "beta", "SKILL.md"), "---\ndescription: 没写 name，用文件夹名\n---\nbeta 正文")
        write(os.path.join(d, "broken", "SKILL.md"), "---\nname: broken\n---\n缺 description")
        write(os.path.join(d, "not_a_skill", "README.md"), "忽略我")
        write(os.path.join(d, "secret.txt"), "技能目录外的机密")
        self.reg = SkillRegistry(d)

    def tearDown(self):
        self.tmp.cleanup()

    def test_scan_and_catalog(self):
        self.assertEqual(self.reg.names(), ["alpha", "beta"])  # broken 被跳过，非技能文件夹被忽略
        self.assertEqual(self.reg.catalog(), "- alpha: 处理 alpha 任务\n- beta: 没写 name，用文件夹名")
        section = self.reg.prompt_section()
        self.assertIn("load_skill", section)
        self.assertNotIn("步骤：先读模板", section)  # 正文不进 system prompt

    def test_load(self):
        text = self.reg.load("alpha")
        self.assertIn("步骤：先读模板", text)
        self.assertIn("- tpl/t.md", text)          # 附带资源文件列表
        self.assertNotIn("SKILL.md", text.split("---")[-1])

    def test_unknown_skill(self):
        for out in (self.reg.load("nope"), self.reg.read_file("nope", "x")):
            self.assertTrue(out.startswith("Error: 未知技能 'nope'"))
            self.assertIn("alpha, beta", out)

    def test_read_file_and_sandbox(self):
        self.assertEqual(self.reg.read_file("alpha", "tpl/t.md"), "模板内容")
        self.assertIn("没有文件", self.reg.read_file("alpha", "missing.md"))
        for bad in ("../secret.txt", "../beta/SKILL.md", "/etc/passwd", "tpl/../../secret.txt", "."):
            self.assertTrue(self.reg.read_file("alpha", bad).startswith("Error:"), bad)

    def test_symlink_escape_blocked(self):
        link = os.path.join(self.tmp.name, "alpha", "link.txt")
        try:
            os.symlink(os.path.join(self.tmp.name, "secret.txt"), link)
        except (OSError, NotImplementedError):
            self.skipTest("当前系统不支持符号链接")
        self.assertIn("禁止访问", self.reg.read_file("alpha", "link.txt"))

    def test_tools_schema(self):
        load, read = self.reg.as_tools()
        self.assertEqual(load.schema()["function"]["parameters"]["required"], ["name"])
        self.assertEqual(read.schema()["function"]["parameters"]["required"], ["name", "path"])

    def test_missing_dir_is_ok(self):
        self.assertEqual(len(SkillRegistry(os.path.join(self.tmp.name, "nope"))), 0)


class TestAgentIntegration(unittest.TestCase):
    def test_full_flow(self):
        reg = SkillRegistry(os.path.join(ROOT, "skills"))
        llm = MockLLM([
            MockLLM.call("load_skill", name="short-drama-script"),
            MockLLM.call("read_skill_file", name="short-drama-script", path="reference/example.md"),
            "脚本完成",
        ])
        trace = TraceRecorder()
        agent = Agent(llm, skills=reg, callbacks=[trace])
        self.assertIn("load_skill", agent.tools.names())
        self.assertEqual(agent.run("写个短剧"), "脚本完成")

        first = llm.calls[0]
        self.assertIn("short-drama-script:", first[0]["content"])  # 目录在 system prompt
        self.assertNotIn("黄金 3 秒", first[0]["content"])          # 正文不在
        last = llm.calls[-1]
        self.assertIn("黄金 3 秒", last[3]["content"])              # 正文作为 tool 消息进入历史
        self.assertIn("替嫁总裁夫人", last[5]["content"])           # 资源文件
        self.assertEqual(agent.loaded_skills, ["short-drama-script"])
        self.assertIn("skill_load", trace.names())

    def test_unknown_skill_no_hook_and_no_skills(self):
        reg = SkillRegistry(os.path.join(ROOT, "skills"))
        agent = Agent(MockLLM([MockLLM.call("load_skill", name="nope"), "ok"]), skills=reg)
        agent.run("x")
        self.assertEqual(agent.loaded_skills, [])
        self.assertIn("Error: 未知技能", agent.memory.messages[2]["content"])
        # 空的技能注册表：不修改 system prompt，也不注册工具
        plain = Agent(MockLLM(["hi"]), skills=SkillRegistry())
        self.assertNotIn("技能", plain.system_prompt)
        self.assertEqual(plain.tools.names(), [])

    def test_bundled_skills_valid(self):
        reg = SkillRegistry(os.path.join(ROOT, "skills"))
        self.assertEqual(reg.names(), ["data-report", "short-drama-script"])
        self.assertIn("{报告标题}", reg.read_file("data-report", "template.md"))


if __name__ == "__main__":
    unittest.main()
