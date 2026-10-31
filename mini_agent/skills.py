"""技能（Skills）：按需加载的"操作手册"，采用渐进式披露（progressive disclosure）。

为什么需要技能？
    把所有领域知识（报告格式、剧本规范、代码规范……）都塞进 system prompt，
    会让上下文又长又贵，还会互相干扰。技能的做法是分三层按需披露：

    第 1 层  目录(catalog)：只有 name + 一句话 description，常驻 system prompt，很短
    第 2 层  SKILL.md 正文：模型判断任务匹配时调用 load_skill(name) 才加载进对话
    第 3 层  资源文件：模板、参考范例、脚本等，正文里提到时再用 read_skill_file 读取

一个技能就是一个文件夹：
    skills/
      data-report/
        SKILL.md          <- 必需：frontmatter 头 + markdown 正文
        template.md       <- 可选：任意资源文件
        reference/...

SKILL.md 格式：
    ---
    name: data-report
    description: 把 CSV 数据整理成结构化分析报告。用户要求"出报告/数据分析"时使用。
    ---
    # 正文（给模型看的操作步骤）
"""
import logging
import os
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple, Union

from .tools import Tool, tool

logger = logging.getLogger("mini_agent.skills")

SKILL_FILE = "SKILL.md"

# 注入 system prompt 的说明文字：告诉模型"目录在哪、怎么用"
SKILLS_PROMPT = """
## 技能（Skills）
你拥有以下技能。每个技能是一份针对某类任务的详细操作手册，目前只列出了名字和简介：
{catalog}

使用规则：
1. 收到任务后，先检查上面的技能列表，判断是否有与任务匹配的技能。
2. 如果有，必须先调用 load_skill(name) 读取完整说明，再开始做任务；不要凭简介猜测具体要求。
3. 严格按照技能说明中的步骤和格式执行；说明中提到的模板/参考文件用 read_skill_file(name, path) 读取。
4. 没有匹配的技能时，按常规方式完成任务。
""".rstrip()


def parse_frontmatter(text: str) -> Tuple[Dict[str, str], str]:
    """手写的极简 frontmatter 解析器（不依赖 yaml 库）。

    只支持我们需要的子集：
        key: value               普通键值；值两侧的引号会被去掉
        key: >  / key: |         后面缩进的多行内容拼成一个值
        缩进的续行               追加到上一个键的值后面
        # 注释行 / 空行          忽略
    没有 frontmatter 时返回 ({}, 原文)。
    """
    text = text.lstrip("\ufeff").replace("\r\n", "\n")  # 去 BOM、统一换行
    lines = text.split("\n")
    if not lines or lines[0].strip() != "---":
        return {}, text
    meta: Dict[str, str] = {}
    key: Optional[str] = None
    for i in range(1, len(lines)):
        line = lines[i]
        if line.strip() == "---":  # frontmatter 结束
            return meta, "\n".join(lines[i + 1:]).strip()
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if line[0] in " \t" and key:  # 续行
            meta[key] = (meta[key] + " " + line.strip()).strip()
            continue
        if ":" in line:
            key, value = line.split(":", 1)
            key, value = key.strip(), value.strip()
            if value in (">", "|", ">-", "|-"):
                value = ""
            if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
                value = value[1:-1]
            meta[key] = value
    return {}, text  # 只有开头的 --- 没有结尾：视为没有 frontmatter


@dataclass
class Skill:
    """一个技能的元数据 + 内容。"""

    name: str
    description: str
    path: str                     # 技能文件夹的绝对路径
    body: str                     # SKILL.md 正文（不含 frontmatter）
    meta: Dict[str, str] = field(default_factory=dict)

    def resources(self) -> List[str]:
        """技能文件夹里除 SKILL.md 外的所有文件（相对路径，/ 分隔）。"""
        out = []
        for root, dirs, files in os.walk(self.path):
            dirs[:] = sorted(d for d in dirs if not d.startswith(".") and d != "__pycache__")
            for f in sorted(files):
                rel = os.path.relpath(os.path.join(root, f), self.path).replace(os.sep, "/")
                if rel != SKILL_FILE and not f.startswith("."):
                    out.append(rel)
        return out


class SkillRegistry:
    """扫描技能目录、生成目录文本、提供 load_skill / read_skill_file 两个工具。

    用法:
        skills = SkillRegistry("skills/")              # 也可以传多个目录的列表
        agent = Agent(llm, tools=[...], skills=skills) # Agent 会自动注入目录并注册工具
    """

    def __init__(self, dirs: Union[str, List[str], None] = None, max_file_chars: int = 20000):
        self.skills: Dict[str, Skill] = {}
        self.max_file_chars = max_file_chars
        for d in ([dirs] if isinstance(dirs, str) else dirs or []):
            self.scan(d)

    # ------------------------------------------------------------------ 发现
    def scan(self, root: str) -> None:
        """扫描 root 下的每个子文件夹，含 SKILL.md 的即为技能。坏技能只告警、不中断。"""
        if not os.path.isdir(root):
            logger.warning("技能目录不存在: %s", root)
            return
        for entry in sorted(os.listdir(root)):
            folder = os.path.join(root, entry)
            if os.path.isfile(os.path.join(folder, SKILL_FILE)):
                try:
                    self.add_folder(folder)
                except Exception as e:
                    logger.warning("跳过无法解析的技能 %s: %s", folder, e)

    def add_folder(self, folder: str) -> Skill:
        with open(os.path.join(folder, SKILL_FILE), "r", encoding="utf-8") as f:
            meta, body = parse_frontmatter(f.read())
        name = meta.get("name") or os.path.basename(os.path.normpath(folder))
        desc = meta.get("description", "").strip()
        if not desc:
            raise ValueError("SKILL.md 缺少 description（模型靠它判断何时使用该技能）")
        if name in self.skills:
            logger.warning("技能 %s 重复定义，后者覆盖前者", name)
        skill = Skill(name=name, description=desc, path=os.path.realpath(folder), body=body, meta=meta)
        self.skills[name] = skill
        return skill

    def names(self) -> List[str]:
        return list(self.skills)

    def get(self, name: str) -> Optional[Skill]:
        return self.skills.get(name)

    def __len__(self) -> int:
        return len(self.skills)

    # ------------------------------------------------------------------ 第 1 层：目录
    def catalog(self) -> str:
        return "\n".join("- %s: %s" % (s.name, s.description) for s in self.skills.values())

    def prompt_section(self) -> str:
        """要追加到 system prompt 末尾的文字；没有技能时返回空串。"""
        return SKILLS_PROMPT.format(catalog=self.catalog()) if self.skills else ""

    # ------------------------------------------------------------------ 第 2 层：正文
    def load(self, name: str) -> str:
        skill = self.skills.get(name)
        if skill is None:
            return self._unknown(name)
        text = "# 技能: %s\n\n%s" % (skill.name, skill.body)
        res = skill.resources()
        if res:
            text += ("\n\n---\n本技能附带的资源文件（用 read_skill_file(\"%s\", path) 读取）:\n" % skill.name
                     + "\n".join("- " + r for r in res))
        return text

    # ------------------------------------------------------------------ 第 3 层：资源
    def read_file(self, name: str, path: str) -> str:
        skill = self.skills.get(name)
        if skill is None:
            return self._unknown(name)
        # 沙箱：解析符号链接和 ../ 之后，必须仍在技能文件夹内
        full = os.path.realpath(os.path.join(skill.path, path))
        if not full.startswith(skill.path + os.sep):
            return "Error: 禁止访问技能目录之外的路径: %s" % path
        if not os.path.isfile(full):
            return "Error: 技能 %s 中没有文件 %s，可用文件: %s" % (name, path, ", ".join(skill.resources()) or "无")
        with open(full, "r", encoding="utf-8", errors="replace") as f:
            content = f.read(self.max_file_chars + 1)
        if len(content) > self.max_file_chars:
            content = content[: self.max_file_chars] + "\n...[文件过长，已截断]"
        return content

    def _unknown(self, name: str) -> str:
        return "Error: 未知技能 '%s'，可用技能: %s" % (name, ", ".join(self.skills) or "（无）")

    # ------------------------------------------------------------------ 暴露成工具
    def as_tools(self, on_load: Optional[Callable[[str], None]] = None) -> List[Tool]:
        """生成 load_skill / read_skill_file 两个工具。

        on_load: 成功加载技能时的回调（Agent 用它触发 on_skill_load 钩子）。
        """
        registry = self

        @tool
        def load_skill(name: str) -> str:
            """加载一个技能的完整操作说明。开始做与某技能匹配的任务前必须先调用。

            Args:
                name: 技能名称，必须是技能列表中的名字
            """
            result = registry.load(name)
            if on_load and not result.startswith("Error:"):
                on_load(name)
            return result

        @tool
        def read_skill_file(name: str, path: str) -> str:
            """读取某个技能文件夹中的资源文件（模板、参考范例等）。

            Args:
                name: 技能名称
                path: 资源文件在技能文件夹内的相对路径，如 "template.md"
            """
            return registry.read_file(name, path)

        return [load_skill, read_skill_file]
