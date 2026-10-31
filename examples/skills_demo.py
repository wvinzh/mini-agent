"""技能（Skills）演示：看清"目录 → load_skill → read_skill_file → 按说明执行"的全过程。

运行:
    python examples/skills_demo.py          # 离线 MockLLM（剧本模拟一个"会用技能"的模型）
    python examples/skills_demo.py --real   # 真实 API（需 OPENAI_API_KEY / OPENAI_BASE_URL / MODEL）

重点看最后打印的消息序列：技能正文是以 role=tool 消息的形式"按需"进入对话历史的，
而 system prompt 里始终只有简短的技能目录。
"""
import os
import sys
import tempfile

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)

from mini_agent import Agent, ConsolePrinter, MockLLM, OpenAIClient, SkillRegistry, auto_approve
from mini_agent.builtin_tools import BUILTIN_TOOLS

REAL = "--real" in sys.argv

SALES_CSV = """区域,销售额
华东,12000
华北,8500
华南,15300
华西,4200
华中,9800
"""

REPORT = """# 《2026 年 10 月区域销售分析报告》

> 数据来源：`sales.csv` ｜ 记录数：5 条 ｜ 统计范围：2026 年 10 月

## 一、核心指标

| 指标 | 数值 | 说明 |
|---|---|---|
| 总销售额 | 49800.00 元 | 5 个区域合计 |
| 平均销售额 | 9960.00 元 | 每区域平均 |
| 最高区域 | 华南 15300.00 元 | 占总额 30.72% |
| 最低区域 | 华西 4200.00 元 | 仅为平均值的 42.17% |

## 二、关键发现

1. 华南以 15300.00 元领先，贡献了 30.72% 的销售额。
2. 华东(12000.00 元)与华南合计占比超过一半，集中度较高。

## 三、异常与风险

- 华西 4200.00 元，低于平均值 50% 以上，属于异常低值，需排查渠道或库存问题。

## 四、行动建议

1. 复盘华南的打法并在华东、华中复制。
2. 对华西做专项诊断，两周内给出改进方案。"""

DRAMA = """**剧名**：《外卖小哥是首富》 第 1 集
**一句话梗概**：被全公司嘲笑的外卖员，竟是来收购这家公司的神秘首富。

**人物表**
| 姓名 | 身份 | 外形关键词 |
|---|---|---|
| 陈野 | 外卖员 / 隐藏首富 | Chinese man 28, short hair, yellow delivery uniform |
| 苏蔓 | 公司前台，势利 | young Chinese woman, office suit, red lipstick |

**分镜表**
| 镜号 | 时长(s) | 景别 | 运镜 | 画面描述 | 台词/音效 | AI 画面提示词(英文) |
|---|---|---|---|---|---|---|
| 1 | 3 | 特写 | 推 | 一杯咖啡泼在陈野脸上 | 苏蔓："送个外卖也敢迟到？" | Chinese man 28, short hair, yellow delivery uniform, coffee splashed on face, office lobby, harsh light, vertical 9:16, cinematic |
| 2 | 4 | 中景 | 固定 | 众员工围观哄笑 | 音效：哄笑声 | office workers laughing, modern lobby, vertical 9:16, cinematic |
| 3 | 5 | 近景 | 摇 | 陈野擦掉咖啡，眼神平静 | 陈野："你会后悔的。" | Chinese man 28, short hair, yellow delivery uniform, calm cold eyes, vertical 9:16, cinematic |
| 4 | 5 | 全景 | 跟 | 一排黑色豪车停在楼下 | 音效：刹车声 | row of black luxury cars arriving, skyscraper, daylight, vertical 9:16, cinematic |
| 5 | 6 | 中景 | 推 | 保镖鞠躬："董事长，收购合同已备好。" | 保镖台词 | bodyguards bowing, black suits, lobby, vertical 9:16, cinematic |
| 6 | 4 | 特写 | 固定 | 苏蔓手中的咖啡杯掉落 | 音效：杯子碎裂 | young Chinese woman, office suit, red lipstick, shocked, cup falling, vertical 9:16, cinematic |
| 7 | 5 | 近景 | 推 | 陈野脱下外卖服，露出定制西装 | 音效：拉链声 | Chinese man 28, short hair, taking off yellow delivery uniform, revealing tailored black suit, vertical 9:16, cinematic |
| 8 | 6 | 中景 | 跟 | 陈野走进会议室，总经理慌忙起身 | 总经理："您、您就是陈董？" | Chinese man 28, short hair, black suit, walking into boardroom, executives standing up, vertical 9:16, cinematic |
| 9 | 5 | 特写 | 固定 | 陈野在收购合同上签字 | 音效：钢笔划纸 | close-up of hand signing contract, fountain pen, warm light, vertical 9:16, cinematic |
| 10 | 4 | 近景 | 摇 | 苏蔓冲进来想道歉，被保安拦住 | 苏蔓："陈总，我错了！" | young Chinese woman, office suit, red lipstick, stopped by security, desperate, vertical 9:16, cinematic |
| 11 | 6 | 中景 | 固定 | 陈野把一杯咖啡放到苏蔓面前 | 陈野："这杯，我请你。" | Chinese man 28, short hair, black suit, placing coffee cup on table, cold smile, vertical 9:16, cinematic |
| 12 | 5 | 全景 | 拉 | 全公司员工鞠躬，陈野转身离开 | 音效：掌声渐弱 | office staff bowing, man in black suit walking away, wide shot, vertical 9:16, cinematic |
| 13 | 3 | 近景 | 拉 | 陈野收到神秘短信："你父亲没死。" | 音效：手机震动 | Chinese man 28, short hair, black suit, reading phone, night, suspense, vertical 9:16, cinematic |

**总时长核对**：3+4+5+5+6+4+5+6+5+4+6+5+3 = 61 秒 ✅（60–90 秒内）

**下集钩子**：发短信的人到底是谁？"""


def mock_llm() -> MockLLM:
    """模拟一个"遵守技能规则"的模型。真实模型会根据 system prompt 自己做出同样的决策。"""
    return MockLLM([
        # ---- 任务 1：数据报告
        MockLLM.call("load_skill", name="data-report"),
        MockLLM.call("read_file", path="sales.csv"),
        MockLLM.call("calculator", expression="12000+8500+15300+4200+9800"),
        MockLLM.call("calculator", expression="49800/5"),
        MockLLM.call("calculator", expression="15300/49800*100"),
        MockLLM.call("calculator", expression="4200/9960*100"),
        MockLLM.call("read_skill_file", name="data-report", path="template.md"),
        MockLLM.call("write_file", path="reports/sales_report.md", content=REPORT),
        REPORT + "\n\n（已保存到 reports/sales_report.md）",
        # ---- 任务 2：短剧分镜
        MockLLM.call("load_skill", name="short-drama-script"),
        MockLLM.call("read_skill_file", name="short-drama-script", path="reference/example.md"),
        MockLLM.call("read_skill_file", name="short-drama-script", path="../data-report/SKILL.md"),  # 越界会被拦截
        MockLLM.call("calculator", expression="3+4+5+5+6+4+5+6+5+4+6+5+3"),
        DRAMA,
    ])


def show_messages(agent: Agent) -> None:
    """打印模型最后一次看到的完整消息序列，并标注技能内容出现的位置。"""
    names = {}  # tool_call_id -> 工具名
    print("\n" + "=" * 70 + "\n📜 消息序列（模型实际看到的上下文）\n" + "=" * 70)
    msgs = [{"role": "system", "content": agent.system_prompt}] + agent.memory.window()
    for i, m in enumerate(msgs):
        content = (m.get("content") or "").replace("\n", "⏎ ")
        note = ""
        if m["role"] == "system":
            note = "  ⬅ 第1层：技能目录常驻 system prompt"
            content = "（基础提示词 + 技能目录，完整内容见开头）"
        for tc in m.get("tool_calls") or []:
            names[tc["id"]] = tc["function"]["name"]
            content += " 🔧%s(%s)" % (tc["function"]["name"], tc["function"]["arguments"][:60])
        if m["role"] == "tool":
            tname = names.get(m.get("tool_call_id"), "")
            if tname == "load_skill":
                note = "  ⬅ 第2层：SKILL.md 正文在此进入历史"
            elif tname == "read_skill_file":
                note = "  ⬅ 第3层：技能资源文件" + ("（越界访问被拦截）" if content.startswith("Error") else "")
        print("[%02d] %-9s| %s%s" % (i, m["role"], content[:110], note))


def main() -> None:
    workdir = tempfile.mkdtemp(prefix="skills_demo_")
    os.environ["AGENT_WORKDIR"] = workdir  # 文件工具的沙箱
    with open(os.path.join(workdir, "sales.csv"), "w", encoding="utf-8") as f:
        f.write(SALES_CSV)

    skills = SkillRegistry(os.path.join(ROOT, "skills"))
    print("发现技能:", skills.names())
    agent = Agent(OpenAIClient() if REAL else mock_llm(), tools=BUILTIN_TOOLS, skills=skills,
                  max_steps=15, callbacks=[ConsolePrinter()],
                  confirm=auto_approve)  # write_file 需要确认，演示中自动批准
    print("\n" + "=" * 70 + "\n🧾 System Prompt（注意：只有技能目录，没有技能正文）\n" + "=" * 70)
    print(agent.system_prompt)
    print("\n可用工具:", agent.tools.names())

    print("\n" + "=" * 70 + "\n任务 1：数据报告（工作目录 %s）\n" % workdir + "=" * 70)
    print("助手>\n" + agent.run("帮我分析一下 sales.csv 的销售数据，出一份报告并保存"))
    show_messages(agent)
    print("\n已加载技能:", agent.loaded_skills)

    agent.reset()  # 换个话题，清空历史（system prompt 中的技能目录不受影响）
    print("\n" + "=" * 70 + "\n任务 2：短剧分镜\n" + "=" * 70)
    print("助手>\n" + agent.run("写一集竖屏短剧的分镜脚本，题材：外卖小哥其实是隐藏首富"))
    show_messages(agent)
    print("\n已加载技能（reset 后重新计数）:", agent.loaded_skills)


if __name__ == "__main__":
    main()
