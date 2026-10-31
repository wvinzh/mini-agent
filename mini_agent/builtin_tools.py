"""几个开箱即用的内置工具：计算器、当前时间、文件读写（限制在沙箱目录内）。

它们同时也是"如何写工具"的范例：类型注解 + docstring 就够了。
"""
import ast
import datetime
import math
import operator
import os
from typing import Optional

from .tools import tool

# ------------------------------------------------------------------ 计算器
# 不使用 eval()！eval 能执行任意代码，模型（或被注入的提示词）可能让它删文件。
# 这里解析成语法树，只允许白名单里的运算符和函数。
_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
        ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod,
        ast.Pow: operator.pow, ast.USub: operator.neg, ast.UAdd: operator.pos}
_FUNCS = {k: getattr(math, k) for k in ("sqrt", "sin", "cos", "tan", "log", "log10", "exp",
                                       "floor", "ceil", "factorial")}
_FUNCS.update(abs=abs, round=round, min=min, max=max)
_NAMES = {"pi": math.pi, "e": math.e}


def _eval(node):
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        left, right = _eval(node.left), _eval(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 1000:
            raise ValueError("指数过大")
        return _OPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.operand))
    if isinstance(node, ast.Name) and node.id in _NAMES:
        return _NAMES[node.id]
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _FUNCS:
        return _FUNCS[node.func.id](*[_eval(a) for a in node.args])
    raise ValueError("不支持的表达式: %s" % ast.dump(node)[:80])


@tool
def calculator(expression: str) -> str:
    """计算数学表达式，支持 + - * / // % **、括号以及 sqrt/sin/cos/log/exp/abs/round 等函数和 pi、e。

    Args:
        expression: 数学表达式，例如 "(3 + 5) * sqrt(16)"
    """
    return str(_eval(ast.parse(expression.replace("^", "**"), mode="eval")))


# ------------------------------------------------------------------ 时间
@tool
def current_time(utc_offset_hours: Optional[float] = None) -> str:
    """获取当前日期和时间。

    Args:
        utc_offset_hours: 时区偏移小时数，如北京时间为 8；不传则使用本机时区
    """
    if utc_offset_hours is None:
        now = datetime.datetime.now().astimezone()
    else:
        now = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=utc_offset_hours)))
    return now.strftime("%Y-%m-%d %H:%M:%S %Z (%A)")


# ------------------------------------------------------------------ 文件（沙箱）
# 工具运行在你的机器上，务必限制可访问范围。通过环境变量 AGENT_WORKDIR 指定沙箱目录。
def _safe_path(path: str) -> str:
    root = os.path.realpath(os.getenv("AGENT_WORKDIR", os.getcwd()))
    full = os.path.realpath(os.path.join(root, path))
    if full != root and not full.startswith(root + os.sep):
        raise PermissionError("禁止访问工作目录之外的路径: %s" % path)
    return full


@tool
def list_dir(path: str = ".") -> str:
    """列出工作目录下某个文件夹的内容。

    Args:
        path: 相对工作目录的路径，默认为工作目录本身
    """
    full = _safe_path(path)
    items = sorted(os.listdir(full))
    return "\n".join(i + ("/" if os.path.isdir(os.path.join(full, i)) else "") for i in items) or "(空目录)"


@tool
def read_file(path: str, max_chars: int = 5000) -> str:
    """读取工作目录内的文本文件。

    Args:
        path: 相对工作目录的文件路径
        max_chars: 最多读取的字符数
    """
    with open(_safe_path(path), "r", encoding="utf-8", errors="replace") as f:
        return f.read(max_chars)


@tool(require_confirm=True)  # 写文件会改动磁盘：执行前需要人工确认
def write_file(path: str, content: str) -> str:
    """把文本写入工作目录内的文件（会覆盖已有内容）。

    Args:
        path: 相对工作目录的文件路径
        content: 要写入的文本
    """
    full = _safe_path(path)
    os.makedirs(os.path.dirname(full) or ".", exist_ok=True)
    with open(full, "w", encoding="utf-8") as f:
        f.write(content)
    return "已写入 %s（%d 字符）" % (path, len(content))


BUILTIN_TOOLS = [calculator, current_time, list_dir, read_file, write_file]
