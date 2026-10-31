"""工具系统：@tool 装饰器 + 注册表 + 安全执行。

核心思想：
    模型看不到你的 Python 函数，它只能看到一份 JSON Schema 描述。
    @tool 做的事就是：读取函数签名、类型注解、docstring，
    自动生成 OpenAI 要求的 `{"type": "function", "function": {...}}` 结构。

docstring 约定（Google 风格，可选）:
    def search(query: str, top_k: int = 3) -> str:
        '''搜索知识库。            <- 第一段作为工具描述

        Args:
            query: 搜索关键词      <- 参数描述
            top_k: 返回条数
        '''
"""
import inspect
import json
import logging
import re
import typing
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Union

from .errors import ToolError

logger = logging.getLogger("mini_agent.tools")

# Python 类型 -> JSON Schema 类型
_PRIMITIVES = {str: "string", int: "integer", float: "number", bool: "boolean",
               dict: "object", list: "array", type(None): "null"}


def _type_to_schema(tp: Any) -> Dict[str, Any]:
    """把类型注解递归转换为 JSON Schema（覆盖常用的类型即可）。"""
    if tp is inspect.Parameter.empty or tp is Any:
        return {}  # 没写注解：不限制类型
    if tp in _PRIMITIVES:
        return {"type": _PRIMITIVES[tp]}
    origin = typing.get_origin(tp)
    args = typing.get_args(tp)
    if origin in (list, List, tuple, set):
        schema: Dict[str, Any] = {"type": "array"}
        if args and args[0] is not Ellipsis:
            schema["items"] = _type_to_schema(args[0])
        return schema
    if origin in (dict, Dict):
        return {"type": "object"}
    if origin is typing.Union:  # Optional[X] == Union[X, None]
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) == 1:
            return _type_to_schema(non_none[0])
        return {"anyOf": [_type_to_schema(a) for a in non_none]}
    if origin is typing.Literal:
        return {"type": _PRIMITIVES.get(type(args[0]), "string"), "enum": list(args)}
    return {"type": "string"}  # 兜底：未知类型按字符串处理


def _parse_docstring(doc: str):
    """从 docstring 中拆出 (总描述, {参数名: 参数描述})。"""
    doc = inspect.cleandoc(doc or "")
    parts = re.split(r"\n\s*(?:Args|Arguments|Parameters|参数)\s*[:：]\s*\n", doc, maxsplit=1)
    summary = parts[0].strip()
    params: Dict[str, str] = {}
    if len(parts) > 1:
        for line in parts[1].splitlines():
            m = re.match(r"\s*(\w+)\s*(?:\([^)]*\))?\s*[:：]\s*(.+)", line)
            if m:
                params[m.group(1)] = m.group(2).strip()
            elif not line.strip() or re.match(r"\s*(Returns|Raises|返回)\s*[:：]", line):
                if params:
                    break
    return summary, params


@dataclass
class Tool:
    """一个可被模型调用的工具。"""

    name: str
    description: str
    parameters: Dict[str, Any]  # JSON Schema
    func: Callable[..., Any]
    return_direct: bool = False  # True 时工具结果直接作为 Agent 最终回答（不再回给模型）
    # 人工确认：True=每次都要确认；也可以是函数 args -> bool，按参数决定是否需要确认
    require_confirm: Union[bool, Callable[[Dict[str, Any]], bool]] = False

    def schema(self) -> Dict[str, Any]:
        """OpenAI `tools` 字段需要的格式。"""
        return {"type": "function",
                "function": {"name": self.name, "description": self.description,
                             "parameters": self.parameters}}

    def needs_confirm(self, arguments: Dict[str, Any]) -> bool:
        if callable(self.require_confirm):
            try:
                return bool(self.require_confirm(arguments))
            except Exception:
                return True  # 判断函数自己出错了：保守起见，要求确认
        return bool(self.require_confirm)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.func(*args, **kwargs)  # 被装饰后依然能当普通函数调用

    @classmethod
    def from_function(cls, func: Callable, name: Optional[str] = None,
                      description: Optional[str] = None, return_direct: bool = False,
                      require_confirm: Union[bool, Callable] = False) -> "Tool":
        sig = inspect.signature(func)
        try:
            hints = typing.get_type_hints(func)  # 兼容 `from __future__ import annotations`
        except Exception:
            hints = {}
        summary, param_docs = _parse_docstring(func.__doc__ or "")
        props: Dict[str, Any] = {}
        required: List[str] = []
        for pname, p in sig.parameters.items():
            if p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
                continue  # *args / **kwargs 无法描述给模型，直接忽略
            prop = _type_to_schema(hints.get(pname, p.annotation))
            if pname in param_docs:
                prop["description"] = param_docs[pname]
            if p.default is p.empty:
                required.append(pname)
            elif _jsonable(p.default):
                prop.setdefault("default", p.default)
            props[pname] = prop
        params = {"type": "object", "properties": props, "required": required}
        return cls(name=name or func.__name__,
                   description=description or summary or func.__name__,
                   parameters=params, func=func, return_direct=return_direct,
                   require_confirm=require_confirm)


def _jsonable(v: Any) -> bool:
    try:
        json.dumps(v)
        return True
    except (TypeError, ValueError):
        return False


def tool(func: Optional[Callable] = None, *, name: Optional[str] = None,
         description: Optional[str] = None, return_direct: bool = False,
         require_confirm: Union[bool, Callable] = False):
    """装饰器：把普通函数变成 Tool。两种写法都支持：

        @tool
        def add(a: int, b: int) -> int: ...

        @tool(name="web_search", description="...")
        def search(q: str) -> str: ...

        @tool(require_confirm=True)          # 执行前需要人工确认（见 confirm.py）
        def delete_file(path: str) -> str: ...
    """
    def wrap(f: Callable) -> Tool:
        return Tool.from_function(f, name=name, description=description,
                                  return_direct=return_direct, require_confirm=require_confirm)
    return wrap(func) if func is not None else wrap


class ToolRegistry:
    """工具注册表：按名字管理工具，并负责"安全执行"。

    安全执行的含义：工具里抛出的任何异常都不会让 Agent 崩溃，
    而是转成一段错误文本返回给模型 —— 模型看到错误后通常会自己修正参数重试。
    """

    def __init__(self, tools: Optional[List[Any]] = None, max_output_chars: int = 8000):
        self._tools: Dict[str, Tool] = {}
        self.max_output_chars = max_output_chars  # 防止工具返回超长内容撑爆上下文
        for t in tools or []:
            self.register(t)

    def register(self, t: Any) -> Tool:
        if not isinstance(t, Tool):
            if not callable(t):
                raise ToolError("只能注册 Tool 或可调用对象: %r" % (t,))
            t = Tool.from_function(t)
        if t.name in self._tools:
            logger.warning("工具 %s 被重复注册，将覆盖旧的", t.name)
        self._tools[t.name] = t
        return t

    def get(self, name: str) -> Optional[Tool]:
        return self._tools.get(name)

    def names(self) -> List[str]:
        return list(self._tools)

    def schemas(self) -> List[Dict[str, Any]]:
        return [t.schema() for t in self._tools.values()]

    def __len__(self) -> int:
        return len(self._tools)

    def execute(self, name: str, arguments: Any) -> str:
        """执行工具并总是返回字符串（成功结果或以 'Error:' 开头的错误说明）。"""
        t = self._tools.get(name)
        if t is None:
            return "Error: 未知工具 '%s'，可用工具: %s" % (name, ", ".join(self._tools))
        # 1) 解析参数：模型给的 JSON 可能不合法
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments) if arguments.strip() else {}
            except ValueError as e:
                return "Error: 参数不是合法 JSON (%s)，原始参数: %s" % (e, arguments[:500])
        if not isinstance(arguments, dict):
            return "Error: 参数必须是 JSON 对象，收到: %r" % (arguments,)
        # 2) 校验必填参数 / 多余参数，给模型清晰反馈
        missing = [p for p in t.parameters.get("required", []) if p not in arguments]
        if missing:
            return "Error: 缺少必填参数 %s" % missing
        sig_params = inspect.signature(t.func).parameters
        if not any(p.kind == p.VAR_KEYWORD for p in sig_params.values()):
            arguments = {k: v for k, v in arguments.items() if k in sig_params}
        # 3) 执行并捕获一切异常
        try:
            result = t.func(**arguments)
        except Exception as e:
            # 默认只记一行警告；开启 DEBUG 日志时才打印完整堆栈
            logger.warning("工具 %s 执行出错: %r", name, e, exc_info=logger.isEnabledFor(logging.DEBUG))
            return "Error: 工具 %s 执行失败: %s: %s" % (name, type(e).__name__, e)
        # 4) 统一转字符串并截断
        if not isinstance(result, str):
            try:
                result = json.dumps(result, ensure_ascii=False, default=str)
            except Exception:
                result = str(result)
        if len(result) > self.max_output_chars:
            result = result[: self.max_output_chars] + "\n...[输出过长，已截断，共 %d 字符]" % len(result)
        return result
