"""LLM 客户端测试。

真实 HTTP 路径用本地 http.server 模拟一个"不靠谱的"OpenAI 兼容服务，
覆盖：正常返回、工具调用、5xx 重试、429 Retry-After、401 不重试、
非 JSON 响应、HTTP 200 内嵌 error、超时、流式 SSE（含分片工具调用）。
"""
import json
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from mini_agent import (LLMAuthError, LLMBadRequestError, LLMConnectionError, LLMRateLimitError,
                        LLMResponse, LLMServerError, LLMTimeoutError, MockLLM, OpenAIClient)


def ok_body(content="hi", tool_calls=None):
    msg = {"role": "assistant", "content": content}
    if tool_calls:
        msg["tool_calls"] = tool_calls
    return json.dumps({"choices": [{"message": msg, "finish_reason": "stop"}],
                       "usage": {"total_tokens": 7}})


class FakeServer:
    """按队列依次返回预设响应：每项是 (状态码, 头, body) 或 "sleep" 或 ("sse", [行...])。"""

    def __init__(self):
        self.queue, self.requests = [], []
        outer = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):  # 静音
                pass

            def do_POST(self):
                body = self.rfile.read(int(self.headers["Content-Length"]))
                outer.requests.append({"body": json.loads(body), "auth": self.headers.get("Authorization")})
                item = outer.queue.pop(0)
                if item == "sleep":
                    time.sleep(1.5)
                    item = (200, {}, ok_body())
                if item[0] == "sse":
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.end_headers()
                    for line in item[1]:
                        self.wfile.write(("data: %s\n\n" % line).encode())
                    return
                status, headers, text = item
                self.send_response(status)
                for k, v in headers.items():
                    self.send_header(k, v)
                data = text.encode()
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.httpd.handle_error = lambda *a: None  # 超时测试中客户端先断开，忽略 BrokenPipe 噪音
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        self.url = "http://127.0.0.1:%d/v1" % self.httpd.server_address[1]

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


class TestOpenAIClient(unittest.TestCase):
    def setUp(self):
        self.srv = FakeServer()
        self.sleeps = []
        self.llm = OpenAIClient(model="m", api_key="k", base_url=self.srv.url, timeout=1,
                                max_retries=3, sleep=self.sleeps.append)

    def tearDown(self):
        self.srv.close()

    def test_basic_and_request_format(self):
        self.srv.queue.append((200, {}, ok_body("你好")))
        r = self.llm.chat([{"role": "user", "content": "hi"}], tools=[{"type": "function"}])
        self.assertEqual(r.content, "你好")
        self.assertEqual(r.usage["total_tokens"], 7)
        req = self.srv.requests[0]
        self.assertEqual(req["auth"], "Bearer k")
        self.assertEqual(req["body"]["model"], "m")
        self.assertIn("tools", req["body"])

    def test_tool_calls_parsed(self):
        tc = [{"id": "c1", "type": "function", "function": {"name": "add", "arguments": '{"a":1}'}}]
        self.srv.queue.append((200, {}, ok_body(None, tc)))
        r = self.llm.chat([{"role": "user", "content": "hi"}])
        self.assertEqual((r.tool_calls[0].id, r.tool_calls[0].name), ("c1", "add"))
        self.assertEqual(r.to_message()["tool_calls"][0]["function"]["arguments"], '{"a":1}')

    def test_retry_on_5xx_then_success(self):
        self.srv.queue += [(502, {}, "bad gateway"), (503, {}, "busy"), (200, {}, ok_body("ok"))]
        self.assertEqual(self.llm.chat([]).content, "ok")
        self.assertEqual(len(self.sleeps), 2)
        self.assertLessEqual(self.sleeps[0], 1.0)   # 指数退避：1s*抖动, 2s*抖动
        self.assertLessEqual(self.sleeps[1], 2.0)

    def test_retry_exhausted(self):
        self.srv.queue += [(500, {}, "x")] * 4
        with self.assertRaises(LLMServerError) as cm:
            self.llm.chat([])
        self.assertEqual(cm.exception.status, 500)
        self.assertEqual(len(self.srv.requests), 4)  # 1 次 + 3 次重试

    def test_429_respects_retry_after(self):
        self.srv.queue += [(429, {"Retry-After": "7"}, "slow down"), (200, {}, ok_body("ok"))]
        self.assertEqual(self.llm.chat([]).content, "ok")
        self.assertEqual(self.sleeps, [7.0])

    def test_auth_error_not_retried(self):
        self.srv.queue.append((401, {}, '{"error":"invalid key"}'))
        with self.assertRaises(LLMAuthError):
            self.llm.chat([])
        self.assertEqual(len(self.srv.requests), 1)

    def test_bad_request_not_retried(self):
        self.srv.queue.append((400, {}, "model not found"))
        with self.assertRaises(LLMBadRequestError) as cm:
            self.llm.chat([])
        self.assertIn("model not found", str(cm.exception))

    def test_invalid_json_retried(self):
        self.srv.queue += [(200, {}, "<html>gateway error</html>"), (200, {}, ok_body("ok"))]
        self.assertEqual(self.llm.chat([]).content, "ok")

    def test_error_inside_200(self):
        self.srv.queue += [(200, {}, '{"error": {"message": "Rate limit reached"}}')] * 4
        with self.assertRaises(LLMRateLimitError):
            self.llm.chat([])

    def test_timeout(self):
        self.llm.max_retries = 0
        self.srv.queue.append("sleep")
        with self.assertRaises(LLMTimeoutError):
            self.llm.chat([])

    def test_connection_refused(self):
        llm = OpenAIClient(api_key="k", base_url="http://127.0.0.1:1/v1", max_retries=1,
                           sleep=lambda s: None, timeout=1)
        with self.assertRaises(LLMConnectionError):
            llm.chat([])

    def test_stream_with_fragmented_tool_call(self):
        chunks = [
            {"choices": [{"delta": {"content": "让我"}}]},
            {"choices": [{"delta": {"content": "算一下"}}]},
            {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "c1",
                                                     "function": {"name": "calc", "arguments": '{"ex'}}]}}]},
            {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": 'p":"1+1"}'}}]},
                          "finish_reason": "tool_calls"}]},
        ]
        self.srv.queue.append(("sse", [json.dumps(c) for c in chunks] + ["[DONE]"]))
        tokens = []
        r = self.llm.chat([], on_token=tokens.append)
        self.assertEqual(tokens, ["让我", "算一下"])
        self.assertEqual(r.content, "让我算一下")
        self.assertEqual(r.tool_calls[0].name, "calc")
        self.assertEqual(json.loads(r.tool_calls[0].arguments), {"exp": "1+1"})
        self.assertEqual(r.finish_reason, "tool_calls")
        self.assertTrue(self.srv.requests[0]["body"]["stream"])


class TestMockLLM(unittest.TestCase):
    def test_script_and_retry(self):
        llm = MockLLM([LLMServerError("x", 503), "ok", lambda m, t: LLMResponse(content=str(len(m)))])
        self.assertEqual(llm.chat([{"role": "user", "content": "a"}]).content, "ok")
        self.assertEqual(llm.chat([{}, {}]).content, "2")
        self.assertIn("剧本已用完", llm.chat([]).content)
        self.assertEqual(len(llm.calls), 4)  # 失败的那次也被记录

    def test_no_retry_after_tokens_emitted(self):
        """流式已输出部分内容后出错，不应重试（否则用户看到重复文字）。"""
        class Flaky(MockLLM):
            def _chat_once(self, messages, tools, on_token):
                on_token("半句")
                raise LLMServerError("断了", 500)
        with self.assertRaises(LLMServerError):
            Flaky([]).chat([], on_token=lambda t: None)


if __name__ == "__main__":
    unittest.main()
