import importlib
import json
import sys
import unittest
from threading import Event
from unittest.mock import Mock, patch

from genai_proxy.streaming import iter_with_heartbeat


class HeartbeatTests(unittest.TestCase):
    def test_source_exception_reaches_consumer(self):
        def source():
            raise RuntimeError("connection failed")
            yield

        stream = iter_with_heartbeat(source(), interval=0.01)
        self.assertIsNone(next(stream))
        with self.assertRaisesRegex(RuntimeError, "connection failed"):
            next(stream)

    def test_heartbeat_precedes_delayed_result_and_disconnect_closes_source(self):
        ready = Event()
        closed = Event()

        def source():
            try:
                ready.wait(1)
                yield {"type": "delta", "content": "OK"}
                yield {"type": "done"}
            finally:
                closed.set()

        stream = iter_with_heartbeat(source(), interval=0.01)
        self.assertIsNone(next(stream))
        self.assertIsNone(next(stream))
        ready.set()
        self.assertEqual(next(stream)["content"], "OK")
        stream.close()
        self.assertTrue(closed.wait(1))


class ToolStreamTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with patch.object(sys, "argv", ["main.py", "--token", "test-only"]):
            cls.server = importlib.import_module("main")

    def chunks(self, events):
        with patch.object(self.server, "stream_genai_events", return_value=iter(events)):
            return list(self.server.stream_buffered_tool_response([], "deepseek-pro", 100))

    def test_length_never_executes_even_a_complete_tool_call(self):
        chunks = self.chunks([
            {"type": "delta", "content": '{"tool_calls":[{"name":"read","arguments":{}}]}'},
            {"type": "done", "finish_reason": "length"},
        ])
        self.assertIn('upstream_error', ''.join(chunks))
        self.assertNotIn('"finish_reason": "tool_calls"', ''.join(chunks))

    def test_plain_chat_preserves_length(self):
        events = [{"type": "delta", "content": "partial"}, {"type": "done", "finish_reason": "length"}]
        with patch.object(self.server, "stream_genai_events", return_value=iter(events)):
            chunks = list(self.server.stream_chat_completions_response([], "chatglm", 10))
        self.assertIn('"finish_reason": "length"', ''.join(chunks))
        self.assertNotIn('"finish_reason": "stop"', ''.join(chunks))

    def test_nonstream_chat_and_responses_preserve_incomplete_status(self):
        for path, body in [
            ('/v1/chat/completions', {"messages": [{"role": "user", "content": "hello"}]}),
            ('/v1/responses', {"input": "hello"}),
        ]:
            events = [{"type": "delta", "content": "partial"}, {"type": "done", "finish_reason": "length"}]
            with patch.object(self.server, "stream_genai_events", return_value=iter(events)):
                response = self.server.app.test_client().post(path, json={"model": "chatglm", **body})
            obj = response.get_json()
            self.assertEqual(response.status_code, 200)
            if path.endswith('completions'):
                self.assertEqual(obj['choices'][0]['finish_reason'], 'length')
            else:
                self.assertEqual(obj['status'], 'incomplete')

    def test_responses_stream_reports_incomplete(self):
        events = [{"type": "delta", "content": "partial"}, {"type": "done", "finish_reason": "length"}]
        with patch.object(self.server, "stream_genai_events", return_value=iter(events)):
            chunks = ''.join(self.server.stream_responses_api([], 'chatglm', 10))
        self.assertIn('response.incomplete', chunks)
        self.assertNotIn('response.completed', chunks)

    def test_screenshot_array_through_stream_and_nonstream_routes(self):
        call = {"id": "call_example", "type": "function", "function": {
            "name": "read_file", "arguments": '{"filePath":".gitignore","startLine":1,"endLine":40}',
        }}
        body = json.dumps([call]) + '\n(https://microsoft.com))'
        for stream in (False, True):
            events = [{"type": "delta", "content": body}, {"type": "done", "finish_reason": "stop"}]
            with patch.object(self.server, "stream_genai_events", return_value=iter(events)):
                response = self.server.app.test_client().post('/v1/chat/completions', json={
                    "model": "chatglm", "stream": stream,
                    "messages": [{"role": "user", "content": "读取文件"}],
                    "tools": [{"type": "function", "function": {"name": "read_file"}}],
                })
                self.assertEqual(response.status_code, 200)
                if stream:
                    payload = next(json.loads(line[6:]) for line in response.get_data(as_text=True).splitlines()
                                   if line.startswith('data: {') and 'tool_calls' in line)
                    actual = payload['choices'][0]['delta']['tool_calls'][0]
                    actual.pop('index')
                else:
                    actual = response.get_json()['choices'][0]['message']['tool_calls'][0]
                self.assertEqual(actual['id'], call['id'])
                self.assertEqual(json.loads(actual['function']['arguments']), json.loads(call['function']['arguments']))

    def test_complete_call_with_missing_wrapper_through_routes(self):
        # 对应真实上下文重放：调用及嵌套参数均完整，只缺外层 ]}。
        body = '{"tool_calls":[{"name":"read_file","arguments":{"filePath":"a.py"}}'
        for stream in (False, True):
            events = [{"type": "delta", "content": body}, {"type": "done", "finish_reason": "stop"}]
            with patch.object(self.server, "stream_genai_events", return_value=iter(events)):
                response = self.server.app.test_client().post('/v1/chat/completions', json={
                    "model": "deepseek-pro", "stream": stream,
                    "messages": [{"role": "user", "content": "读取文件"}],
                    "tools": [{"type": "function", "function": {"name": "read_file"}}],
                })
                self.assertEqual(response.status_code, 200)
                if stream:
                    text = response.get_data(as_text=True)
                    self.assertIn('"finish_reason": "tool_calls"', text)
                    self.assertNotIn('upstream_error', text)
                else:
                    choice = response.get_json()['choices'][0]
                    self.assertEqual(choice['finish_reason'], 'tool_calls')
                    args = json.loads(choice['message']['tool_calls'][0]['function']['arguments'])
                    self.assertEqual(args, {'filePath': 'a.py'})

    def test_empty_and_failed_upstream_are_explicit_errors(self):
        for events in ([{"type": "done"}], [{"type": "error", "error": "upstream unavailable"}]):
            chunks = self.chunks(events)
            self.assertEqual(chunks[0], ": keep-alive\n\n")
            self.assertIn('"type": "upstream_error"', "".join(chunks))
            self.assertEqual(chunks[-1], "data: [DONE]\n\n")
            self.assertNotIn('"finish_reason": "stop"', "".join(chunks))

    def test_tool_fragments_are_collected_before_parsing(self):
        chunks = self.chunks([
            {"type": "delta", "content": '{"tool_calls":[{"name":"read",'},
            {"type": "delta", "content": '"arguments":{"path":"a.py"}}]}'},
            {"type": "done"},
        ])
        payloads = [json.loads(chunk[6:]) for chunk in chunks if chunk.startswith("data: {")]
        self.assertEqual(payloads[0]["choices"][0]["delta"]["tool_calls"][0]["function"]["name"], "read")
        self.assertEqual(payloads[-1]["choices"][0]["finish_reason"], "tool_calls")

    def test_route_returns_headers_before_upstream_finishes(self):
        ready = Event()

        def source(*args):
            ready.wait(1)
            yield {"type": "delta", "content": "OK"}
            yield {"type": "done"}

        with patch.object(self.server, "stream_genai_events", side_effect=source):
            response = self.server.app.test_client().post("/v1/chat/completions", json={
                "model": "deepseek-pro", "stream": True,
                "messages": [{"role": "user", "content": "hello"}],
                "tools": [{"type": "function", "function": {"name": "read"}}],
            }, buffered=False)
            try:
                self.assertEqual(response.status_code, 200)
                self.assertEqual(next(response.response), b": keep-alive\n\n")
                ready.set()
                self.assertIn(b"OK", b"".join(response.response))
            finally:
                ready.set()
                response.close()

    def test_upstream_payload_sets_nonempty_chat_info_and_max_completion_tokens(self):
        fake = Mock(status_code=200)
        fake.iter_lines.return_value = iter([
            b'data: {"choices":[{"delta":{"content":"OK"},"finish_reason":"stop"}]}',
        ])
        with patch.object(self.server.requests, "post", return_value=fake) as upstream:
            response = self.server.app.test_client().post("/v1/chat/completions", json={
                "model": "Kimi-k3",
                "stream": False,
                "max_tokens": None,
                "max_completion_tokens": 1234,
                "messages": [
                    {"role": "developer", "content": "Always answer concisely."},
                    {"role": "user", "content": "hi"},
                ],
            })
        self.assertEqual(response.status_code, 200)
        sent_json = upstream.call_args.kwargs["json"]
        self.assertEqual(sent_json["maxToken"], 1234)
        self.assertEqual(sent_json["chatInfo"], "hi")
        self.assertEqual(sent_json["messages"], [{"role": "system", "content": "Always answer concisely."}])
        if self.server.GENAI_PROXIES is not None:
            self.assertEqual(upstream.call_args.kwargs["proxies"], self.server.GENAI_PROXIES)

    def test_responses_instructions_and_validate_cached_token(self):
        with patch.object(self.server, "stream_genai_events") as stream_mock:
            stream_mock.return_value = iter([{"type": "delta", "content": "OK"}, {"type": "done"}])
            resp = self.server.app.test_client().post("/v1/responses", json={
                "model": "qwen-instruct",
                "instructions": "Be brief.",
                "input": [{"role": "developer", "content": "Use plain text."}, {"role": "user", "content": "say OK"}],
                "max_output_tokens": None,
                "max_tokens": 321,
            })
        self.assertEqual(resp.status_code, 200)
        passed_msgs = stream_mock.call_args.args[0]
        self.assertEqual(passed_msgs[0]["role"], "system")
        self.assertIn("Be brief.", passed_msgs[0]["content"])
        self.assertIn("Use plain text.", passed_msgs[0]["content"])
        self.assertEqual(stream_mock.call_args.args[2], 321)

        ok_resp = Mock(status_code=200)
        ok_resp.json.return_value = {"success": True, "result": {"records": []}}
        bad_resp = Mock(status_code=200)
        bad_resp.json.return_value = {"success": False, "message": "Token失效"}
        with patch.object(self.server.requests, "get", side_effect=[ok_resp, bad_resp]):
            self.assertTrue(self.server.validate_cached_token("valid"))
            self.assertFalse(self.server.validate_cached_token("expired"))
