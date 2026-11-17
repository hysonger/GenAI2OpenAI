import unittest
import json

from genai_proxy.tools import (
    build_tool_calling_messages,
    normalize_messages_for_genai,
    parse_tool_calls_from_content,
    split_messages_for_genai,
)
from genai_proxy.upstream import iter_sse_events


class UpstreamSseTests(unittest.TestCase):
    def test_eof_is_not_reported_as_normal_completion(self):
        events = list(iter_sse_events(['data: {"choices":[{"delta":{"content":"partial"}}]}']))
        self.assertEqual(events[-1]["type"], "error")

    def test_length_and_done_marker_are_preserved(self):
        events = list(iter_sse_events(['data: {"choices":[{"delta":{"content":"last"},"finish_reason":"length"}]}']))
        self.assertEqual(events[0]["content"], "last")
        self.assertEqual(events[-1]["finish_reason"], "length")
        events = list(iter_sse_events([': keep-alive', 'data: [DONE]']))
        self.assertEqual(events[-1]["finish_reason"], "stop")

    def test_tool_protocol_follows_client_context_without_mutating_it(self):
        messages = [
            {"role": "system", "content": "Use native tools to inspect files."},
            {"role": "user", "content": "Inspect this project."},
        ]
        tools = [{"type": "function", "function": {"name": "glob"}}]

        result = build_tool_calling_messages(messages, tools, "auto")

        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0]["content"], "Use native tools to inspect files.")
        self.assertEqual(messages[1]["content"], "Inspect this project.")
        self.assertEqual(result[0]["role"], "system")
        self.assertTrue(all(msg["role"] != "system" for msg in result[1:]))
        self.assertIn("Use native tools to inspect files.", result[0]["content"])
        self.assertIn('"tool_calls"', result[0]["content"])
        self.assertIn('"name": "glob"', result[0]["content"])
        self.assertNotIn("<tool_call>", result[0]["content"])
        self.assertEqual(result[-1]["role"], "user")
        self.assertIn("Inspect this project.", result[-1]["content"])
        self.assertIn("工具协议提醒", result[-1]["content"])

    def test_tool_prompt_and_chat_info_split_with_trailing_assistant(self):
        messages = normalize_messages_for_genai([
            {"role": "developer", "content": "Follow rules."},
            {"role": "user", "content": "Check repo."},
            {"role": "assistant", "content": "Checking."},
        ])
        tools = [{"type": "function", "function": {"name": "glob"}}]

        prepared = build_tool_calling_messages(messages, tools, "required", parallel_tool_calls=False)
        payload_msgs, chat_info = split_messages_for_genai(prepared)

        self.assertEqual(prepared[0]["role"], "system")
        self.assertIn("每轮最多调用一个工具", prepared[0]["content"])
        self.assertEqual(prepared[-1]["role"], "user")
        self.assertIn("必须调用至少一个工具", prepared[-1]["content"])
        self.assertEqual(chat_info, prepared[-1]["content"])
        self.assertTrue(all(m["role"] != "system" for m in payload_msgs[1:]))

    def test_last_delta_is_kept_when_finish_reason_is_present(self):
        fragments = ['{"tool_calls":[', '{"name":"weather","arguments":{}}]}']
        lines = [
            'data: ' + json.dumps({"choices": [{
                "delta": {"content": fragment},
                "finish_reason": "stop" if index else None,
            }]})
            for index, fragment in enumerate(fragments)
        ]

        events = list(iter_sse_events(lines))
        content = "".join(
            event.get("content") or ""
            for event in events
            if event["type"] == "delta"
        )

        self.assertEqual(content, "".join(fragments))
        self.assertEqual(parse_tool_calls_from_content(content)[0]["function"]["name"], "weather")
        self.assertEqual(events[-1]["type"], "done")

    def test_business_error_is_not_an_empty_success(self):
        for payload in [
            {"code": 500, "errMsg": "未找到模型"},
            {"success": False, "message": "Token失效，请重新登录"},
        ]:
            with self.subTest(payload=payload):
                events = list(iter_sse_events(['data: ' + json.dumps(payload)]))
                self.assertEqual([event["type"] for event in events], ["error"])
                self.assertEqual(events[0]["error"], payload.get("errMsg") or payload["message"])

    def test_tool_call_json_is_parsed_after_stream_aggregation(self):
        content = '{"tool_calls":[{"name":"weather","arguments":{"city":"上海"}}]}'

        calls = parse_tool_calls_from_content(content)

        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["function"]["name"], "weather")
        self.assertEqual(calls[0]["function"]["arguments"], '{"city": "上海"}')


if __name__ == "__main__":
    unittest.main()
