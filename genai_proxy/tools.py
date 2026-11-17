"""Message normalization and prompt-based tool-call compatibility."""

import json
import logging
import re
import uuid


def convert_messages_to_genai_format(messages):
    """从消息列表中提取 GenAI 所需的 `chatInfo`。

    当前上游实际请求中 `chatInfo` 只使用最后一条用户消息内容，因此这里
    仅做最小提取。

    Args:
        messages (list[dict]): OpenAI 风格的消息列表。

    Returns:
        str: 最后一条用户消息的文本内容；若不存在则返回空字符串。
    """
    # 上游会单独接收一份 chatInfo，这里取最后一条 user 消息与网页行为对齐。
    for msg in reversed(messages):
        if not isinstance(msg, dict):
            continue
        if msg.get("role") == "user":
            text = normalize_content_for_genai(msg.get("content", ""))
            if text:
                return text

    return ""


def normalize_content_for_genai(content):
    """将 OpenAI 消息 content 归一化为上游可读文本。"""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts = []
        for part in content:
            if isinstance(part, dict):
                text = part.get("text")
                if isinstance(text, dict):
                    text = text.get("value")
                if text:
                    text_parts.append(text)
            elif isinstance(part, str):
                text_parts.append(part)
        return "\n".join(text_parts)
    return str(content)


def _message_has_image_part(message):
    content = message.get("content")
    if not isinstance(content, list):
        return False
    return any(
        isinstance(part, dict) and part.get("type") in {"image_url", "input_image"}
        for part in content
    )


def _collect_tool_call_names(messages):
    """收集历史 assistant 消息中的 tool_call_id -> 函数名映射。"""
    names = {}
    for message in messages:
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        for call in message.get("tool_calls") or []:
            if not isinstance(call, dict):
                continue
            call_id = call.get("id")
            fn_name = (call.get("function") or {}).get("name") if isinstance(call.get("function"), dict) else call.get("name")
            if isinstance(call_id, str) and call_id and isinstance(fn_name, str) and fn_name:
                names[call_id] = fn_name
    return names


def _ensure_leading_system_only(messages):
    """合并开头 system 消息，并将中间 system 消息转为 user 提示。

    部分上游模型（如 Qwen）严格要求 system 消息只能位于消息列表最前端。
    """
    leading_system_parts = []
    rest = []
    seen_non_system = False

    for message in messages:
        role = message.get("role", "user")
        content = message.get("content", "")
        if role == "system" and not seen_non_system:
            if content:
                leading_system_parts.append(content)
            continue

        seen_non_system = True
        if role == "system":
            if content:
                rest.append({"role": "user", "content": f"[系统提示]\n{content}"})
            continue
        rest.append(dict(message))

    result = []
    if leading_system_parts:
        result.append({"role": "system", "content": "\n\n".join(leading_system_parts)})
    result.extend(rest)
    return result


def normalize_messages_for_genai(messages, default_image_text=""):
    """归一化 OpenAI 消息为上游兼容格式。

    - 将 `developer` 统一映射为 `system`，避免部分模型报 422 或忽略指令；
    - 将 `tool`/`function` 消息降级为普通 `user` 文本；
    - 合并开头 `system` 并将非开头 `system` 转为 `user` 提示，满足首条 `system` 约束。
    """
    call_names = _collect_tool_call_names(messages)
    normalized_messages = []

    for message in messages:
        if not isinstance(message, dict):
            continue

        role = message.get("role", "user")
        content = normalize_content_for_genai(message.get("content"))

        if role == "developer":
            role = "system"

        if role in ("tool", "function"):
            tool_call_id = message.get("tool_call_id")
            explicit_name = message.get("name")
            resolved_name = call_names.get(tool_call_id) if tool_call_id else None
            if explicit_name and tool_call_id:
                tool_label = f"{explicit_name} ({tool_call_id})"
            else:
                tool_label = explicit_name or (f"{resolved_name} ({tool_call_id})" if resolved_name else tool_call_id) or "tool"
            normalized_messages.append({
                "role": "user",
                "content": f"工具 {tool_label} 返回结果：\n{content}",
            })
            continue

        tool_calls = message.get("tool_calls")
        function_call = message.get("function_call")
        if role == "assistant" and (tool_calls or function_call):
            serialized = json.dumps(tool_calls if tool_calls else [function_call], ensure_ascii=False)
            normalized_messages.append({
                "role": "assistant",
                "content": (content + "\n" if content else "") + "已请求调用工具：\n" + serialized,
            })
            continue

        if role == "user" and not content:
            if _message_has_image_part(message) and default_image_text:
                content = default_image_text
            else:
                continue

        if role in ("system", "assistant") and not content:
            continue

        normalized_messages.append({
            "role": role,
            "content": content,
        })

    return _ensure_leading_system_only(normalized_messages)


def split_messages_for_genai(messages):
    """拆分发送给 GenAI 上游的 `messages` 与 `chatInfo`。

    GenAI 后端会自动把 `chatInfo` 作为最后一条 `user` 消息追加到 `messages` 末尾。
    若保留 `chatInfo=""`，上游会在末尾注入空 user 消息导致 Kimi-k3 等模型直接报错。
    """
    copied = [dict(msg) for msg in messages if isinstance(msg, dict)]
    if copied and copied[-1].get("role") == "user":
        last_content = copied[-1].get("content") or ""
        if isinstance(last_content, str) and last_content.strip():
            return copied[:-1], last_content
    return copied, "请继续。"


def should_enable_tools(tools, tool_choice):
    """判断当前请求是否需要启用本地工具调用兼容层。"""
    return bool(tools) and tool_choice != "none"


def get_request_tools(req_data):
    """读取新版 tools 或旧版 functions 入参，统一为 OpenAI tools 结构。"""
    tools = req_data.get("tools")
    if tools:
        return tools

    functions = req_data.get("functions")
    if not functions:
        return []

    return [
        {
            "type": "function",
            "function": function,
        }
        for function in functions
        if isinstance(function, dict)
    ]


def get_request_tool_choice(req_data):
    """读取新版 tool_choice 或旧版 function_call 入参。"""
    if "tool_choice" in req_data:
        return req_data.get("tool_choice")

    function_call = req_data.get("function_call")
    if isinstance(function_call, dict) and function_call.get("name"):
        return {"name": function_call["name"]}
    return function_call


def normalize_tool_choice(tool_choice):
    """将 OpenAI 的 tool_choice 归一化为提示词中便于描述的约束。"""
    if tool_choice is None:
        return "auto"
    if isinstance(tool_choice, str):
        return tool_choice
    if isinstance(tool_choice, dict):
        if tool_choice.get("name"):
            return {"name": tool_choice["name"]}
        function_name = tool_choice.get("function", {}).get("name")
        if function_name:
            return {"name": function_name}
    return "auto"


def build_tool_calling_messages(messages, tools, tool_choice, parallel_tool_calls=True):
    """通过提示词工程让无原生工具调用能力的上游返回可解析的工具调用 JSON。"""
    normalized_choice = normalize_tool_choice(tool_choice)
    tool_prompt = [
        "你可以调用调用方提供的工具，但上游 API 没有原生 tool calling 能力。",
        "当你决定调用工具时，只输出一个合法的 JSON 对象，不要输出 Markdown 代码块、XML 标签、解释或额外文本。",
        "JSON 格式必须为：{\"tool_calls\":[{\"name\":\"工具名\",\"arguments\":{}}]}。",
        "arguments 必须是符合工具 JSON Schema 的对象；若工具无参数，arguments 设为 {}。",
    ]
    if parallel_tool_calls is False:
        tool_prompt.append("每轮最多调用一个工具。")
    tool_prompt.extend([
        "如果不需要调用工具，则正常回答用户，不要输出上述 JSON。",
        "如果需要通过工具查看文件或执行操作，本轮必须直接输出工具调用 JSON，不能只描述接下来要做什么。",
        f"tool_choice: {json.dumps(normalized_choice, ensure_ascii=False)}",
        "可用工具：",
        json.dumps(tools, ensure_ascii=False),
    ])

    if normalized_choice == "required":
        tool_prompt.append("本次请求必须调用至少一个工具。")
        reminder = "[工具协议提醒：本轮必须调用至少一个工具，请直接输出 {\"tool_calls\":[{\"name\":\"...\",\"arguments\":{}}]} JSON，不要输出普通正文、Markdown 或 XML 标签。]"
    elif isinstance(normalized_choice, dict):
        tool_prompt.append(f"本次请求必须调用工具 {normalized_choice['name']}。")
        reminder = (
            f"[工具协议提醒：本轮必须调用工具 {normalized_choice['name']}，"
            f"请直接输出 {{\"tool_calls\":[{{\"name\":\"{normalized_choice['name']}\",\"arguments\":{{}}}}]}} JSON，不要输出普通正文、Markdown 或 XML 标签。]"
        )
    else:
        reminder = "[工具协议提醒：若需调用工具，请直接输出 {\"tool_calls\":[{\"name\":\"...\",\"arguments\":{}}]} JSON，不要只输出行动预告、Markdown 或 XML 标签；若无需调用工具，请直接给出最终回答。]"

    prepared = _ensure_leading_system_only([dict(m) for m in messages if isinstance(m, dict)])
    tool_prompt_text = "\n".join(tool_prompt)

    # 将完整工具定义放入首条 system，避免尾部 system 导致 Qwen 等模型报 400/500。
    if prepared and prepared[0].get("role") == "system":
        existing = prepared[0].get("content") or ""
        prepared[0]["content"] = f"{existing}\n\n{tool_prompt_text}" if existing else tool_prompt_text
    else:
        prepared.insert(0, {"role": "system", "content": tool_prompt_text})

    # 同时在末尾 user 轮附带精简强提示，防止长上下文下模型只输出行动预告而不调用工具。
    if prepared and prepared[-1].get("role") == "user":
        last_content = prepared[-1].get("content") or ""
        prepared[-1]["content"] = f"{last_content}\n\n{reminder}" if last_content else reminder
    else:
        prepared.append({"role": "user", "content": reminder})

    return prepared


def strip_json_code_fence(text):
    """去掉模型偶尔包裹的 JSON Markdown 代码块。"""
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped

    lines = stripped.splitlines()
    if len(lines) >= 2 and lines[-1].strip() == "```":
        return "\n".join(lines[1:-1]).strip()
    return stripped


def _strip_leading_think_block(text):
    """若文本以闭合的 <think>...</think> 开头且后续仍有内容，剥离思考块以便提取工具 JSON。"""
    match = re.match(r"^\s*<think>.*?</think>\s*(.+)$", text, flags=re.DOTALL)
    return match.group(1) if match else text


def is_tool_call_object(value):
    """普通 JSON 的 name 字段不代表工具调用。"""
    if not isinstance(value, dict):
        return False
    function = value.get("function")
    return (
        isinstance(function, dict) and "name" in function
        or "name" in value and "arguments" in value
    )


def _is_tool_payload(value):
    if isinstance(value, dict):
        return "tool_calls" in value or is_tool_call_object(value)
    if isinstance(value, list):
        return any(is_tool_call_object(item) for item in value)
    return False


def looks_like_tool_container(text):
    """只检查看似 JSON 的容器，不用正文里的字段名猜测工具协议。"""
    if not re.match(r'^\s*[\[{]\s*(?:[\[{\"]|$)', text):
        return False
    return bool(re.search(
        r'"tool_calls"\s*:|"function"\s*:\s*\{|"name"\s*:\s*"[^"\n]+"\s*,\s*"arguments"\s*:',
        text,
    ))


def repair_tool_wrapper(text):
    """只补齐完整调用外缺少的数组/包装对象闭合，不补参数或字符串。"""
    stack = []
    in_string = False
    escaped = False
    for char in text:
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
        elif char == '"':
            in_string = True
        elif char in "[{":
            stack.append(char)
        elif char in "]}":
            if not stack or stack.pop() != ("[" if char == "]" else "{"):
                return None
    if in_string or stack not in (["["], ["{"], ["{", "["]):
        return None
    suffix = "".join("]" if char == "[" else "}" for char in reversed(stack))
    try:
        parsed = json.loads(text + suffix)
    except json.JSONDecodeError:
        return None
    calls = parsed.get("tool_calls") if isinstance(parsed, dict) else parsed
    if not isinstance(calls, list) or not calls or not all(is_tool_call_object(call) for call in calls):
        return None
    # 调用对象和参数必须在原输出中闭合；只允许其外部包装缺少闭合。
    if isinstance(parsed, dict) and set(parsed) != {"tool_calls"}:
        return None
    logging.getLogger("genai-proxy").warning(
        "Repaired missing outer tool wrapper: closing_chars=%s", len(suffix),
    )
    return parsed


def extract_json_object(text):
    """从文本中提取工具调用容器或完整 JSON；优先识别工具协议对象。"""
    stripped = strip_json_code_fence(_strip_leading_think_block(text))
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass

    decoder = json.JSONDecoder()
    pos = 0
    first_non_tool_value = None
    collected_bare_calls = []

    while pos < len(stripped):
        match = re.search(r"[\[{]", stripped[pos:])
        if not match:
            break
        start = pos + match.start()
        candidate = stripped[start:]
        try:
            value, end_offset = decoder.raw_decode(stripped, start)
            if _is_tool_payload(value):
                if isinstance(value, dict) and "tool_calls" not in value and is_tool_call_object(value):
                    collected_bare_calls.append(value)
                    pos = end_offset
                    continue
                return value
            if first_non_tool_value is None:
                first_non_tool_value = value
            pos = end_offset
        except json.JSONDecodeError:
            if looks_like_tool_container(candidate):
                repaired = repair_tool_wrapper(candidate)
                if repaired is not None:
                    return repaired
                raise ValueError("Invalid or incomplete tool call JSON")
            pos = start + 1

    if collected_bare_calls:
        return collected_bare_calls if len(collected_bare_calls) > 1 else collected_bare_calls[0]
    return first_non_tool_value


def normalize_tool_call_arguments(arguments):
    """OpenAI 要求 function.arguments 是 JSON 字符串。"""
    if isinstance(arguments, str):
        return arguments
    if arguments is None:
        return "{}"
    return json.dumps(arguments, ensure_ascii=False)


def _try_parse_whole_json_payload(content):
    """仅当整段输出可直接解析为工具 JSON 时优先走 JSON 通道，避免参数内含 <tool_call> 字样误触发 XML 分支。"""
    stripped = strip_json_code_fence(_strip_leading_think_block(content))
    if not stripped.startswith(("{", "[")):
        return None
    try:
        parsed = json.loads(stripped)
        if _is_tool_payload(parsed):
            return parsed
    except json.JSONDecodeError:
        pass
    return None


def parse_tool_calls_from_content(content, tools=None, tool_choice=None):
    """解析工具对象、数组和 XML；只返回已声明且参数为 JSON 对象的调用。"""
    if not content:
        return []

    content_for_parse = _strip_leading_think_block(content)
    whole_json = _try_parse_whole_json_payload(content_for_parse)

    if whole_json is None and "<tool_call>" in content_for_parse:
        blocks = re.findall(r"<tool_call>\s*(.*?)\s*</tool_call>", content_for_parse, flags=re.DOTALL)
        if len(blocks) != content_for_parse.count("<tool_call>"):
            raise ValueError("Incomplete tool_call block")
        raw_calls = []
        for block in blocks:
            parsed = extract_json_object(block)
            if not isinstance(parsed, dict):
                raise ValueError("Invalid tool_call JSON")
            raw_calls.append(parsed)
    else:
        parsed = whole_json if whole_json is not None else extract_json_object(content_for_parse)
        if isinstance(parsed, list):
            # 普通答案也可能是 JSON 数组，只有调用形状才当作工具协议。
            if not any(is_tool_call_object(item) for item in parsed):
                raw_calls = []
            else:
                raw_calls = parsed
        elif isinstance(parsed, dict):
            raw_calls = parsed.get("tool_calls")
            if raw_calls is None and is_tool_call_object(parsed):
                raw_calls = [parsed]
            if raw_calls is None:
                raw_calls = []
        else:
            raw_calls = []

    if not isinstance(raw_calls, list):
        raise ValueError("tool_calls must be an array")
    allowed_names = None if tools is None else {
        tool.get("function", {}).get("name") for tool in tools if isinstance(tool, dict)
    }
    tool_calls = []
    for raw_call in raw_calls:
        if not isinstance(raw_call, dict):
            raise ValueError("Each tool call must be an object")
        function = raw_call.get("function") if isinstance(raw_call.get("function"), dict) else raw_call
        name = function.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Tool call is missing a function name")
        if allowed_names is not None and name not in allowed_names:
            raise ValueError(f"Unknown tool: {name}")
        arguments = function.get("arguments", {})
        if arguments is None:
            arguments = {}
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON arguments for tool: {name}") from exc
        if not isinstance(arguments, dict):
            raise ValueError(f"Tool arguments must be an object: {name}")
        call_id = raw_call.get("id")
        tool_calls.append({
            "id": call_id if isinstance(call_id, str) and call_id else f"call_{uuid.uuid4().hex[:24]}",
            "type": "function",
            "function": {"name": name, "arguments": normalize_tool_call_arguments(arguments)},
        })

    choice = normalize_tool_choice(tool_choice)
    if choice == "required" and not tool_calls:
        raise ValueError("Upstream did not return the required tool call")
    if isinstance(choice, dict) and (not tool_calls or any(call["function"]["name"] != choice["name"] for call in tool_calls)):
        raise ValueError(f"Upstream did not call the selected tool: {choice['name']}")
    return tool_calls
