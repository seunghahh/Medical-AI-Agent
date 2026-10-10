"""Small Chat Completions client for Ollama and configurable compatible servers."""
import json
import os
import time
import urllib.error
import urllib.request


class ModelError(RuntimeError):
    pass


class ResponseFormatError(ModelError):
    pass


def parse_json(text):
    text = text.strip()
    if text.startswith("```") and text.endswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    obj = json.loads(text)
    if not isinstance(obj, dict):
        raise ValueError("Expected one JSON object")
    return obj


class ChatClient:
    def __init__(self, base_url, model, max_tokens=4096, reasoning="low", timeout=180):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.max_tokens = max_tokens
        self.reasoning = reasoning
        self.timeout = timeout
        self.calls = []

    def complete(self, system, payload, role, deadline):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise ModelError("Case deadline exceeded")
        own = [c for c in self.calls if c["role"] == role]
        if role == "doctor" and len(own) >= 200:
            raise ModelError("Doctor call budget exceeded")
        # These are session-wide limits for this process. Simulator calls are separate.
        if role == "doctor" and (sum(c["input_tokens"] for c in own) >= 500000
                                 or sum(c["output_tokens"] for c in own) >= 100000):
            raise ModelError("Doctor token budget exceeded")
        body = {"model": self.model, "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}],
            "temperature": 0, "max_tokens": self.max_tokens,
            "reasoning_effort": self.reasoning, "stream": False,
            "response_format": {"type": "json_object"}}
        key = os.getenv("CLINIC_API_KEY", "ollama")
        req = urllib.request.Request(self.base_url + "/chat/completions",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json", "Authorization": "Bearer " + key})
        started = time.monotonic()
        try:
            with urllib.request.urlopen(req, timeout=min(self.timeout, remaining)) as response:
                result = json.load(response)
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            # Do not emit raw provider errors: they can contain request data or credentials.
            raise ModelError(f"Model endpoint unavailable ({type(exc).__name__}). Check server, model ID and CLINIC_API_KEY.") from None
        usage = result.get("usage") or {}
        self.calls.append({"role": role, "seconds": time.monotonic() - started,
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
            "usage_reported": "prompt_tokens" in usage and "completion_tokens" in usage})
        try:
            choice = result["choices"][0]
            self.calls[-1]["final_source"] = "content"
            self.calls[-1]["finish_reason"] = choice.get("finish_reason")
            if choice.get("finish_reason") == "length":
                raise ModelError("Generation truncated; increase --max-tokens or shorten prompt")
            message = choice["message"]
            content = message.get("content")
            if not isinstance(content, str) or not content.strip():
                tool_calls = message.get("tool_calls") or []
                if tool_calls:
                    if (len(tool_calls) != 1 or tool_calls[0].get("type") != "function"
                            or tool_calls[0].get("function", {}).get("name") not in {"assistant", "ACTION"}):
                        raise ResponseFormatError("Unexpected tool call; expected one final JSON object")
                    # Observed Ollama wrappers contain JSON data, never executable functions.
                    content = tool_calls[0]["function"].get("arguments")
                    self.calls[-1]["final_source"] = tool_calls[0]["function"]["name"] + "_arguments"
            if not isinstance(content, str) or not content.strip():
                raise ResponseFormatError("No final answer content returned")
            return parse_json(content)
        except json.JSONDecodeError as exc:
            # shortcut: only one missing outer brace; use an enforcing backend if other syntax failures recur.
            if (choice.get("finish_reason") == "stop" and exc.pos == len(exc.doc)
                    and exc.doc.startswith("{") and exc.doc.endswith("}")):
                try:
                    repaired = parse_json(exc.doc + "}")
                except ValueError:
                    pass
                else:
                    self.calls[-1]["format_repair"] = {"kind": "missing_outer_closing_brace", "raw_final": content}
                    return repaired
            self.calls[-1]["format_error"] = {"message": exc.msg, "line": exc.lineno,
                "column": exc.colno, "position": exc.pos, "raw_final": content}
            raise ResponseFormatError(f"Invalid final JSON (JSONDecodeError): {exc.msg} at line {exc.lineno}, column {exc.colno}") from None
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise ResponseFormatError(f"Invalid final JSON ({type(exc).__name__})") from None
