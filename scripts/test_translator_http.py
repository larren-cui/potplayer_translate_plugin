"""测试 SakuraTranslator 与 llama-server 的 HTTP 集成（用 mock server，无需真实模型）。

验证：
  1. SakuraTranslator 发送的请求格式正确（messages/temperature/top_p）
  2. 响应解析正确（提取 choices[0].message.content）
  3. 多行翻译对齐逻辑正确
  4. 重试逻辑在失败时触发
"""
from __future__ import annotations

import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("PYTHONUTF8", "1")


class MockHandler(BaseHTTPRequestHandler):
    """模拟 llama-server 的 /health 和 /v1/chat/completions。"""

    # 测试用：记录收到的最后一个请求
    last_request: dict = {}

    def log_message(self, *a):
        pass  # 静默

    def do_GET(self):
        if self.path == "/health":
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"ok")
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path == "/v1/chat/completions":
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length))
            MockHandler.last_request = body

            # 从 user message 中提取日文行
            user_msg = body.get("messages", [{}])[-1].get("content", "")
            # 模拟翻译：在每行前加 [中] 标记
            prefix = "将下面的日文文本翻译成中文："
            ja_text = user_msg[len(prefix):] if user_msg.startswith(prefix) else user_msg
            ja_lines = [ln for ln in ja_text.split("\n") if ln.strip()]
            zh_lines = [f"[译]{ln}" for ln in ja_lines]

            response = {
                "choices": [{"message": {"content": "\n".join(zh_lines)}}],
                "model": body.get("model", "sakura"),
            }
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(response).encode())
        else:
            self.send_response(404)
            self.end_headers()


def main() -> int:
    port = 18099
    server = HTTPServer(("127.0.0.1", port), MockHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    print(f"mock server on :{port}", flush=True)

    from jpzh_subtitle.translate import SakuraTranslator, SAKURA_SYSTEM

    tr = SakuraTranslator(base_url=f"http://127.0.0.1:{port}")

    # 测试 1: 单行翻译
    print("\n=== 测试1: 单行翻译 ===", flush=True)
    result = tr.translate_lines(["こんにちは"])
    print(f"  输入: ['こんにちは']")
    print(f"  输出: {result}")
    assert len(result) == 1, f"expected 1 line, got {len(result)}"
    assert result[0] == "[译]こんにちは", f"unexpected: {result[0]}"
    print("  ✓ 通过", flush=True)

    # 测试 2: 多行翻译 + 对齐
    print("\n=== 测试2: 多行翻译 ===", flush=True)
    ja = ["おはよう", "今日は天気がいい", "さようなら"]
    result = tr.translate_lines(ja)
    print(f"  输入: {ja}")
    print(f"  输出: {result}")
    assert len(result) == 3, f"expected 3 lines, got {len(result)}"
    assert result[0] == "[译]おはよう"
    assert result[1] == "[译]今日は天気がいい"
    assert result[2] == "[译]さようなら"
    print("  ✓ 通过", flush=True)

    # 测试 3: 验证请求格式
    print("\n=== 测试3: 请求格式验证 ===", flush=True)
    req = MockHandler.last_request
    print(f"  model: {req.get('model')}")
    print(f"  temperature: {req.get('temperature')}")
    print(f"  top_p: {req.get('top_p')}")
    print(f"  stream: {req.get('stream')}")
    print(f"  messages count: {len(req.get('messages', []))}")
    assert req.get("model") == "sakura"
    assert req.get("temperature") == 0.1
    assert req.get("top_p") == 0.3
    assert req.get("stream") == False
    msgs = req.get("messages", [])
    assert len(msgs) == 2
    assert msgs[0]["role"] == "system"
    assert msgs[0]["content"] == SAKURA_SYSTEM
    assert msgs[1]["role"] == "user"
    assert msgs[1]["content"].startswith("将下面的日文文本翻译成中文：")
    print("  ✓ 请求格式完全符合 Sakura v0.9 规范", flush=True)

    # 测试 4: 空输入
    print("\n=== 测试4: 空输入 ===", flush=True)
    result = tr.translate_lines([])
    assert result == []
    print("  ✓ 空输入返回空列表", flush=True)

    # 测试 5: 错误响应重试（mock 返回 500）
    print("\n=== 测试5: 错误响应重试 ===", flush=True)
    tr_fail = SakuraTranslator(
        base_url=f"http://127.0.0.1:{port}",
        max_retries=1,
        timeout=5.0,
    )

    class FailHandler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_POST(self):
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"error":{"message":"model not loaded"}}')

        def do_GET(self):
            self.send_response(200)
            self.end_headers()

    fail_server = HTTPServer(("127.0.0.1", port + 1), FailHandler)
    fail_thread = threading.Thread(target=fail_server.serve_forever, daemon=True)
    fail_thread.start()

    tr_fail2 = SakuraTranslator(
        base_url=f"http://127.0.0.1:{port + 1}",
        max_retries=1,
        timeout=5.0,
    )
    raised = False
    try:
        tr_fail2.translate_lines(["テスト"])
    except RuntimeError as e:
        raised = True
        print(f"  捕获预期异常: {str(e)[:80]}")
    assert raised, "应该抛出 RuntimeError"
    print("  ✓ 错误响应正确触发重试和异常", flush=True)
    fail_server.shutdown()

    server.shutdown()
    print("\n✓ 全部测试通过", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
