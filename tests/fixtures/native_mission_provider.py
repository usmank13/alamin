"""Loopback Anthropic Messages fixture for the native Rhizome mission check.

Run with Python, set ANTHROPIC_BASE_URL to the printed URL and configure the
native agent's anthropic key file to contain alamin-fixture. Only the exact
test mission triggers the predefined tool call. The agent owns all tool I/O.
"""
import json
from http.server import BaseHTTPRequestHandler, HTTPServer

MISSION = "Drive straight ahead 1.2 meters."
KEY = "alamin-fixture"


def reply(messages):
    mission = next((i for i, m in enumerate(messages) if m["role"] == "user"
                    and isinstance(m["content"], str)
                    and m["content"].endswith("] " + MISSION)), None)
    if mission is None:
        return {"type": "text", "text": "Awaiting fixture mission."}
    turns = messages[mission:]
    called = {part["name"] for turn in turns if turn["role"] == "assistant"
              for part in turn["content"] if isinstance(part, dict) and part.get("type") == "tool_use"}
    if "push_waypoints" not in called:
        return {"type": "tool_use", "id": "tool_leg", "name": "push_waypoints",
                "input": {"points": [[1.2, 0.0]], "frame": "body", "wake_before_s": 0}}
    return {"type": "text", "text": "Fixture turn complete."}


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path.split("?")[0] != "/v1/messages" or self.headers.get("x-api-key") != KEY:
            self.send_error(403)
            return
        request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        block = reply(request["messages"])
        print(block.get("name", "end_turn"), flush=True)
        response = json.dumps({"id": "msg_fixture", "type": "message", "role": "assistant",
                               "model": request["model"], "content": [block],
                               "stop_reason": "tool_use" if block["type"] == "tool_use" else "end_turn",
                               "stop_sequence": None, "usage": {"input_tokens": 1, "output_tokens": 1}}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(response)))
        self.end_headers()
        self.wfile.write(response)

    def log_message(self, *_):
        pass


if __name__ == "__main__":
    with HTTPServer(("127.0.0.1", 0), Handler) as server:
        print(f"http://127.0.0.1:{server.server_port}", flush=True)
        server.serve_forever()
