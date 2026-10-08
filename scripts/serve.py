"""Serves a Laya checkpoint with Jev's request and response format on any POST path.

python scripts/serve.py laya-routing
HOST, PORT (default 8000), DEVICE and API_KEY (optional Bearer key) configure it.
"""
import json, os, sys, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import laya, torch

DEVICE = os.environ.get("DEVICE") or ("cuda" if torch.cuda.is_available()
                                      else "mps" if torch.backends.mps.is_available() else "cpu")
KEY = os.environ.get("API_KEY", "")
agent = laya.load(sys.argv[1], device=DEVICE)
lock = threading.Lock()
print("loaded", sys.argv[1], "on", DEVICE, flush=True)

class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def reply(self, code, body, ms=None):
        data = json.dumps(body, default=float).encode()
        self.send_response(code)
        self.send_header("content-type", "application/json")
        if ms is not None:
            self.send_header("x-server-ms", f"{ms:.1f}")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        self.reply(200, {"status": "ok", "model": os.path.basename(sys.argv[1].rstrip("/"))})

    def do_POST(self):
        if KEY and self.headers.get("Authorization", "") != "Bearer " + KEY:
            return self.reply(401, {"error": "unauthorized"})
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        with lock:
            start = time.perf_counter()
            out = agent.system_one(body["state"], body["questions"])
            ms = (time.perf_counter() - start) * 1000
        print(f"ok {len(body['questions'])}q {ms:.0f}ms", flush=True)
        self.reply(200, out, ms)

ThreadingHTTPServer((os.environ.get("HOST", "127.0.0.1"), int(os.environ.get("PORT", "8000"))), Handler).serve_forever()
