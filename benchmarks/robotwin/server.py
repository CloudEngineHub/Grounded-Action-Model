"""GAM policy server for RoboTwin.

RoboTwin runs in its own Python environment; its policy client (client/deploy_policy.py) posts
each observation here and receives an action chunk.

    python benchmarks/robotwin/server.py --config configs/robotwin/shake_bottle.yaml [--ckpt my.pt] [--port 18120]

POST /reset {instruction, seed}                      -> {slots, eval}
POST /act   {rgb, depth, K, E, state, prefix?}       -> {actions (H, 14), boxes}
POST /boxes {rgb, depth, K, E, state}                -> {boxes}            (overlay videos)

The perception + policy is GAM.eval.runner.GAMRunner (same code path as the training-cache precompute);
this file only speaks JSON.
"""
import argparse, base64, io, json, os, sys, threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from GAM.config import load_config
from GAM.eval.runner import GAMRunner
from GAM.hub import fetch


def _unb64(s):
    return np.load(io.BytesIO(base64.b64decode(s)))["a"]


class Server:
    """GAMRunner behind the JSON protocol (base64 arrays in, lists out)."""

    def __init__(self, cfg, ckpt):
        self.cfg = cfg; self.runner = GAMRunner(cfg, ckpt); self.slots = self.runner.slots; self.lock = threading.Lock()

    @property
    def t(self):
        return self.runner.t

    def _args(self, req):
        return (_unb64(req["rgb"]), _unb64(req["depth"]), req["K"], req["E"], req["state"])

    def reset(self, req):
        self.runner.reset(req.get("instruction", ""), int(req.get("seed", 0)))
        return {"ok": True, "slots": self.slots, "eval": self.cfg["eval"]}

    def boxes(self, req):
        return {"boxes": self.runner.boxes(*self._args(req))}

    def act(self, req):
        pf = req.get("prefix")
        chunk, boxes = self.runner.act(*self._args(req), prefix=(np.asarray(pf, np.float32) if pf else None))
        return {"actions": chunk.tolist(), "n_found": self.runner.last_n_found, "ms": self.runner.last_ms, "boxes": boxes}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True); ap.add_argument("--ckpt", default=None); ap.add_argument("--port", type=int, default=18120)
    a = ap.parse_args()
    cfg = load_config(a.config)
    S = Server(cfg, fetch(a.ckpt or cfg["checkpoints"]["policy"]))

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args): pass

        def _send(self, obj, code=200):
            b = json.dumps(obj).encode(); self.send_response(code); self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)

        def do_GET(self): self._send({"ok": True, "t": S.t})

        def do_POST(self):
            n = int(self.headers.get("Content-Length", 0)); req = json.loads(self.rfile.read(n) or b"{}")
            try:
                with S.lock:
                    self._send({"/reset": S.reset, "/boxes": S.boxes}.get(self.path, S.act)(req))
            except Exception as e:  # noqa: BLE001
                import traceback; traceback.print_exc(); self._send({"error": str(e)}, 500)

    print(f"[server] listening on :{a.port}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", a.port), Handler).serve_forever()   # local only (no authentication)


if __name__ == "__main__":
    main()
