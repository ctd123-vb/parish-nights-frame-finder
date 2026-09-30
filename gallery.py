#!/usr/bin/env python3
"""Step 9: approval gallery on a small local server.

    python gallery.py                 # then open http://localhost:8765
    python gallery.py --port 9000 --keepers /path/to/keepers

Keyboard: K = keep, J = reject, U = undo, left/right arrows = move.
Keep copies the full-resolution frame to keepers/. Reject moves it from output/<video>/shortlist/
to output/<video>/rejected/ (and removes it from keepers/ if it was kept before).
Every decision is saved to output/decisions.json and to the decision column of output/results.csv.
"""
import argparse
import json
import shutil
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote

from framefinder.results import build_results, load_json

HERE = Path(__file__).resolve().parent
lock = threading.Lock()


class App:
    def __init__(self, output: Path, keepers: Path):
        self.output, self.keepers = output.resolve(), keepers.resolve()
        self.keepers.mkdir(parents=True, exist_ok=True)
        self.decisions_path = self.output / "decisions.json"

    def items(self):
        return build_results(self.output)

    def _video_file(self, item_id: str):
        video, _, name = item_id.partition("/")
        vdir = (self.output / video).resolve()
        if vdir.parent != self.output or "/" in name or not name.endswith(".jpg"):
            raise ValueError("bad id")
        return vdir, name

    def full_path(self, item_id: str) -> Path | None:
        vdir, name = self._video_file(item_id)
        for sub in ("shortlist", "rejected"):
            if (vdir / sub / name).exists():
                return vdir / sub / name
        return None

    def decide(self, item_id: str, decision: str):
        vdir, name = self._video_file(item_id)
        with lock:
            decisions = load_json(self.decisions_path, {})
            short, rej, keep = vdir / "shortlist" / name, vdir / "rejected" / name, self.keepers / name
            # always start from "in shortlist, not kept"
            if rej.exists():
                shutil.move(rej, short)
            keep.unlink(missing_ok=True)
            if decision == "keep":
                shutil.copy2(short, keep)
                decisions[item_id] = "keep"
            elif decision == "reject":
                rej.parent.mkdir(exist_ok=True)
                shutil.move(short, rej)
                decisions[item_id] = "reject"
            else:
                decisions.pop(item_id, None)
            self.decisions_path.write_text(json.dumps(decisions, indent=2))
            build_results(self.output)
        return decisions.get(item_id, "")


def make_handler(app: App):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, body: bytes, ctype="application/json"):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store" if ctype == "application/json" else "max-age=3600")
            self.end_headers()
            self.wfile.write(body)

        def _file(self, path: Path | None):
            if path and path.exists():
                self._send(200, path.read_bytes(), "image/jpeg")
            else:
                self._send(404, b"not found", "text/plain")

        def do_GET(self):
            p = unquote(self.path.split("?")[0])
            try:
                if p == "/":
                    self._send(200, (HERE / "gallery.html").read_bytes(), "text/html; charset=utf-8")
                elif p == "/api/items":
                    self._send(200, json.dumps({"items": app.items(), "keepers": str(app.keepers)}).encode())
                elif p.startswith("/thumb/"):
                    vdir, name = app._video_file(p[len("/thumb/"):])
                    self._file(vdir / "thumbs" / name)
                elif p.startswith("/full/"):
                    self._file(app.full_path(p[len("/full/"):]))
                else:
                    self._send(404, b"not found", "text/plain")
            except ValueError:
                self._send(400, b"bad request", "text/plain")

        def do_POST(self):
            if self.path != "/api/decide":
                return self._send(404, b"not found", "text/plain")
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                if body.get("decision") not in ("keep", "reject", "undo"):
                    raise ValueError("bad decision")
                result = app.decide(body["id"], body["decision"])
                self._send(200, json.dumps({"id": body["id"], "decision": result}).encode())
            except (ValueError, KeyError, FileNotFoundError) as e:
                self._send(400, json.dumps({"error": str(e)}).encode())

    return Handler


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output", type=Path, default=Path("output"))
    ap.add_argument("--keepers", type=Path, default=Path("keepers"))
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()
    app = App(args.output, args.keepers)
    n = len(app.items())
    server = ThreadingHTTPServer((args.host, args.port), make_handler(app))
    print(f"Gallery: http://localhost:{args.port}  ({n} frames, keepers -> {app.keepers})")
    print("Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
