"""Serve a viewer directory over http and collect the timings the pages POST back.

The dense pages fetch their binary payloads, which a file:// page is not allowed to do, so they
have to be measured from a real origin. The pages also post their frame timings to /__report when
they are opened with ?bench, and this server appends them to a JSON file:

    .venv/bin/python method/geometry/serve_viewer.py <dir> --port 8811 --report out/bench.json
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" --headless=new \\
        --user-data-dir=$PWD/.chrome-bench --no-sandbox --enable-unsafe-swiftshader \\
        --window-size=1400,900 "http://127.0.0.1:8811/<page>.html?bench"
"""
import argparse
import json
import sys
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

REPORT = {"path": None}


class Handler(SimpleHTTPRequestHandler):
    def do_POST(self):  # noqa: N802  (http.server naming)
        if self.path != "/__report":
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length)
        try:
            record = json.loads(body)
        except ValueError:
            self.send_error(400)
            return
        if REPORT["path"] is not None:
            with Path(REPORT["path"]).open("a") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        print("report: " + json.dumps(record, ensure_ascii=False), flush=True)
        self.send_response(204)
        self.end_headers()

    def log_message(self, pattern, *values):
        sys.stderr.write("%s\n" % (pattern % values))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("directory")
    parser.add_argument("--port", type=int, default=8811)
    parser.add_argument("--report", default="")
    args = parser.parse_args()
    if args.report:
        REPORT["path"] = args.report
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text("")
    handler = lambda *a, **k: Handler(*a, directory=args.directory, **k)  # noqa: E731
    server = ThreadingHTTPServer(("127.0.0.1", args.port), handler)
    print("serving %s on http://127.0.0.1:%d" % (args.directory, args.port), flush=True)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
