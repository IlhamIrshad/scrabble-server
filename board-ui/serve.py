#!/usr/bin/env python3
"""Serve this directory with caching turned off.

    python serve.py [port]        # default 8200

`python -m http.server` sends no cache headers at all, which leaves the browser free to
guess how long a file stays fresh. It guesses generously, so an edited module keeps
being served from cache and the page goes on running the previous version - which looks
exactly like a change that did not work, and costs an hour before anyone suspects the
cache. Saying `no-store` outright removes the guess.

Nothing else differs from the standard server, and nothing here is needed in
production: these are static files and any web server will do.
"""

from __future__ import annotations

import sys
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


class NoCacheHandler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cache-Control", "no-store, must-revalidate")
        self.send_header("Expires", "0")
        super().end_headers()

    def log_message(self, fmt, *args):
        # One line per request is noise when the page pulls a dozen modules; errors
        # still come through, which is the part worth seeing.
        if not str(args[1] if len(args) > 1 else "").startswith("2"):
            super().log_message(fmt, *args)


def main() -> None:
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8200
    here = Path(__file__).resolve().parent
    handler = partial(NoCacheHandler, directory=str(here))
    with ThreadingHTTPServer(("127.0.0.1", port), handler) as httpd:
        print(f"serving {here.name} on http://127.0.0.1:{port}  (caching off)")
        print("the board reader is a separate service; see README.md")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nstopped")


if __name__ == "__main__":
    main()
