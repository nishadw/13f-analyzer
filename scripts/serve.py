"""
Start the API backend server.

Locally: runs HTTPS (self-signed cert) to avoid mixed-content issues.
Production (Render): runs plain HTTP — Render handles TLS at the edge.

Usage:
    python scripts/serve.py           # auto-detects SSL
    python scripts/serve.py --reload  # dev hot-reload
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import uvicorn

_CERT = Path(__file__).parent.parent / "api_backend" / "cert.pem"
_KEY  = Path(__file__).parent.parent / "api_backend" / "key.pem"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reload", action="store_true")
    parser.add_argument("--port",   type=int, default=7779)
    parser.add_argument("--host",   default="0.0.0.0")
    args = parser.parse_args()

    use_ssl = _CERT.exists() and _KEY.exists()
    scheme  = "https" if use_ssl else "http"
    print(f"\n  Backend: {scheme}://{args.host}:{args.port}\n")

    uvicorn.run(
        "api_backend.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        **({"ssl_certfile": str(_CERT), "ssl_keyfile": str(_KEY)} if use_ssl else {}),
    )
