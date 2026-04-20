"""
Start the API backend server over HTTPS.

HTTPS avoids browser mixed-content issues when a secure frontend calls localhost.

First run: visit https://localhost:7779 in your browser and click
"Advanced → Proceed to localhost" to accept the self-signed certificate.
Then point your frontend client to https://localhost:7779.

Usage:
    python scripts/serve.py           # HTTPS on port 7779
    python scripts/serve.py --reload  # dev mode with hot-reload
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
    parser.add_argument("--port", type=int, default=7779)
    parser.add_argument("--host", default="0.0.0.0")
    args = parser.parse_args()

    if not _CERT.exists() or not _KEY.exists():
        print("ERROR: TLS cert/key not found. Run from project root:")
        print("  openssl req -x509 -newkey rsa:2048 -keyout api_backend/key.pem "
              "-out api_backend/cert.pem -days 825 -nodes -subj '/CN=localhost'")
        sys.exit(1)

    print(f"\n  Server: https://localhost:{args.port}")
    print("  STEP 1: Open https://localhost:7779 in your browser")
    print("          Click Advanced -> Proceed to localhost (unsafe)")
    print("  STEP 2: Point your frontend/backend client to: https://localhost:7779\n")

    uvicorn.run(
        "api_backend.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        ssl_certfile=str(_CERT),
        ssl_keyfile=str(_KEY),
    )
