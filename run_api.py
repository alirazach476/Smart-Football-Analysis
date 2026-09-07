"""
Convenience launcher for the FastAPI backend.

    python run_api.py                 # 127.0.0.1:8000
    python run_api.py --host 0.0.0.0 --port 8080 --reload

Equivalent to `uvicorn app.main:app`.
"""

from __future__ import annotations

import argparse

import uvicorn


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Smart Football Analysis API.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true", help="Auto-reload on code changes.")
    parser.add_argument("--workers", type=int, default=1, help="Uvicorn worker processes.")
    args = parser.parse_args()

    uvicorn.run(
        "app.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=args.workers if not args.reload else 1,
    )


if __name__ == "__main__":
    main()
