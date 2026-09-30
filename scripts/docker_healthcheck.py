"""Container health check for the API or the Streamlit UI."""

from __future__ import annotations

import os
import urllib.request


def main() -> None:
    service = os.environ.get("SERVICE", "api")
    port = os.environ.get("PORT", "8000" if service == "api" else "8501")
    path = "/_stcore/health" if service == "ui" else "/health"
    urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=3)


if __name__ == "__main__":
    main()
