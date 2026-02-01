import os
import sys
import time
from typing import Optional

import requests


def _check_ollama(base_url: str, retries: int = 5, delay_s: float = 1.0) -> Optional[str]:
    for attempt in range(1, retries + 1):
        try:
            response = requests.get(f"{base_url}/api/tags", timeout=3)
            if response.status_code == 200:
                return None
            return f"Ollama returned status {response.status_code}"
        except requests.RequestException as exc:
            if attempt >= retries:
                return str(exc)
        time.sleep(delay_s)
    return "Ollama did not respond"


def main() -> None:
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").strip()
    error = _check_ollama(base_url)
    if error:
        print(f"[run_all] Ollama is not available at {base_url}: {error}")
        print("[run_all] Start Ollama with: ollama serve")
        sys.exit(1)

    host = os.getenv("BACKEND_HOST", "0.0.0.0")
    port = os.getenv("BACKEND_PORT", "8000")
    os.execvp(
        "uvicorn",
        [
            "uvicorn",
            "api.main:app",
            "--host",
            host,
            "--port",
            str(port),
        ],
    )


if __name__ == "__main__":
    main()
