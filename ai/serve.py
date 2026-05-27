from __future__ import annotations

import os

import uvicorn


def main() -> None:
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", os.getenv("AI_PORT", "8001")))
    uvicorn.run("ai.dashboard_api:app", host=host, port=port)


if __name__ == "__main__":
    main()
