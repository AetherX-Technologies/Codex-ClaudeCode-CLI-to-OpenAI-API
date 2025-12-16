import asyncio
import os
import sys

import uvicorn
from dotenv import load_dotenv

# Switch to Selector loop on Windows to avoid Proactor WinError 10014 when forking
if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

load_dotenv(override=False)

API_HOST = os.getenv("API_HOST", "127.0.0.1")
try:
    API_PORT = int(os.getenv("API_PORT", "8000"))
except ValueError:
    API_PORT = 8000

if __name__ == "__main__":
    print(f"Starting server at http://{API_HOST}:{API_PORT}")
    uvicorn.run(
        "app:app",
        host=API_HOST,
        port=API_PORT,
        reload=False,
        loop="asyncio",
    )
