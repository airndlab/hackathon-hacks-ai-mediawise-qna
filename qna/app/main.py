import logging
import sys
from pathlib import Path

import uvicorn
from fastapi import FastAPI

if __name__ == "__main__":
    from dotenv import load_dotenv

    env_path = Path('../../.env')
    load_dotenv(dotenv_path=env_path)

from app import api, hook

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

logger = logging.getLogger(__name__)

app = FastAPI(lifespan=hook.lifespan)

app.include_router(api.router)


@app.get("/")
async def status():
    return {"status": "UP"}


if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8088)
