import logging
import sys
import uvicorn
from fastapi import FastAPI
from pathlib import Path

if __name__ == "__main__":
    from dotenv import load_dotenv

    env_path = Path('../../.env')
    load_dotenv(dotenv_path=env_path)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

from app import api, hook

logger = logging.getLogger(__name__)

app = FastAPI(lifespan=hook.lifespan)

app.include_router(api.router)


@app.get("/")
async def status():
    return {"status": "UP"}


if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8080)
