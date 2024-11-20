import logging
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app import pooling, pipelines

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info('Startup hook')
    pipelines.do_reindex()
    thread = threading.Thread(target=pooling.do_pooling)
    thread.daemon = True  # Позволяет завершить поток при остановке приложения
    thread.start()
    yield
    logger.info('Shutdown hook')
