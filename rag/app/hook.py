import logging
import os
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app import pooling

logger = logging.getLogger(__name__)

POOLING_ENABLED = os.getenv("POOLING_ENABLED", "true").lower() == "true"

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info('Startup hook')

    if POOLING_ENABLED:
        logger.info('Pooling is enabled by POOLING_ENABLED, starting the pooling thread...')
        thread = threading.Thread(target=pooling.do_pooling)
        thread.daemon = True  # Позволяет завершить поток при остановке приложения
        thread.start()
    else:
        logger.info('Pooling is disabled by POOLING_ENABLED.')

    yield
    logger.info('Shutdown hook')
