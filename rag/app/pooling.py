import logging
import requests
import time
from app import pipelines

logger = logging.getLogger(__name__)


def do_pooling():
    error_count = 0
    while True:
        try:
            # Выполнение generator_poll
            pipelines.generator_poll()
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                # Специальная обработка для статуса 404
                time.sleep(1)
            else:
                error_count += 1
                logger.warn(f"Exception of polling: {e}")
                # Если количество ошибок достигает 10, делаем паузу
                if error_count >= 10:
                    logger.warn("Too many errors. Sleeping for 10 seconds.")
                    time.sleep(10)  # Пауза в 10 секунд
                    error_count = 0  # Сбрасываем счетчик после паузы
        except Exception as e:
            error_count += 1
            logger.warn(f"Exception of polling: {e}")
            # Если количество ошибок достигает 10, делаем паузу
            if error_count >= 10:
                logger.warn("Too many errors. Sleeping for 10 seconds.")
                time.sleep(10)  # Пауза в 10 секунд
                error_count = 0  # Сбрасываем счетчик после паузы
