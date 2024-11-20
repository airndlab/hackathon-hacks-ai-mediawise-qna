import requests
import time
from app import pipelines


def do_pooling():
    while True:
        try:
            # Выполнение generator_poll
            pipelines.generator_poll()
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                # Специальная обработка для статуса 404
                time.sleep(1)
            else:
                print(e)
        except Exception as e:
            print(e)
            time.sleep(1)
