# API CONFIG
API_CONFIG_PATH = os.getenv("API_CONFIG_PATH")
with open(API_CONFIG_PATH, "r", encoding='utf-8') as config_file:
    api_config = yaml.safe_load(config_file)

# Извлечение URL и эндпоинтов из конфигурации
BASE_URL = api_config["api"]["base_url"]
POLL_ENDPOINT = f"{BASE_URL}{api_config['api']['endpoints']['poll']}"
COMPLETE_ENDPOINT = f"{BASE_URL}{api_config['api']['endpoints']['complete']}"


# Функция для опроса POLL_ENDPOINT
def poll():
    response = requests.get(POLL_ENDPOINT)
    response.raise_for_status()  # Проверка на успешный статус ответа
    return response.json()  # Возвращает JSON-ответ от сервера


# Функция для отправки результата на COMPLETE_ENDPOINT
def complete(request_id, message="", sources=[], suggestions=[]):
    complete_response = requests.post(
        COMPLETE_ENDPOINT,
        json={
            "requestId": request_id,
            "message": message,
            "sources": sources,
            "suggestions": suggestions,

        }
    )
    complete_response.raise_for_status()  # Проверка на успешный статус ответа
    return complete_response.json()


# Основная функция для обработки генерации ответа
def generator_poll():
    # Опрос POLL_ENDPOINT
    response_data = poll()

    # Извлечение необходимых данных из ответа
    request_id = response_data.get("requestId")
    question = response_data.get("query")

    # Если получены необходимые данные
    if request_id:
        if not question:  # Если question пустой (пустая строка)
            conversational = response_data.get("history", [])
            related_questions = generate_related_questions(conversational)
            complete(
                request_id=request_id,
                suggestions=related_questions,
            )
        else:
            categories = response_data.get("categories", None)
            space = response_data.get("space", None)
            filename = response_data.get("filename", None)
            input_data = {
                "question": question,
                "categories": categories,  #
                "space": space,  #
                "filename": filename,  #
            }
            model_response = get_chat_response_json(input_data)

            complete(
                message=model_response.get("message", ""),
                request_id=request_id,
                sources=model_response.get("sources", []),
            )


# Бесконечный цикл для выполнения generator_poll и check_s3_poll
def do_pooling():
    while True:
        try:
            # Выполнение generator_poll
            generator_poll()
        except Exception as e:
            # Задержка перед следующей итерацией
            time.sleep(1)
