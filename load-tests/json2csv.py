import json
import os
import pandas as pd
import sys

# Директория с JSON файлами
input_directory = sys.argv[1]
output_csv = f"{input_directory}/req.csv"


# Функция для извлечения данных из имени файла
def extract_file_info(filename):
    parts = filename.replace("result-5m-", "").replace(".json", "").split("-")
    return int(parts[0]), int(parts[1])  # интенсивность, количество пользователей


# Список для хранения данных
data = []

# Обрабатываем все JSON файлы в директории
for filename in os.listdir(input_directory):
    if filename.endswith(".json"):
        intensity, users = extract_file_info(filename)
        filepath = os.path.join(input_directory, filename)

        with open(filepath, "r") as f:
            json_data = json.load(f)

        # Извлекаем необходимые метрики
        avg_response_time = json_data["metrics"]["http_req_duration"]["values"]["avg"]
        min_response_time = json_data["metrics"]["http_req_duration"]["values"]["min"]
        med_response_time = json_data["metrics"]["http_req_duration"]["values"]["med"]
        max_response_time = json_data["metrics"]["http_req_duration"]["values"]["max"]
        p90_response_time = json_data["metrics"]["http_req_duration"]["values"]["p(90)"]
        p95_response_time = json_data["metrics"]["http_req_duration"]["values"]["p(95)"]
        error_rate = json_data["metrics"]["http_req_failed"]["values"]["rate"] * 100  # в процентах
        request_rate = json_data["metrics"]["http_reqs"]["values"]["rate"]

        # Добавляем строку данных
        data.append({
            "Интенсивность": intensity,
            "Кол-во пользователей": users,
            "Среднее время ответа (мс)": avg_response_time,
            "Минимальное время ответа (мс)": min_response_time,
            "Максимальное время ответа (мс)": max_response_time,
            "Медианное время ответа (мс)": med_response_time,
            "90 персентиль (мс)": p90_response_time,
            "95 персентиль (мс)": p95_response_time,
            "Процент ошибок": error_rate,
            "Rate (запросов/сек)": request_rate
        })

# Создаем DataFrame
df = pd.DataFrame(data)

# Сортируем по интенсивности и количеству пользователей
df = df.sort_values(by=["Интенсивность", "Кол-во пользователей"])

# Сохраняем в CSV
df.to_csv(output_csv, index=False)

print(f"CSV файл сохранен: {output_csv}")
