import json
import matplotlib.pyplot as plt
import sys
import os

# Папка с результатами
if len(sys.argv) < 2:
    results_folder = './reports/llm'
else:
    results_folder = sys.argv[1]

# Массивы для хранения данных
vus_values = []   # Количество пользователей (K6_VUS)
avg_times = []    # Среднее время (http_req_duration)
max_times = []    # Максимальное время (http_req_duration)
min_times = []    # Минимальное время (http_req_duration)
error_rates = []  # Процент ошибок (http_req_failed.rate)

# Чтение всех JSON-файлов в директории
for filename in os.listdir(results_folder):
    if filename.endswith('.json'):
        file_path = os.path.join(results_folder, filename)

        # Открытие и загрузка JSON-файла
        with open(file_path, 'r') as file:
            data = json.load(file)

            # Извлечение информации из имени файла
            try:
                vus = int(filename.split('-')[2].split('.')[0])  # Извлекаем K6_VUS из имени файла
            except (IndexError, ValueError):
                print(f"Неправильное имя файла: {filename}, пропускаем.")
                continue

            # Извлечение данных из JSON
            avg_time = data['metrics']['http_req_duration']['values']['avg']  # Среднее время
            max_time = data['metrics']['http_req_duration']['values']['max']  # Максимальное время
            min_time = data['metrics']['http_req_duration']['values']['min']  # Минимальное время

            # Извлечение процента ошибок
            error_rate = data['metrics'].get('http_req_failed', {}).get('values', {}).get('rate', 0)

            # Добавляем значения в массивы
            vus_values.append(vus)
            avg_times.append(avg_time)
            max_times.append(max_time)
            min_times.append(min_time)
            error_rates.append(error_rate)

# Сортировка данных по количеству пользователей (VUS)
sorted_data = sorted(zip(vus_values, avg_times, max_times, min_times, error_rates))
vus_values, avg_times, max_times, min_times, error_rates = map(list, zip(*sorted_data))

# Построение графиков
plt.figure(figsize=(12, 8))

# Определяем шаг для меток на оси X
x_ticks = range(0, max(vus_values) + 1, 5)  # Шаг 5

# График для среднего времени
plt.subplot(2, 2, 1)
plt.plot(vus_values, avg_times, marker='o', linestyle='-', color='b', label='Среднее время')
plt.xlabel('Количество пользователей (VUS)')
plt.ylabel('Среднее время (ms)')
plt.title('Среднее время запроса')
plt.xticks(x_ticks)  # Настраиваем шаг на оси X
plt.grid(True)

# График для максимального времени
plt.subplot(2, 2, 2)
plt.plot(vus_values, max_times, marker='o', linestyle='-', color='r', label='Максимальное время')
plt.xlabel('Количество пользователей (VUS)')
plt.ylabel('Максимальное время (ms)')
plt.title('Максимальное время запроса')
plt.xticks(x_ticks)  # Настраиваем шаг на оси X
plt.grid(True)

# График для минимального времени
plt.subplot(2, 2, 3)
plt.plot(vus_values, min_times, marker='o', linestyle='-', color='g', label='Минимальное время')
plt.xlabel('Количество пользователей (VUS)')
plt.ylabel('Минимальное время (ms)')
plt.title('Минимальное время запроса')
plt.xticks(x_ticks)  # Настраиваем шаг на оси X
plt.grid(True)

# График для процента ошибок
plt.subplot(2, 2, 4)
plt.plot(vus_values, [e * 100 for e in error_rates], marker='o', linestyle='-', color='purple', label='Процент ошибок')
plt.xlabel('Количество пользователей (VUS)')
plt.ylabel('Процент ошибок (%)')
plt.title('Процент ошибок')
plt.xticks(x_ticks)  # Настраиваем шаг на оси X
plt.grid(True)

# Показать все графики
plt.tight_layout()
plt.show()
