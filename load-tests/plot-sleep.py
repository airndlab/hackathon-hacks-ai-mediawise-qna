import os
import sys
import json
import matplotlib.pyplot as plt

def parse_filename(filename):
    # Извлечение значения sleep и числа пользователей из имени файла
    parts = filename.split('-')
    sleep = int(parts[2])
    users = int(parts[3].split('.')[0])
    return sleep, users

def load_data(directory):
    data = {}

    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            filepath = os.path.join(directory, filename)
            with open(filepath, 'r') as f:
                content = json.load(f)

            sleep, users = parse_filename(filename)
            avg_time = content["metrics"]["http_req_duration"]["values"]["avg"]
            rate_per_second = content["metrics"]["http_reqs"]["values"]["rate"]
            rate_per_minute = rate_per_second * 60

            if sleep not in data:
                data[sleep] = []

            data[sleep].append({
                "users": users,
                "avg_time": avg_time,
                "rate_per_minute": rate_per_minute
            })

    # Сортировка данных по числу пользователей для каждого sleep
    for sleep in data:
        data[sleep].sort(key=lambda x: x["users"])

    return data

def plot_graphs(data):
    # Создание одного окна для всех графиков
    fig, axs = plt.subplots(len(data), 2, figsize=(16, 4 * len(data)))

    if len(data) == 1:
        axs = [axs]  # Приведение к списку для единообразной обработки

    for ax_row, (sleep, results) in zip(axs, data.items()):
        users = [item["users"] for item in results]
        avg_times = [item["avg_time"] for item in results]
        rates_per_minute = [item["rate_per_minute"] for item in results]

        # График среднего времени
        ax_row[0].plot(users, avg_times, marker='o', label=f"Среднее время (Sleep {sleep})")
        ax_row[0].set_xlabel("Количество пользователей")
        ax_row[0].set_ylabel("Среднее время (ms)")
        ax_row[0].set_title(f"Среднее время vs Количество пользователей (Sleep {sleep})")
        ax_row[0].set_xticks(users)  # Установка явных значений по оси X
        ax_row[0].grid(True)
        ax_row[0].legend()

        # График запросов в минуту
        ax_row[1].plot(users, rates_per_minute, marker='o', label=f"Запросы в минуту (Sleep {sleep})", color='orange')
        ax_row[1].set_xlabel("Количество пользователей")
        ax_row[1].set_ylabel("Запросы в минуту")
        ax_row[1].set_title(f"Запросы в минуту vs Количество пользователей (Sleep {sleep})")
        ax_row[1].set_xticks(users)  # Установка явных значений по оси X
        ax_row[1].grid(True)
        ax_row[1].legend()

    # Показать графики
    plt.tight_layout()
    plt.show()

def main():
    if len(sys.argv) < 2:
        directory = "reports/vllm-prompt-llm"
    else:
        directory = sys.argv[1]
    data = load_data(directory)
    plot_graphs(data)

if __name__ == "__main__":
    main()
