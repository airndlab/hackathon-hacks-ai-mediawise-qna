import csv
import matplotlib.pyplot as plt
import sys

results_csv = sys.argv[1]

# Чтение данных из файла
intensities = set()
data = {}

with open(results_csv, newline='', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile, delimiter=',')
    for row in reader:
        intensity = row['Интенсивность']
        users = int(row['Кол-во пользователей'])
        avg_response_time = float(row['Среднее время ответа (мс)'])

        if intensity not in data:
            data[intensity] = {'users': [], 'avg_response_time': []}

        data[intensity]['users'].append(users)
        data[intensity]['avg_response_time'].append(avg_response_time)
        intensities.add(intensity)

# Цвета для разных интенсивностей
color_map = {'15': 'red', '30': 'yellow', '45': 'green'}

# Построение графика для среднего времени ответа
fig, ax = plt.subplots(figsize=(10, 6))

for intensity in sorted(intensities):
    users = data[intensity]['users']
    avg_response_time = data[intensity]['avg_response_time']
    # Используем цвет из color_map, если интенсивность есть в словаре, иначе ставим серый
    color = color_map.get(intensity, 'gray')
    ax.plot(users, avg_response_time, label=f'Интенсивность {intensity}', color=color)

# Устанавливаем метки на оси X как уникальные значения количества пользователей
all_users = sorted(set(user for intensity in data for user in data[intensity]['users']))
ax.set_xticks(all_users)

# Заголовок и подписи
ax.set_xlabel('Количество пользователей')
ax.set_ylabel('Среднее время ответа (мс)')
ax.set_title('Среднее время ответа и количество пользователей по интенсивности')
ax.legend(title="Интенсивность")
ax.grid(True)

plt.tight_layout()
plt.show()
