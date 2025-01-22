import csv
import matplotlib.pyplot as plt
import sys

# Получаем имя файла CSV из аргументов командной строки
results_csv = sys.argv[1]

# Чтение данных из файла
names = set()
data = {}
with open(results_csv, newline='', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile, delimiter=',')
    for row in reader:
        name = row['Название']
        users = int(row['Кол-во пользователей'])
        avg_response_time = float(row['Среднее время ответа (мс)'])
        if name not in data:
            data[name] = {'users': [], 'avg_response_time': []}
        data[name]['users'].append(users)
        data[name]['avg_response_time'].append(avg_response_time)
        names.add(name)

# Цвета для разных названий (можно расширить или изменить по необходимости)
color_map = {
    '1 core': 'yellow',
    '2 core (hor)': 'green',
    '2 core (ver)': 'blue',
    '2 core (32B)': 'red',
}

# Построение графика для среднего времени ответа
fig, ax = plt.subplots(figsize=(10, 6))
for name in sorted(names):
    users = data[name]['users']
    avg_response_time = data[name]['avg_response_time']
    # Используем цвет из color_map, если название есть в словаре, иначе ставим серый
    color = color_map.get(name, 'gray')
    ax.plot(users, avg_response_time, label=name, color=color)

# Устанавливаем метки на оси X как уникальные значения количества пользователей
all_users = sorted(set(user for name in data for user in data[name]['users']))
ax.set_xticks(all_users)

# Заголовок и подписи
ax.set_xlabel('Количество пользователей')
ax.set_ylabel('Среднее время ответа (мс)')
ax.set_title('Среднее время ответа и количество пользователей')
ax.legend(title="Название")
ax.grid(True)
plt.tight_layout()
plt.show()