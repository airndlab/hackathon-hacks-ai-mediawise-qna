import csv
import matplotlib.pyplot as plt
from datetime import datetime
import sys
import matplotlib.ticker as ticker
from matplotlib.ticker import MaxNLocator

# Чтение данных из файла
if len(sys.argv) < 2:
    filename = 'reports/llm/stats.csv'
else:
    filename = sys.argv[1]

timestamps = []
service_names = set()
cpu_usage = {}
memory_usage = {}
gpu_usage = {}
gpu_memory_usage = {}

# Чтение CSV файла
with open(filename, mode='r') as file:
    reader = csv.DictReader(file)
    for row in reader:
        timestamp = datetime.strptime(row['timestamp'], '%Y-%m-%d %H:%M:%S')
        service_name = row['service_name']
        cpu = float(row['cpu_usage(%)'])
        memory = float(row['memory_usage(MB)'])
        gpu = float(row['gpu_usage(%)'])
        gpu_memory = float(row['gpu_memory_usage(MB)'])

        if service_name not in service_names:
            service_names.add(service_name)

        if service_name not in cpu_usage:
            cpu_usage[service_name] = []
            memory_usage[service_name] = []
            gpu_usage[service_name] = []
            gpu_memory_usage[service_name] = []

        cpu_usage[service_name].append((timestamp, cpu))
        memory_usage[service_name].append((timestamp, memory))
        gpu_usage[service_name].append((timestamp, gpu))
        gpu_memory_usage[service_name].append((timestamp, gpu_memory))

        if not timestamps:
            timestamps.append(timestamp)
        elif timestamp not in timestamps:
            timestamps.append(timestamp)

# Создание графиков
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# График для CPU Usage
for service in service_names:
    times = [entry[0] for entry in cpu_usage[service]]
    cpu_values = [entry[1] for entry in cpu_usage[service]]
    axes[0, 0].plot(times, cpu_values, label=service)
axes[0, 0].set_title('CPU Usage (%)')
axes[0, 0].set_xlabel('Timestamp')
axes[0, 0].set_ylabel('CPU Usage (%)')
axes[0, 0].legend()

# График для Memory Usage
for service in service_names:
    times = [entry[0] for entry in memory_usage[service]]
    memory_values = [entry[1] for entry in memory_usage[service]]
    axes[0, 1].plot(times, memory_values, label=service)
axes[0, 1].set_title('Memory Usage (MB)')
axes[0, 1].set_xlabel('Timestamp')
axes[0, 1].set_ylabel('Memory Usage (MB)')
axes[0, 1].legend()

# График для GPU Usage
for service in service_names:
    times = [entry[0] for entry in gpu_usage[service]]
    gpu_values = [entry[1] for entry in gpu_usage[service]]
    axes[1, 0].plot(times, gpu_values, label=service)
axes[1, 0].set_title('GPU Usage (%)')
axes[1, 0].set_xlabel('Timestamp')
axes[1, 0].set_ylabel('GPU Usage (%)')
axes[1, 0].legend()

# График для GPU Memory Usage
for service in service_names:
    times = [entry[0] for entry in gpu_memory_usage[service]]
    gpu_mem_values = [entry[1] for entry in gpu_memory_usage[service]]
    axes[1, 1].plot(times, gpu_mem_values, label=service)
axes[1, 1].set_title('GPU Memory Usage (MB)')
axes[1, 1].set_xlabel('Timestamp')
axes[1, 1].set_ylabel('GPU Memory Usage (MB)')
axes[1, 1].legend()

# Настройка отображения чисел на оси Y без научной нотации и удаление дублей
for ax in axes.flat:
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x:.0f}'))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))  # Убираем дубли и отображаем только уникальные значения

# Автоматическая настройка отображения
plt.tight_layout()
plt.show()
