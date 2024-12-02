#!/bin/bash

mkdir -p reports

# Файл для вывода
OUTPUT_FILE="reports/stats.csv"

# Создать CSV файл с заголовками, если его нет
if [ ! -f "$OUTPUT_FILE" ]; then
  echo "timestamp,service_name,cpu_usage(%),memory_usage(MB),gpu_usage(%),gpu_memory_usage(MB)" > "$OUTPUT_FILE"
fi

# Интервал записи данных (в секундах)
INTERVAL=1

while true; do
  # Текущая дата и время
  TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')

  # Список работающих контейнеров
  CONTAINERS=$(docker ps --format '{{.Names}}')

  for CONTAINER in $CONTAINERS; do
    # Получение использования CPU и RAM через docker stats
    STATS=$(docker stats --no-stream --format "{{.CPUPerc}},{{.MemUsage}}" "$CONTAINER")
    CPU_USAGE=$(echo "$STATS" | cut -d',' -f1 | tr -d '%')
    MEM_USAGE=$(echo "$STATS" | cut -d',' -f2 | awk -F'/' '{print $1}' | sed 's/[^0-9.]//g')

    # Получение использования GPU и GPU RAM через nvidia-smi
    GPU_STATS=$(docker exec "$CONTAINER" nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits 2>/dev/null)
    if [ -n "$GPU_STATS" ]; then
      GPU_USAGE=$(echo "$GPU_STATS" | awk -F', ' '{print $1}')
      GPU_MEM_USAGE=$(echo "$GPU_STATS" | awk -F', ' '{print $2}')
    else
      GPU_USAGE=0
      GPU_MEM_USAGE=0
    fi

    # Запись данных в CSV
    echo "$TIMESTAMP,$CONTAINER,$CPU_USAGE,$MEM_USAGE,$GPU_USAGE,$GPU_MEM_USAGE" >> "$OUTPUT_FILE"
  done

  sleep "$INTERVAL"
done
