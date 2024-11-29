#!/bin/bash

# Файл для записи метрик
OUTPUT_FILE="resource_usage.csv"
echo "Time,Container,CPU%,Memory(MB),GPU-Usage(%),GPU-Memory(GB)" > $OUTPUT_FILE

# Функция для сбора метрик GPU
function get_gpu_metrics {
  nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits |
  awk -F "," '{print $1","$2}'
}

# Функция для сбора метрик контейнеров
function get_docker_stats {
  docker stats --no-stream --format "{{.Name}},{{.CPUPerc}},{{.MemUsage}}" |
  awk -F "," '{gsub(/MiB|GiB|%/,"",$3); print $1","$2","$3}'
}

# Запуск тестов в фоновом режиме
echo "Запуск тестов..."
k6 run --vus 10 --duration 30s test.js &  # Запустите ваш тест
TEST_PID=$!

# Цикл сбора метрик каждые 2 секунды
while kill -0 $TEST_PID 2>/dev/null; do
  TIMESTAMP=$(date "+%Y-%m-%d %H:%M:%S")

  # Сбор метрик контейнеров
  while IFS= read -r line; do
    DOCKER_METRICS="$line"

    # Сбор метрик GPU
    GPU_METRICS=$(get_gpu_metrics)
    for gpu_line in $GPU_METRICS; do
      echo "$TIMESTAMP,$DOCKER_METRICS,$gpu_line" >> $OUTPUT_FILE
    done
  done < <(get_docker_stats)

  sleep 2
done

echo "Тесты завершены. Метрики сохранены в $OUTPUT_FILE"
