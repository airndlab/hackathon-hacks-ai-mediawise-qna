# Нагрузочное тестирование

## Результаты

Intel Broadwell with NVIDIA® Tesla® V100 (gpu-standard-v1)

| Number of GPUs | VRAM, GB | Number of vCPUs | RAM, GB |
|----------------|----------|-----------------|---------|
| 1              | 32       | 8               | 96      |

Кол-во пользователей: 1 2 4 8 12 24 32

### LLM

Запросы:

![llm-req.png](charts/llm-req.png)

Ресурсы:

![llm-stats.png](charts/llm-stats.png)

### VLM

Запросы:

![vlm-req.png](charts/vlm-req.png)

Ресурсы:

![vlm-stats.png](charts/vlm-stats.png)

## Запуск

Сначала запускается скрипт сбора статистики потребления ресурсов на ВМ, дальше на ПК запускаются k6 тесты.
После завершения тестов, скрипт сборка статистики завершается.

### На ВМ

```shell
./stats.sh
```

### На ПК

Для LLM:

```shell
./runner-llm.sh
```

Для VLM:

```shell
./runner-vlm.sh
```

## Вопросы

Списки вопросов в [questions](questions):

- для LLM - [llm.csv](questions/llm.csv)
- для VLM - [vlm.csv](questions/vlm.csv)

## Отчеты

Результаты работы k6 в [reports](reports).

## Графики

Построить графики для LLM:

```shell
python plot.py reports/llm
python plot-stats.py reports/llm/stats.csv
```

Построить графики для VLM:

```shell
python plot.py reports/vlm
python plot-stats.py reports/vlm/stats.csv
```
