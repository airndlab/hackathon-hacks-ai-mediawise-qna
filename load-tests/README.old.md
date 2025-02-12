# Нагрузочное тестирование

## Запросы ответов на вопросы

Сценарий: каждый пользователь задает вопрос после 15 секунд "прочтения" ответа на предыдущий вопрос
(среднее время прочтения человеком текста ответа)

Запускается на разном кол-ве пользователей: 1 2 4 8 12 24 32

### ВМ

Intel Broadwell with NVIDIA® Tesla® V100 (gpu-standard-v1)

| Number of GPUs | VRAM, GB | Number of vCPUs | RAM, GB |
|----------------|----------|-----------------|---------|
| 1              | 32       | 8               | 96      |

### Запросы к vLLM (LLM) с prompt и разным sleep

#### 1 GPU 32 GB

Запросы:

![vllm-prompt-llm-req.png](charts/vllm-prompt-llm-req.png)

Ресурсы:

![vllm-prompt-llm-stats.png](charts/vllm-prompt-llm-stats.png)

### Запросы к vLLM (VLM) с prompt и разным sleep

#### 1 GPU 32 GB

Запросы:

![vllm-prompt-vlm-req.png](charts/vllm-prompt-vlm-req.png)

Ресурсы:

![vllm-prompt-vlm-stats.png](charts/vllm-prompt-vlm-stats.png)

### Полный LLM пайплайн

![img.png](images/img.png)

LLM: *Qwen/Qwen2.5-14B-Instruct-GPTQ-Int4*

Embedder: *deepvk/USER-bge-m3*

Ranker: *qilowoq/bge-reranker-v2-m3-en-ru*

### VLM модели

Для индексациии: *vidore/colpali-v1.2*

Для генерации (VLM): *Qwen/Qwen2-VL-2B-Instruct-GPTQ-Int4*

### Результаты

#### LLM

Запросы:

![llm-req.png](charts/llm-req.png)

Ресурсы:

![llm-stats.png](charts/llm-stats.png)

#### VLM

Запросы:

![vlm-req.png](charts/vlm-req.png)

Ресурсы:

![vlm-stats.png](charts/vlm-stats.png)

## Запросы однострочные напрямую к модели (VLLM)

Однострочные вопросы на разные темы.

Запросы:

![vllm-req.png](charts/vllm-req.png)

Ресурсы:

![vllm-stats.png](charts/vllm-stats.png)

## Запросы однострочные к модели (VLLM) через сервис (rag)

Однострочные вопросы на разные темы.

Видно как увеличивается время работы, когда rag выступает по сути прокси для vllm.

Запросы:

![rag-vllm-req.png](charts/rag-vllm-req.png)

Ресурсы:

![rag-vllm-stats.png](charts/rag-vllm-stats.png)

## Запросы с prompt напрямую к модели (VLLM)

Запрос с prompt по документу.

Запросы:

![vllm-doc-req.png](charts/vllm-doc-req.png)

Ресурсы:

![vllm-doc-stats.png](charts/vllm-doc-stats.png)

## Запросы с prompt напрямую к VLM модели (VLLM)

Запрос с prompt по документу.

Запросы:

![vllm-vlm-doc-req.png](charts/vllm-vlm-doc-req.png)

Ресурсы:

![vllm-vlm-doc-stats.png](charts/vllm-vlm-doc-stats.png)

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
