# Нагрузочное тестирование

Отправка [вопросов](questions/llm-prompt.csv) с информацией о документах с промптом напрямую к LLM запущенную в vLLM.

Тест запускается с разной интенсивностью (время между запросами от одно пользователя): 15, 30 и 45 секунд.

Тест: [vllm-prompt-llm.js](scripts/vllm-prompt-llm.js)

## На одном ядре GPU

Intel Broadwell with NVIDIA® Tesla® V100

| Number of GPUs | VRAM, GB | Number of vCPUs | RAM, GB |
|----------------|----------|-----------------|---------|
| 1              | 32       | 8               | 96      |

### Схема

![v100.png](images/v100.png)

### Модель

[Qwen/Qwen2.5-14B-Instruct-GPTQ-Int4](https://huggingface.co/Qwen/Qwen2.5-14B-Instruct-GPTQ-Int4)

### Результат

Среднее время ответа:

![v100-vllm-prompt-llm-req.png](charts/v100-vllm-prompt-llm-req.png)

Потребление ресурсов:

![v100-vllm-prompt-llm-stats.png](charts/v100-vllm-prompt-llm-stats.png)

## На двух ядрах GPU

Intel Cascade Lake with NVIDIA® Tesla® V100

| Number of GPUs | VRAM, GB | Number of vCPUs | RAM, GB |
|----------------|----------|-----------------|---------|
| 2              | 64       | 16              | 96      |

### С горизонтальным масштабированием

Запуск двух vLLM на каждое ядро.

#### Схема

![v100x2h.png](images/v100x2h.png)

#### Модель

[Qwen/Qwen2.5-14B-Instruct-GPTQ-Int4](https://huggingface.co/Qwen/Qwen2.5-14B-Instruct-GPTQ-Int4)

#### Результат

Среднее время ответа:

![v100x2h-vllm-prompt-llm-req.png](charts/v100x2h-vllm-prompt-llm-req.png)

Потребление ресурсов:

![v100x2h-vllm-prompt-llm-stats.png](charts/v100x2h-vllm-prompt-llm-stats.png)

### С вертикальным масштабированием

Запуск одного vLLM на двух ядрах.

#### Схема

![v100x2v.png](images/v100x2v.png)

#### Модель

[Qwen/Qwen2.5-14B-Instruct-GPTQ-Int4](https://huggingface.co/Qwen/Qwen2.5-14B-Instruct-GPTQ-Int4)

#### Результат

Среднее время ответа:

![v100x2v-vllm-prompt-llm-req.png](charts/v100x2v-vllm-prompt-llm-req.png)

### С вертикальным масштабированием + модель на 32B

Запуск одного vLLM на двух ядрах.

#### Схема

![v100x32.png](images/v100x32.png)

#### Модель

[Qwen/Qwen2.5-32B-Instruct-GPTQ-Int4](https://huggingface.co/Qwen/Qwen2.5-32B-Instruct-GPTQ-Int4)

#### Результат

Среднее время ответа:

![v100x32-vllm-prompt-llm-req.png](charts/v100x32-vllm-prompt-llm-req.png)
