# QnA чат-бот с поиском информации по библиотеке знаний

[![Docker Image Version (latest semver)](https://img.shields.io/docker/v/airndlab/perplexica-frontend?label=perplexica-frontend)](https://hub.docker.com/r/airndlab/perplexica-frontend)
[![Docker Image Version (latest semver)](https://img.shields.io/docker/v/airndlab/perplexica-backend?label=perplexica-backend)](https://hub.docker.com/r/airndlab/perplexica-backend)
[![Docker Image Version (latest semver)](https://img.shields.io/docker/v/airndlab/mediawise-rag?label=mediawise-rag)](https://hub.docker.com/r/airndlab/mediawise-rag)

Решение команды **НейроДрайв**
по [кейсу](docs/media-wise.pdf)
от [Media Wise](https://mediadirectiongroup.ru/agency/mediawise/)
на хакатоне Цифровой прорыв сезон: ИИ.

## vLLM

Модель работает как отдельный сервис запущенный
на [vLLM](https://docs.vllm.ai/) (имеет совместимый с OpenAI API)
локально в Docker.

Для проверки, что vLLM запущен - открыть в браузере: `http://hostname:8000/v1/models`,
где `hostname` заменить на адрес сервера где работает vLLM.
