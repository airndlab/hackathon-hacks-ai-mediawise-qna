# QnA чат-бот с поиском информации по библиотеке знаний

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
