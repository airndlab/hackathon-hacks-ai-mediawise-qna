import http from 'k6/http'
import { check, sleep } from 'k6'
import { htmlReport } from 'https://raw.githubusercontent.com/benc-uk/k6-reporter/main/dist/bundle.js'

const k6VUS = __ENV.K6_VUS
const k6DURATION = __ENV.K6_DURATION
const questionType = 'llm-prompt'
const resultType = 'vllm-doc'

// Функция для чтения вопросов из CSV файла
function loadQuestions(filePath) {
  const file = open(filePath) // Открытие файла
  const lines = file.split('\n') // Разделение на строки
  const questions = lines.map(line => line.trim()).filter(line => line.length > 0) // Очистка и фильтрация пустых строк
  return questions
}

// Загрузка вопросов
const questions = loadQuestions(`../questions/${questionType}.csv`)

export default function() {
  const url = 'http://158.160.85.147:8000/v1/chat/completions'

  const question = questions[Math.floor(Math.random() * questions.length)]

  const payload = JSON.stringify({
    messages: [
      {
        role: 'user',
        content: question
      }
    ],
    model: 'Qwen/Qwen2.5-14B-Instruct-GPTQ-Int4'
  })

  const params = {
    headers: {
      'Content-Type': 'application/json',
    },
  }

  const res = http.post(url, payload, params)

  check(res, {
    'is status 200': (r) => r.status === 200,
  })

  sleep(15)
}

export function handleSummary(data) {
  return {
    [`reports/${resultType}/result-${k6DURATION}-${k6VUS}.json`]: JSON.stringify(data, null, 2),
    [`reports/${resultType}/result-${k6DURATION}-${k6VUS}.html`]: htmlReport(data),
  }
}
