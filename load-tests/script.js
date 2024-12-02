import http from 'k6/http'
import { check, sleep } from 'k6'
import { htmlReport } from 'https://raw.githubusercontent.com/benc-uk/k6-reporter/main/dist/bundle.js'

const k6VUS = __ENV.K6_VUS
const k6DURATION = __ENV.K6_DURATION

const questions = [
  'Какие есть подкасты?',
  'Кто самый популярный стример?',
  'Какие существуют рекламные площадки?',
  'Кто такой Хазбик?',
  'Какие есть методы рекламы?',
  'Какие есть крупные спонсоры?',
  'Что сейчас в тренде?',
]

export default function() {
  const url = 'http://158.160.85.147:8080/asking'

  const question = questions[Math.floor(Math.random() * questions.length)];

  const payload = JSON.stringify({
    question: question,
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
    [`reports/result-${k6DURATION}-${k6VUS}.json`]: JSON.stringify(data, null, 2),
    [`reports/result-${k6DURATION}-${k6VUS}.html`]: htmlReport(data),
  }
}
