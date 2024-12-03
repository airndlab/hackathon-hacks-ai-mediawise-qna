import http from 'k6/http'
import { check, sleep } from 'k6'
import { htmlReport } from 'https://raw.githubusercontent.com/benc-uk/k6-reporter/main/dist/bundle.js'
import papaparse from 'https://jslib.k6.io/papaparse/5.1.1/index.js';
import { SharedArray } from 'k6/data';

const k6VUS = __ENV.K6_VUS
const k6DURATION = __ENV.K6_DURATION
const questionType = 'vlm-prompt'
const resultType = 'vllm-vlm-doc'


const datum = new SharedArray('another data name', function () {
  // Load CSV file and parse it using Papa Parse
  return papaparse.parse(open(`../questions/${questionType}.csv`), { header: true }).data;
});

export default function() {
  const url = 'http://158.160.85.147:8000/v1/chat/completions'

  const data = datum[Math.floor(Math.random() * datum.length)]
  const payload = JSON.stringify(
    {
      messages: [
        {
          role: 'system',
          content: 'Вы — интеллектуальная модель, обученная анализировать изображения и предоставлять ответы на основе их содержимого. Ваша задача — внимательно интерпретировать предоставленное изображение, выделить ключевую информацию и ответить на заданный вопрос пользователя. Отвечайте четко, лаконично и на основе видимого на изображении. Если информация на изображении недостаточна для точного ответа, вежливо уточните или сообщите об этом пользователю. При необходимости используйте контекст предоставленного текста или данных.'
        },
        {
          role: 'user',
          content: [
            {
              type: 'text',
              text: data.question
            },
            {
              type: 'image_url',
              image_url: {
                url: data.image_url
              }
            }
          ]
        }
      ],
      model: 'Qwen/Qwen2-VL-2B-Instruct-GPTQ-Int4'
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

  if (!check(res, { 'is status 200': (r) => r.status === 200 })) {
    console.log(`Request failed. Status: ${res.status} | Body: ${res.body}`)
  }

  sleep(15)
}

export function handleSummary(data) {
  return {
    [`reports/${resultType}/result-${k6DURATION}-${k6VUS}.json`]: JSON.stringify(data, null, 2),
    [`reports/${resultType}/result-${k6DURATION}-${k6VUS}.html`]: htmlReport(data),
  }
}
