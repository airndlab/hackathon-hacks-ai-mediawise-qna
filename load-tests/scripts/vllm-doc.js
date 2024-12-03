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
        role: "system",
        content: "Your task is to answer the user's questions using only the information from the provided documents.\n\n *Rules to follow*:\n - Say *exactly* \"Я не знаю ответа на ваш вопрос\" if:\n      1. The input is not a question.\n      2. The answer is not in the provided context.\n - Never generate information outside the provided context.\n - Answer in detail, making maximum use of the information provided in context if it is relevant to the question.\n - After answering, generate a Python list of JSON objects representing the documents you used to construct the answer. Each JSON object should contain only \"filename\" and \"page_number\" fields.\n\n **Examples:**\n <examples>\n <documents>\n [{\"filename\": \"Document1.pdf\", \"page_number\": 1, \"content\": \"The Renaissance period saw a revival of classical philosophy and art.\"}]\n </documents>\n Question: \"What was the Renaissance known for?\"\n Answer: \n The Renaissance was known for its revival of classical philosophy and art, as well as significant advancements in literature and science.\n\n [{\"filename\": \"Document1.pdf\", \"page_number\": 1}]\n <documents>\n [{\"filename\": \"Document2.pdf\", \"page_number\": 1, \"content\": \"Photosynthesis is the process by which plants convert sunlight into energy.\"}, {\"filename\": \"Document3.pdf\", \"page_number\": 2, \"content\": \"Chlorophyll in plants absorbs sunlight, which then drives the process of photosynthesis.\"}]\n </documents>\n Question: \"How do plants produce energy, and what role does chlorophyll play?\"\n Answer: \n Plants produce energy through a process called photosynthesis, where they convert sunlight into usable energy. Chlorophyll in plants absorbs the sunlight, which drives this photosynthesis process.\n\n [{\"filename\": \"Document2.pdf\", \"page_number\": 1},{\"filename\": \"Document3.pdf\", \"page_number\": 2}]\n </examples>\n\n A lot depends on this answer—triple-check it!"
      },
      {
        role: "user",
        content: question
      }
    ],
    model: "Qwen/Qwen2.5-14B-Instruct-GPTQ-Int4"
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
