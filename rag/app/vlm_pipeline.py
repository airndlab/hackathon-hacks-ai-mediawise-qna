import os

import re

import io

import pandas as pd


from typing import List, Union, Optional, Dict, Any, Tuple

from pathlib import Path
from byaldi import RAGMultiModalModel

import torch
import gc

from numpy.random import default_rng
from pdf2image import convert_from_path


import yaml

from openai import OpenAI


PDF_IMGS_DIR = os.getenv("PDF_IMGS_DIR")
INDEX_ROOT_DIR = os.getenv("INDEX_ROOT_DIR")
SUMMARY_DF_PATH = os.getenv("SUMMARY_DF_PATH")
MAIN_DOCS_DIR = os.getenv("MAIN_DOCS_DIR")

# API CONFIG
API_CONFIG_PATH = os.getenv("API_CONFIG_PATH")
with open(API_CONFIG_PATH, "r", encoding='utf-8') as config_file:
  api_config = yaml.safe_load(config_file)

# Извлечение URL и эндпоинтов из конфигурации
BASE_URL = api_config["api"]["base_url"]
VLM_POLL_ENDPOINT = f"{BASE_URL}{api_config['api']['endpoints']['vlm_poll']}"
COMPLETE_ENDPOINT = f"{BASE_URL}{api_config['api']['endpoints']['complete']}"


# PROMPT CONFIG
PROMPTS_CONFIG_PATH = os.getenv("PROMPTS_CONFIG_PATH")
with open(PROMPTS_CONFIG_PATH, 'r', encoding='utf-8') as file:
    prompt_config = yaml.safe_load(file)

vlm_system_prompt = prompt_config['vlm_system_prompt']

# RAG CONFIG
VLM_RAG_CONFIG_PATH = os.getenv("VLM_RAG_CONFIG_PATH")
with open(VLM_RAG_CONFIG_PATH, "r") as file:
    vlm_rag_config = yaml.safe_load(file)

vlm_model = vlm_rag_config.get("vlm_model")
colvision_model = vlm_rag_config.get("colvision_model")

vlm_top_k = vlm_rag_config.get("vlm_top_k", 1)

index_config = vlm_rag_config.get("index", {})
index_name = index_config.get("name", "index")

VLLM_URL = os.getenv("VLLM_VLM_URL")

summary_df = pd.read_csv(SUMMARY_DF_PATH)

def extract_page_number(path):
  match = re.search(r'page_(\d+)\.png$', path.name)
  if match:
    return int(match.group(1))
  else:
    return 0


def initialize_image_paths():
  """
  Инициализирует словарь image_paths, сканируя директорию PDF_IMGS_DIR
  и добавляя существующие пути к изображениям в правильном порядке.
  """
  global image_paths
  image_paths = {}

  # Проверяем, существует ли PDF_IMGS_DIR
  if not os.path.exists(PDF_IMGS_DIR):
    print(f"Директория {PDF_IMGS_DIR} не существует. Создаём её.")
    os.makedirs(PDF_IMGS_DIR, exist_ok=True)
    return

  # Проходим по всем поддиректориям в PDF_IMGS_DIR
  for dir_entry in os.scandir(PDF_IMGS_DIR):
    if dir_entry.is_dir():
      pdf_filename = dir_entry.name + ".pdf"  # Предполагаем, что имя папки соответствует PDF файлу
      image_dir = Path(dir_entry.path)
      image_files = sorted(
        image_dir.glob("page_*.png"),
        key=extract_page_number  # Сортируем по номеру страницы
      )

      if image_files:
        image_paths[pdf_filename] = [str(image_file.resolve()) for image_file in image_files]
        print(f"Найдено {len(image_files)} изображений для файла {pdf_filename}.")

  print("Инициализация image_paths завершена.")


def index_single_file(file_path):
    """
    Индексирует отдельный файл.

    Args:
        file_path (Path): Путь к файлу.
        organization_name (str): Название организации.
        space_name (str): Название пространства.
        metadata (dict): Метаданные для файла.
    """
    if file_path.suffix.lower() != ".pdf":
      return

    filename = file_path.name
    space_name = file_path.parent.name
    organization_name = file_path.parent.parent.name

    print(f"Индексирую Filename:{filename}\nSpace:{space_name}\nOrganization:{organization_name}")

    row = summary_df[
      (summary_df["organization"] == organization_name) &
      (summary_df["space"] == space_name) &
      (summary_df["filename"] == filename)
      ]

    if not row.empty:
      title = row.iloc[0]["title"]
      summary = row.iloc[0]["summary"]
      category = row.iloc[0].get("category", None)
    else:
      title = ""
      summary = ""
      category = ""

    metadata = {
      "organization": organization_name,
      "space": space_name,
      "title": title,
      "filename": filename,
      "summary": summary,
      "category": category,
    }

    if main_index.model.index_name is not None:
      main_index.add_to_index(
        input_item=file_path,
        store_collection_with_index=False,
        metadata=metadata,
      )
    else:
      main_index.index(
        input_path=file_path,
        index_name=index_name,
        store_collection_with_index=False,
        metadata=[metadata],
        overwrite=True,
      )

    if filename in image_paths:
      print(f"Изображения для файла {filename} уже существуют. Пропускаем нарезку на картинки.")
      return

    curr_doc_imgs_path = os.path.join(PDF_IMGS_DIR, filename.replace(".pdf", ""))
    curr_doc_images = convert_from_path(file_path)
    os.makedirs(curr_doc_imgs_path, exist_ok=True)

    curr_image_paths = []
    for i, image in enumerate(curr_doc_images):
      path = os.path.join(curr_doc_imgs_path, f"page_{i + 1}.png")
      image.save(path, "PNG")
      curr_image_paths.append(path)

    image_paths[filename] = curr_image_paths

    print(f"Изображения для файла {filename} сохранены.")


def index_all_subdirs():
  """
  Основная функция для индексации всех файлов в поддиректориях.
  """
  orgs_dirs = Path(MAIN_DOCS_DIR)
  for org_dir in orgs_dirs.iterdir():
    if org_dir.is_dir():
      for space_dir in org_dir.iterdir():
        if space_dir.is_dir():
          for file_path in space_dir.iterdir():
            index_single_file(file_path=file_path)


image_paths = {}

initialize_image_paths()

if index_config.get("reindex"):
    main_index = RAGMultiModalModel.from_pretrained(
        pretrained_model_name_or_path = colvision_model,
        index_root = INDEX_ROOT_DIR,
        )
    index_all_subdirs()
else:
    main_index = RAGMultiModalModel.from_index(
      index_path = index_name,
      index_root = INDEX_ROOT_DIR,
      )


client = OpenAI(
    api_key="VLLM-PLACEHOLDER-API-KEY",
    base_url=VLLM_URL,
)


def get_retrieve_response(text_query, category=None, space=None, filename=None):
  # Формируем filter_metadata
  filter_metadata = {}
  if category:
    filter_metadata["category"] = category
  if space:
    filter_metadata["space"] = space
  if filename:
    filter_metadata["filename"] = filename

  results = main_index.search(text_query, k=1, filter_metadata=filter_metadata)

  return results


def get_rag_response(text_query, category=None, space=None, filename=None):
  # Формируем filter_metadata
  filter_metadata = {}
  if category:
    filter_metadata["category"] = category
  if space:
    filter_metadata["space"] = space
  if filename:
    filter_metadata["filename"] = filename

  results = main_index.search(text_query, k=1, filter_metadata=filter_metadata)

  if not results:
    return "Я не знаю ответ на ваш вопрос.", None

  # Получаем doc_id из результатов
  doc_id = results[0]["doc_id"]
  page_num = results[0]["page_num"]

  doc_ids_to_file_names = main_index.get_doc_ids_to_file_names()
  file_name = os.path.basename(doc_ids_to_file_names[doc_id])

  # Получаем индекс изображения для конкретного файла
  image_index = page_num - 1
  image_path = image_paths[file_name][image_index]  # Достаем изображение из словаря по имени файла

  # Добавляем префикс file:// перед локальным путем
  image_url = f"file://{os.path.abspath(image_path)}"  # Преобразуем в абсолютный путь, но передаем его как file://

  image_data = {"filename": file_name, "image_index": image_index}

  print(image_url)

  # Готовим сообщение для API, добавляя системный промпт
  messages = [
    {
      "role": "system",
      "content": vlm_system_prompt
    },
    {
      "role": "user",
      "content": [
        {"type": "text", "text": text_query},
        {"type": "image_url", "image_url": {"url": image_url}},  # Передаем путь с префиксом file://
      ],
    }
  ]

  # Отправляем запрос в OpenAI-like API
  chat_response = client.chat.completions.create(
    model=vlm_model,
    messages=messages
  )

  output_text = chat_response.choices[0].message.content

  gc.collect()
  torch.cuda.empty_cache()

  return output_text, image_data


def model_response_to_json(
          model_response,
  ) -> Dict[str, Any]:

    sources = []
    file_data = model_response[1]
    documents = main_index.search(query="", k=1, filter_metadata={"filename": file_data["filename"]})
    document = documents[0] if documents else None  # Получаем первый документ или None, если документов нет

    if document:
      slide_snippet = document['metadata']['summary']
      filename = document['metadata'].get("filename")
      slide_number = file_data['image_index'] + 1

      category = document['metadata'].get("category")
      space = document['metadata'].get("space")

      # Извлечение названия презентации из имени файла (без расширения)
      file_title = filename.rsplit('.', 1)[0] if filename else "Unknown"
      title = f"{file_title} Слайд №{slide_number}" if slide_number != 0 else file_title
      slide_number_str = f"_{slide_number}" if slide_number != 0 else ""

      # Формирование метаданных с использованием space и categories
      metadata = {
        "title": title,
        "file_title": file_title,
        "file_name": filename,
        "file_category": category if category else [None],
        "file_space": space,
        "slide_number": slide_number,
        "url": f"https://storage.yandexcloud.net/hacks-ai-storage/{filename}",
        "thumbnail": f"https://storage.yandexcloud.net/hacks-ai-storage/thumb_{file_title}{slide_number_str}.png"
      }

      # Формирование источника
      source = {
        "pageContent": slide_snippet,
        "metadata": metadata
      }

      sources.append(source)
    else:
      print(f"Документ не найден для {file_data}")

    # Формирование итогового JSON
    json_output = {
      "message": model_response[0],
      "sources": sources
    }

    return json_output


def get_chat_response_json(input_data: Dict[str, Any]) -> dict[str, Any]:
  # Извлечение значений из входного JSON-данных
  category = input_data.get("category", None)
  space = input_data.get("space", None)
  filename = input_data.get("filename", None)
  question = input_data.get("question", "")

  # Получаем ответ модели с учетом извлеченных параметров
  model_response = get_rag_response(
    text_query=question,
    category=category,
    space=space,
    filename=filename
  )

  # Преобразуем ответ модели в JSON с передачей space и categories
  json_output = model_response_to_json(model_response)

  return json_output