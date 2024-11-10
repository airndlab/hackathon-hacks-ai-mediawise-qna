from haystack import Pipeline, Document, component
from haystack.components.converters import PyPDFToDocument, DOCXToDocument, PPTXToDocument
from haystack.components.converters.csv import CSVToDocument
from haystack.components.preprocessors import DocumentSplitter, DocumentCleaner
from haystack.components.routers import FileTypeRouter
from haystack.components.joiners import DocumentJoiner
from haystack_integrations.document_stores.chroma import ChromaDocumentStore
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from haystack.components.writers import DocumentWriter
from haystack.utils import ComponentDevice, Device, Secret
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever, InMemoryBM25Retriever
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.dataclasses import ChatMessage, ChatRole
from haystack.components.builders import ChatPromptBuilder

from pathlib import Path
from pprint import pprint

import yaml
from PyPDF2 import PdfReader

import torch
import json
from typing import List, Optional

import os
import pandas as pd

# s3 client

# pdf2image converter
# from pdf2image import convert_from_path

if torch.cuda.is_available():
    device = ComponentDevice.from_single(Device.gpu(id=0))
else:
    device = ComponentDevice.from_single(Device.cpu())

VLLM_URL = os.getenv("VLLM_URL")
CHROMA_HOSTNAME = os.getenv("CHROMA_HOSTNAME")
CHROMA_PORT = os.getenv("CHROMA_PORT")

YANDEX_API_TOKEN = os.getenv("YANDEX_API_TOKEN")

MAIN_DOCS_DIR = os.getenv('MAIN_DOCS_DIR')

# PROMPT CONFIG
PROMPTS_CONFIG_PATH = os.getenv("PROMPTS_CONFIG_PATH")
with open(PROMPTS_CONFIG_PATH, 'r', encoding='utf-8') as file:
    prompt_config = yaml.safe_load(file)

# RAG CONFIG
RAG_CONFIG_PATH = os.getenv("RAG_CONFIG_PATH")
with open(RAG_CONFIG_PATH, "r") as file:
    rag_config = yaml.safe_load(file)

model = rag_config.get("model")
embedding_model = rag_config.get("embedding_model")

split_function_config = rag_config.get("split_function", {})
max_chunk_size = split_function_config.get("max_chunk_size", 300)
overlap = split_function_config.get("overlap", 0)

rag_gen_kwargs = rag_config.get("rag_gen_kwargs", {})
json_gen_kwargs = rag_config.get("json_gen_kwargs", {})

# API CONFIG
API_CONFIG_PATH = os.getenv("API_CONFIG_PATH")
with open(API_CONFIG_PATH, "r", encoding='utf-8') as config_file:
    api_config = yaml.safe_load(config_file)

# Извлечение URL и эндпоинтов из конфигурации
BASE_URL = api_config["api"]["base_url"]
POLL_ENDPOINT = f"{BASE_URL}{api_config['api']['endpoints']['poll']}"
COMPLETE_ENDPOINT = f"{BASE_URL}{api_config['api']['endpoints']['complete']}"


@component
class QueryExpander:
    def __init__(
            self,
            system_prompt: str,
            user_prompt_template: str,
            json_gen_kwargs,
    ):
        self.system_prompt = system_prompt
        self.user_prompt_template = user_prompt_template

        builder = ChatPromptBuilder(variables=["query", "user_info"])
        llm = create_generator(json_gen_kwargs)

        self.pipeline = Pipeline()
        self.pipeline.add_component(name="builder", instance=builder)
        self.pipeline.add_component(name="llm", instance=llm)
        self.pipeline.connect("builder", "llm")

    @component.output_types(queries=List[str])
    def run(
            self,
            query: str,
    ):
        messages = [
            ChatMessage.from_system(self.system_prompt),
            ChatMessage.from_user(self.user_prompt_template)
        ]

        result = self.pipeline.run({
            'builder': {
                'template': messages,
                'query': query,
            }
        })

        response_text = result['llm']['replies'][0].content

        def extract_json_array(text):
            last_bracket_idx = text.rfind(']')
            if last_bracket_idx == -1:
                return None, text
            first_bracket_idx = text.rfind('[', 0, last_bracket_idx)
            if first_bracket_idx == -1:
                return None, text
            json_str = text[first_bracket_idx:last_bracket_idx + 1]
            remaining_text = text[:first_bracket_idx].strip()
            return json_str, remaining_text

        json_str, remaining_text = extract_json_array(response_text)

        expanded_queries = []

        if json_str:
            try:
                expanded_queries = json.loads(json_str)

            except Exception as e:
                print(e)
                return {"queries": [query]}

        expanded_queries.append(query)

        return {"queries": expanded_queries}


@component
class MultiQueryTextEmbedder:
    def __init__(self, embedder: SentenceTransformersTextEmbedder, top_k: int = 1):
        self.embedder = embedder
        self.embedder.warm_up()
        self.results = []
        self.ids = set()
        self.top_k = top_k

    @component.output_types(embeddings=List[List[str]])
    def run(self, queries: List[str]):
        self.results = []
        for query in queries:
            self.results.append(self.embedder.run(query))

        return {"embeddings": self.results}


@component
class MultiQueryInMemoryRetriever:
    def __init__(self, retriever: InMemoryEmbeddingRetriever, filters=None, top_k: int = 3,
                 score_threshold: float = 0.0):

        self.retriever = retriever
        self.results = []
        self.ids = set()
        self.top_k = top_k
        self.filters = filters
        self.score_threshold = score_threshold

    def add_document(self, document: Document):
        if (document.id not in self.ids) and (document.score > self.score_threshold):
            self.results.append(document)
            self.ids.add(document.id)

    @component.output_types(documents=List[Document])
    def run(self, emdeddings: List[List[str]], filters=None):
        self.results = []
        self.ids = set()

        for emdedding in emdeddings:
            result = self.retriever.run(query_embedding=emdedding['embedding'], filters=filters, top_k=self.top_k)
            for doc in result['documents']:
                self.add_document(doc)

        self.results.sort(key=lambda x: x.score, reverse=True)

        return {"documents": self.results}


@component
class MultiQueryInMemoryBM25Retriever:
    def __init__(self, retriever: InMemoryBM25Retriever, top_k: int = 3, filters=None):

        self.retriever = retriever
        self.results = []
        self.ids = set()
        self.filters = filters
        self.top_k = top_k

    def add_document(self, document: Document):
        if document.id not in self.ids:
            self.results.append(document)
            self.ids.add(document.id)

    @component.output_types(documents=List[Document])
    def run(self, queries: List[str], top_k: int = None, filters=None):
        self.results = []
        self.ids = set()
        if top_k != None:
            self.top_k = top_k
        for query in queries:
            result = self.retriever.run(query=query, filters=filters, top_k=self.top_k)
            for doc in result['documents']:
                self.add_document(doc)
        self.results.sort(key=lambda x: x.score, reverse=True)
        return {"documents": self.results}


def create_in_memory_document_store():
    document_store = InMemoryDocumentStore()

    return document_store


def create_chroma_document_store():
    document_store = ChromaDocumentStore(
        host=CHROMA_HOSTNAME,
        port=CHROMA_PORT
    )

    return document_store


def create_indexing_pipeline(document_store):
    file_type_router = FileTypeRouter(mime_types=[
        "application/pdf",
        "text/csv",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    ])

    pdf_converter = PyPDFToDocument()
    docx_converter = DOCXToDocument()
    csv_converter = CSVToDocument()
    pptx_converter = PPTXToDocument()

    document_joiner = DocumentJoiner()
    document_cleaner = DocumentCleaner()
    document_splitter = DocumentSplitter(split_by="page", split_length=1, split_overlap=0)
    document_writer = DocumentWriter(document_store)
    document_embedder = SentenceTransformersDocumentEmbedder(model=embedding_model, device=device)

    indexing_pipeline = Pipeline()

    indexing_pipeline.add_component(instance=file_type_router, name="file_type_router")
    indexing_pipeline.add_component(instance=pdf_converter, name="pypdf_converter")
    indexing_pipeline.add_component(instance=docx_converter, name="docx_converter")
    indexing_pipeline.add_component(instance=csv_converter, name="csv_converter")
    indexing_pipeline.add_component(instance=pptx_converter, name="pptx_converter")

    indexing_pipeline.add_component(instance=document_joiner, name="document_joiner")
    indexing_pipeline.add_component(instance=document_cleaner, name="document_cleaner")
    indexing_pipeline.add_component(instance=document_splitter, name="document_splitter")
    indexing_pipeline.add_component(instance=document_embedder, name="document_embedder")
    indexing_pipeline.add_component(instance=document_writer, name="document_writer")

    indexing_pipeline.connect("file_type_router.application/pdf", "pypdf_converter.sources")
    indexing_pipeline.connect(
        "file_type_router.application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "docx_converter.sources")
    indexing_pipeline.connect(
        "file_type_router.application/vnd.openxmlformats-officedocument.presentationml.presentation",
        "pptx_converter.sources"
    )
    indexing_pipeline.connect("file_type_router.text/csv", "csv_converter.sources")

    indexing_pipeline.connect("pypdf_converter", "document_joiner")
    indexing_pipeline.connect("docx_converter", "document_joiner")
    indexing_pipeline.connect("csv_converter", "document_joiner")
    indexing_pipeline.connect("pptx_converter", "document_joiner")

    indexing_pipeline.connect("document_joiner", "document_splitter")
    indexing_pipeline.connect("document_splitter", "document_cleaner")
    indexing_pipeline.connect("document_cleaner", "document_embedder")
    indexing_pipeline.connect("document_embedder", "document_writer")

    return indexing_pipeline


document_store = create_in_memory_document_store()

indexing_pipeline = create_indexing_pipeline(document_store=document_store)


def create_generator(gen_kwargs=None):
    return OpenAIChatGenerator(
        api_key=Secret.from_token("VLLM-PLACEHOLDER-API-KEY"),
        model=model,
        api_base_url=VLLM_URL,
        generation_kwargs=gen_kwargs,
        timeout=600,
    )


def create_summarize_pipeline():
    pipeline = Pipeline()

    prompt_builder = ChatPromptBuilder(variables=['document_fragment'])

    generator = create_generator(rag_gen_kwargs)

    pipeline.add_component("prompt_builder", prompt_builder)
    pipeline.add_component("generator", generator)

    pipeline.connect("prompt_builder", "generator")

    return pipeline


summarize_pipeline = create_summarize_pipeline()

gen_questions_system_message = ChatMessage.from_system(prompt_config['summarize_system_prompt'])
summarize_messages = [gen_questions_system_message]


def generate_summarization(document_content):
    # Выполняем запрос к пайплайну суммаризации
    response = summarize_pipeline.run(
        {"prompt_builder": {"template": summarize_messages, "document_fragment": document_content}})

    # Извлекаем текст ответа
    response_text = response['generator']['replies'][0].content

    # Поиск первого и последнего вхождения фигурных скобок для выделения JSON
    json_str = None
    try:
        first_brace = response_text.find('{')
        last_brace = response_text.rfind('}')

        # Если фигурные скобки найдены, извлекаем текст между ними
        if first_brace != -1 and last_brace != -1:
            json_str = response_text[first_brace:last_brace + 1]
            response_dict = json.loads(json_str)
        else:
            print("Error: JSON structure not found in response.")
            return {}

    except json.JSONDecodeError as e:
        print(f"Error parsing JSON response: {e}")
        print(f"Attempted JSON string: {json_str}")
        return {}

    return response_dict


def extract_context_from_pdf(pdf_path: str) -> str:
    context = ""
    try:
        # Read PDF content using PyPDF2
        with open(pdf_path, 'rb') as f:
            reader = PdfReader(f)
            # Set the number of pages to extract (up to 3)
            num_pages_to_extract = min(7, len(reader.pages))

            # Extract text from each page up to the third page
            for page_num in range(num_pages_to_extract):
                page = reader.pages[page_num]
                text = page.extract_text()
                if text:
                    context += text

            # Return the extracted context or a fallback if empty
            return context.strip() if context else "No text extracted from the first three pages."

    except Exception as e:
        print(f"An error occurred while extracting context from '{pdf_path}': {e}")
        return 'Error extracting context'


def do_reindex():
    orgs_dirs = Path(MAIN_DOCS_DIR)
    for org_dir in orgs_dirs.iterdir():
        if org_dir.is_dir():
            source_docs = list(org_dir.glob("**/*"))
            for doc in source_docs:
                # Проверка типа файла и обработка PDF
                if doc.suffix.lower() == ".pdf":
                    # Извлечение контекста из PDF и генерация суммаризации
                    context = extract_context_from_pdf(doc)
                    summarization = generate_summarization(context)

                    # Заполнение метаданных из суммаризации
                    meta_data = {
                        "space": org_dir.name,
                        "filename": os.path.basename(doc),
                        "title": summarization.get("title", ""),
                        "summary": summarization.get("summary", ""),
                        "categories": summarization.get("categories", [])
                    }
                else:
                    # Для других типов файлов - заполняем метаданные пустыми значениями
                    meta_data = {
                        "space": org_dir.name,
                        "filename": os.path.basename(doc),
                        "title": "",
                        "summary": "",
                        "categories": []
                    }

                # Запуск пайплайна индексации с заполненными метаданными
                indexing_pipeline.run({
                    "file_type_router": {"sources": [doc]},
                    "pypdf_converter": {"meta": meta_data},
                    "csv_converter": {"meta": meta_data},
                    "docx_converter": {"meta": meta_data},
                    "pptx_converter": {"meta": meta_data},
                })

                # Очистка памяти
                gc.collect()
                torch.cuda.empty_cache()


def do_index_single_file(file_path):
    # Преобразуем путь к файлу в объект Path
    file_path = Path(file_path)

    # Проверяем, что файл существует
    if not file_path.exists() or not file_path.is_file():
        raise ValueError(f"The file {file_path} does not exist or is not a valid file.")

    # Извлекаем директорию, в которой находится файл, для использования в метаданных
    space = file_path.parent.name
    filename = os.path.basename(file_path)

    # Проверка типа файла и обработка PDF
    if file_path.suffix.lower() == ".pdf":
        # Извлечение контекста из PDF и генерация суммаризации
        context = extract_context_from_pdf(file_path)
        summarization = generate_summarization(context)

        # Заполнение метаданных из суммаризации
        meta_data = {
            "space": space,
            "filename": filename,
            "title": summarization.get("title", ""),
            "summary": summarization.get("summary", ""),
            "categories": summarization.get("categories", [])
        }
    else:
        # Для других типов файлов - заполняем метаданные пустыми значениями
        meta_data = {
            "space": space,
            "filename": filename,
            "title": "",
            "summary": "",
            "categories": []
        }

    # Запуск пайплайна индексации для одного файла с заполненными метаданными
    indexing_pipeline.run({
        "file_type_router": {"sources": [file_path]},
        "pypdf_converter": {"meta": meta_data},
        "csv_converter": {"meta": meta_data},
        "docx_converter": {"meta": meta_data},
        "pptx_converter": {"meta": meta_data},
    })

def create_rag_pipeline(
        document_store,
) -> Pipeline:
    expander = QueryExpander(
        system_prompt = prompt_config['query_expander_system_prompt'],
        user_prompt_template = prompt_config['query_expander_user_prompt_template'],
        json_gen_kwargs = json_gen_kwargs,
    )
    document_joiner = DocumentJoiner(top_k = rag_config.get("document_joiner").get("top_k", 4), join_mode = "distribution_based_rank_fusion")
    text_embedder = MultiQueryTextEmbedder(SentenceTransformersTextEmbedder(model=embedding_model, device=device))
    embedding_retriever = MultiQueryInMemoryRetriever(InMemoryEmbeddingRetriever(document_store), top_k = rag_config.get("embedding_retriever").get("top_k", 3))
    bm25_retriever = MultiQueryInMemoryBM25Retriever(InMemoryBM25Retriever(document_store), top_k = rag_config.get("bm25_retriever").get("top_k", 3))

    final_answer_prompt_builder = ChatPromptBuilder(variables=["documents", "question"])

    doc_relevant_generator = create_generator(rag_gen_kwargs)
    final_answer_generator = create_generator(rag_gen_kwargs)

    rag_pipeline = Pipeline()

    rag_pipeline.add_component("expander", expander)
    rag_pipeline.add_component("text_embedder", text_embedder)
    rag_pipeline.add_component("embedding_retriever", embedding_retriever)
    rag_pipeline.add_component("bm25_retriever", bm25_retriever)
    rag_pipeline.add_component("document_joiner", document_joiner)

    rag_pipeline.add_component("final_answer_prompt_builder", final_answer_prompt_builder)
    rag_pipeline.add_component("final_answer_llm", final_answer_generator)


    rag_pipeline.connect("expander.queries", "text_embedder.queries")
    rag_pipeline.connect("expander.queries", "bm25_retriever.queries")
    rag_pipeline.connect("text_embedder.embeddings", "embedding_retriever")
    rag_pipeline.connect("bm25_retriever", "document_joiner")
    rag_pipeline.connect("embedding_retriever", "document_joiner")

    rag_pipeline.connect("document_joiner", "final_answer_prompt_builder.documents")
    rag_pipeline.connect("final_answer_prompt_builder", "final_answer_llm")

    return rag_pipeline

rag_pipeline = create_rag_pipeline(document_store = document_store)

chat_system_message = ChatMessage.from_system(prompt_config['qwen_chat_system_prompt'])
chat_user_message = ChatMessage.from_user(prompt_config['qwen_chat_user_prompt_template'])

final_answer_messages = [chat_system_message, chat_user_message]

class Reference(BaseModel):
    filename: str
    page_number: str

class ModelResponse(BaseModel):
    answer: str
    references: Optional[List[Reference]] = None

def get_chat_response(
        question: str,
        categories: Optional[List[str]] = None,
        space: Optional[str] = None,
        filename: Optional[str] = None,
) -> ModelResponse:
    # Формирование фильтров для bm25_retriever и embedding_retriever
    filters = {"operator": "AND", "conditions": []}
    if categories:
        filters["conditions"].append({"field": "meta.categories", "operator": "in", "value": categories})
    if space:
        filters["conditions"].append({"field": "meta.space", "operator": "==", "value": space})
    if filename:
        filters["conditions"].append({"field": "meta.filename", "operator": "==", "value": filename})
    # Запускаем RAG-пайплайн с фильтрами
    response = rag_pipeline.run({
        "expander": {
            "query": question,
        },
        "bm25_retriever": {
            "filters": filters,
        },
        "embedding_retriever": {
            "filters": filters,
        },
        "final_answer_prompt_builder": {
            "question": question,
            "template": final_answer_messages,
        },
    }, include_outputs_from={"expander", "document_joiner", "final_answer_prompt_builder"})

    response_text = response['final_answer_llm']['replies'][0].content

    references = None
    try:
        # Находим индекс последнего символа ']'
        last_bracket_idx = response_text.rfind(']')
        if last_bracket_idx != -1:
            # Находим индекс соответствующего символа '[' перед ']'
            first_bracket_idx = response_text.rfind('[', 0, last_bracket_idx)
            if first_bracket_idx != -1:
                json_str = response_text[first_bracket_idx:last_bracket_idx+1]

                try:
                    # Парсим JSON-строку
                    references_list = json.loads(json_str)
                    references = [Reference(**item) for item in references_list]
                    response_text = response_text[:first_bracket_idx].strip()
                except json.JSONDecodeError as e:
                    print(f"Ошибка при парсинге JSON: {e}")
    except Exception as e:
        print(f"Общая ошибка при обработке ответа: {e}")

    gc.collect()
    torch.cuda.empty_cache()

    return ModelResponse(answer=response_text, references=references)

def model_response_to_json(
        model_response: ModelResponse,
        categories: Optional[List[str]] = None,
        space: Optional[str] = None
) -> Dict[str, Any]:
    sources = []

    if model_response.references:
        for ref in model_response.references:
            # Фильтруем только по page_number и filename
            filters = {
                "operator": "AND",
                "conditions": [
                    {"field": "meta.page_number", "operator": "==", "value": int(ref.page_number)},
                    {"field": "meta.filename", "operator": "==", "value": ref.filename},
                ]
            }

            # Извлечение единственного документа из document_store по фильтрам
            documents = document_store.filter_documents(filters=filters)
            document = documents[0] if documents else None  # Получаем первый документ или None, если документов нет

            if document and document.meta.get("filename") == ref.filename:
                # Извлечение необходимых полей
                slide_snippet = document.content
                filename = document.meta.get("filename")
                slide_number = document.meta.get("page_number")

                # Извлечение названия презентации из имени файла (без расширения)
                file_title = filename.rsplit('.', 1)[0] if filename else "Unknown"

                # Формирование метаданных с использованием space и categories
                metadata = {
                    "title": f"{file_title} Слайд №{slide_number}",
                    "file_title": file_title,
                    "file_name": filename,
                    "file_category": categories if categories else [None],  # Используем categories или [null] в JSON
                    "file_space": space,       # Используем space или null в JSON
                    "slide_number": slide_number,
                    "url": f"https://storage.yandexcloud.net/hacks-ai-storage/{filename}",
                    "thumbnail": f"https://storage.yandexcloud.net/hacks-ai-storage/thumb_{filename}_{slide_number}.png"
                }

                # Формирование источника
                source = {
                    "pageContent": slide_snippet,
                    "metadata": metadata
                }

                sources.append(source)
            else:
                # Обработка случая, когда документ не найден
                print(f"Документ не найден для filename: {ref.filename}, page_number: {ref.page_number}")

    # Формирование итогового JSON
    json_output = {
        "message": model_response.answer,
        "sources": sources
    }

    return json_output

def get_chat_response_json(input_data: Dict[str, Any]) -> str:
    # Извлечение значений из входного JSON-данных
    categories = input_data.get("categories", None)
    space = input_data.get("space", None)
    filename = input_data.get("filename", None)
    question = input_data.get("question", "")

    # Получаем ответ модели с учетом извлеченных параметров
    model_response = get_chat_response(
        question=question,
        categories=categories,
        space=space,
        filename=filename
    )

    # Преобразуем ответ модели в JSON с передачей space и categories
    json_output = model_response_to_json(model_response, categories=categories, space=space)

    return json_output

def create_gen_rel_questions_pipeline():
    pipeline = Pipeline()

    prompt_builder = ChatPromptBuilder()
    generator = create_generator() # Можно параметры чтобы поразнообразнее

    pipeline.add_component("prompt_builder", prompt_builder)
    pipeline.add_component("generator", generator)

    pipeline.connect("prompt_builder", "generator")

    return pipeline

gen_questions_pipeline = create_gen_rel_questions_pipeline()

gen_questions_system_message = ChatMessage.from_system(prompt_config['qwen_gen_related_system_prompt'])

def generate_related_questions(conversational: List[List[Union[str, str]]]) -> List[str]:
    # Инициализируем сообщения, добавляя системное сообщение
    chat_messages = [gen_questions_system_message]

    # Фильтруем и ограничиваем историю переписки до последних 5 сообщений
    filtered_conversational = conversational[-5:]

    # Формируем сообщения на основе фильтрованных данных
    for role, content in filtered_conversational:
        if role in {"human", "user"}:
            chat_messages.append(ChatMessage.from_user(content))
        else:  # Любая другая роль трактуется как сообщение ассистента
            chat_messages.append(ChatMessage.from_assistant(content))

    # Запуск пайплайна с подготовленными сообщениями
    result = gen_questions_pipeline.run({"prompt_builder": {"template": chat_messages}})

    # Извлечение и парсинг сгенерированного ответа
    generated_replies = result["generator"]["replies"][0].content
    questions = []
    try:
        # Находим индекс последнего символа ']' и первого '[' перед ним
        last_bracket_idx = generated_replies.rfind(']')
        if last_bracket_idx != -1:
            first_bracket_idx = generated_replies.rfind('[', 0, last_bracket_idx)
            if first_bracket_idx != -1:
                json_str = generated_replies[first_bracket_idx:last_bracket_idx + 1]

                try:
                    # Парсим JSON-строку как список вопросов
                    questions_list = json.loads(json_str)
                    if isinstance(questions_list, list) and all(isinstance(item, str) for item in questions_list):
                        questions = questions_list
                except json.JSONDecodeError as e:
                    print(f"JSON parsing error: {e}")
    except Exception as e:
        print(f"General error processing response: {e}")

    # Возвращаем распарсенные вопросы или пустой список при ошибке
    return questions
