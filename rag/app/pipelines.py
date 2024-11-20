from haystack import Pipeline, Document, component, tracing
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
from haystack.dataclasses import ChatMessage
from haystack.components.builders import ChatPromptBuilder
from haystack.tracing.logging_tracer import LoggingTracer

from pathlib import Path

import yaml
import requests
from datetime import datetime
from PyPDF2 import PdfReader

import gc
import io
import torch
import json
from pydantic import BaseModel
from typing import List, Optional, Dict, Any, Union

import os
import time

import logging
import pickle

logger = logging.getLogger(__name__)

tracing.tracer.is_content_tracing_enabled = True
tracing.enable_tracing(LoggingTracer(tags_color_strings={"haystack.component.input": "\x1b[1;31m", "haystack.component.name": "\x1b[1;34m"}))

if torch.cuda.is_available():
    device = ComponentDevice.from_single(Device.gpu(id=0))
else:
    device = ComponentDevice.from_single(Device.cpu())

VLLM_URL = os.getenv("VLLM_URL")
CHROMA_HOSTNAME = os.getenv("CHROMA_HOSTNAME")
CHROMA_PORT = os.getenv("CHROMA_PORT")

YANDEX_API_TOKEN = os.getenv("YANDEX_API_TOKEN")

MAIN_DOCS_DIR = os.getenv('MAIN_DOCS_DIR')
DOCUMENT_STORES_DIR = os.getenv('DOCUMENT_STORES_DIR')

# PROMPT CONFIG
PROMPTS_CONFIG_PATH = os.getenv("PROMPTS_CONFIG_PATH")
with open(PROMPTS_CONFIG_PATH, 'r', encoding='utf-8') as file:
    prompt_config = yaml.safe_load(file)

# DICTS CONFIG
DICTS_CONFIG_PATH = os.getenv("DICTS_CONFIG_PATH")
with open(DICTS_CONFIG_PATH, 'r') as file:
    dicts_config = yaml.safe_load(file)

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
            json_str = text[first_bracket_idx:last_bracket_idx+1]
            remaining_text = text[:first_bracket_idx].strip()
            return json_str, remaining_text

        json_str, remaining_text = extract_json_array(response_text)

        expanded_queries = []

        if json_str:
            try:
                expanded_queries = json.loads(json_str)

            except Exception as e:
                logger.warn(e)
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
    def __init__(self, retriever: InMemoryEmbeddingRetriever, filters=None, top_k: int = 3, score_threshold: float = 0.0):

        self.retriever = retriever
        self.results = []
        self.ids = set()
        self.top_k = top_k
        self.filters=filters
        self.score_threshold = score_threshold

    def add_document(self, document: Document):
        if (document.id not in self.ids) and (document.score > self.score_threshold):
            logger.info(f"Embed: Adding document with score {document.score}")
            self.results.append(document)
            self.ids.add(document.id)

    @component.output_types(documents=List[Document])
    def run(self, emdeddings: List[List[str]], filters=None):
        self.results = []
        self.ids = set()

        for emdedding in emdeddings:
            result = self.retriever.run(query_embedding=emdedding['embedding'], filters = filters, top_k = self.top_k)
            for doc in result['documents']:
                self.add_document(doc)

        self.results.sort(key=lambda x: x.score, reverse=True)

        return {"documents": self.results}

@component
class MultiQueryInMemoryBM25Retriever:
    def __init__(self, retriever: InMemoryBM25Retriever, top_k: int = 3, filters=None, score_threshold: float = 0.0):

        self.retriever = retriever
        self.results = []
        self.ids = set()
        self.filters=filters
        self.top_k = top_k
        self.score_threshold = score_threshold

    def add_document(self, document: Document):
        if (document.id not in self.ids) and (document.score > self.score_threshold):
            logger.info(f"BM25: Adding document with score {document.score}")
            self.results.append(document)
            self.ids.add(document.id)

    @component.output_types(documents=List[Document])
    def run(self, queries: List[str], top_k: int = None, filters=None):
        self.results = []
        self.ids = set()

        if top_k != None:
            self.top_k = top_k

        for query in queries:
            result = self.retriever.run(query = query, filters = filters, top_k = self.top_k)
            for doc in result['documents']:
                self.add_document(doc)

        self.results.sort(key=lambda x: x.score, reverse=True)

        return {"documents": self.results}

def create_in_memory_document_store():
    document_store = InMemoryDocumentStore()

    return document_store

def create_chroma_document_store():
    document_store = ChromaDocumentStore(
        host = CHROMA_HOSTNAME,
        port = CHROMA_PORT
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
    document_embedder = SentenceTransformersDocumentEmbedder(model = embedding_model, device = device)

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

def save_documents_from_store(document_store_name: str):
    if not document_store_name.endswith(".pkl"):
        document_store_name += ".pkl"

    with open(os.path.join(DOCUMENT_STORES_DIR, document_store_name), "wb") as f:
        pickle.dump(document_store.filter_documents(), f)

def load_documents_to_store(document_store, document_store_name: str):
    if not document_store_name.endswith(".pkl"):
        document_store_name += ".pkl"

    with open(os.path.join(DOCUMENT_STORES_DIR, document_store_name), 'rb') as f:
        all_documents = pickle.load(f)

    document_store.write_documents(all_documents)

    gc.collect()
    torch.cuda.empty_cache()

    # return document_store

document_store = create_in_memory_document_store()

load_documents_to_store(document_store, "test_3_document_store")

indexing_pipeline = create_indexing_pipeline(document_store = document_store)

def create_generator(gen_kwargs = None):
    return OpenAIChatGenerator(
        api_key = Secret.from_token("VLLM-PLACEHOLDER-API-KEY"),
        model = model,
        api_base_url = VLLM_URL,
        generation_kwargs = gen_kwargs,
        timeout = 600,
    )

def create_summarize_pipeline():
    pipeline = Pipeline()

    prompt_builder = ChatPromptBuilder(variables = ['document_fragment'])

    generator = create_generator(rag_gen_kwargs)

    pipeline.add_component("prompt_builder", prompt_builder)
    pipeline.add_component("generator", generator)

    pipeline.connect("prompt_builder", "generator")

    return pipeline

summarize_pipeline = create_summarize_pipeline()

gen_questions_system_message = ChatMessage.from_system(prompt_config['summarize_system_prompt'])
summarize_messages = [gen_questions_system_message]

def generate_summarization(document_content):
    response = summarize_pipeline.run(
        {
            "prompt_builder":
                {
                    "template": summarize_messages,
                    "document_fragment": document_content
                }
        }
    )

    response_text = response['generator']['replies'][0].content

    json_str = None
    try:
        first_brace = response_text.find('{')
        last_brace = response_text.rfind('}')

        # Если фигурные скобки найдены, извлекаем текст между ними
        if first_brace != -1 and last_brace != -1:
            json_str = response_text[first_brace:last_brace + 1]
            response_dict = json.loads(json_str)
        else:
            logger.warn("Error: JSON structure not found in response.")
            return {}

    except json.JSONDecodeError as e:
        logger.warn(f"Error parsing JSON response: {e}")
        logger.warn(f"Attempted JSON string: {json_str}")
        return {}

    return response_dict

def extract_context_from_pdf(pdf_path: str) -> str:
    context = ""
    try:
        with open(pdf_path, 'rb') as f:
            reader = PdfReader(f)
            num_pages_to_extract = min(7, len(reader.pages))

            for page_num in range(num_pages_to_extract):
                page = reader.pages[page_num]
                text = page.extract_text()
                if text:
                    context += text

            return context.strip() if context else "No text extracted from the first three pages."

    except Exception as e:
        logger.warn(f"An error occurred while extracting context from '{pdf_path}': {e}")
        return 'Error extracting context'

def do_reindex():
    orgs_dirs = Path(MAIN_DOCS_DIR)
    for org_dir in orgs_dirs.iterdir():
        if org_dir.is_dir():
            source_docs = list(org_dir.glob("**/*"))
            for doc in source_docs:
                space = doc.parent.name
                filename = os.path.basename(doc)
                organization = org_dir.name

                if doc.suffix.lower() == ".pdf":
                    context = extract_context_from_pdf(doc)
                    summary_dict = generate_summarization(context)
                else:
                    summary_dict = {}

                title = summary_dict.get("title", "")
                summary = summary_dict.get("summary", "")
                category = summary_dict.get("category", "")

                meta_data = {
                    "organization": organization,
                    "space": space,
                    "filename": filename,
                    "title": title,
                    "summary": summary,
                    "category": category,
                }

                if summary != "":
                    meta_data.update({"page_number": 0})
                    meta_document = Document(
                        content=summary,
                        meta=meta_data,
                    )

                    document_store.write_documents([meta_document])

                indexing_pipeline.run({
                    "file_type_router": {"sources": [doc]},
                    "pypdf_converter": {"meta": meta_data},
                    "csv_converter": {"meta": meta_data},
                    "docx_converter": {"meta": meta_data},
                    "pptx_converter": {"meta": meta_data},
                })

                gc.collect()
                torch.cuda.empty_cache()


def do_index_single_file(file_path):
    file_path = Path(file_path)

    if not file_path.exists() or not file_path.is_file():
        raise ValueError(f"The file {file_path} does not exist or is not a valid file.")

    organization = file_path.parent.parent.name
    space = file_path.parent.name
    filename = os.path.basename(file_path)

    if file_path.suffix.lower() == ".pdf":
        context = extract_context_from_pdf(file_path)
        summary_dict = generate_summarization(context)
    else:
        summary_dict = {}

    title = summary_dict.get("title", "")
    summary = summary_dict.get("summary", "")
    category = summary_dict.get("category", "")

    meta_data = {
        "organization": organization,
        "space": space,
        "filename": filename,
        "title": title,
        "summary": summary,
        "category": category,
    }

    indexing_pipeline.run({
        "file_type_router": {"sources": [file_path]},
        "pypdf_converter": {"meta": meta_data},
        "csv_converter": {"meta": meta_data},
        "docx_converter": {"meta": meta_data},
        "pptx_converter": {"meta": meta_data},
    })

    if summary != "":
        meta_data.update({"page_number": 0})
        meta_document = Document(
            content=summary,
            meta=meta_data,
        )

        document_store.write_documents([meta_document])

    gc.collect()
    torch.cuda.empty_cache()

def find_manual_summary_documents():
    # Получаем все документы из document_store
    all_documents = document_store.filter_documents()

    # Фильтруем только те документы, где content совпадает с meta["summary"]
    manual_summary_docs = [
        doc for doc in all_documents
        if doc.content == doc.meta.get("summary")
    ]

    return manual_summary_docs

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
    page_number: int

class ModelResponse(BaseModel):
    answer: str
    references: Optional[List[Reference]] = None

def get_chat_response(
        question: str,
        category: Optional[str] = None,
        space: Optional[str] = None,
        filename: Optional[str] = None,
) -> ModelResponse:

    filters = {"operator": "AND", "conditions": []}
    if space:
        filters["conditions"].append({"field": "meta.space", "operator": "==", "value": space})
    if filename:
        filters["conditions"].append({"field": "meta.filename", "operator": "==", "value": filename})
    if category:
        filters["conditions"].append({"field": "meta.category", "operator": "==", "value": category})

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
        documents = response.get('document_joiner', {}).get('documents', [])
        document_joiner_references = [
            Reference(
                filename=doc.meta.get('filename', ''),
                page_number=int(doc.meta.get('page_number')) if doc.meta.get('page_number') is not None else 0
            ) for doc in documents
        ]
    except Exception as e:
        logger.warn(f"Ошибка при формировании references из документов: {e}")

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
                    logger.warn(f"Ошибка при парсинге JSON: {e}")
    except Exception as e:
        logger.warn(f"Общая ошибка при обработке ответа: {e}")

    if references and document_joiner_references:
        for doc_joiner_ref in document_joiner_references:
            if doc_joiner_ref not in references:
                references.append(doc_joiner_ref)

    if references is None and document_joiner_references:
        references = document_joiner_references

    if "Я не знаю ответа на ваш вопрос" in response_text:
        response_text = "Я не знаю ответа на ваш вопрос"

    gc.collect()
    torch.cuda.empty_cache()

    return ModelResponse(answer=response_text, references=references)

def model_response_to_json(
        model_response: ModelResponse,
        category: Optional[str] = None,
        space: Optional[str] = None,
) -> Dict[str, Any]:
    sources = []

    if model_response.references:
        for ref in model_response.references:
            # Фильтруем только по page_number и filename
            filters = {
                "operator": "AND",
                "conditions": [
                    {"field": "meta.page_number", "operator": "==", "value": ref.page_number},
                    {"field": "meta.filename", "operator": "==", "value": ref.filename},
                ]
            }

            # Извлечение единственного документа из document_store по фильтрам
            documents = document_store.filter_documents(filters=filters)
            document = documents[0] if documents else None  # Получаем первый документ или None, если документов нет

            if document and document.meta.get("filename") == ref.filename:
                slide_snippet = document.content
                filename = document.meta.get("filename")
                slide_number = document.meta.get("page_number")

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
                logger.warn(f"Документ не найден для filename: {ref.filename}, page_number: {ref.page_number}")

    # Формирование итогового JSON
    json_output = {
        "message": model_response.answer,
        "sources": sources
    }

    return json_output

def get_chat_response_json(input_data: Dict[str, Any]) -> str:
    # Извлечение значений из входного JSON-данных
    category = input_data.get("category", None)
    space = input_data.get("space", None)
    filename = input_data.get("filename", None)
    question = input_data.get("question", "")

    # Получаем ответ модели с учетом извлеченных параметров
    model_response = get_chat_response(
        question=question,
        category=category,
        space=space,
        filename=filename
    )

    # Преобразуем ответ модели в JSON с передачей space и categories
    json_output = model_response_to_json(model_response, category=category, space=space)

    return json_output

def create_gen_rel_questions_pipeline():
    pipeline = Pipeline()

    prompt_builder = ChatPromptBuilder()
    generator = create_generator()

    pipeline.add_component("prompt_builder", prompt_builder)
    pipeline.add_component("generator", generator)

    pipeline.connect("prompt_builder", "generator")

    return pipeline

gen_questions_pipeline = create_gen_rel_questions_pipeline()

gen_questions_system_message = ChatMessage.from_system(prompt_config['qwen_gen_related_system_prompt'])

def generate_related_questions(conversational: List[List[Union[str, str]]]) -> List[str]:
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
                    logger.warn(f"JSON parsing error: {e}")
    except Exception as e:
        logger.warn(f"General error processing response: {e}")

    # Возвращаем распарсенные вопросы или пустой список при ошибке
    return questions

import pandas as pd
import os

def generate_submission_csv(input_csv: str, output_csv: str):
    """
    Reads questions from a CSV file, processes each question using get_chat_response,
    and writes the results to a CSV file.

    Parameters:
    - input_csv (str): Path to the input CSV file (questions.csv).
    - output_csv (str): Path to the output CSV file (submission.csv).
    """
    # Чтение CSV-файла в DataFrame
    try:
        df = pd.read_csv(input_csv, dtype={'question': str, 'filename': str, 'slide_number': str})
    except Exception as e:
        logger.warn(f"Ошибка при чтении файла CSV: {e}")
        return

    # Проверка наличия столбца question
    if 'question' not in df.columns:
        logger.warn(f"Отсутствует необходимый столбец: question\n{df.columns}")
        return

    # Добавление колонки answer, если ее нет
    if 'answer' not in df.columns:
        df['answer'] = ""

    # Итерация по каждой строке и обработка вопроса
    for index, row in df.iterrows():
        question = row['question']
        filename = f"{row['filename']}.pdf"  # Чтение существующего значения filename
        slide_number = row['slide_number']  # Чтение существующего значения slide_number

        if pd.isna(question) or not isinstance(question, str):
            # Если вход не является валидным вопросом
            df.at[index, 'filename'] = filename
            df.at[index, 'slide_number'] = slide_number
            # df.at[index, 'answer'] = "Я не знаю ответа на ваш вопрос"
            continue
        # Получение ответа от модели с передачей question и filename
        model_response = get_chat_response(question, filename=filename)


        # Заполнение полей answer, filename и slide_number
        # df.at[index, 'answer'] = model_response.answer

        # Заполнение filename и slide_number из ответа, если есть ссылки
        if model_response.references:
            filenames = [ref.filename for ref in model_response.references]
            slide_numbers = [ref.page_number for ref in model_response.references]

            # Обновляем filename и slide_number первым значением из списка, убирая расширение у filename
            df.at[index, 'filename'] = os.path.splitext(filenames[0])[0]
            df.at[index, 'slide_number'] = slide_numbers[0]
        else:
            # Если ссылки не найдены, оставляем исходные значения
            filename = os.path.splitext(filename)[0]
            df.at[index, 'filename'] = filename
            top_slide_number = response['document_joiner']['documents'][0].meta.get('page_number', 0)
            if top_slide_number:
                df.at[index, 'slide_number'] = top_slide_number
            else:
                df.at[index, 'slide_number'] = 0

    # Запись обновленного DataFrame в CSV-файл
    try:
        df.to_csv(output_csv, index=False, encoding='utf-8')
        logger.info(f"Файл успешно сохранен как {output_csv}")
    except Exception as e:
        logger.warn(f"Ошибка при записи файла CSV: {e}")

def get_eval_response(
        question: str,
):
    response = rag_pipeline.run({
        "expander": {
            "query": question,
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
                    logger.warn(f"Ошибка при парсинге JSON: {e}")
    except Exception as e:
        logger.warn(f"Общая ошибка при обработке ответа: {e}")

    gc.collect()
    torch.cuda.empty_cache()
    try:
        r_slide_number = references[0].page_number if references else None
        r_filename = references[0].filename if references else None
    except Exception as e:
        r_slide_number = None
        r_filename = None

    contexts = []
    for doc in response['document_joiner']['documents']:
        contexts.append(doc.content)

    res_dict = {
        "response": response_text,
        "r_filename": r_filename,
        "r_slide_number": r_slide_number,
        "contexts": contexts,
    }

    return res_dict

def add_rag_responses_to_csv(
        input_csv="ground_truth_data.csv",
        output_csv="gt_evaluation.csv",
        get_response_function=None,
        save_dir="",
        sub_dir_name=None  # Новый параметр для имени поддиректории
):
    """
    Читает CSV-файл с документами и вопросами, получает ответы с помощью RAG пайплайна,
    добавляет их в DataFrame и сохраняет результат в новый CSV-файл в указанной поддиректории с добавлением datetime.

    Параметры:
    - input_csv (str): Путь к входному файлу ground_truth_data.csv.
    - output_csv (str): Имя выходного файла с добавленными ответами.
    - get_response_function (callable): Функция для получения ответа из RAG пайплайна.
    - save_dir (str): Директория для сохранения выходного файла.
    - sub_dir_name (str, optional): Имя поддиректории внутри save_dir для сохранения файла. По умолчанию None.
    """

    if get_response_function is None:
        raise ValueError("Функция get_response_function должна быть передана и не должна быть None.")

    # Полный путь к входному файлу
    input_path = os.path.join(save_dir, input_csv) if save_dir else input_csv

    # Чтение CSV-файла
    try:
        df = pd.read_csv(input_path)
        logger.info(f"Файл {input_path} успешно прочитан.")
    except FileNotFoundError:
        logger.warn(f"Файл {input_path} не найден.")
        return
    except Exception as e:
        logger.warn(f"Ошибка при чтении файла {input_path}: {e}")
        return

    # Проверка наличия необходимых колонок
    required_columns = ["questions", "ground_truth_documents", "filename", "slide_number", "ground_truth"]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        logger.warn(f"Входной файл не содержит следующие необходимые колонки: {missing_columns}")
        return

    responses = []
    r_filenames = []
    r_slide_numbers = []
    contexts_list = []

    # Итерация по строкам с прогрессбаром
    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Обработка строк"):
        question = row["questions"]

        if pd.isna(question) or question.strip() == "":
            logger.info(f"Строка {index}: Вопрос пуст. Пропуск.")
            responses.append("")
            r_filenames.append("")
            r_slide_numbers.append("")
            contexts_list.append("")
            continue

        # Получение ответа через RAG пайплайн
        try:
            response_dict = get_response_function(question)
            responses.append(response_dict.get("response", ""))
            r_filenames.append(response_dict.get("r_filename", ""))
            r_slide_numbers.append(response_dict.get("r_slide_number", ""))

            contexts = response_dict.get("contexts", [])
            contexts_list.append(contexts)
        except Exception as e:
            logger.warn(f"Строка {index}: Ошибка при получении ответа: {e}")
            responses.append("")
            r_filenames.append("")
            r_slide_numbers.append("")
            contexts_list.append([""])

    # Добавление новых колонок в DataFrame
    df["responses"] = responses
    df["r_filename"] = r_filenames
    df["r_slide_number"] = r_slide_numbers
    df["contexts"] = contexts_list

    # Определение поддиректории и создание её при необходимости
    if sub_dir_name:
        current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
        sub_dir_path = os.path.join(save_dir, f"{sub_dir_name}_{current_datetime}") if save_dir else sub_dir_name
        os.makedirs(sub_dir_path, exist_ok=True)
    else:
        sub_dir_path = save_dir  # Если поддиректория не задана, использовать save_dir


    # Разделение имени файла и расширения
    filename, ext = os.path.splitext(output_csv)

    # Создание нового имени файла с добавлением datetime
    output_filename_with_datetime = output_csv

    # Полный путь к выходному файлу
    output_path = os.path.join(sub_dir_path, output_filename_with_datetime) if sub_dir_path else output_filename_with_datetime

    # Сохранение DataFrame в CSV
    try:
        df.to_csv(output_path, index=False, encoding='utf-8')
        logger.info(f"Результаты успешно сохранены в {output_path}")
    except Exception as e:
        logger.warn(f"Ошибка при записи файла CSV: {e}")

    return df

def get_summary_by_filename(filename):
    # Проверка, что filename не пустой
    if not filename:
        return None

    # Формирование фильтра для поиска документов по filename
    filters = {
        "field": "meta.filename",
        "operator": "==",
        "value": filename
    }

    # Получение списка документов, соответствующих фильтру
    documents = document_store.filter_documents(filters=filters)

    # Проверка, что найден хотя бы один документ
    if not documents:
        return None

    # Извлечение значения 'summary' из метаданных первого документа
    summary = documents[0].meta.get("summary")

    return summary

# Функция для опроса POLL_ENDPOINT
def poll():
    response = requests.get(POLL_ENDPOINT)
    if response.status_code == 404:
        # Если статус 404 (очередь пуста), ждём 1 секунду и возвращаем None
        time.sleep(1)
        return None
    response.raise_for_status()  # Проверка на успешный статус ответа
    return response.json()  # Возвращает JSON-ответ от сервера

# Функция для отправки результата на COMPLETE_ENDPOINT
def complete(request_id, message="", sources=[], suggestions=[], image=None):
    json_data = {
        "requestId": request_id,
        "message": message,
        "sources": sources,
        "suggestions": suggestions,
    }

    # Формируем данные для отправки
    files = {}
    if image is not None:
        # Конвертируем изображение в формат PNG и сохраняем в буфер
        image_buffer = io.BytesIO()
        image.save(image_buffer, format="PNG")
        image_buffer.seek(0)  # Возвращаемся в начало буфера

        # Добавляем картинку в запрос
        files["image"] = ("image.png", image_buffer, "image/png")
    else:
        # Если изображения нет, добавить пустой файл (по необходимости)
        json_data["image"] = None  # Или можно просто не включать поле

    # Отправляем запрос
    complete_response = requests.post(
        COMPLETE_ENDPOINT,
        json=json_data if not files else None,  # Если нет файла, отправляем JSON как есть
        files=files if files else None,        # Если файл есть, отправляем его как часть multipart
        data=json_data if files else None,     # JSON идет как поле `data` при наличии файлов
    )

    # Обработка ошибки
    try:
        complete_response.raise_for_status()
        return complete_response.json()
    except requests.exceptions.RequestException as e:
        return {"error": str(e), "status_code": complete_response.status_code}


def generator_poll():
    # Опрос POLL_ENDPOINT
    response_data = poll()
    if response_data is None:
        return  # Если очередь пуста (404), пропускаем итерацию

    # Добавляем проверку на pipeline_type
    if response_data.get("pipeline_type", None) not in ("haystack", None):
        return

    # Извлечение необходимых данных из ответа
    logger.info(response_data)
    request_id = response_data.get("requestId")
    question = response_data.get("query")

    # Если получены необходимые данные
    if request_id:
        if not question:  # Если question пустой (пустая строка)
            conversational = response_data.get("history", [])
            related_questions = generate_related_questions(conversational)
            complete(
                request_id=request_id,
                suggestions=related_questions,
            )
        else:
            # Проверка на наличие "Сводка:"
            if question.startswith("Сводка:"):
                # Извлекаем имя файла из вопроса, удаляя "Сводка:"
                filename = question[len("Сводка:"):].strip()
                # Получаем базовое имя файла
                file_basename = os.path.basename(filename) if filename else None

                # Ищем summary по базовому имени файла
                summary = get_summary_by_filename(file_basename)

                # Отправляем результат в complete
                complete(
                    request_id=request_id,
                    message=summary if summary else "Сводка по документу не найдена.",
                )

            else:
                # Пытаемся получить значение по ключу "category" как строку
                category = response_data.get("category")

                # Если "category" пустой, проверяем старый ключ "categories"
                if category is None:
                    categories = response_data.get("categories", [])
                    # Если categories является строкой, используем её значение
                    if isinstance(categories, str):
                        category = categories
                    # Если categories является списком, берём первое значение, если список не пустой
                    elif isinstance(categories, list) and categories:
                        category = categories[0]
                    else:
                        category = None

                # Присваиваем None, если категория пуста или пустая строка
                if not category:  # Проверяет как на None, так и на пустую строку
                    category = None


                space = response_data.get("space") or None  # Присваиваем None, если пустая строка
                filename = response_data.get("filename") or None  # Присваиваем None, если пустая строка

                input_data = {
                    "question": question,
                    "category": category,
                    "space": space,
                    "filename": filename,
                }
                model_response = get_chat_response_json(input_data)

                complete(
                    message=model_response.get("message", ""),
                    request_id=request_id,
                    sources=model_response.get("sources", []),
                )
