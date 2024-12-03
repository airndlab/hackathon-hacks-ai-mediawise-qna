import os
from typing import Optional, List, Any

from fastapi import APIRouter
from pydantic import BaseModel

from openai import OpenAI

import yaml

# RAG CONFIG
RAG_CONFIG_PATH = os.getenv("RAG_CONFIG_PATH")
with open(RAG_CONFIG_PATH, "r") as file:
    rag_config = yaml.safe_load(file)

model = rag_config.get("model")

# VLM_RAG CONFIG
VLM_RAG_CONFIG_PATH = os.getenv("VLM_RAG_CONFIG_PATH")
with open(VLM_RAG_CONFIG_PATH, "r") as file:
    vlm_rag_config = yaml.safe_load(file)

vlm_model = vlm_rag_config.get("vlm_model")

PIPELINE_TYPE = os.getenv("PIPELINE_TYPE")

if PIPELINE_TYPE == "LLM":
    api_base_url = os.getenv("VLLM_URL")
elif PIPELINE_TYPE == "VLM":
    api_base_url = os.getenv("VLLM_VLM_URL")

router = APIRouter()

client = OpenAI(
    api_key="VLLM-PLACEHOLDER-API-KEY",
    base_url=api_base_url,
)


class QuestionRequest(BaseModel):
    question: Optional[str] = None
    category: Optional[str] = None
    space: Optional[str] = None
    filename: Optional[str] = None


class AnswerResponse(BaseModel):
    message: Optional[str] = None
    sources: Optional[List[Any]] = None


class RetrieveResponse(BaseModel):
    documents: Optional[List[Any]] = []

if PIPELINE_TYPE == "LLM":
    @router.post("/asking")
    async def ask(request: QuestionRequest):
        from app import pipelines
        input_data = {
            "question": request.question,
            "category": request.category,
            "space": request.space,
            "filename": request.filename,
        }
        model_response = pipelines.get_chat_response_json(input_data)
        return AnswerResponse(
            message=model_response.get("message", ""),
            sources=model_response.get("sources", []),
        )

    @router.post("/simple_asking")
    async def ask(request: QuestionRequest):
        messages = [
            {
                "role": "user",
                "content": request.question
            }
        ]

        chat_response = client.chat.completions.create(
            model=model,
            messages=messages
        )

        output_text = chat_response.choices[0].message.content

        return AnswerResponse(
            message=output_text,
        )


    @router.post("/simple_retrieve")
    async def simple_retrieve(request: QuestionRequest):
        from app import pipelines
        retrieved_documents = pipelines.run_simple_retrieval(
            question =  request.question,
            category =  request.category,
            space = request.space,
            filename = request.filename,
        )

        return RetrieveResponse(documents=retrieved_documents)


    @router.post("/base_retrieve")
    async def base_retrieve(request: QuestionRequest):
        from app import pipelines
        retrieved_documents = pipelines.run_base_retrieval(
            question =  request.question,
            category =  request.category,
            space = request.space,
            filename = request.filename,
        )

        return RetrieveResponse(documents=retrieved_documents)


    @router.post("/full_retrieve")
    async def full_retrieve(request: QuestionRequest):
        from app import pipelines
        retrieved_documents = pipelines.run_full_retrieval(
            question =  request.question,
            category =  request.category,
            space = request.space,
            filename = request.filename,
        )

        return RetrieveResponse(documents=retrieved_documents)

elif PIPELINE_TYPE == "VLM":
    @router.post("/asking")
    async def ask(request: QuestionRequest):
        from app import vlm_pipeline
        input_data = {
            "question": request.question,
            "category": request.category,
            "space": request.space,
            "filename": request.filename,
        }
        model_response = vlm_pipeline.get_chat_response_json(input_data)
        return AnswerResponse(
            message=model_response.get("message", ""),
            sources=model_response.get("sources", []),
        )

