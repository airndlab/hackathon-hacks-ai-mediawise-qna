from typing import Optional, List, Any

from fastapi import APIRouter
from pydantic import BaseModel

from haystack import Document

from app import pipelines

router = APIRouter()


class QuestionRequest(BaseModel):
    question: Optional[str] = None
    category: Optional[str] = None
    space: Optional[str] = None
    filename: Optional[str] = None


class AnswerResponse(BaseModel):
    message: Optional[str] = None
    sources: Optional[List[Any]] = None


class RetrieveResponse(BaseModel):
    documents: Optional[List[Document]] = []


@router.post("/asking")
async def ask(request: QuestionRequest):
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


@router.post("/simple_retrieve")
async def simple_retrieve(request: QuestionRequest):
    retrieved_documents = pipelines.run_simple_retrieval(
        question =  request.question,
        category =  request.category,
        space = request.space,
        filename = request.filename,
    )

    return RetrieveResponse(retrieved_documents)


@router.post("/base_retrieve")
async def base_retrieve(request: QuestionRequest):
    retrieved_documents = pipelines.run_base_retrieval(
        question =  request.question,
        category =  request.category,
        space = request.space,
        filename = request.filename,
    )

    return RetrieveResponse(retrieved_documents)


@router.post("/full_retrieve")
async def full_retrieve(request: QuestionRequest):
    retrieved_documents = pipelines.run_full_retrieval(
        question =  request.question,
        category =  request.category,
        space = request.space,
        filename = request.filename,
    )

    return RetrieveResponse(retrieved_documents)

