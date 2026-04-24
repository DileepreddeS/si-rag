from __future__ import annotations
from pydantic import BaseModel
from typing import List, Optional


class ChatMessage(BaseModel):
    role: str        # "user" or "assistant"
    content: str


class ChatRequest(BaseModel):
    question: str
    session_id: str = "default"
    stream: bool = False


class ClaimResult(BaseModel):
    claim: str
    label: str       # ENTAILED / BASELESS / CONTRADICTED
    score: float


class PipelineTraceResponse(BaseModel):
    original_query: str
    optimized_query: str
    retrieved_docs: List[dict]
    confidence: float
    retries: int
    strategy_used: Optional[str] = None
    claims: List[ClaimResult] = []


class ChatResponse(BaseModel):
    answer: str
    confidence: float
    citations: List[str]
    retries: int
    session_id: str
    trace: Optional[PipelineTraceResponse] = None


class UploadResponse(BaseModel):
    filename: str
    chunks_added: int
    total_docs: int
    message: str


class SessionResponse(BaseModel):
    session_id: str
    messages: List[ChatMessage]


class ErrorResponse(BaseModel):
    error: str
    detail: str