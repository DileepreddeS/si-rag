from __future__ import annotations
from fastapi import APIRouter, HTTPException
from api.schemas import ChatRequest, ChatResponse, PipelineTraceResponse, ClaimResult
from api.conversation import ConversationMemory
from core.orchestrator import SIOrchestrator
from utils.logger import log_step

router  = APIRouter()
memory  = ConversationMemory()

# Single orchestrator instance shared across requests
_orchestrator: SIOrchestrator | None = None


def get_orchestrator() -> SIOrchestrator:
    global _orchestrator
    if _orchestrator is None:
        log_step("API", "Initializing SIOrchestrator...")
        _orchestrator = SIOrchestrator()
    return _orchestrator


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        orchestrator = get_orchestrator()

        # Inject conversation history into query
        history_text = memory.format_history_for_prompt(request.session_id)
        full_query   = f"{history_text}Current question: {request.question}" \
                       if history_text else request.question

        log_step("API", f"Session={request.session_id} Query={request.question}")

        # Run full SI-RAG pipeline
        response = orchestrator.run(full_query)

        # Save turn to memory
        memory.add_turn(
            session_id=request.session_id,
            question=request.question,
            answer=response.answer,
        )

        # Build trace response
        trace = None
        if response.trace:
            t = response.trace
            trace = PipelineTraceResponse(
                original_query=t.query_state.original_query,
                optimized_query=t.query_state.optimized_query,
                retrieved_docs=[
                    {
                        "id": d.id,
                        "text": d.text[:200],
                        "source": d.source,
                        "score": round(d.score, 3),
                    }
                    for d in t.reranked_docs
                ],
                confidence=t.final_confidence,
                retries=t.total_retries,
                strategy_used=(
                    t.retry_decisions[-1].strategy.value
                    if t.retry_decisions else None
                ),
                claims=[
                    ClaimResult(
                        claim=v.claim[:100],
                        label=v.label.value,
                        score=round(v.score, 3),
                    )
                    for v in (t.verification.verdicts if t.verification else [])
                ],
            )

        return ChatResponse(
            answer=response.answer,
            confidence=response.confidence,
            citations=response.citations,
            retries=response.total_retries,
            session_id=request.session_id,
            trace=trace,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/session/{session_id}")
async def clear_session(session_id: str):
    memory.clear_session(session_id)
    return {"message": f"Session {session_id} cleared"}


@router.get("/session/{session_id}")
async def get_session(session_id: str):
    history = memory.get_history(session_id)
    return {
        "session_id": session_id,
        "messages": [m.dict() for m in history],
    }