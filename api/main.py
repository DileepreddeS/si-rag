from __future__ import annotations
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routes.chat import router as chat_router
from api.routes.upload import router as upload_router

app = FastAPI(
    title="SI-RAG API",
    description="Self-Verifying Adaptive RAG with Super-Intelligence Layer",
    version="1.0.0",
)

# CORS — allows React frontend to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routes
app.include_router(chat_router,   prefix="/api", tags=["chat"])
app.include_router(upload_router, prefix="/api", tags=["upload"])


@app.get("/")
async def root():
    return {
        "name": "SI-RAG API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
    }


@app.get("/health")
async def health():
    return {"status": "healthy"}