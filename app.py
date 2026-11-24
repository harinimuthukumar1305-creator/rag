import os
import re
import threading
from typing import Optional

import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

KB_PATH = os.getenv("KB_PATH", "knowledge_base.json")
MODEL_NAME = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
MIN_SCORE = float(os.getenv("MIN_SCORE", "0.30"))
MIN_MARGIN = float(os.getenv("MIN_MARGIN", "0.025"))

with open(KB_PATH, "r", encoding="utf-8") as f:
    PASSAGES = __import__("json").load(f)

# Load once at process startup; protect model inference for concurrent requests.
model = SentenceTransformer(MODEL_NAME)
model_lock = threading.Lock()
embeddings = model.encode(
    PASSAGES,
    convert_to_numpy=True,
    normalize_embeddings=True,
    show_progress_bar=False
).astype("float32")

app = FastAPI(title="RAG Colosseum API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1)
    top_k: int = Field(default=3, ge=1, le=10)

class QueryResponse(BaseModel):
    answer: Optional[str]
    sources: list[int]
    scores: list[float]
    confidence: float

WORD_RE = re.compile(r"[a-zA-Z0-9]+")
STOP = {
    "what","which","who","whom","whose","when","where","why","how","is","are",
    "was","were","the","a","an","of","to","for","in","on","at","by","with",
    "and","or","does","do","did","can","could","would","should","please",
    "tell","me","about","give","show","explain","according","passage"
}

def tokens(text: str) -> set[str]:
    return {w.lower() for w in WORD_RE.findall(text)} - STOP

def sentence_split(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]

def rank_passages(question: str, top_k: int):
    with model_lock:
        q = model.encode([question], convert_to_numpy=True,
                         normalize_embeddings=True, show_progress_bar=False)[0]
    sims = embeddings @ q
    order = np.argsort(-sims)[:top_k]
    return [(int(i), float(np.clip(sims[i], 0.0, 1.0))) for i in order]

def select_evidence(question: str, ranked: list[tuple[int, float]]) -> list[tuple[int, float, str]]:
    q_tokens = tokens(question)
    evidence = []
    for idx, score in ranked:
        sentences = sentence_split(PASSAGES[idx])
        scored = []
        for sent in sentences:
            st = tokens(sent)
            overlap = len(q_tokens & st) / max(1, len(q_tokens))
            # Prefer sentences with query-term overlap while retaining semantic ranking.
            scored.append((overlap, sent))
        overlap, best = max(scored, default=(0.0, PASSAGES[idx]), key=lambda x: x[0])
        evidence.append((idx, score, best if overlap > 0 else PASSAGES[idx]))
    return evidence

def looks_off_topic(question: str, best_score: float) -> bool:
    # The supplied KB covers five technical domains only. A conservative score gate
    # avoids answering unrelated questions from accidental nearest-neighbour matches.
    return best_score < MIN_SCORE

def answer_from_evidence(question: str, evidence: list[tuple[int, float, str]]) -> Optional[str]:
    if not evidence:
        return None
    q = tokens(question)
    # Keep only evidence sentences with meaningful lexical support, but allow semantic
    # questions whose wording differs from the passage.
    chosen = []
    for idx, score, sentence in evidence:
        st = tokens(sentence)
        overlap = len(q & st)
        if overlap >= 1 or score >= 0.48:
            if sentence not in chosen:
                chosen.append(sentence)
    if not chosen:
        return None
    # Avoid returning a long unrelated passage. Multiple sentences are useful for
    # cross-passage questions; the endpoint remains strictly extractive and grounded.
    return " ".join(chosen[:3])

@app.get("/")
def root():
    return {"status": "ok", "service": "RAG Colosseum API", "passages": len(PASSAGES)}

@app.get("/health")
def health():
    return {"status": "healthy", "passages": len(PASSAGES), "embedding_model": MODEL_NAME}

@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    ranked = rank_passages(req.question.strip(), req.top_k)
    if not ranked or looks_off_topic(req.question, ranked[0][1]):
        return QueryResponse(answer=None, sources=[], scores=[], confidence=0.15)

    evidence = select_evidence(req.question, ranked)
    answer = answer_from_evidence(req.question, evidence)
    if answer is None:
        return QueryResponse(answer=None, sources=[], scores=[], confidence=0.20)

    # Confidence combines best semantic match and separation from the next result.
    best = ranked[0][1]
    second = ranked[1][1] if len(ranked) > 1 else 0.0
    margin = max(0.0, best - second)
    confidence = float(np.clip(0.62 + 0.55 * (best - 0.35) + 0.25 * margin, 0.61, 0.97))
    return QueryResponse(
        answer=answer,
        sources=[x[0] for x in evidence],
        scores=[round(x[1], 4) for x in evidence],
        confidence=round(confidence, 4)
    )
