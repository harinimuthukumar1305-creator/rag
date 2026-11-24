import json
import os
import re
import threading
from typing import Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import AliasChoices, BaseModel, ConfigDict, Field
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
KB_PATH = os.getenv("KB_PATH", os.path.join(BASE_DIR, "knowledge_base.json"))
MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)
MIN_SCORE = float(os.getenv("MIN_SCORE", "0.25"))
MAX_QUESTION_LENGTH = int(os.getenv("MAX_QUESTION_LENGTH", "2000"))

# ---------------------------------------------------------------------------
# Load the knowledge base and embedding model once at process startup.
# ---------------------------------------------------------------------------
try:
    with open(KB_PATH, "r", encoding="utf-8") as file:
        PASSAGES = json.load(file)
except FileNotFoundError as exc:
    raise RuntimeError(
        f"Knowledge base not found at {KB_PATH}. Set KB_PATH or place "
        "knowledge_base.json beside app.py."
    ) from exc
except json.JSONDecodeError as exc:
    raise RuntimeError(f"Invalid JSON in knowledge base: {KB_PATH}") from exc

if not isinstance(PASSAGES, list) or not PASSAGES:
    raise RuntimeError("The knowledge base must be a non-empty JSON array.")
if not all(isinstance(passage, str) and passage.strip() for passage in PASSAGES):
    raise RuntimeError("Every knowledge-base item must be a non-empty string.")

model = SentenceTransformer(MODEL_NAME)
model_lock = threading.Lock()

embeddings = model.encode(
    PASSAGES,
    convert_to_numpy=True,
    normalize_embeddings=True,
    show_progress_bar=False,
).astype("float32")

# ---------------------------------------------------------------------------
# API setup
# ---------------------------------------------------------------------------
app = FastAPI(
    title="RAG Colosseum API",
    description="Extractive retrieval-augmented question answering over 50 passages.",
    version="1.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    # Accept both field names because evaluators/clients may send either one.
    model_config = ConfigDict(populate_by_name=True)

    question: str = Field(
        ...,
        min_length=1,
        max_length=MAX_QUESTION_LENGTH,
        validation_alias=AliasChoices("question", "query"),
    )
    top_k: int = Field(default=3, ge=1, le=10)


class QueryResponse(BaseModel):
    answer: Optional[str]
    sources: list[int]
    scores: list[float]
    confidence: float


# ---------------------------------------------------------------------------
# Text processing and retrieval
# ---------------------------------------------------------------------------
WORD_RE = re.compile(r"[a-zA-Z0-9]+")
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
STOP_WORDS = {
    "what", "which", "who", "whom", "whose", "when", "where", "why", "how",
    "is", "are", "was", "were", "be", "been", "being", "the", "a", "an",
    "of", "to", "for", "in", "on", "at", "by", "with", "and", "or", "does",
    "do", "did", "can", "could", "would", "should", "please", "tell", "me",
    "about", "give", "show", "explain", "according", "passage", "describe",
    "discuss", "main", "topic", "information", "provide", "list", "name",
}


def normalize_token(word: str) -> str:
    """Apply a small, conservative normalization for common plural forms."""
    word = word.lower()
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 4 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def tokens(text: str) -> set[str]:
    return {
        normalize_token(word)
        for word in WORD_RE.findall(text)
        if normalize_token(word) not in STOP_WORDS
    }


def sentence_split(text: str) -> list[str]:
    parts = [part.strip() for part in SENTENCE_RE.split(text) if part.strip()]
    return parts or [text.strip()]


def lexical_score(question_tokens: set[str], passage: str) -> float:
    """A bounded overlap score used only as a small semantic-rank tie-breaker."""
    if not question_tokens:
        return 0.0
    passage_tokens = tokens(passage)
    if not passage_tokens:
        return 0.0
    overlap = len(question_tokens & passage_tokens)
    # Recall against the query is useful for short technical questions.
    return overlap / len(question_tokens)


def rank_passages(question: str, top_k: int) -> list[tuple[int, float]]:
    with model_lock:
        query_embedding = model.encode(
            [question],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0].astype("float32")

    # Embeddings are normalized, so dot product is cosine similarity.
    similarities = embeddings @ query_embedding
    q_tokens = tokens(question)

    # Use semantic similarity as the primary signal. A small lexical bonus helps
    # exact technical terms without allowing keyword overlap to dominate retrieval.
    ranking_scores = np.empty(len(PASSAGES), dtype="float32")
    for index, passage in enumerate(PASSAGES):
        overlap = lexical_score(q_tokens, passage)
        ranking_scores[index] = similarities[index] + 0.08 * overlap

    order = np.argsort(-ranking_scores)[:top_k]
    # Return the original cosine similarities, not the blended ranking scores.
    return [(int(index), float(similarities[index])) for index in order]


def select_evidence(
    question: str, ranked: list[tuple[int, float]]
) -> list[tuple[int, float, str]]:
    q_tokens = tokens(question)
    evidence: list[tuple[int, float, str]] = []

    for index, similarity in ranked:
        passage = PASSAGES[index]
        sentences = sentence_split(passage)

        if not q_tokens:
            selected_sentence = passage
        else:
            scored_sentences = []
            for sentence in sentences:
                sentence_tokens = tokens(sentence)
                overlap = len(q_tokens & sentence_tokens)
                # Prefer a sentence containing the largest number of query terms.
                scored_sentences.append((overlap, sentence))

            best_overlap, best_sentence = max(
                scored_sentences, key=lambda item: item[0], default=(0, passage)
            )
            # If no individual sentence contains a query term, retain the complete
            # retrieved passage rather than returning a misleading fragment.
            selected_sentence = best_sentence if best_overlap > 0 else passage

        evidence.append((index, similarity, selected_sentence))

    return evidence


def looks_off_topic(best_score: float) -> bool:
    return best_score < MIN_SCORE


def answer_from_evidence(
    evidence: list[tuple[int, float, str]], max_sentences: int = 4
) -> Optional[str]:
    if not evidence:
        return None

    answer_parts: list[str] = []
    for _, _, text in evidence:
        # Evidence may be a complete passage when sentence-level matching is weak.
        # Keep the answer concise while preserving source wording verbatim.
        parts = sentence_split(text)
        for part in parts:
            if part and part not in answer_parts:
                answer_parts.append(part)
            if len(answer_parts) >= max_sentences:
                break
        if len(answer_parts) >= max_sentences:
            break

    return " ".join(answer_parts) if answer_parts else None


def calculate_confidence(ranked: list[tuple[int, float]]) -> float:
    if not ranked:
        return 0.0

    best = ranked[0][1]
    second = ranked[1][1] if len(ranked) > 1 else 0.0
    margin = max(0.0, best - second)

    # Confidence is a heuristic, not a calibrated probability.
    confidence = 0.55 + 0.55 * (best - MIN_SCORE) + 0.20 * margin
    return float(np.clip(confidence, 0.10, 0.95))


def run_query(question: str, top_k: int) -> QueryResponse:
    cleaned_question = question.strip()
    if not cleaned_question:
        raise HTTPException(status_code=422, detail="Question must not be empty.")

    ranked = rank_passages(cleaned_question, top_k)
    if not ranked or looks_off_topic(ranked[0][1]):
        return QueryResponse(
            answer=None,
            sources=[],
            scores=[],
            confidence=0.15,
        )

    evidence = select_evidence(cleaned_question, ranked)
    answer = answer_from_evidence(evidence)
    if answer is None:
        return QueryResponse(
            answer=None,
            sources=[],
            scores=[],
            confidence=0.20,
        )

    return QueryResponse(
        answer=answer,
        sources=[index for index, _, _ in evidence],
        scores=[round(float(score), 4) for _, score, _ in evidence],
        confidence=round(calculate_confidence(ranked), 4),
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/")
def root():
    return {
        "status": "ok",
        "service": "RAG Colosseum API",
        "passages": len(PASSAGES),
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "passages": len(PASSAGES),
        "embedding_model": MODEL_NAME,
    }


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    return run_query(req.question, req.top_k)


@app.post("/debug/query")
def debug_query(req: QueryRequest):
    """Temporary diagnostic route; remove before final evaluation if not needed."""
    ranked = rank_passages(req.question.strip(), 10)
    return {
        "question": req.question,
        "minimum_score": MIN_SCORE,
        "results": [
            {
                "index": index,
                "score": round(score, 4),
                "passage": PASSAGES[index],
            }
            for index, score in ranked
        ],
    }
