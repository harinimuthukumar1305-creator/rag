---
title: RAG Colosseum
emoji: 🏛️
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
---

# RAG Colosseum – 20-Probe Deployment Gauntlet

FastAPI retrieval service using `sentence-transformers/all-MiniLM-L6-v2` and cosine similarity over the supplied 50-passage knowledge base.

## Local run

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 7860
```

Open `http://localhost:7860/docs`.

## Test

```bash
curl -X POST http://localhost:7860/query \
  -H "Content-Type: application/json" \
  -d '{"question":"What heartbeat interval does Raft use?","top_k":3}'
```

## Deploy to Hugging Face Spaces

1. Create a new Space at https://huggingface.co/new-space.
2. Choose **Docker** as the Space SDK.
3. Upload `app.py`, `knowledge_base.json`, `requirements.txt`, and `Dockerfile` (or push this folder using Git).
4. Wait for the Space to build and start.
5. Test `https://YOUR-SPACE.hf.space/health`.
6. Submit `https://YOUR-SPACE.hf.space` as the public base URL (do not append `/query`).

## API

`POST /query`

Request:
```json
{"question":"What heartbeat interval does Raft use?","top_k":3}
```

Response:
```json
{
  "answer":"...",
  "sources":[0],
  "scores":[0.88],
  "confidence":0.88
}
```

## Notes

- The service loads the embedding model and builds the in-memory vector index once at startup.
- It returns extractive evidence from the supplied passages; it does not call an external LLM.
- The score and abstention thresholds are configurable with `MIN_SCORE` and `MIN_MARGIN`.
- Benchmark all 20 probes before submission. No implementation can guarantee a score without seeing the hidden probe questions.
- The supplied passage text is preserved as-is, including its stated numerical values. The system does not independently validate the technical correctness of those passages.
