from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["passages"] == 50

def test_query_contract():
    r = client.post("/query", json={"question": "What heartbeat interval does Raft use?", "top_k": 3})
    assert r.status_code == 200
    data = r.json()
    assert set(data) == {"answer", "sources", "scores", "confidence"}
    assert isinstance(data["sources"], list)
    assert len(data["sources"]) == len(data["scores"])
    assert 0.0 <= data["confidence"] <= 1.0
