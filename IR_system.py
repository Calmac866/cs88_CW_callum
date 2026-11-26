import json
import math
import os
import re
from collections import Counter
from html import unescape
from typing import Any
from pathlib import Path
from datetime import datetime, timezone

import httpx
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

UHRI_BASE_URL = "https://dataex.ohchr.org/uhri/api"
UHRI_TOKEN = "0aa417c2-91ed-4786-8ffd-186c952232dd".strip()


# Text Cleaning & Temporary Corpus Builder 
def strip_html(raw_html: str | None) -> str:
    """Removes HTML tags and decodes HTML entities."""
    if not raw_html:
        return ""
    no_tags = re.sub(r"<[^>]+>", " ", raw_html)
    cleaned = unescape(no_tags)
    return re.sub(r"\s+", " ", cleaned).strip()


def build_temp_corpus(
    api_response: dict[str, Any],
) -> tuple[dict[str, str], dict[str, dict[str, Any]]]:
    """
    Extracts a temporary {annotationId: cleaned_text} corpus and companion
    metadata from a UHRI /search/mechanism_result response.
    """
    corpus: dict[str, str] = {}
    metadata: dict[str, dict[str, Any]] = {}

    for rec in api_response.get("listOfRecommendations", []):
        ann_id = rec.get("annotationId")
        if not ann_id:
            continue

        clean_text = strip_html(rec.get("text"))
        if not clean_text:
            continue

        corpus[ann_id] = clean_text
        metadata[ann_id] = {
            "symbol": rec.get("symbol"),
            "documentId": rec.get("documentId"),
            "publicationDate": rec.get("publicationDate"),
            "rawText": rec.get("text"),
        }

    return corpus, metadata


# TF-IDF + Cosine Similarity Search Engine
def tokenize(text: str) -> list[str]:
    return re.findall(r"\b\w+\b", text.lower())


class SimpleSearchEngine:
    def __init__(self, docs: dict[str, str]):
        self.docs = docs
        self.tokenized_docs = {doc_id: tokenize(text) for doc_id, text in docs.items()}
        self.vocab = {w for tokens in self.tokenized_docs.values() for w in tokens}
        self.idf = self._compute_idf()
        self.doc_vectors = {
            doc_id: self._compute_tfidf(tokens)
            for doc_id, tokens in self.tokenized_docs.items()
        }

    def _compute_idf(self) -> dict[str, float]:
        n_docs = len(self.docs)
        idf: dict[str, float] = {}
        for word in self.vocab:
            doc_count = sum(1 for tokens in self.tokenized_docs.values() if word in tokens)
            idf[word] = math.log((n_docs + 1) / (doc_count + 1)) + 1
        return idf

    def _compute_tfidf(self, tokens: list[str]) -> dict[str, float]:
        tf = Counter(tokens)
        total = len(tokens) or 1
        return {w: (c / total) * self.idf.get(w, 0.0) for w, c in tf.items()}

    def _cosine_similarity(self, vec1: dict[str, float], vec2: dict[str, float]) -> float:
        common = set(vec1.keys()) & set(vec2.keys())
        dot = sum(vec1[w] * vec2[w] for w in common)
        norm1 = math.sqrt(sum(v ** 2 for v in vec1.values()))
        norm2 = math.sqrt(sum(v ** 2 for v in vec2.values()))
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot / (norm1 * norm2)

    def search(self, query: str, top_k: int = 10) -> list[dict[str, Any]]:
        query_vec = self._compute_tfidf(tokenize(query))
        scored: list[dict[str, Any]] = []
        for doc_id, doc_vec in self.doc_vectors.items():
            score = self._cosine_similarity(query_vec, doc_vec)
            if score > 0:
                scored.append({
                    "id": doc_id,
                    "score": round(score, 4),
                    "text": self.docs[doc_id],
                })
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]


# Request Schema & UHRI Fetcher 
class UhriSearchRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Query string used for local TF-IDF ranking")
    mechanismId: str
    culture: str = "en"
    currentRecoDisplayed: int = 0
    top_k: int = Field(10, ge=1, le=40)

    # UHRI filter arrays
    countries: list[str] = Field(default_factory=list)
    mechanisms: list[str] = Field(default_factory=list)
    affectedPersons: list[str] = Field(default_factory=list)
    sdgs: list[str] = Field(default_factory=list)
    themes: list[str] = Field(default_factory=list)
    documentCodes: list[str] = Field(default_factory=list)
    documentTypes: list[str] = Field(default_factory=list)
    uprCycles: list[str] = Field(default_factory=list)
    uprPositions: list[str] = Field(default_factory=list)
    uprStates: list[str] = Field(default_factory=list)
    regions: list[str] = Field(default_factory=list)
    uprSessions: list[str] = Field(default_factory=list)
    uprRegions: list[str] = Field(default_factory=list)
    searchText: str = ""
    fromDate: str | None = None
    toDate: str | None = None
    fromDatePublishedFrontEnd: str | None = None
    toDatePublishedFrontEnd: str | None = None


async def fetch_uhri_mechanism_results(payload: UhriSearchRequest) -> dict[str, Any]:
    if not UHRI_TOKEN:
        raise HTTPException(
            status_code=500,
            detail="UHRI_API_TOKEN is not set on the server.",
        )

    uhri_body = payload.model_dump(
        exclude={"query", "culture", "top_k"}
    )

    headers = {
        "x-uhri-api-public-token": UHRI_TOKEN,
        "Content-Type": "application/json",
    }

    url = f"{UHRI_BASE_URL}/search/mechanism_result"
    params = {"culture": payload.culture}

    async with httpx.AsyncClient(timeout=20.0) as client:
        response = await client.post(url, params=params, json=uhri_body, headers=headers)

    if response.status_code != 200:
        raise HTTPException(
            status_code=response.status_code,
            detail=f"UHRI API error: {response.text}",
        )

    return response.json()


# FastAPI App & Routes
app = FastAPI(title="UHRI Temporary-Corpus IR API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/")
def serve_webpage():
    return FileResponse("index.html")


@app.post("/api/search")
async def search_uhri_temp_corpus(req: UhriSearchRequest):
    raw_data = await fetch_uhri_mechanism_results(req)
    corpus, metadata = build_temp_corpus(raw_data)

    if not corpus:
        return {
            "query": req.query,
            "corpusSize": 0,
            "count": 0,
            "results": [],
        }
    

    # Build temporary engine on this fetched batch and rank against req.query
    temp_engine = SimpleSearchEngine(corpus)
    ranked = temp_engine.search(req.query, top_k=req.top_k)

    enriched_results = [
        {
            **item,
            **metadata.get(item["id"], {}),
        }
        for item in ranked
    ]

    return {
        "query": req.query,
        "corpusSize": len(corpus),
        "count": len(enriched_results),
        "results": enriched_results,
    }
@app.get("/api/filters")
async def get_uhri_filters(culture: str = "en"):
    if not UHRI_TOKEN:
        raise HTTPException(
            status_code=500,
            detail="UHRI_API_TOKEN is not set on the server.",
        )

    headers = {
        "x-uhri-api-public-token": UHRI_TOKEN,
        "Accept": "application/json",
    }

    async with httpx.AsyncClient(timeout=20.0) as client:
        response = await client.get(
            f"{UHRI_BASE_URL}/FilterList",
            params={"culture": culture},
            headers=headers,
        )

    if response.status_code != 200:
        raise HTTPException(
            status_code=response.status_code,
            detail=f"UHRI FilterList error: {response.text}",
        )

    return response.json()

@app.get("/api/document/{doc_id}")
async def get_uhri_document_info(doc_id: str, culture: str = "en"):
    clean_token = "".join(ch for ch in UHRI_TOKEN.strip() if 32 <= ord(ch) < 127)
    if not clean_token:
        raise HTTPException(
            status_code=500,
            detail="UHRI_API_TOKEN is not set on the server.",
        )

    headers = {
        "x-uhri-api-public-token": clean_token,
        "Accept": "application/json",
    }

    async with httpx.AsyncClient(timeout=20.0) as client:
        response = await client.get(
            f"{UHRI_BASE_URL}/Document/document_info/{doc_id}",
            params={"culture": culture},
            headers=headers,
        )

    if response.status_code != 200:
        raise HTTPException(
            status_code=response.status_code,
            detail=f"UHRI Document info error: {response.text}",
        )

    return response.json()

# User Testing Telemetry Logging
TELEMETRY_LOG_PATH = Path("user_testing_logs.jsonl")


class TelemetryEvent(BaseModel):
    sessionId: str = Field(..., min_length=1, max_length=100)
    eventType: str = Field(..., min_length=1, max_length=60)  # e.g. "click", "linger", "search"
    target: str = Field(..., min_length=1, max_length=200)
    clickCount: int = 0
    lingerMs: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


@app.post("/api/telemetry")
async def record_telemetry(event: TelemetryEvent):
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **event.model_dump(),
    }
    with TELEMETRY_LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return {"status": "ok"}


@app.get("/api/telemetry/summary")
async def get_telemetry_summary():
    if not TELEMETRY_LOG_PATH.exists():
        return {
            "totalEvents": 0,
            "sessions": 0,
            "totalClicks": 0,
            "avgLingerSeconds": 0.0,
            "topClickedTargets": [],
        }

    sessions: set[str] = set()
    max_clicks_by_session: dict[str, int] = {}
    linger_durations_ms: list[int] = []
    target_counts: Counter[str] = Counter()
    total_events = 0

    with TELEMETRY_LOG_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total_events += 1
            entry = json.loads(line)
            sid = entry.get("sessionId", "unknown")
            sessions.add(sid)

            clicks = int(entry.get("clickCount", 0) or 0)
            if clicks > max_clicks_by_session.get(sid, 0):
                max_clicks_by_session[sid] = clicks

            if entry.get("eventType") == "click":
                target_counts[entry.get("target", "unknown")] += 1

            linger = entry.get("lingerMs")
            if isinstance(linger, (int, float)) and linger > 0:
                linger_durations_ms.append(int(linger))

    avg_linger_sec = (
        round((sum(linger_durations_ms) / len(linger_durations_ms)) / 1000, 2)
        if linger_durations_ms
        else 0.0
    )

    return {
        "totalEvents": total_events,
        "sessions": len(sessions),
        "totalClicks": sum(max_clicks_by_session.values()),
        "avgLingerSeconds": avg_linger_sec,
        "topClickedTargets": [
            {"target": name, "clicks": count}
            for name, count in target_counts.most_common(15)
        ],
    }