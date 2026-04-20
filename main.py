"""MySELAgent local implementation using Ollama (Gemma 3).

This is a compact, local-first replica of the paper's architecture:
- Hypervisor router
- Researcher agent (theory RAG)
- Profiler agent (classroom assessment explanation)
- SEL activities recommender (activity search + learning path)

It uses:
- Ollama for the LLM (e.g. gemma3)
- Ollama embeddings if available (recommended model: nomic-embed-text or bge-m3)
- A lightweight JSON/SQLite-backed knowledge store
- NetworkX for graph traversal

Run:
    python myselagent_local.py init --data ./data
    python myselagent_local.py chat --data ./data

Expected data layout:
    data/
      theory_docs/               # .txt/.md/.pdf text-extracted docs
      activities.json            # list of SEL activities
      classroom_assessments.json # classroom profiles/history

Optional:
    data/seed_theory/            # starter documents to index

This code is intentionally self-contained so you can extend it toward the paper's full setup.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import re
import sqlite3
import sys
import textwrap
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import networkx as nx
import numpy as np

try:
    import requests
except Exception as e:  # pragma: no cover
    raise RuntimeError("This script requires the requests package.") from e


# ----------------------------
# Config
# ----------------------------

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
DEFAULT_LLM = os.environ.get("OLLAMA_MODEL", "gemma3:12b")
DEFAULT_EMBED_MODEL = os.environ.get("OLLAMA_EMBED_MODEL", "nomic-embed-text")
TOP_K = int(os.environ.get("MYSEL_TOP_K", "5"))
DB_PATH = "myselagent.db"
GRAPH_PATH = "myselagent_graph.graphml"
ACTIVITIES_JSON = "activities.json"
ASSESSMENTS_JSON = "classroom_assessments.json"
THEORY_DIR = "theory_docs"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 120


# ----------------------------
# Utility functions
# ----------------------------

def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    va = np.array(a, dtype=np.float32)
    vb = np.array(b, dtype=np.float32)
    denom = (np.linalg.norm(va) * np.linalg.norm(vb)) + 1e-8
    return float(np.dot(va, vb) / denom)


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    text = normalize_text(text)
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunk = text[start:end]
        chunks.append(chunk)
        if end == len(text):
            break
        start = max(0, end - overlap)
    return chunks


def safe_json_loads(text: str) -> Any:
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        # Try to recover a JSON block.
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            return json.loads(m.group(0))
        m = re.search(r"\[.*\]", text, re.DOTALL)
        if m:
            return json.loads(m.group(0))
        raise


# ----------------------------
# Ollama client
# ----------------------------

class OllamaClient:
    def __init__(self, host: str = OLLAMA_HOST, model: str = DEFAULT_LLM, embed_model: str = DEFAULT_EMBED_MODEL):
        self.host = host.rstrip("/")
        self.model = model
        self.embed_model = embed_model

    def chat(self, messages: List[Dict[str, str]], temperature: float = 0.0, max_tokens: int = 1024) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
        r = requests.post(f"{self.host}/api/chat", json=payload, timeout=300)
        r.raise_for_status()
        data = r.json()
        return data["message"]["content"]

    def embed(self, text: str) -> List[float]:
        payload = {"model": self.embed_model, "prompt": text}
        r = requests.post(f"{self.host}/api/embeddings", json=payload, timeout=300)
        r.raise_for_status()
        data = r.json()
        return data["embedding"]


# ----------------------------
# Data structures
# ----------------------------

@dataclass
class ChunkRecord:
    chunk_id: str
    doc_id: str
    title: str
    text: str
    embedding: List[float]
    metadata: Dict[str, Any]


@dataclass
class Activity:
    activity_id: str
    title: str
    summary: str
    age_group: str
    competencies: List[str]
    didactic_strategies: List[str]
    topics: List[str]
    popularity: float = 0.0
    impact: float = 0.0
    comments: List[str] = dataclasses.field(default_factory=list)
    next_activities: List[str] = dataclasses.field(default_factory=list)


@dataclass
class ClassroomAssessment:
    classroom_id: str
    grade: str
    school_level: str
    timestamp: str
    metrics: Dict[str, float]
    notes: str = ""


# ----------------------------
# Store
# ----------------------------

class MySELStore:
    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.db_path = self.root / DB_PATH
        self.graph_path = self.root / GRAPH_PATH
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        self.graph = nx.DiGraph()
        self._init_db()
        self._load_graph()

    def _init_db(self) -> None:
        cur = self.conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS chunks (
                chunk_id TEXT PRIMARY KEY,
                doc_id TEXT NOT NULL,
                title TEXT NOT NULL,
                text TEXT NOT NULL,
                embedding TEXT NOT NULL,
                metadata TEXT NOT NULL
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS activities (
                activity_id TEXT PRIMARY KEY,
                payload TEXT NOT NULL
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS assessments (
                classroom_id TEXT PRIMARY KEY,
                payload TEXT NOT NULL
            )
            """
        )
        self.conn.commit()

    def _load_graph(self) -> None:
        if self.graph_path.exists():
            try:
                self.graph = nx.read_graphml(self.graph_path)
            except Exception:
                self.graph = nx.DiGraph()

    def save_graph(self) -> None:
        nx.write_graphml(self.graph, self.graph_path)

    def upsert_chunk(self, record: ChunkRecord) -> None:
        cur = self.conn.cursor()
        cur.execute(
            """
            INSERT INTO chunks (chunk_id, doc_id, title, text, embedding, metadata)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(chunk_id) DO UPDATE SET
                doc_id=excluded.doc_id,
                title=excluded.title,
                text=excluded.text,
                embedding=excluded.embedding,
                metadata=excluded.metadata
            """,
            (
                record.chunk_id,
                record.doc_id,
                record.title,
                record.text,
                json.dumps(record.embedding),
                json.dumps(record.metadata),
            ),
        )
        self.conn.commit()

    def load_chunks(self) -> List[ChunkRecord]:
        rows = self.conn.execute("SELECT * FROM chunks").fetchall()
        out = []
        for row in rows:
            out.append(
                ChunkRecord(
                    chunk_id=row["chunk_id"],
                    doc_id=row["doc_id"],
                    title=row["title"],
                    text=row["text"],
                    embedding=json.loads(row["embedding"]),
                    metadata=json.loads(row["metadata"]),
                )
            )
        return out

    def upsert_activity(self, activity: Activity) -> None:
        self.conn.execute(
            """
            INSERT INTO activities (activity_id, payload)
            VALUES (?, ?)
            ON CONFLICT(activity_id) DO UPDATE SET payload=excluded.payload
            """,
            (activity.activity_id, json.dumps(asdict(activity))),
        )
        self.conn.commit()

    def load_activities(self) -> List[Activity]:
        rows = self.conn.execute("SELECT payload FROM activities").fetchall()
        return [Activity(**json.loads(r[0])) for r in rows]

    def upsert_assessment(self, assessment: ClassroomAssessment) -> None:
        self.conn.execute(
            """
            INSERT INTO assessments (classroom_id, payload)
            VALUES (?, ?)
            ON CONFLICT(classroom_id) DO UPDATE SET payload=excluded.payload
            """,
            (assessment.classroom_id, json.dumps(asdict(assessment))),
        )
        self.conn.commit()

    def load_assessments(self) -> List[ClassroomAssessment]:
        rows = self.conn.execute("SELECT payload FROM assessments").fetchall()
        return [ClassroomAssessment(**json.loads(r[0])) for r in rows]


# ----------------------------
# Ingestion
# ----------------------------

class Ingestor:
    def __init__(self, client: OllamaClient, store: MySELStore):
        self.client = client
        self.store = store

    def ingest_theory_docs(self, theory_dir: Path) -> None:
        if not theory_dir.exists():
            return
        for path in theory_dir.iterdir():
            if not path.is_file():
                continue
            if path.suffix.lower() not in {".txt", ".md"}:
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            self._ingest_document(path.stem, path.name, text)

    def _ingest_document(self, doc_id: str, title: str, text: str) -> None:
        for idx, chunk in enumerate(chunk_text(text)):
            emb = self.client.embed(chunk)
            rec = ChunkRecord(
                chunk_id=f"{doc_id}::chunk::{idx}",
                doc_id=doc_id,
                title=title,
                text=chunk,
                embedding=emb,
                metadata={"chunk_index": idx, "source": title},
            )
            self.store.upsert_chunk(rec)
            self.store.graph.add_node(rec.chunk_id, kind="chunk", doc_id=doc_id, title=title)
            if idx > 0:
                prev = f"{doc_id}::chunk::{idx-1}"
                self.store.graph.add_edge(prev, rec.chunk_id, relation="next")

        # lightweight LLM entity extraction to build graph connections
        try:
            snippet = text[:6000]
            prompt = (
                "Extract 8-15 SEL-related entities and relations from the text as JSON with keys: "
                "entities=[{name,type}], relations=[{source,relation,target}]. "
                "Keep it compact and only include useful concepts.\n\nTEXT:\n" + snippet
            )
            raw = self.client.chat([
                {"role": "system", "content": "Return only valid JSON."},
                {"role": "user", "content": prompt},
            ], temperature=0.0, max_tokens=800)
            data = safe_json_loads(raw)
            for ent in data.get("entities", []):
                self.store.graph.add_node(ent["name"], kind="entity", entity_type=ent.get("type", "concept"))
            for rel in data.get("relations", []):
                self.store.graph.add_edge(rel["source"], rel["target"], relation=rel.get("relation", "related_to"))
        except Exception:
            pass

    def ingest_activities(self, activities_path: Path) -> None:
        if not activities_path.exists():
            return
        raw = json.loads(activities_path.read_text(encoding="utf-8"))
        for item in raw:
            activity = Activity(
                activity_id=str(item.get("activity_id") or item.get("id") or item["title"]),
                title=item["title"],
                summary=item.get("summary", item.get("description", "")),
                age_group=item.get("age_group", item.get("target_age", "unknown")),
                competencies=item.get("competencies", []),
                didactic_strategies=item.get("didactic_strategies", []),
                topics=item.get("topics", []),
                popularity=float(item.get("popularity", 0.0)),
                impact=float(item.get("impact", 0.0)),
                comments=item.get("comments", []),
                next_activities=item.get("next_activities", []),
            )
            self.store.upsert_activity(activity)
            self.store.graph.add_node(activity.activity_id, kind="activity", title=activity.title)
            for nxt in activity.next_activities:
                self.store.graph.add_node(nxt, kind="activity")
                self.store.graph.add_edge(activity.activity_id, nxt, relation="sequence")
        self.store.save_graph()

    def ingest_assessments(self, assessments_path: Path) -> None:
        if not assessments_path.exists():
            return
        raw = json.loads(assessments_path.read_text(encoding="utf-8"))
        for item in raw:
            assessment = ClassroomAssessment(
                classroom_id=str(item["classroom_id"]),
                grade=str(item.get("grade", "")),
                school_level=str(item.get("school_level", "")),
                timestamp=str(item.get("timestamp", "")),
                metrics={k: float(v) for k, v in item.get("metrics", {}).items()},
                notes=item.get("notes", ""),
            )
            self.store.upsert_assessment(assessment)


# ----------------------------
# Retrieval helpers
# ----------------------------

class Retriever:
    def __init__(self, client: OllamaClient, store: MySELStore):
        self.client = client
        self.store = store

    def search_chunks(self, query: str, top_k: int = TOP_K) -> List[Tuple[ChunkRecord, float]]:
        q_emb = self.client.embed(query)
        scored = []
        for rec in self.store.load_chunks():
            scored.append((rec, cosine_similarity(q_emb, rec.embedding)))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def search_activities(self, query: str, top_k: int = TOP_K) -> List[Tuple[Activity, float]]:
        q_emb = self.client.embed(query)
        scored = []
        for act in self.store.load_activities():
            text = " | ".join([act.title, act.summary, " ".join(act.competencies), " ".join(act.didactic_strategies), " ".join(act.topics)])
            emb = self.client.embed(text)
            score = cosine_similarity(q_emb, emb)
            score += 0.08 * act.popularity + 0.12 * act.impact
            scored.append((act, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def get_classroom_assessment(self, classroom_id: str) -> Optional[ClassroomAssessment]:
        for a in self.store.load_assessments():
            if a.classroom_id == classroom_id:
                return a
        return None

    def graph_neighbors(self, node_id: str, depth: int = 2) -> List[str]:
        if node_id not in self.store.graph:
            return []
        nodes = {node_id}
        frontier = {node_id}
        for _ in range(depth):
            nxt = set()
            for n in frontier:
                nxt.update(self.store.graph.successors(n))
                nxt.update(self.store.graph.predecessors(n))
            nxt -= nodes
            nodes.update(nxt)
            frontier = nxt
        return list(nodes)


# ----------------------------
# Agents
# ----------------------------

class ResearcherAgent:
    def __init__(self, client: OllamaClient, retriever: Retriever):
        self.client = client
        self.retriever = retriever

    def answer(self, question: str) -> str:
        chunks = self.retriever.search_chunks(question, top_k=6)
        context = "\n\n".join([f"[Chunk score={score:.3f}] {rec.title}: {rec.text}" for rec, score in chunks])
        prompt = f"""You are the Researcher Agent for SEL.
Answer strictly from the provided context. If context is insufficient, say so clearly.

Question: {question}

Context:
{context}

Write a concise, accurate answer."""
        return self.client.chat([
            {"role": "system", "content": "You are a careful educational assistant. Do not invent facts."},
            {"role": "user", "content": prompt},
        ], temperature=0.0, max_tokens=900)


class ProfilerAgent:
    def __init__(self, client: OllamaClient, retriever: Retriever):
        self.client = client
        self.retriever = retriever

    def explain(self, classroom_id: str) -> str:
        a = self.retriever.get_classroom_assessment(classroom_id)
        if not a:
            return f"No classroom assessment found for classroom_id={classroom_id}."

        baseline = self._baseline_for_level(a.school_level)
        prompt = f"""You are the Profiler Agent.
Explain the classroom assessment at group level only. Avoid any personal identification.

Classroom assessment:
{json.dumps(asdict(a), indent=2)}

Baseline for comparison:
{json.dumps(baseline, indent=2)}

Tasks:
1) Summarize strengths and concerns.
2) Compare metrics against baseline.
3) Mention trends only if the assessment data supports them.
4) Suggest what SEL competencies may need attention.

Return a clear, teacher-friendly summary."""
        return self.client.chat([
            {"role": "system", "content": "Be precise, privacy-preserving, and educational."},
            {"role": "user", "content": prompt},
        ], temperature=0.0, max_tokens=900)

    def _baseline_for_level(self, school_level: str) -> Dict[str, float]:
        # Simple default baseline. Replace with data-driven baselines if available.
        return {
            "group_emotional_awareness": 0.62,
            "group_emotional_regulation": 0.60,
            "group_emotional_climate": 0.63,
            "association_index": 0.58,
            "classroom_social_cohesion": 0.61,
            "isolated_students_ratio": 0.15,
        }


class RecommenderAgent:
    def __init__(self, client: OllamaClient, retriever: Retriever):
        self.client = client
        self.retriever = retriever

    def recommend(self, user_request: str) -> str:
        criteria = self._extract_criteria(user_request)
        query_text = self._criteria_to_query(criteria)
        activities = self.retriever.search_activities(query_text, top_k=10)
        path = self._build_learning_path([act for act, _ in activities], criteria)
        prompt = f"""You are the SEL Activities Recommender Agent.
Use the criteria and the retrieved activities to produce a helpful recommendation.
Do not claim certainty beyond the retrieved data.

User request:
{user_request}

Extracted criteria:
{json.dumps(criteria, indent=2)}

Candidate activities:
{json.dumps([asdict(a) for a, _ in activities], indent=2)}

Proposed path:
{json.dumps(path, indent=2)}

Return:
- a brief explanation of why these activities match
- the recommended activities or learning path
- a note that the teacher should adapt them to the classroom context."""
        return self.client.chat([
            {"role": "system", "content": "Be practical, concise, and pedagogically sound."},
            {"role": "user", "content": prompt},
        ], temperature=0.0, max_tokens=1200)

    def _extract_criteria(self, text: str) -> Dict[str, Any]:
        # Local heuristic extraction; can be replaced with an LLM JSON extractor.
        age_group = None
        m = re.search(r"\b(6-8|9-12|13-18)\b", text)
        if m:
            age_group = m.group(1)
        num = None
        m = re.search(r"\b(\d{1,2})\s*(?:activities?|steps?)\b", text, re.I)
        if m:
            num = int(m.group(1))
        return {
            "age_group": age_group,
            "num_activities": num or 5,
            "competencies": self._keyword_hits(text, [
                "emotional awareness", "empathy", "emotion regulation", "self-awareness",
                "self-management", "social awareness", "relationship skills", "responsible decision-making",
                "social cohesion", "group emotional climate"
            ]),
            "strategies": self._keyword_hits(text, ["role playing", "project-based learning", "experiential learning", "arts-based learning"]),
            "topic_words": self._extract_topics(text),
        }

    def _keyword_hits(self, text: str, keywords: List[str]) -> List[str]:
        low = text.lower()
        return [k for k in keywords if k in low]

    def _extract_topics(self, text: str) -> List[str]:
        words = re.findall(r"[A-Za-z][A-Za-z\- ]{3,}", text)
        return [w.strip().lower() for w in words[:8]]

    def _criteria_to_query(self, c: Dict[str, Any]) -> str:
        parts = []
        if c.get("age_group"):
            parts.append(c["age_group"])
        parts.extend(c.get("competencies", []))
        parts.extend(c.get("strategies", []))
        parts.extend(c.get("topic_words", []))
        return " ".join(parts) if parts else "SEL activity"

    def _build_learning_path(self, activities: List[Activity], criteria: Dict[str, Any]) -> Dict[str, Any]:
        # Prefer existing sequence links, then add high-impact items.
        if not activities:
            return {"path": [], "note": "No activities found."}

        # Build a simple DAG path using candidate activities and explicit next_activities.
        candidate_ids = {a.activity_id for a in activities}
        order = []
        visited = set()

        def dfs(act_id: str):
            if act_id in visited or act_id not in candidate_ids:
                return
            visited.add(act_id)
            order.append(act_id)
            for _, nxt, ed in self.retriever.store.graph.out_edges(act_id, data=True):
                if ed.get("relation") == "sequence":
                    dfs(nxt)

        dfs(activities[0].activity_id)
        for a in activities:
            if a.activity_id not in visited and len(order) < criteria.get("num_activities", 5):
                order.append(a.activity_id)

        order = order[: max(1, int(criteria.get("num_activities", 5)))]
        selected = [a for a in activities if a.activity_id in order]
        return {
            "path": [asdict(a) for a in selected],
            "goal": "Create a sequential SEL learning path",
            "target_length": int(criteria.get("num_activities", 5)),
        }


# ----------------------------
# Hypervisor router
# ----------------------------

class HypervisorAgent:
    def __init__(self, researcher: ResearcherAgent, profiler: ProfilerAgent, recommender: RecommenderAgent):
        self.researcher = researcher
        self.profiler = profiler
        self.recommender = recommender

    def route(self, user_input: str) -> str:
        lower = user_input.lower()
        if any(k in lower for k in ["classroom", "assessment", "baseline", "otp", "profile", "scores", "emotional climate"]):
            # Prefer profiler when classroom data or metric interpretation is requested.
            classroom_id = self._extract_classroom_id(user_input)
            if classroom_id:
                return self.profiler.explain(classroom_id)
        if any(k in lower for k in ["activity", "learning path", "program", "recommend", "sequence", "intervention"]):
            return self.recommender.recommend(user_input)
        if any(k in lower for k in ["what is", "explain", "how does", "summarize", "define", "sel", "social and emotional"]):
            return self.researcher.answer(user_input)
        # fallback
        return self.researcher.answer(user_input)

    def _extract_classroom_id(self, text: str) -> Optional[str]:
        m = re.search(r"classroom[_\s-]?id\s*[:=]\s*([A-Za-z0-9_-]+)", text, re.I)
        if m:
            return m.group(1)
        m = re.search(r"\b([A-Za-z]{1,3}\d{1,4})\b", text)
        if m:
            return m.group(1)
        return None


# ----------------------------
# Bootstrapping
# ----------------------------

def build_system(data_root: Path, client: OllamaClient) -> Tuple[MySELStore, HypervisorAgent]:
    store = MySELStore(data_root)
    ingestor = Ingestor(client, store)
    ingestor.ingest_theory_docs(data_root / THEORY_DIR)
    ingestor.ingest_activities(data_root / ACTIVITIES_JSON)
    ingestor.ingest_assessments(data_root / ASSESSMENTS_JSON)
    retriever = Retriever(client, store)
    researcher = ResearcherAgent(client, retriever)
    profiler = ProfilerAgent(client, retriever)
    recommender = RecommenderAgent(client, retriever)
    hypervisor = HypervisorAgent(researcher, profiler, recommender)
    return store, hypervisor


def cmd_init(args: argparse.Namespace) -> None:
    data_root = Path(args.data)
    data_root.mkdir(parents=True, exist_ok=True)
    (data_root / THEORY_DIR).mkdir(parents=True, exist_ok=True)

    # Seed example files if absent.
    seed_activities = data_root / ACTIVITIES_JSON
    if not seed_activities.exists():
        seed_activities.write_text(
            json.dumps(
                [
                    {
                        "activity_id": "a1",
                        "title": "How are you feeling?",
                        "summary": "A warm-up activity to name feelings and normalize emotional vocabulary.",
                        "age_group": "6-8",
                        "competencies": ["emotional awareness", "emotion expression"],
                        "didactic_strategies": ["arts-based-learning"],
                        "topics": ["feelings", "vocabulary"],
                        "popularity": 0.8,
                        "impact": 0.7,
                        "next_activities": ["a2"],
                    },
                    {
                        "activity_id": "a2",
                        "title": "Mood Meter",
                        "summary": "Students place emotions on an arousal/pleasantness map and discuss differences.",
                        "age_group": "9-12",
                        "competencies": ["self-awareness", "emotional regulation"],
                        "didactic_strategies": ["experiential-learning"],
                        "topics": ["emotion scale", "self-regulation"],
                        "popularity": 0.9,
                        "impact": 0.85,
                        "next_activities": ["a3"],
                    },
                    {
                        "activity_id": "a3",
                        "title": "The Guardian Angels",
                        "summary": "A reflection activity on peer support, belonging, and empathy.",
                        "age_group": "13-18",
                        "competencies": ["empathy", "social awareness"],
                        "didactic_strategies": ["role-playing"],
                        "topics": ["peer support", "empathy"],
                        "popularity": 0.7,
                        "impact": 0.72,
                    },
                ],
                indent=2,
            ),
            encoding="utf-8",
        )

    seed_assessments = data_root / ASSESSMENTS_JSON
    if not seed_assessments.exists():
        seed_assessments.write_text(
            json.dumps(
                [
                    {
                        "classroom_id": "A1",
                        "grade": "5",
                        "school_level": "primary",
                        "timestamp": "2026-03-01",
                        "metrics": {
                            "group_emotional_awareness": 0.55,
                            "group_emotional_regulation": 0.48,
                            "group_emotional_climate": 0.59,
                            "association_index": 0.52,
                            "classroom_social_cohesion": 0.61,
                            "isolated_students_ratio": 0.20,
                        },
                        "notes": "Classroom appears somewhat fragmented, with a few isolated students.",
                    }
                ],
                indent=2,
            ),
            encoding="utf-8",
        )

    (data_root / THEORY_DIR / "sel_intro.txt").write_text(
        """Social and Emotional Learning (SEL) supports self-awareness, self-management, social awareness, relationship skills, and responsible decision-making.\n\nTeachers can improve SEL through structured activities, reflective discussion, and sequential learning paths.\n\nGraph retrieval is useful when facts are interconnected across multiple documents.""",
        encoding="utf-8",
    )
    print(f"Initialized data folder at {data_root.resolve()}")


def cmd_chat(args: argparse.Namespace) -> None:
    data_root = Path(args.data)
    client = OllamaClient(model=args.model, embed_model=args.embed_model)
    _, hypervisor = build_system(data_root, client)
    print("MySELAgent local chat. Type 'exit' to quit.")
    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break
        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit"}:
            break
        try:
            answer = hypervisor.route(user_input)
        except requests.HTTPError as e:
            answer = f"Ollama request failed: {e}\nMake sure Ollama is running and the model is pulled."  # noqa: E501
        except Exception as e:
            answer = f"Error: {e}"
        print(f"\nAssistant: {answer}")


def cmd_rebuild(args: argparse.Namespace) -> None:
    data_root = Path(args.data)
    client = OllamaClient(model=args.model, embed_model=args.embed_model)
    store, _ = build_system(data_root, client)
    print(f"Indexed {len(store.load_chunks())} chunks, {len(store.load_activities())} activities, {len(store.load_assessments())} assessments.")


def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Local MySELAgent implementation with Ollama")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_init = sub.add_parser("init", help="Create sample data files")
    p_init.add_argument("--data", default="./data")
    p_init.set_defaults(func=cmd_init)

    p_chat = sub.add_parser("chat", help="Run interactive chat")
    p_chat.add_argument("--data", default="./data")
    p_chat.add_argument("--model", default=DEFAULT_LLM)
    p_chat.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL)
    p_chat.set_defaults(func=cmd_chat)

    p_rebuild = sub.add_parser("rebuild", help="Rebuild index and graph")
    p_rebuild.add_argument("--data", default="./data")
    p_rebuild.add_argument("--model", default=DEFAULT_LLM)
    p_rebuild.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL)
    p_rebuild.set_defaults(func=cmd_rebuild)
    return p


def main(argv: Optional[List[str]] = None) -> None:
    parser = make_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
