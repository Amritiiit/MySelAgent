# MySELAgent: Local-First Agentic Graph RAG System

> An agentic Social and Emotional Learning (SEL) assistant for educators — runs fully locally, no cloud required.

---

## Overview

MySELAgent is a multi-agent Retrieval-Augmented Generation (RAG) system designed to support teachers with three core tasks:

- 📚 **SEL Theory Q&A** — Answer questions grounded in SEL literature
- 📊 **Classroom Assessment Interpretation** — Analyze group-level socio-emotional data with privacy in mind
- 🗺️ **Activity Recommendation & Learning Paths** — Suggest evidence-informed SEL activities and sequential paths

Inspired by: *Fotopoulou et al., "Expert Agents for Social and Emotional Learning: An Agentic Graph Retrieval Augmented Generation Approach," IEEE Access, 2026.*

---

## Tech Stack

| Component | Technology |
|---|---|
| LLM Runtime | [Ollama](https://ollama.com/) + Gemma 3 |
| Embeddings | Ollama embed endpoint |
| Database | SQLite |
| Graph Layer | NetworkX (GraphML) |
| Language | Python 3 |

---

## Project Structure

```
MySELAgent_Local/
│
├── main.py                          # Entry point — agents, routing, inference
│
└── data/
    ├── activities.json              # SEL activity definitions
    ├── classroom_assessments.json   # Classroom-level assessment records
    ├── myselagent.db                # SQLite persistent store
    ├── myselagent_graph.graphml     # NetworkX directed graph
    │
    └── theory_docs/
        ├── sel_intro.txt            # SEL theory introduction
        ├── sel_assessment_guide.txt # Assessment methodology guide
        └── teacher_handbook.txt     # Practical teacher handbook
```

---

## Architecture

The system follows a **multi-agent cascade**:

```
User Query
    │
    ▼
┌─────────────────────┐
│   Hypervisor Agent  │  ← Routes query by keyword heuristics
└──────────┬──────────┘
           │
    ┌──────┴──────┬──────────────┐
    ▼             ▼              ▼
Researcher     Profiler      Recommender
  Agent          Agent          Agent
    │             │              │
SEL Theory   Classroom      SEL Activity
  Q&A        Analysis    Recommendations &
             (Group)       Learning Paths
    │             │              │
    └──────┬──────┴──────────────┘
           ▼
    ┌─────────────┐
    │   Gemma 3   │  ← Local LLM generates final response
    └─────────────┘
```

### Agents

| Agent | Responsibility |
|---|---|
| **Hypervisor** | Classifies query and routes to the correct sub-agent |
| **Researcher** | Semantic chunk retrieval from theory docs → grounded LLM answer |
| **Profiler** | Looks up classroom by ID, compares to baseline, generates privacy-safe summary |
| **Recommender** | Scores activities (similarity + popularity + impact), builds graph-traversal paths |

### Role of the Graph

- **Document Chunk Connectivity** — `next` edges preserve local ordering of theory content
- **Activity Sequencing** — `sequence` edges enable full learning path generation
- **Conceptual Enrichment** — Entity/relation nodes support semantic reasoning

---

## Setup & Usage

### Prerequisites

- Python 3.8+
- [Ollama](https://ollama.com/) installed and running
- Gemma 3 model pulled: `ollama pull gemma3:4b`

### Installation

```bash
git clone https://github.com/AmritSagar/MySELAgent_Local
cd MySELAgent_Local
pip install -r requirements.txt
```

### Running the script

```bash
python main.py init --data ./data
python main.py rebuild --data ./data --model gemma3:4b
python main.py chat --data ./data --model gemma3:4b
```

---

## Example Interaction

```
You: Analyze classroom D4

Assistant: Classroom Assessment Summary — Group Level (D4, Grade 3)

Strengths: Moderate social cohesion; students generally connect with one another.

Concerns:
  - Emotional Awareness:   0.49  (baseline 0.62)
  - Emotional Regulation:  0.45  (baseline 0.60)
  - Emotional Climate:     0.51  (baseline 0.63)
  - ~23% of students show signs of social isolation

Suggested Focus Areas:
  1. Emotional Awareness — activities to broaden emotion identification
  2. Emotional Regulation — techniques for managing challenging emotions
  3. Social Cohesion & Inclusion — strategies to reduce isolation
```

---

## Known Limitations

| Limitation | Detail |
|---|---|
| Scale | Designed for single-machine research; not tested on large institutional datasets |
| Heuristic Routing | Hypervisor uses keyword rules, not a learned classifier |
| Isolated Agent Loops | Agents do not yet collaborate iteratively without user re-prompting |
| Vector Search | Brute-force similarity; no FAISS or ANN index — will not scale well |

---

## Roadmap

- [ ] Replace keyword router with an LLM-based decision layer
- [ ] Upgrade to Neo4j for native multi-hop graph traversal
- [ ] Enable direct Profiler → Recommender pipeline handoff
- [ ] Add FAISS index for scalable vector retrieval

---

## Authors

| Name | Roll Number |
|---|---|
| Amrit Sagar | 2023UG3028 |
| Ujjwal Prakash Bajpayee | 2023UG3017 |

---

## References

1. Fotopoulou et al., *Expert Agents for Social and Emotional Learning*, IEEE Access, 2026.
2. Microsoft Research, *GraphRAG: Leveraging Knowledge Graphs for RAG*, 2024.
3. Lewis et al., *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*, NeurIPS, 2020.
4. Chase et al., *LangGraph: Building Stateful Multi-Agent Applications with LLMs*, 2024.
5. Zhao et al., *A Survey of Large Language Model-based Agents*, 2024.

---

*Course: Agentic AI / Generative Models*
