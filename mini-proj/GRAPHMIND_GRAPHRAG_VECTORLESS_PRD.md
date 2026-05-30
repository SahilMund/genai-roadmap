# 🕸️ GraphMind — Graph RAG & Vectorless RAG
> **Type:** Minor Project · Phase 2 Deep Dive (Advanced RAG Architectures)
> **Stack:** FastAPI · Neo4j · NetworkX · Qdrant (optional) · OpenAI · Groq · React 18
> **Timeline:** 2–3 weeks (25–30 hrs)
> **Teaches:** Knowledge graph construction · Graph traversal for retrieval · Multi-hop reasoning · Vectorless (BM25 + keyword) RAG · When each approach wins

---

## 📑 Table of Contents

1. [The Two Ideas](#1-the-two-ideas)
2. [Why This Project](#2-why-this-project)
3. [Part A — GraphRAG](#part-a--graphrag)
   - [What is GraphRAG](#31-what-is-graphrag)
   - [Architecture](#32-architecture)
   - [Knowledge Graph Construction](#33-knowledge-graph-construction)
   - [Graph Retrieval](#34-graph-retrieval)
   - [Full GraphRAG Pipeline](#35-full-graphrag-pipeline)
4. [Part B — Vectorless RAG](#part-b--vectorless-rag)
   - [What is Vectorless RAG](#41-what-is-vectorless-rag)
   - [BM25 + Keyword Pipeline](#42-bm25--keyword-pipeline)
   - [When Vectorless Wins](#43-when-vectorless-wins)
5. [Side-by-Side Comparison](#5-side-by-side-comparison)
6. [Combined Mode — Best of Both](#6-combined-mode--best-of-both)
7. [API Reference](#7-api-reference)
8. [Database Schema](#8-database-schema)
9. [Folder Structure](#9-folder-structure)
10. [Phase-by-Phase Build Plan](#10-phase-by-phase-build-plan)
11. [Resume Deliverables](#11-resume-deliverables)

---

## 1. The Two Ideas

### Part A — GraphRAG: When Vector Search Fails Multi-Hop Questions

```
VANILLA RAG fails this question:
  "Compare how Python and JavaScript handle async programming,
   and explain which companies in our docs use each and why."

Why it fails:
  This is a multi-hop, comparative question.
  Step 1: find Python async content
  Step 2: find JavaScript async content
  Step 3: find companies using Python → cross-reference with async section
  Step 4: same for JavaScript
  Step 5: synthesise comparison

Vector search returns the top-5 most similar chunks globally.
It can't do "traverse the graph from Python → async → companies using Python."

GraphRAG adds a knowledge graph layer:
  Python → [HAS_FEATURE] → AsyncIO → [USED_BY] → Netflix, Uber
  JavaScript → [HAS_FEATURE] → Promises/async-await → [USED_BY] → LinkedIn, PayPal
  
  Query: traverse Python → async → users AND JavaScript → async → users
  → return connected nodes → LLM synthesises comparison
```

### Part B — Vectorless RAG: Sometimes You Don't Need Embeddings

```
VECTOR EMBEDDINGS cost money and time:
  10K documents × $0.02/1M tokens × 500 tokens avg = $0.10 to embed
  Qdrant setup, maintenance, scaling
  Embedding model updates break existing index

VECTORLESS RAG uses classic information retrieval:
  BM25: probabilistic keyword ranking (what Elasticsearch uses under the hood)
  TF-IDF: term frequency × inverse document frequency
  SQLite FTS5: full-text search built into SQLite — zero extra infra
  Postgres tsvector: full-text search in your existing Postgres

Best for:
  Exact term matching: "What is the penalty under Section 32B?"
  Technical docs with precise vocabulary: error codes, API names, config keys
  Small knowledge bases (< 10K documents): BM25 quality ≈ vector quality
  Privacy requirements: can't send docs to embedding API
  Zero-infra environments: llama.cpp + SQLite FTS5 → fully offline RAG
```

---

## 2. Why This Project

```
WHAT MOST CANDIDATES BUILD:
  Qdrant + text-embedding-3-small + GPT-4o → done.
  This is correct, but everyone does it. It's table stakes in 2026.

WHAT THIS PROJECT SHOWS:
  You understand WHY vanilla vector RAG fails on multi-hop questions.
  You know alternative retrieval strategies and when to apply each.
  You've implemented a knowledge graph and traversed it for Q&A.
  You've built vectorless RAG — you know it's not always about embeddings.
  You can architect the retrieval layer for any problem, not just copy-paste Qdrant.

INTERVIEW TALKING POINT:
  "I've implemented three retrieval strategies in the same application:
   dense vector search for semantic similarity, GraphRAG for multi-hop
   relational questions, and BM25 for exact term matching. I route queries
   to the right retriever based on query type. For 'compare X and Y' or
   'how is X related to Z' questions, GraphRAG outperforms vector search
   significantly. For 'find all mentions of error code 429', BM25 wins."
```

---

## Part A — GraphRAG

### 3.1 What is GraphRAG

```
KNOWLEDGE GRAPH:
  Entities:     things mentioned in your documents (products, companies, concepts)
  Relationships: how they connect (USES, COMPETES_WITH, HAS_FEATURE, PART_OF)
  
  Example from a tech company wiki:
  
  [Python] ──HAS_FEATURE──▶ [AsyncIO]
  [Python] ──USED_BY──────▶ [Data Team]
  [FastAPI] ──BUILT_WITH──▶ [Python]
  [FastAPI] ──SIMILAR_TO──▶ [Flask]
  [Data Team] ──OWNS──────▶ [ML Pipeline]
  [ML Pipeline] ──USES────▶ [Python]

GRAPH RETRIEVAL:
  "What does the Data Team use?"
  → Start at [Data Team] → traverse outgoing edges → [Python, ML Pipeline]
  → What does [ML Pipeline] use? → [Python]
  → Answer: Data Team uses Python (directly) via FastAPI and ML Pipeline.

  Vector search for same query:
  → Returns top-5 chunks mentioning "Data Team"
  → May miss that ML Pipeline (owned by Data Team) also uses Python
  → Misses the transitive relationship
```

### 3.2 Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        GraphMind                                    │
│                                                                     │
│  INGESTION PIPELINE                                                 │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │  Raw Documents (PDF, MD, TXT)                                │  │
│  │         ↓                                                    │  │
│  │  Entity Extraction (GPT-4o-mini, spaCy NER)                 │  │
│  │  [{entity, type, description}]                              │  │
│  │         ↓                                                    │  │
│  │  Relationship Extraction (GPT-4o-mini)                      │  │
│  │  [{source, relation, target, sentence}]                     │  │
│  │         ↓                    ↓                              │  │
│  │  Neo4j (graph DB)       Postgres (raw chunks + source)      │  │
│  │  Nodes + Edges           BM25 index (vectorless)            │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                                                                     │
│  QUERY PIPELINE                                                     │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │  User Query                                                  │  │
│  │         ↓                                                    │  │
│  │  Query Router (classify: simple | multi-hop | exact-term)   │  │
│  │     ↓           ↓              ↓                            │  │
│  │  Vector       GraphRAG       BM25                           │  │
│  │  (Qdrant)     (Neo4j)        (Postgres FTS)                │  │
│  │     ↓           ↓              ↓                            │  │
│  │         Merge + Rerank                                      │  │
│  │                  ↓                                          │  │
│  │         LLM Generation (Groq 70b)                          │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                                                                     │
│  Neo4j · Postgres FTS · Qdrant (optional) · FastAPI · React        │
└─────────────────────────────────────────────────────────────────────┘
```

### 3.3 Knowledge Graph Construction

```python
# app/graph_rag/kg_builder.py
# pip install neo4j openai spacy
# python -m spacy download en_core_web_md

import spacy
import instructor
from openai import AsyncOpenAI
from neo4j import AsyncGraphDatabase
from pydantic import BaseModel

nlp    = spacy.load("en_core_web_md")
client = instructor.from_openai(AsyncOpenAI())
driver = AsyncGraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "password"))

# ── Pydantic models for extraction ────────────────────────────────────
class Entity(BaseModel):
    name:        str
    type:        str   # PRODUCT | PERSON | COMPANY | CONCEPT | TECHNOLOGY | PROCESS
    description: str   # one sentence summary

class Relationship(BaseModel):
    source:      str   # entity name
    relation:    str   # USES | OWNS | COMPETES_WITH | HAS_FEATURE | PART_OF | BUILT_WITH
    target:      str   # entity name
    sentence:    str   # the sentence this came from (for provenance)

class KGExtraction(BaseModel):
    entities:      list[Entity]
    relationships: list[Relationship]

async def extract_kg_from_chunk(text: str, source: str) -> KGExtraction:
    """Extract entities + relationships from a document chunk using LLM"""
    return await client.chat.completions.create(
        model="gpt-4o-mini",
        response_model=KGExtraction,
        messages=[{
            "role": "user",
            "content": f"""Extract entities and relationships from this text.

Text:
{text}

Source: {source}

Entity types: PRODUCT, PERSON, COMPANY, CONCEPT, TECHNOLOGY, PROCESS, FEATURE
Relationship types: USES, OWNS, COMPETES_WITH, HAS_FEATURE, PART_OF, BUILT_WITH,
                    SIMILAR_TO, DEPENDS_ON, CREATED_BY, USED_BY

Extract all meaningful entities and every stated relationship between them.
Only extract what is explicitly stated — do not infer.""",
        }],
    )

async def store_in_neo4j(extraction: KGExtraction, source: str):
    """Store extracted entities and relationships in Neo4j"""
    async with driver.session() as session:
        # Upsert entities (MERGE = create if not exists, else match)
        for entity in extraction.entities:
            await session.run("""
                MERGE (e:Entity {name: $name})
                SET e.type = $type,
                    e.description = $description,
                    e.source = $source,
                    e.updated_at = datetime()
            """, name=entity.name, type=entity.type,
                 description=entity.description, source=source)

        # Upsert relationships
        for rel in extraction.relationships:
            # Dynamic relationship type (USES, OWNS, etc.)
            query = f"""
                MATCH (s:Entity {{name: $source}})
                MATCH (t:Entity {{name: $target}})
                MERGE (s)-[r:{rel.relation}]->(t)
                SET r.sentence = $sentence,
                    r.doc_source = $doc_source
            """
            await session.run(query,
                source=rel.source, target=rel.target,
                sentence=rel.sentence, doc_source=source)

async def build_knowledge_graph(documents: list[dict]):
    """
    Full pipeline: chunk documents → extract entities/relationships → store in Neo4j
    documents: [{text, source, chunk_id}]
    """
    from tqdm import tqdm
    for doc in tqdm(documents, desc="Building KG"):
        extraction = await extract_kg_from_chunk(doc["text"], doc["source"])
        await store_in_neo4j(extraction, doc["source"])
    print(f"Knowledge graph built: {len(documents)} chunks processed")

# ── Utility: visualise the graph ──────────────────────────────────────
async def get_graph_stats() -> dict:
    async with driver.session() as session:
        result = await session.run("""
            MATCH (e:Entity) WITH count(e) as entities
            MATCH ()-[r]->() WITH entities, count(r) as relationships
            RETURN entities, relationships
        """)
        row = await result.single()
        return {"entities": row["entities"], "relationships": row["relationships"]}
```

### 3.4 Graph Retrieval

```python
# app/graph_rag/graph_retriever.py

from neo4j import AsyncGraphDatabase
from pydantic import BaseModel
from typing import Literal

class GraphQueryPlan(BaseModel):
    """LLM decomposes user query into graph traversal steps"""
    entity_seeds:       list[str]   # entities to start traversal from
    traversal_depth:    int         # how many hops (1=direct, 2=neighbours, 3=extended)
    relationship_types: list[str]   # filter to these relationship types (empty = all)
    query_type:         Literal["single_entity", "multi_hop", "comparison", "path"]

class GraphContext(BaseModel):
    nodes:         list[dict]   # entities found
    edges:         list[dict]   # relationships found
    text_passages: list[str]    # original sentences from document
    summary:       str          # LLM summary of what the graph found

async def plan_graph_query(user_query: str) -> GraphQueryPlan:
    """Ask LLM to decompose the query into graph traversal instructions"""
    return await client.chat.completions.create(
        model="gpt-4o-mini",
        response_model=GraphQueryPlan,
        messages=[{
            "role": "user",
            "content": f"""You are planning a knowledge graph query.

User question: {user_query}

Identify:
1. entity_seeds: main entities the question is about (1-3 entities)
2. traversal_depth: 1 for direct facts, 2 for connected info, 3 for extended network
3. relationship_types: relevant relationship types (or empty for all)
4. query_type: single_entity | multi_hop | comparison | path

Example: "How is Python related to our ML pipeline?"
→ seeds: ["Python", "ML Pipeline"], depth: 2, type: multi_hop""",
        }],
    )

async def traverse_graph(plan: GraphQueryPlan) -> GraphContext:
    """Execute graph traversal based on the plan"""
    async with driver.session() as session:
        nodes, edges, passages = [], [], []

        for seed in plan.entity_seeds:
            # Variable-depth traversal
            if plan.relationship_types:
                rel_filter = "|".join(plan.relationship_types)
                query = f"""
                    MATCH path = (start:Entity {{name: $seed}})
                                 -[:{rel_filter}*1..{plan.traversal_depth}]->
                                 (end:Entity)
                    RETURN nodes(path) as nodes, relationships(path) as rels
                    LIMIT 20
                """
            else:
                query = f"""
                    MATCH path = (start:Entity {{name: $seed}})
                                 -[*1..{plan.traversal_depth}]->
                                 (end:Entity)
                    RETURN nodes(path) as nodes, relationships(path) as rels
                    LIMIT 20
                """

            result = await session.run(query, seed=seed)
            async for record in result:
                for node in record["nodes"]:
                    node_dict = dict(node.items())
                    if node_dict not in nodes:
                        nodes.append(node_dict)
                for rel in record["rels"]:
                    rel_dict = {
                        "source":   rel.start_node["name"],
                        "relation": rel.type,
                        "target":   rel.end_node["name"],
                        "sentence": rel.get("sentence", ""),
                    }
                    if rel_dict not in edges:
                        edges.append(rel_dict)
                        if rel_dict["sentence"]:
                            passages.append(rel_dict["sentence"])

        # Ask LLM to summarise what the graph found
        graph_text = (
            "Entities: " + ", ".join(n["name"] for n in nodes) + "\n" +
            "Relationships: " + "\n".join(
                f"  {e['source']} --{e['relation']}--> {e['target']}"
                for e in edges
            )
        )
        summary_resp = await groq.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{
                "role": "user",
                "content": f"Summarise these graph findings in 2-3 sentences:\n{graph_text}"
            }],
            max_tokens=150,
        )

        return GraphContext(
            nodes=nodes,
            edges=edges,
            text_passages=passages,
            summary=summary_resp.choices[0].message.content,
        )

# ── Comparison queries: find shortest path between two entities ────────
async def find_path_between(entity_a: str, entity_b: str, max_hops: int = 4) -> list[dict]:
    """
    "How is FastAPI related to the ML Pipeline?"
    → Find shortest path: FastAPI → Python → ML Pipeline
    """
    async with driver.session() as session:
        result = await session.run("""
            MATCH path = shortestPath(
                (a:Entity {name: $entity_a})-[*..%d]-(b:Entity {name: $entity_b})
            )
            RETURN [node in nodes(path) | node.name] as path_names,
                   [rel in relationships(path) | type(rel)] as rel_types,
                   length(path) as hops
            LIMIT 3
        """ % max_hops, entity_a=entity_a, entity_b=entity_b)

        paths = []
        async for record in result:
            paths.append({
                "path":      record["path_names"],
                "relations": record["rel_types"],
                "hops":      record["hops"],
            })
        return paths
```

### 3.5 Full GraphRAG Pipeline

```python
# app/graph_rag/pipeline.py

from langfuse.decorators import observe

@observe(name="graphrag_query")
async def graphrag_query(user_query: str) -> dict:
    """
    Full GraphRAG pipeline:
    1. Plan: decompose query into graph traversal
    2. Traverse: walk the knowledge graph
    3. Augment: combine graph context with optional vector search
    4. Generate: LLM synthesises answer from graph + text
    """

    # Step 1: Plan the graph traversal
    plan    = await plan_graph_query(user_query)

    # Step 2: Traverse the graph
    graph_ctx = await traverse_graph(plan)

    # Step 3: Build context for LLM
    context_parts = []
    if graph_ctx.nodes:
        context_parts.append(
            "Knowledge Graph Findings:\n" + graph_ctx.summary
        )
    if graph_ctx.text_passages:
        context_parts.append(
            "Source Passages:\n" + "\n".join(f"• {p}" for p in graph_ctx.text_passages[:5])
        )

    # Step 4: Generate answer
    context = "\n\n".join(context_parts)
    response = await groq.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a knowledgeable assistant. Answer using the knowledge "
                    "graph context provided. Show relationships clearly. "
                    "If asked to compare, compare explicitly."
                ),
            },
            {
                "role": "user",
                "content": f"Context from knowledge graph:\n{context}\n\nQuestion: {user_query}",
            },
        ],
        max_tokens=500,
        temperature=0.2,
    )

    return {
        "answer":         response.choices[0].message.content,
        "graph_nodes":    [n["name"] for n in graph_ctx.nodes],
        "graph_edges":    len(graph_ctx.edges),
        "traversal_plan": plan.model_dump(),
        "retriever":      "graph",
    }
```

---

## Part B — Vectorless RAG

### 4.1 What is Vectorless RAG

```
VECTORLESS RAG = retrieval WITHOUT vector embeddings.
Uses classical Information Retrieval (IR) algorithms instead.

The algorithms:

BM25 (Best Match 25):
  An improved TF-IDF formula used by Elasticsearch, Solr, and Lucene.
  Considers: term frequency, inverse document frequency, document length.
  Formula: BM25(D, Q) = Σ IDF(qi) × [f(qi,D)×(k+1)] / [f(qi,D) + k×(1-b+b×|D|/avgdl)]
  Where: f = term frequency, k=1.5, b=0.75 (tunable constants)
  
TF-IDF (Term Frequency × Inverse Document Frequency):
  TF: how often a term appears in this document
  IDF: log(total docs / docs containing term) — rare terms score higher
  Score = TF × IDF
  → "async programming" in a doc about async = high TF
  → "the" = low IDF (appears everywhere) = low score

Postgres FTS (Full-Text Search):
  Built into Postgres via tsvector/tsquery.
  No extra infra. Works on your existing database.
  Supports: stemming, stop words, ranking (ts_rank), AND/OR/NOT queries.

SQLite FTS5:
  Built into SQLite — completely offline, zero dependencies.
  Perfect for: edge devices, offline apps, privacy-sensitive environments.
```

### 4.2 BM25 + Keyword Pipeline

```python
# app/vectorless/bm25_retriever.py
# pip install rank-bm25

from rank_bm25 import BM25Okapi
import re, json, asyncpg

# ── In-memory BM25 index ──────────────────────────────────────────────
class BM25Retriever:
    """
    In-memory BM25 index over your document chunks.
    Load at startup, refresh when documents are updated.
    For large collections (> 100K docs): use Elasticsearch instead.
    """

    def __init__(self):
        self._corpus:   list[dict]       = []   # [{id, content, source, metadata}]
        self._tokenised:list[list[str]]  = []
        self._bm25:     BM25Okapi | None = None

    def _tokenise(self, text: str) -> list[str]:
        text = text.lower()
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        tokens = text.split()
        # Remove stopwords (keep domain terms)
        STOPWORDS = {"the", "a", "an", "is", "are", "was", "were", "be",
                     "to", "of", "and", "or", "in", "it", "for", "on"}
        return [t for t in tokens if t not in STOPWORDS and len(t) > 1]

    def build_index(self, documents: list[dict]):
        """
        documents: [{id, content, source, metadata}]
        Call at startup and after knowledge base updates.
        """
        self._corpus    = documents
        self._tokenised = [self._tokenise(d["content"]) for d in documents]
        self._bm25      = BM25Okapi(self._tokenised, k1=1.5, b=0.75)
        print(f"BM25 index built: {len(documents)} documents")

    def search(self, query: str, k: int = 5) -> list[dict]:
        """Return top-k documents with BM25 scores"""
        if not self._bm25:
            raise RuntimeError("Index not built. Call build_index() first.")

        query_tokens = self._tokenise(query)
        scores       = self._bm25.get_scores(query_tokens)

        # Get top-k indices
        top_k        = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:k]

        results = []
        for idx, score in top_k:
            if score > 0:
                results.append({
                    **self._corpus[idx],
                    "bm25_score": round(float(score), 4),
                })
        return results

# ── Postgres FTS (for production, no in-memory index) ─────────────────
class PostgresFTSRetriever:
    """
    Uses Postgres tsvector for full-text search.
    No extra infrastructure — just Postgres.
    Scales to millions of documents.
    """

    def __init__(self, pool: asyncpg.Pool):
        self._pool = pool

    async def setup(self):
        """Create FTS index on chunks table"""
        async with self._pool.acquire() as conn:
            # Add tsvector column and index (run once)
            await conn.execute("""
                ALTER TABLE chunks
                ADD COLUMN IF NOT EXISTS fts_vector tsvector
                    GENERATED ALWAYS AS (
                        to_tsvector('english', coalesce(content, ''))
                    ) STORED;
                CREATE INDEX IF NOT EXISTS chunks_fts_idx
                    ON chunks USING GIN (fts_vector);
            """)

    async def search(self, query: str, org_id: str, k: int = 5) -> list[dict]:
        """BM25-ranked full-text search via Postgres ts_rank"""
        async with self._pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT
                    id,
                    content,
                    source,
                    ts_rank(fts_vector, query) AS rank,
                    ts_headline('english', content, query,
                                'MaxWords=30, MinWords=10') AS snippet
                FROM chunks,
                     plainto_tsquery('english', $1) query
                WHERE fts_vector @@ query
                  AND org_id = $2
                  AND is_active = TRUE
                ORDER BY rank DESC
                LIMIT $3
            """, query, org_id, k)

            return [
                {
                    "id":      str(row["id"]),
                    "content": row["content"],
                    "source":  row["source"],
                    "score":   float(row["rank"]),
                    "snippet": row["snippet"],    # highlighted match
                }
                for row in rows
            ]

# ── SQLite FTS5 — fully offline vectorless RAG ────────────────────────
import sqlite3
from pathlib import Path

class SQLiteFTS5Retriever:
    """
    Zero-infra vectorless RAG using SQLite FTS5.
    Works completely offline. Perfect for:
      - Edge devices (Raspberry Pi, embedded systems)
      - Privacy-sensitive data that cannot leave the machine
      - Demos with zero cloud dependencies
    """

    def __init__(self, db_path: str = "knowledge.db"):
        self.db_path = db_path
        self._setup()

    def _setup(self):
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS docs
            USING fts5(
                id UNINDEXED,
                content,
                source UNINDEXED,
                tokenize = 'porter ascii'  -- stemming: runs=run=running
            )
        """)
        conn.commit()
        conn.close()

    def index_document(self, doc_id: str, content: str, source: str):
        conn = sqlite3.connect(self.db_path)
        # REPLACE handles re-indexing same doc
        conn.execute("INSERT OR REPLACE INTO docs(id, content, source) VALUES (?,?,?)",
                     (doc_id, content, source))
        conn.commit()
        conn.close()

    def index_batch(self, documents: list[dict]):
        """Bulk index for efficiency"""
        conn = sqlite3.connect(self.db_path)
        conn.executemany(
            "INSERT OR REPLACE INTO docs(id, content, source) VALUES (?,?,?)",
            [(d["id"], d["content"], d["source"]) for d in documents],
        )
        conn.commit()
        conn.close()
        print(f"SQLite FTS5: indexed {len(documents)} documents")

    def search(self, query: str, k: int = 5) -> list[dict]:
        conn = sqlite3.connect(self.db_path)
        rows = conn.execute("""
            SELECT id, content, source,
                   rank,
                   snippet(docs, 1, '<b>', '</b>', '...', 20) as snippet
            FROM docs
            WHERE docs MATCH ?
            ORDER BY rank
            LIMIT ?
        """, (query, k)).fetchall()
        conn.close()

        return [
            {"id": r[0], "content": r[1], "source": r[2],
             "score": abs(r[3]), "snippet": r[4]}
            for r in rows
        ]

# ── Full vectorless RAG pipeline ─────────────────────────────────────
class VectorlessRAGPipeline:
    """BM25 retrieval → LLM generation. Zero vector embeddings."""

    def __init__(self, retriever, llm_client):
        self.retriever = retriever
        self.llm       = llm_client

    async def query(self, question: str, org_id: str = "default") -> dict:
        # Retrieve with BM25
        if isinstance(self.retriever, PostgresFTSRetriever):
            chunks = await self.retriever.search(question, org_id)
        else:
            chunks = self.retriever.search(question)

        if not chunks:
            return {
                "answer":    "I couldn't find relevant information for that question.",
                "retriever": "bm25",
                "chunks":    [],
            }

        context = "\n\n".join(
            f"[{c['source']}]\n{c.get('snippet') or c['content'][:500]}"
            for c in chunks
        )

        response = await groq.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system",
                 "content": "Answer only from the provided context. Cite sources. Be concise."},
                {"role": "user",
                 "content": f"Context:\n{context}\n\nQuestion: {question}"},
            ],
            max_tokens=400,
        )

        return {
            "answer":    response.choices[0].message.content,
            "retriever": "bm25",
            "chunks":    chunks,
        }
```

### 4.3 When Vectorless Wins

```
USE BM25 / VECTORLESS when:
  ✅ Exact term matching matters: error codes, legal citations, section numbers
     "Section 32B", "error code 404", "API endpoint /v2/users"
     Vector search: "Section 32B" ≈ "section 33B" (cosine similar!)
     BM25: "Section 32B" must appear exactly (keyword match)

  ✅ Small knowledge base (< 10K docs)
     Quality gap between BM25 and vector search: minimal
     Infrastructure complexity gap: massive (no Qdrant needed)

  ✅ No GPU / embedding API
     Fully offline: llama.cpp (local LLM) + SQLite FTS5 = zero cloud
     Privacy: data never leaves the machine

  ✅ Existing Postgres
     Add tsvector column + GIN index = vectorless RAG in 30 minutes
     No new infrastructure, no embedding costs

USE VECTOR SEARCH when:
  ✅ Semantic similarity: "how do I reset my password" ≈ "password recovery steps"
  ✅ Multilingual: user queries in Hindi, docs in English
  ✅ Large collections: > 50K documents
  ✅ Long-tail queries: uncommon phrasings that BM25 misses

HYBRID (BM25 + Vector) — best of both:
  Dense score + BM25 score combined (Reciprocal Rank Fusion)
  This is what Qdrant's hybrid search does under the hood
  Outperforms either alone on most benchmarks
```

---

## 5. Side-by-Side Comparison

```python
# app/query/router.py

from pydantic import BaseModel
from typing import Literal

class QueryClassification(BaseModel):
    query_type:  Literal["factual", "multi_hop", "comparison", "exact_term"]
    retriever:   Literal["vector", "graph", "bm25", "hybrid"]
    reasoning:   str

async def route_query(question: str) -> QueryClassification:
    """
    Classify the query → route to the right retriever.
    
    factual:    simple factual Q → vector or bm25
    multi_hop:  "how is X related to Y" → graph
    comparison: "compare X and Y" → graph
    exact_term: "section 32B" "error 429" → bm25
    """
    return await client.chat.completions.create(
        model="gpt-4o-mini",
        response_model=QueryClassification,
        messages=[{
            "role": "user",
            "content": f"""Classify this query for optimal retrieval.

Query: {question}

Choose retriever:
  graph:  multi-hop ("how is X related to Y"), comparisons ("compare X and Y"),
          relationship questions ("what does X depend on?")
  bm25:   exact terms (section numbers, error codes, API names, product IDs)
  vector: semantic similarity (paraphrase, synonym, concept questions)
  hybrid: general Q&A where you want both semantic + keyword coverage""",
        }],
    )

async def unified_query(question: str, org_id: str) -> dict:
    """Route to right retriever, return unified response"""
    classification = await route_query(question)

    match classification.retriever:
        case "graph":
            result = await graphrag_query(question)
        case "bm25":
            result = await vectorless_pipeline.query(question, org_id)
        case "vector":
            result = await vector_rag_query(question, org_id)
        case "hybrid":
            # Run BM25 + vector in parallel, merge with RRF
            bm25_r, vec_r = await asyncio.gather(
                vectorless_pipeline.query(question, org_id),
                vector_rag_query(question, org_id),
            )
            result = await rrf_merge_and_generate(question, bm25_r, vec_r)

    result["routing"] = classification.model_dump()
    return result

def reciprocal_rank_fusion(
    bm25_results:   list[dict],
    vector_results: list[dict],
    k: int = 60,
) -> list[dict]:
    """
    Merge BM25 and vector search results using Reciprocal Rank Fusion.
    Score = 1/(k + rank_bm25) + 1/(k + rank_vector)
    k=60 is the standard default (reduces influence of top-ranked outliers)
    """
    scores: dict[str, float] = {}
    docs:   dict[str, dict]  = {}

    for rank, doc in enumerate(bm25_results):
        key = doc["id"]
        scores[key] = scores.get(key, 0) + 1 / (k + rank + 1)
        docs[key]   = doc

    for rank, doc in enumerate(vector_results):
        key = doc["id"]
        scores[key] = scores.get(key, 0) + 1 / (k + rank + 1)
        docs[key]   = doc

    sorted_ids = sorted(scores, key=scores.get, reverse=True)
    return [{"rrf_score": scores[i], **docs[i]} for i in sorted_ids[:5]]
```

---

## 6. Combined Mode — Best of Both

```
GraphRAG + BM25 + Vector in one unified query endpoint:

POST /query
  {
    "question": "What are the async patterns used by teams that own Python services?",
    "mode": "auto"   // auto | graph | bm25 | vector | hybrid
  }

RESPONSE:
  {
    "answer": "The Data and Platform teams own Python services.
               Both use AsyncIO for concurrent operations.
               The Data Team uses it in the ML Pipeline for parallel
               batch processing. The Platform Team uses it in the API
               Gateway for high-throughput request handling. [Source: Team Wiki]",
    "routing": {
      "query_type": "multi_hop",
      "retriever": "graph",
      "reasoning": "Question asks about relationship: teams → services → patterns"
    },
    "graph_nodes": ["Data Team", "Platform Team", "Python", "AsyncIO", "ML Pipeline"],
    "graph_edges": 8
  }

COMPARISON TABLE (run this yourself on your dataset):
┌────────────────────────────────────────────────────────────────────────────────┐
│  Query                           │ Best Retriever │  Why                       │
│──────────────────────────────────┼────────────────┼────────────────────────────│
│ "What is the return policy?"     │ BM25 / Vector  │ Factual, keyword match fine │
│ "How is X related to Y?"         │ Graph          │ Relational, multi-hop       │
│ "Compare async in Py vs JS"      │ Graph          │ Comparison across entities  │
│ "Section 32B of the contract"    │ BM25           │ Exact term, BM25 wins       │
│ "password reset" → "reset pass"  │ Vector         │ Semantic, synonym matching  │
│ "What does Team X depend on?"    │ Graph          │ Graph traversal question    │
│ "error code 404 in our API"      │ BM25           │ Exact code matching         │
└────────────────────────────────────────────────────────────────────────────────┘
```

---

## 7. API Reference

```
── GRAPH ENDPOINTS ───────────────────────────────────────────────────
POST  /ingest/graph          Ingest docs → extract KG → store Neo4j
GET   /graph/stats           Node/edge counts
GET   /graph/entity/{name}   Entity + its relationships
GET   /graph/path            Find shortest path between two entities
                             ?from=Python&to=ML+Pipeline&max_hops=3
POST  /query/graph           Graph RAG query

── VECTORLESS ENDPOINTS ─────────────────────────────────────────────
POST  /ingest/bm25           Index documents in BM25 / Postgres FTS
POST  /query/bm25            BM25 / FTS query
POST  /query/sqlite          SQLite FTS5 query (offline mode)

── UNIFIED ENDPOINT ─────────────────────────────────────────────────
POST  /query                 Auto-route: query → router → best retriever
      Body: {question, mode: "auto"|"graph"|"bm25"|"vector"|"hybrid"}

── COMPARE ──────────────────────────────────────────────────────────
POST  /query/compare         Run same query on ALL retrievers, return comparison
      Body: {question}
      Response: {graph: {...}, bm25: {...}, vector: {...}, winner: "graph"}
```

---

## 8. Database Schema

```sql
-- ── Neo4j (graph) — defined as Cypher constraints ──────────────────
-- CREATE CONSTRAINT entity_name_unique ON (e:Entity) ASSERT e.name IS UNIQUE;

-- ── Postgres (chunks for BM25 + metadata) ──────────────────────────
CREATE TABLE documents (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source      TEXT NOT NULL,
    title       TEXT,
    raw_content TEXT NOT NULL,
    chunk_count INTEGER DEFAULT 0,
    indexed_at  TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE chunks (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    doc_id      UUID REFERENCES documents(id),
    source      TEXT NOT NULL,
    content     TEXT NOT NULL,
    chunk_index INTEGER,
    -- BM25 / FTS column
    fts_vector  tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED,
    created_at  TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX chunks_fts_idx ON chunks USING GIN (fts_vector);

-- ── Graph extraction log (for debugging + audit) ───────────────────
CREATE TABLE kg_extractions (
    id         UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    chunk_id   UUID REFERENCES chunks(id),
    entities   JSONB,
    relations  JSONB,
    model_used TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- ── Query log (compare retriever quality over time) ─────────────────
CREATE TABLE query_log (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    question    TEXT NOT NULL,
    retriever   TEXT NOT NULL,      -- graph | bm25 | vector | hybrid
    answer      TEXT,
    latency_ms  INTEGER,
    thumbs_up   BOOLEAN,            -- user feedback
    created_at  TIMESTAMPTZ DEFAULT NOW()
);
```

---

## 9. Folder Structure

```
graphmind/
├── app/
│   ├── main.py
│   ├── graph_rag/
│   │   ├── kg_builder.py         ← entity + relationship extraction
│   │   ├── graph_retriever.py    ← Neo4j traversal, path finding
│   │   └── pipeline.py           ← full GraphRAG query pipeline
│   ├── vectorless/
│   │   ├── bm25_retriever.py     ← in-memory BM25 (rank-bm25)
│   │   ├── postgres_fts.py       ← Postgres tsvector/tsquery
│   │   ├── sqlite_fts5.py        ← SQLite FTS5 (fully offline)
│   │   └── pipeline.py           ← vectorless RAG query pipeline
│   ├── query/
│   │   ├── router.py             ← LLM-based query classifier
│   │   └── rrf.py                ← Reciprocal Rank Fusion merger
│   └── api/
│       ├── ingest.py
│       └── query.py
├── scripts/
│   ├── build_kg.py               ← one-shot KG construction from docs
│   ├── index_bm25.py             ← build BM25 index from docs
│   └── compare_retrievers.py     ← benchmark all 3 on a query set
├── data/
│   ├── sample_docs/              ← 10-20 sample markdown/PDF docs
│   └── queries.json              ← test queries with expected answers
├── notebooks/
│   └── retriever_comparison.ipynb ← interactive benchmarking notebook
├── docker-compose.yml            ← neo4j + postgres + api
└── README.md
```

---

## 10. Phase-by-Phase Build Plan

### Phase 0 — Setup (1 day)
```
[ ] docker-compose: Neo4j + Postgres + FastAPI
[ ] Load 10-20 sample markdown docs into /data/sample_docs/
[ ] Postgres chunks table created, FTS index applied
[ ] Neo4j running, Cypher constraint applied
Showable: Neo4j browser at localhost:7474, Postgres FTS5 searchable
```

### Phase 1 — Vectorless RAG (3 days)
```
[ ] BM25Retriever: build_index() + search() tested on sample docs
[ ] PostgresFTSRetriever: setup() + search() with ts_headline
[ ] SQLiteFTS5Retriever: fully offline, zero dependencies
[ ] VectorlessRAGPipeline: BM25 → LLM → answer
[ ] POST /query/bm25 endpoint working
[ ] POST /query/sqlite endpoint working
Showable: "Section 3.2 of the return policy" → correct chunk found
          Same query on BM25 vs vector (if you have both): compare results
```

### Phase 2 — Knowledge Graph Construction (4 days)
```
[ ] extract_kg_from_chunk() using GPT-4o-mini + instructor
[ ] store_in_neo4j(): nodes + edges upserted
[ ] build_knowledge_graph(): batch pipeline over all docs
[ ] GET /graph/entity/{name} returns node + its relationships
[ ] Neo4j browser: visualise the extracted graph
[ ] GET /graph/stats returns node/edge counts
Showable: Neo4j browser showing entity network from your docs
          "Python" node → connected to [AsyncIO, FastAPI, Data Team, ...]
```

### Phase 3 — Graph Retrieval + Query (4 days)
```
[ ] plan_graph_query(): LLM decomposes query into traversal plan
[ ] traverse_graph(): Neo4j variable-depth traversal
[ ] find_path_between(): shortest path query
[ ] graphrag_query(): full pipeline (plan → traverse → generate)
[ ] POST /query/graph endpoint
[ ] POST /query/compare: run all retrievers, return side-by-side
Showable: "Compare how Python and JS handle async in our codebase"
          → Graph retrieves connected entities, LLM synthesises comparison
```

### Phase 4 — Router + Unified Endpoint (2 days)
```
[ ] route_query(): LLM classifies query → retriever
[ ] unified_query(): dispatches to right retriever
[ ] rrf_merge_and_generate(): Reciprocal Rank Fusion for hybrid
[ ] POST /query with mode=auto
[ ] Benchmark: query log table, /query/compare endpoint
Showable: demo with 5 query types, each routed to the right retriever
          Comparison table: which retriever won and why
```

---

## 11. Resume Deliverables

```
GitHub repo:  graphmind — with architecture diagram, Neo4j screenshots
Notebook:     retriever_comparison.ipynb — BM25 vs vector vs graph on same queries
Benchmark:    "On multi-hop queries, GraphRAG answered 73% correctly vs 41% for vector"
Demo:         /query/compare endpoint showing all 3 retrievers side by side

Interview story (60 seconds):
"Standard vector RAG fails on multi-hop questions — questions that require
 traversing relationships, like 'compare how Python and JavaScript are used
 across our engineering teams.' Vector search returns the top-5 similar chunks
 globally, but it can't traverse: Python → used by → Data Team →
 also uses → ML Pipeline → which uses → AsyncIO.

 I built GraphRAG on top of Neo4j. During ingestion, an LLM extracts entities
 and relationships from each document chunk — things like 'FastAPI BUILT_WITH Python'
 or 'Data Team OWNS ML Pipeline.' These go into a graph database. At query time,
 another LLM decomposes the question into a traversal plan: which entities to start
 from, how many hops, which relationship types. Then I traverse the graph and feed
 the connected nodes and edges as context to the generation LLM.

 I also implemented vectorless RAG — BM25 via Postgres FTS and SQLite FTS5 for
 fully offline use. On exact-term queries like 'Section 32B' or 'error code 429',
 BM25 outperforms vector search because it does exact keyword matching rather than
 semantic similarity.

 The system has a query router that classifies each question and sends it to the
 right retriever. On our benchmark, graph retrieval had 73% accuracy on multi-hop
 questions versus 41% for vector-only."
```

---

*GraphMind — GraphRAG & Vectorless RAG Minor Project · Phase 2 Deep Dive*
*Neo4j · BM25 · Postgres FTS · SQLite FTS5 · Reciprocal Rank Fusion · Query Routing*
