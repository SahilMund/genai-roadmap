# 📚 Phase 2 — RAG Systems
> **Complete Study Notes | Interview Prep | Real-Life Examples | Beginner-Friendly Explanations**  
> Part of: GenAI + LLMOps Engineering Roadmap 2026  
> Estimated time: 70–90 hrs over 4–5 weeks  
> Prerequisites: Phase 0 (Embeddings, Transformers) + Phase 1 (LLM APIs, Structured Outputs)

---

## 📑 Table of Contents

1. [What is RAG and Why It Matters](#-what-is-rag)
2. [2.1 — Data Ingestion Pipelines](#21--data-ingestion-pipelines)
3. [2.2 — Chunking Strategies](#22--chunking-strategies)
4. [2.3 — Embeddings Deep Dive](#23--embeddings-deep-dive)
5. [2.4 — Vector Databases](#24--vector-databases)
6. [2.5 — Advanced Retrieval](#25--advanced-retrieval)
7. [2.6 — RAG Architectures](#26--rag-architectures)
8. [2.7 — RAG Evaluation](#27--rag-evaluation)
9. [Phase 2 Projects](#-phase-2-projects)
10. [Interview Cheat Sheet](#-interview-cheat-sheet)
11. [Quick Revision Cards](#-quick-revision-cards)

---

## 🚀 What is RAG?

### 🧑‍🎓 Explain Like I'm in BTech 2nd Year

Picture this: you're appearing for an open-book exam. The question is something very specific from a textbook the teacher updated just last week. Your memory (the LLM) was trained months ago — it has no idea about last week's updates. But if you could bring the textbook into the exam and look up the answer, you'd get it right.

**RAG = bringing the right pages of the right book into the exam at the right time.**

```
Without RAG:
  User: "What is our company's current leave policy?"
  LLM:  "I don't have access to your company's internal policies." ❌
  OR:
  LLM:  "You are entitled to 15 days of paid leave per year..." ← hallucinated! ❌

With RAG:
  Step 1: Search company HR documents for "leave policy"
  Step 2: Find: "Employees are entitled to 18 days PL + 12 days CL per year..."
  Step 3: Feed this chunk to the LLM as context
  Step 4: LLM: "Based on our HR policy, you get 18 days Privilege Leave
           and 12 days Casual Leave per year." ✅ Grounded. Citable.
```

---

### 📖 Formal Definition

> RAG (Retrieval-Augmented Generation) is an architecture pattern that enhances LLM responses by first retrieving relevant information from an external knowledge base and injecting it into the prompt as context before generation. It grounds the model's output in actual data rather than parametric memory.

RAG was introduced by Meta AI researchers (Lewis et al.) in 2020 and has since become the **default architecture** for knowledge-grounded AI applications in production.

---

### ⚔️ RAG vs Fine-Tuning vs Prompt Engineering

| Approach | When to use | Cost | Knowledge update | Hallucination risk |
|----------|------------|------|-----------------|-------------------|
| **Prompt Engineering** | Simple factual tasks within training data | Cheapest | None | High |
| **RAG** | Dynamic/private knowledge, policies, docs | Low | Minutes | Low (grounded) |
| **Fine-Tuning** | Domain tone/style, tasks not in base model | High | Retrain needed | Medium |
| **RAG + Fine-Tune** | Best of both: tone + live knowledge | High | Partial | Lowest |

**Rule of thumb:** Always try RAG first. Fine-tune only if RAG alone can't solve the quality problem.

---

[Why do we need RAG/fine tuning?](https://medium.com/@saqibbuzdar/why-do-we-need-rag-or-finetuning-llm-isnt-alone-sufficient-1201f46eed33)

### 🗺️ The Full RAG System — Two Pipelines

```
╔══════════════════════════════════════════════════════════════════════╗
║           INGESTION PIPELINE  (runs offline, once or scheduled)     ║
╠══════════════════════════════════════════════════════════════════════╣
║  Raw Docs (PDF/DOCX/HTML/API)                                        ║
║       ↓                                                              ║
║  [Loader]   → Extract text + preserve metadata (source, page, date) ║
║       ↓                                                              ║
║  [Cleaner]  → Strip noise, fix encoding, remove headers/footers     ║
║       ↓                                                              ║
║  [Chunker]  → Split into semantic units with overlap                ║
║       ↓                                                              ║
║  [Embedder] → Convert each chunk to a dense vector                  ║
║       ↓                                                              ║
║  [VectorDB] → Store vectors + original text + metadata              ║
╚══════════════════════════════════════════════════════════════════════╝

╔══════════════════════════════════════════════════════════════════════╗
║           QUERY PIPELINE  (runs real-time per user request)         ║
╠══════════════════════════════════════════════════════════════════════╣
║  User: "What is the refund timeline for electronics?"               ║
║       ↓                                                              ║
║  [Query Rewriter] → Clarify/expand the user's query (optional)      ║
║       ↓                                                              ║
║  [Embedder]       → Convert query to dense vector                   ║
║       ↓                                                              ║
║  [VectorDB]       → Find top-K most similar chunks (cosine/hybrid)  ║
║       ↓                                                              ║
║  [Reranker]       → Re-score candidates with cross-encoder          ║
║       ↓                                                              ║
║  [Context Builder]→ Assemble final prompt with retrieved context    ║
║       ↓                                                              ║
║  [LLM]            → Generate grounded answer                        ║
║       ↓                                                              ║
║  Answer + Source citations                                           ║
╚══════════════════════════════════════════════════════════════════════╝
```

---

## 2.1 — Data Ingestion Pipelines

### 🧑‍🎓 Explain Like I'm in BTech 2nd Year

Before the RAG system can answer anything, you need to feed it documents. This is like loading books into a library before students can borrow them. The hard part: real-world documents are messy. PDFs have tables, images, scanned pages, multi-column layouts. HTML has ads, menus, and tracking pixels. Your ingestion pipeline needs to handle all of this and produce clean, well-structured text.

Think of it as a factory assembly line:
```
Raw material (PDFs) → Clean → Cut into pieces → Label each piece → Store in warehouse
```

---

### 📄 Document Loaders

#### PDF — PyMuPDF (fastest, best layout preservation)

```python
# pip install pymupdf
import fitz  # PyMuPDF

def load_pdf_with_metadata(pdf_path: str) -> list[dict]:
    """Load PDF preserving page-level metadata for filtering later"""
    doc = fitz.open(pdf_path)
    pages = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text()

        # Skip mostly empty pages (headers/footers/blank pages)
        if len(text.strip()) < 50:
            continue

        pages.append({
            "text": text,
            "metadata": {
                "source": pdf_path,
                "page": page_num + 1,
                "total_pages": len(doc),
                "file_name": pdf_path.split("/")[-1],
            }
        })

    doc.close()
    return pages

pages = load_pdf_with_metadata("company_policy.pdf")
print(f"Loaded {len(pages)} pages from PDF")
```

#### PDF with Tables — pdfplumber

```python
# pip install pdfplumber
import pdfplumber

def load_pdf_tables(pdf_path: str) -> list[dict]:
    """Extract both text and tables from PDFs"""
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text() or ""

            # Extract tables and convert to readable Markdown format
            tables = page.extract_tables()
            table_markdown = ""
            for table in tables:
                if not table:
                    continue
                # Header row
                header = " | ".join([str(c or "") for c in table[0]])
                separator = " | ".join(["---"] * len(table[0]))
                rows = "\n".join(
                    " | ".join([str(c or "") for c in row])
                    for row in table[1:]
                )
                table_markdown += f"\n{header}\n{separator}\n{rows}\n"

            combined = text + ("\n\n" + table_markdown if table_markdown else "")

            pages.append({
                "text": combined,
                "metadata": {
                    "source": pdf_path,
                    "page": i + 1,
                    "has_tables": len(tables) > 0,
                }
            })
    return pages
```

#### Production-Grade — Unstructured.io

```python
# pip install unstructured[pdf,docx,html]
from unstructured.partition.auto import partition
from unstructured.documents.elements import (
    Title, NarrativeText, Table, Image, ListItem
)

def load_with_unstructured(file_path: str) -> list[dict]:
    """
    Best for: complex PDFs, scanned docs, mixed formats
    Handles: tables, images, multi-column layouts, OCR
    """
    elements = partition(
        filename=file_path,
        strategy="hi_res",           # uses vision model for scanned PDFs
        infer_table_structure=True,  # extract tables as HTML
        include_page_breaks=True,
    )

    chunks = []
    for element in elements:
        if isinstance(element, (Title, NarrativeText, ListItem)):
            chunks.append({
                "text": str(element),
                "type": type(element).__name__,
                "metadata": element.metadata.to_dict(),
            })
        elif isinstance(element, Table):
            chunks.append({
                "text": element.metadata.text_as_html,  # table as HTML
                "type": "Table",
                "metadata": element.metadata.to_dict(),
            })
    return chunks
```

#### HTML / Web Pages

```python
# pip install httpx beautifulsoup4
import httpx
from bs4 import BeautifulSoup
import asyncio

async def load_webpage_clean(url: str) -> dict:
    """Load a webpage, removing navigation/ads/footer noise"""
    async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
        r = await client.get(url, headers={"User-Agent": "RAG-Bot/1.0"})
        r.raise_for_status()

    soup = BeautifulSoup(r.text, "html.parser")

    # Remove noise elements
    for tag in soup(["nav", "footer", "header", "script",
                     "style", "aside", "advertisement", "cookie"]):
        tag.decompose()

    # Prefer main article content
    main = (
        soup.find("main") or
        soup.find("article") or
        soup.find(id="content") or
        soup.body
    )
    text = main.get_text(separator="\n", strip=True) if main else ""

    return {
        "text": text,
        "metadata": {
            "source": url,
            "title": soup.title.string if soup.title else "",
            "doc_type": "webpage",
        }
    }

# Bulk sitemap crawl
async def crawl_from_sitemap(sitemap_url: str, limit: int = 50) -> list[dict]:
    import xml.etree.ElementTree as ET
    async with httpx.AsyncClient() as c:
        r = await c.get(sitemap_url)
    root = ET.fromstring(r.text)
    ns = "{http://www.sitemaps.org/schemas/sitemap/0.9}"
    urls = [loc.text for loc in root.iter(f"{ns}loc")][:limit]

    results = await asyncio.gather(
        *[load_webpage_clean(u) for u in urls],
        return_exceptions=True
    )
    return [r for r in results if isinstance(r, dict)]
```

#### LangChain Loaders (convenience wrappers)

```python
from langchain_community.document_loaders import (
    PyMuPDFLoader,
    Docx2txtLoader,
    CSVLoader,
    JSONLoader,
    WebBaseLoader,
    DirectoryLoader,
)

# Load an entire folder of PDFs in parallel
loader = DirectoryLoader(
    "./knowledge_base/",
    glob="**/*.pdf",
    loader_cls=PyMuPDFLoader,
    show_progress=True,
    use_multithreading=True,
    max_concurrency=8,
)
all_docs = loader.load()
print(f"Loaded {len(all_docs)} document pages")
```

---

### 🏭 Production ETL Design

```
Each stage is an independent Celery worker — scale them separately:

STAGE 1: Extract       STAGE 2: Clean         STAGE 3: Chunk
  PDF Loader     ──▶    Strip noise     ──▶    Split text
  HTML Loader           Fix encoding           Add metadata
  API Poller            Dedup check            Overlap
  DOCX Loader           Length filter          Parent/child split

STAGE 4: Embed         STAGE 5: Store
  Batch embed    ──▶    Qdrant upsert
  Rate limit            Progress update in Redis
  Retry on fail         Hash stored for dedup
```

```python
import hashlib
from redis.asyncio import Redis

async def should_reingest(doc_id: str, content: str, redis: Redis) -> bool:
    """Skip ingestion if document content hasn't changed (hash-based dedup)"""
    new_hash = hashlib.sha256(content.encode()).hexdigest()
    stored = await redis.get(f"doc_hash:{doc_id}")

    if stored and stored.decode() == new_hash:
        return False  # unchanged — skip

    await redis.set(f"doc_hash:{doc_id}", new_hash, ex=86400 * 30)  # 30d TTL
    return True  # new or changed — process

# Celery task with progress tracking
from celery import Celery
app = Celery("rag_ingestion", broker="redis://localhost:6379/0")

@app.task(bind=True)
def ingest_document(self, doc_id: str, file_path: str, org_id: str):
    """Full ingestion pipeline as a Celery task"""
    self.update_state(state="LOADING", meta={"progress": 10})
    pages = load_pdf_with_metadata(file_path)

    self.update_state(state="CHUNKING", meta={"progress": 30})
    chunks = chunk_documents(pages)

    self.update_state(state="EMBEDDING", meta={"progress": 60})
    embeddings = embed_chunks(chunks)

    self.update_state(state="STORING", meta={"progress": 85})
    store_in_qdrant(chunks, embeddings, org_id)

    self.update_state(state="DONE", meta={"progress": 100})
    return {"doc_id": doc_id, "chunks_created": len(chunks)}
```

---

### 💼 Interview Questions — Data Ingestion

**Q1: What are the biggest challenges with PDF ingestion for RAG?**
> PDFs are layout-based, not semantic documents. Common problems: multi-column layouts merge text incorrectly; tables are fragmented into disconnected cells; text inside images (scanned pages) needs OCR; headers and footers on every page pollute content; footnotes interrupt the main flow. Production RAG systems use Unstructured.io with high-res strategy for complex documents, pdfplumber for table-heavy PDFs, and PyMuPDF for fast text-focused extraction. Always filter pages with fewer than 50 characters — they're likely decorative.

**Q2: Why is hash-based deduplication critical in ingestion pipelines?**
> Without deduplication, re-running the ingestion pipeline creates duplicate vectors. When a user queries the system, duplicate chunks appear in results — wasting context window space, inflating confidence, and increasing cost. SHA-256 hash of the document content is stored on first ingestion. On subsequent runs, if the hash matches, the document is skipped. This also enables incremental ingestion — only changed documents are reprocessed.

---

## 2.2 — Chunking Strategies

### 🧑‍🎓 Explain Like I'm in BTech 2nd Year

You have a 500-page technical manual. You can't feed all 500 pages to the LLM — context window limit + cost. So you cut it into pieces. But *how* you cut matters enormously.

Imagine photocopying pages from a textbook and giving them to a student. If you cut mid-sentence — the copy is useless. If the copy is too short — it lacks context. If too long — the relevant part is buried.

**Chunking = deciding how to cut text so each piece is independently meaningful and retrievable.**

```
Bad chunking:
  Chunk 17: "...The return policy for electronics requires the original
             packaging. In addition, customers must present a"
  Chunk 18: "valid receipt dated within 30 days of purchase. Exceptions..."

No single chunk contains the full rule — retrieval fails both.

Good chunking (with overlap):
  Chunk 17: "...The return policy for electronics requires the original
             packaging. In addition, customers must present a valid
             receipt dated within 30 days of purchase."
  Chunk 18: "...customers must present a valid receipt dated within 30
             days of purchase. Exceptions apply for..."

Both chunks contain the core fact — retrieval succeeds.
```

---

### ✂️ Strategy 1: Fixed-Size Chunking

Split every N characters regardless of content. Simple. Fast. Low quality.

```python
def fixed_size_chunk(text: str, size: int = 1000, overlap: int = 200) -> list[str]:
    chunks, start = [], 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        start += size - overlap
    return chunks

# Problem demo:
text = "The transformer architecture was revolutionary. It replaced recurrent networks."
chunks = fixed_size_chunk(text, size=40, overlap=10)
# ['The transformer architecture was revolution', 'revolution ary. It replaced recurrent ', ...]
# "revolutionary" is cut in half → broken word at chunk boundary
```

**Use when:** Quick prototype, pre-structured text (e.g., already one paragraph per line).

---

### ✂️ Strategy 2: Recursive Character Splitting (Default)

Tries boundaries in order: `\n\n` → `\n` → `. ` → ` ` → character. Falls back only if needed.

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", ". ", " ", ""],
    length_function=len,
)

from langchain.schema import Document
docs = splitter.create_documents(
    texts=[full_text],
    metadatas=[{"source": "policy.pdf", "page": 1, "org_id": "acme"}]
)

for d in docs[:2]:
    print(f"Length: {len(d.page_content)} | Preview: {d.page_content[:80]}...")
```

**Use when:** General production RAG. Your safe default for most document types.

---

### ✂️ Strategy 3: Sentence-Level Chunking

Split at sentence boundaries. Preserves grammatical units.

```python
import nltk
nltk.download("punkt", quiet=True)

def sentence_chunk(text: str, sentences_per_chunk: int = 5, overlap: int = 1) -> list[str]:
    sentences = nltk.sent_tokenize(text)
    chunks = []
    for i in range(0, len(sentences), sentences_per_chunk - overlap):
        chunk = " ".join(sentences[i : i + sentences_per_chunk])
        if chunk.strip():
            chunks.append(chunk)
    return chunks

# Clean sentence boundaries — no mid-sentence cuts
```

**Use when:** Well-structured prose documents where sentence integrity matters.

---

### ✂️ Strategy 4: Semantic Chunking

Use embeddings to detect where the *meaning* changes, then split there.

```python
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings

chunker = SemanticChunker(
    embeddings=OpenAIEmbeddings(model="text-embedding-3-small"),
    breakpoint_threshold_type="percentile",
    breakpoint_threshold_amount=95,  # split at top 5% most dissimilar transitions
)

text = """
The transformer architecture uses attention mechanisms to process sequences.
Self-attention allows every token to directly attend to every other token.
This enables parallel computation unlike sequential RNNs.

In contrast, convolutional neural networks excel at spatial feature extraction.
CNNs use learnable filters to detect local patterns in fixed-size receptive fields.
They process images hierarchically from low-level to high-level features.
"""

chunks = chunker.create_documents([text])
# Result: splits at the blank line (topic shift from Transformers to CNNs)
for c in chunks:
    print(f"--- Chunk ({len(c.page_content)} chars) ---")
    print(c.page_content[:120])
```

**Use when:** High-quality production RAG, technical documents with clear topic shifts.

---

### ✂️ Strategy 5: Parent-Child Chunking ⭐ Production Default

**The most important strategy.** Solves the precision vs context trade-off.

```
The dilemma:
  Small chunks (400 chars) → Precise retrieval (query matches closely)
                           → BUT: poor context for LLM (answer is fragmented)

  Large chunks (2000 chars) → Rich context for LLM
                            → BUT: poor retrieval (query signal diluted)

Parent-child solution:
  Index:    Small child chunks (400 chars) → stored in vector DB for retrieval
  Store:    Large parent chunks (2000 chars) → stored in docstore, linked to children

  At query time:
    1. Vector search finds the most similar CHILD chunks (precise match)
    2. Fetch their PARENT chunks (full context for the LLM)
    3. LLM answers using rich parent context
```

```python
from langchain.storage import InMemoryStore
from langchain.retrievers import ParentDocumentRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = QdrantVectorStore.from_existing_collection(
    embedding=embeddings,
    collection_name="child_chunks",
    url="http://localhost:6333",
)

# Splitters
child_splitter  = RecursiveCharacterTextSplitter(chunk_size=400,  chunk_overlap=50)
parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)

# Docstore holds parents; vectorstore holds children
docstore = InMemoryStore()  # swap for Redis/Postgres in production

retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=docstore,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
)

retriever.add_documents(documents)  # indexes children, stores parents

# Query — finds child, returns parent
results = retriever.invoke("What is the electronics return window?")
print(len(results[0].page_content))  # ~2000 chars — full context returned
```

---

### ✂️ Strategy 6: Late Chunking (2024 technique)

Embed the full document first, then pool embeddings to chunk level. Each chunk retains full-document context in its embedding.

```
Standard chunking:
  Chunk: "The CEO resigned yesterday."
  Embedded in isolation → doesn't know which company's CEO

Late chunking:
  Full doc embedded first: "Tata Motors... CEO Shailesh Chandra... resigned..."
  Chunk embedding pooled from full-doc context
  → "The CEO resigned" embedding knows it's about Tata Motors ← better retrieval
```

Currently experimental in production; supported in newer sentence-transformer models.

---

### ✂️ Strategy 7: Code-Aware Chunking

For technical documentation and code repositories.

```python
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter

# Python-aware splitter — respects function/class boundaries
python_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,
    chunk_size=1500,
    chunk_overlap=100,
)

# JavaScript, Go, Java, Rust, etc. also supported
js_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.JS,
    chunk_size=1500,
    chunk_overlap=100,
)

# Splits at class → function → block → line boundaries
# Never cuts mid-function body
python_code = """
def calculate_rag_score(retrieved: list, ground_truth: str) -> float:
    if not retrieved:
        return 0.0
    relevant = sum(1 for doc in retrieved if ground_truth in doc)
    return relevant / len(retrieved)

class RAGEvaluator:
    def __init__(self, threshold: float = 0.75):
        self.threshold = threshold
"""
chunks = python_splitter.create_documents([python_code])
```

---

### 📊 Metadata Enrichment — Non-Negotiable in Production

Every chunk MUST carry rich metadata. Metadata enables pre-filtering before vector search — orders of magnitude faster than post-filtering.

```python
from langchain.schema import Document
from datetime import datetime

def enrich_chunk(text: str, source_metadata: dict, chunk_idx: int) -> Document:
    return Document(
        page_content=text,
        metadata={
            # Source information
            "source":       source_metadata["file_path"],
            "file_name":    source_metadata["file_name"],
            "page":         source_metadata.get("page", 0),
            "section":      source_metadata.get("section", ""),

            # Document classification
            "doc_type":     source_metadata.get("doc_type", "general"),  # policy/faq/product
            "language":     source_metadata.get("language", "en"),

            # Freshness
            "last_updated": source_metadata.get("updated", datetime.utcnow().isoformat()),
            "ingested_at":  datetime.utcnow().isoformat(),

            # Multi-tenancy — CRITICAL
            "org_id":       source_metadata["org_id"],
            "user_id":      source_metadata.get("user_id"),

            # Parent-child linkage
            "parent_doc_id": source_metadata.get("doc_id"),
            "chunk_index":   chunk_idx,
        }
    )

# Usage in Qdrant query — filter BEFORE vector search
results = qdrant_client.search(
    collection_name="rag_docs",
    query_vector=query_embedding,
    query_filter=Filter(must=[
        FieldCondition(key="org_id",   match=MatchValue(value="acme")),
        FieldCondition(key="doc_type", match=MatchValue(value="policy")),
    ]),
    limit=10,
)
```

---

### ⚔️ Chunking Strategy Comparison

| Strategy | Semantic Quality | Speed | Cost | When to Use |
|----------|----------------|-------|------|-------------|
| Fixed-size | Low | Fastest | Lowest | Prototyping only |
| Recursive character | Medium | Fast | Low | **General production default** |
| Sentence-level | Medium-High | Medium | Low | Clean prose docs |
| Semantic | High | Slow | Higher | Quality-critical RAG |
| **Parent-child** | **Highest** | Medium | Medium | **Production default** |
| Late chunking | High | Slow | Higher | Experimental, context-rich |
| Code-aware | High (for code) | Fast | Low | Code repos/tech docs |

---

### 💼 Interview Questions — Chunking

**Q1: What is parent-child chunking and why is it the production default?**
> It solves the precision-vs-context dilemma. Small chunks (400 chars) give high cosine similarity with the query — precise retrieval. But small chunks lack context for the LLM to generate a good answer. Large chunks (2000 chars) give the LLM enough context but dilute the embedding signal, hurting retrieval. Parent-child stores small child chunks in the vector store for retrieval and large parent chunks in a document store. At query time, we find precise child matches, then retrieve their richer parent chunks for generation.

**Q2: Why is metadata as important as the chunk text itself?**
> Metadata enables pre-filtering — constraining the vector search to a relevant subset before comparing embeddings. In a multi-tenant system with 10M chunks across 1000 organisations, searching all 10M vectors per query is slow and leaks data. With `org_id` metadata filtering, we search only the 10K chunks belonging to the querying org. This is orders of magnitude faster and enforces data isolation. Without metadata, you'd need separate vector DB collections per tenant — operationally expensive.

---

## 2.3 — Embeddings Deep Dive

### 🧑‍🎓 Explain Like I'm in BTech 2nd Year

Embeddings convert text into coordinates in a high-dimensional space. Similar meaning = nearby coordinates. Dissimilar meaning = far apart.

```
"dog"   → [0.2, 0.8, 0.1, 0.9 ...]  ← 1536 numbers
"puppy" → [0.21, 0.79, 0.11, 0.88 ...] ← very close!
"SQL"   → [0.9, 0.1, 0.7, 0.3 ...]  ← far away

Cosine similarity:
  cos("dog", "puppy") = 0.94  ← very similar
  cos("dog", "SQL")   = 0.18  ← very different

RAG uses this: embed the user question, find chunks with high cosine 
similarity (close coordinates = related meaning).
```

---

### 🏆 Embedding Models in 2026

#### OpenAI Embeddings

```python
from openai import AsyncOpenAI
import tiktoken

client = AsyncOpenAI()

async def embed_texts(texts: list[str], model: str = "text-embedding-3-small") -> list[list[float]]:
    """Embed a batch of texts"""
    response = await client.embeddings.create(model=model, input=texts)
    return [item.embedding for item in sorted(response.data, key=lambda x: x.index)]

# text-embedding-3-small: 1536 dims, $0.02/1M tokens — default choice
# text-embedding-3-large: 3072 dims, $0.13/1M tokens — higher quality

# Matryoshka: truncate to smaller dims without retraining
async def embed_matryoshka(texts: list[str], dims: int = 256) -> list[list[float]]:
    """Use fewer dimensions — 6× smaller index, ~90% quality retained"""
    response = await client.embeddings.create(
        model="text-embedding-3-small",
        input=texts,
        dimensions=dims,   # 256, 512, 768, 1024, or 1536
    )
    return [item.embedding for item in response.data]

# Cost tracking
enc = tiktoken.get_encoding("cl100k_base")
token_count = sum(len(enc.encode(t)) for t in texts)
cost = token_count * 0.02 / 1_000_000
print(f"Embedding cost: ${cost:.6f} for {token_count} tokens")
```

#### Open-Source Embeddings — BGE-M3 (Best Free Option)

```python
from sentence_transformers import SentenceTransformer
import numpy as np

# BGE-M3: multilingual, multi-granularity, 1024 dims, free
model = SentenceTransformer("BAAI/bge-m3")

def embed_batch_local(texts: list[str], batch_size: int = 64) -> np.ndarray:
    """Embed texts locally — free, private, works offline"""
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,   # L2 normalize for cosine similarity
        device="cuda" if torch.cuda.is_available() else "cpu",
        convert_to_numpy=True,
    )
    return embeddings   # shape: (len(texts), 1024)

# nomic-embed-text: good alternative, 768 dims
nomic_model = SentenceTransformer(
    "nomic-ai/nomic-embed-text-v1.5",
    trust_remote_code=True
)
embeddings = nomic_model.encode(
    texts,
    task_name="search_document",  # or "search_query" for queries
)
```

#### Cohere Embeddings — Best for Multilingual Production

```python
import cohere

co = cohere.Client("COHERE_API_KEY")

# KEY INSIGHT: use different input_type for documents vs queries
doc_embeddings = co.embed(
    texts=document_texts,
    model="embed-multilingual-v3.0",
    input_type="search_document",  # ← for indexing documents
).embeddings

query_embedding = co.embed(
    texts=["What is the return policy?"],
    model="embed-multilingual-v3.0",
    input_type="search_query",     # ← for searching
).embeddings[0]

# Asymmetric embedding: doc and query spaces are optimised to match each other
# Using wrong input_type gives noticeably worse retrieval
```

---

### 📦 Batch Embedding 10,000+ Documents

```python
import asyncio
from openai import AsyncOpenAI

client = AsyncOpenAI()

async def embed_all_chunks(
    chunks: list[str],
    batch_size: int = 100,
    max_concurrent: int = 5,
    model: str = "text-embedding-3-small",
) -> list[list[float]]:
    """
    Embed large document sets efficiently with:
    - Batching (100 texts per API call)
    - Concurrency control (5 parallel calls max)
    - Automatic retry on rate limit
    """
    batches = [chunks[i:i+batch_size] for i in range(0, len(chunks), batch_size)]
    semaphore = asyncio.Semaphore(max_concurrent)

    async def embed_one_batch(batch: list[str]) -> list:
        async with semaphore:
            for attempt in range(3):
                try:
                    resp = await client.embeddings.create(model=model, input=batch)
                    return [d.embedding for d in sorted(resp.data, key=lambda x: x.index)]
                except Exception as e:
                    if attempt == 2:
                        raise
                    await asyncio.sleep(2 ** attempt)

    results = await asyncio.gather(*[embed_one_batch(b) for b in batches])
    return [emb for batch_result in results for emb in batch_result]

# Embed 10,000 chunks in ~2 minutes
all_embeddings = await embed_all_chunks(chunks, batch_size=100, max_concurrent=5)
```

---

### 🔄 Embedding Freshness

```
When to re-embed your entire corpus:
  ✅ OpenAI upgrades text-embedding-3-small (different vector space)
  ✅ You switch embedding providers (OpenAI → Cohere)
  ✅ Your documents change significantly (> 30%)
  ✅ You change chunk size (alters what's in each vector)

When you DON'T need to re-embed:
  ✗ You change your prompt template
  ✗ You switch LLM providers (GPT-4o → Claude)
  ✗ Small additions to document set (just embed new docs)

Track embedding model version in metadata:
  {"embedding_model": "text-embedding-3-small", "embedding_version": "2"}
  If version changes → trigger full re-ingestion
```

---

### ⚔️ Embedding Model Comparison

| Model | Dims | Cost | Speed | Multilingual | Privacy | Best For |
|-------|------|------|-------|-------------|---------|---------|
| `text-embedding-3-small` | 1536 | $0.02/1M | Fast | Partial | Cloud | Default OpenAI stack |
| `text-embedding-3-large` | 3072 | $0.13/1M | Medium | Partial | Cloud | High-accuracy needs |
| `BAAI/bge-m3` | 1024 | Free | GPU: fast | 100+ langs | **Local** | **Open-source default** |
| `nomic-embed-text-v1.5` | 768 | Free | Fast | Partial | **Local** | Fast local dev |
| `embed-multilingual-v3` | 1024 | $0.10/1M | Fast | 100+ langs | Cloud | Multilingual production |

---

### 💼 Interview Questions — Embeddings

**Q1: Why use `search_document` vs `search_query` in Cohere embeddings?**
> Cohere trains asymmetric embeddings — documents and queries are embedded into different but compatible spaces. `search_document` creates embeddings optimised to be retrieved; `search_query` creates embeddings optimised to find those documents. Using the wrong type (e.g., `search_document` for a user query) degrades retrieval quality significantly because you're comparing vectors from incompatible spaces.

**Q2: What are Matryoshka embeddings and when would you use them?**
> Matryoshka Representation Learning (MRL) trains models so that the first N dimensions of a full embedding are themselves a valid lower-dimensional embedding. OpenAI's `text-embedding-3` models support this — you can request 256 instead of 1536 dimensions and get ~90% of the quality at 1/6th the storage and compute cost. Useful for tiered search: fast approximate search at 256 dims to get 100 candidates, then full 1536-dim re-scoring of those candidates for final ranking.

**Q3: When should you re-embed your entire corpus?**
> Re-embed when the embedding model changes (different vector space — old and new embeddings are incompatible), when you switch providers, or when chunk sizes change. You don't need to re-embed when you change your LLM, prompt, or retrieval strategy. Track `embedding_model` and a version string in each chunk's metadata — if the version changes on next ingestion, trigger a full re-embed.

---

## 2.4 — Vector Databases

### 🧑‍🎓 Explain Like I'm in BTech 2nd Year

A normal SQL database: `SELECT * FROM products WHERE name = 'iPhone'` — exact match.

A vector database: *"Find the 10 product descriptions most similar in meaning to this query"* — similarity match.

```
SQL search:
  Query: "phone for photography"
  Results: rows containing exactly those words
  Misses: "camera-focused smartphone", "best mobile for pictures" ← synonyms missed

Vector search:
  Query embedded → similar meaning vector space
  Results: most semantically similar chunks
  Finds: "camera-focused smartphone" ✅, "best mobile for pictures" ✅
```

---

### 🔍 HNSW — How Vector Search Actually Works

Most production vector databases use HNSW (Hierarchical Navigable Small World):

```
Problem: 1 million 1536-dim vectors
Brute force: compare query to ALL 1M vectors = 1.5 billion multiplications = SLOW

HNSW builds a layered graph:
  Layer 2 (top):   ●────────────●────────────● (few nodes, long-range jumps)
  Layer 1 (mid):   ●──●─────●──●──●─────●──● (medium density)
  Layer 0 (base):  ●─●─●─●─●─●─●─●─●─●─●─● (all nodes, dense)

Search:
  1. Start at Layer 2, navigate toward query
  2. Descend to Layer 1, refine navigation
  3. Descend to Layer 0, find nearest neighbours

Complexity: O(log n) vs O(n) brute force → 1000x faster for 1M vectors
Trade-off: Approximate (may miss true nearest), but >99% recall achievable

Key parameters:
  m:               connections per node (16 = good default, higher = more memory)
  ef_construction: quality of index build (64 = good default)
  ef_search:       quality at query time (higher = better recall, slower)
```

---

### 🗃️ ChromaDB — Local Development

```python
import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

# Persistent local storage
client = chromadb.PersistentClient(path="./chroma_db")

ef = OpenAIEmbeddingFunction(api_key="sk-...", model_name="text-embedding-3-small")

collection = client.get_or_create_collection(
    name="rag_docs",
    embedding_function=ef,
    metadata={"hnsw:space": "cosine"},
)

# Add documents — ChromaDB auto-embeds
collection.add(
    documents=[chunk.page_content for chunk in chunks],
    metadatas=[chunk.metadata for chunk in chunks],
    ids=[f"chunk_{i}" for i in range(len(chunks))],
)

# Query with metadata filter
results = collection.query(
    query_texts=["What is the return policy for electronics?"],
    n_results=5,
    where={"doc_type": "policy"},   # metadata pre-filter
    include=["documents", "metadatas", "distances"],
)

for doc, meta, dist in zip(
    results["documents"][0],
    results["metadatas"][0],
    results["distances"][0]
):
    print(f"Score: {1-dist:.3f} | {meta['source']} | {doc[:100]}")
```

---

### 🏭 Qdrant — Production Choice

```python
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct,
    Filter, FieldCondition, MatchValue,
    OptimizersConfigDiff, HnswConfigDiff,
)

# Local or cloud
client = QdrantClient(url="http://localhost:6333")
# client = QdrantClient(url="https://xxx.qdrant.io", api_key="...")

# Create optimised collection
client.recreate_collection(
    collection_name="rag_production",
    vectors_config=VectorParams(
        size=1536,
        distance=Distance.COSINE,
        on_disk=True,            # vectors stored on disk (save RAM for large collections)
    ),
    hnsw_config=HnswConfigDiff(
        m=16,
        ef_construct=100,
        full_scan_threshold=10_000,
    ),
    optimizers_config=OptimizersConfigDiff(
        indexing_threshold=20_000,  # start indexing after 20K vectors
    ),
)

# Batch upsert with rich payloads
BATCH_SIZE = 100
for i in range(0, len(chunks), BATCH_SIZE):
    batch_chunks = chunks[i:i+BATCH_SIZE]
    batch_embeddings = all_embeddings[i:i+BATCH_SIZE]

    points = [
        PointStruct(
            id=i + j,
            vector=batch_embeddings[j],
            payload={
                "text":         batch_chunks[j].page_content,
                "source":       batch_chunks[j].metadata["source"],
                "page":         batch_chunks[j].metadata.get("page", 0),
                "org_id":       batch_chunks[j].metadata["org_id"],
                "doc_type":     batch_chunks[j].metadata.get("doc_type", "general"),
                "last_updated": batch_chunks[j].metadata.get("last_updated"),
            }
        )
        for j in range(len(batch_chunks))
    ]
    client.upsert(collection_name="rag_production", points=points)
    print(f"Upserted {i + len(batch_chunks)}/{len(chunks)}")

# Search with metadata pre-filter
results = client.search(
    collection_name="rag_production",
    query_vector=query_embedding,
    limit=10,
    query_filter=Filter(must=[
        FieldCondition(key="org_id",   match=MatchValue(value="acme")),
        FieldCondition(key="doc_type", match=MatchValue(value="policy")),
    ]),
    with_payload=True,
    score_threshold=0.5,   # only return results above 0.5 cosine similarity
)

for hit in results:
    print(f"Score: {hit.score:.3f}")
    print(f"Text:  {hit.payload['text'][:200]}")
    print(f"From:  {hit.payload['source']}, page {hit.payload['page']}\n")
```

#### Qdrant Multi-Tenancy Pattern

```python
# Option 1: Separate collection per tenant (strongest isolation)
client.recreate_collection(collection_name=f"rag_{org_id}")

# Option 2: Single collection + org_id filter (more efficient)
# Use payload filtering on every query — org_id is ALWAYS in filter
results = client.search(
    collection_name="rag_shared",
    query_vector=query_emb,
    query_filter=Filter(must=[FieldCondition(key="org_id", match=MatchValue(value=org_id))]),
    limit=5,
)
```

---

### 🐘 pgvector — Vector Search Inside Postgres

```sql
-- Enable once per database
CREATE EXTENSION IF NOT EXISTS vector;

-- Documents table with vector column
CREATE TABLE rag_chunks (
    id          BIGSERIAL PRIMARY KEY,
    content     TEXT NOT NULL,
    embedding   vector(1536),
    source      TEXT,
    page_num    INTEGER,
    org_id      TEXT NOT NULL,
    doc_type    TEXT DEFAULT 'general',
    created_at  TIMESTAMPTZ DEFAULT NOW()
);

-- HNSW index for fast approximate search
CREATE INDEX ON rag_chunks
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- Index on org_id for fast pre-filtering
CREATE INDEX ON rag_chunks (org_id);

-- Semantic search with metadata pre-filter
SELECT
    id,
    content,
    source,
    page_num,
    1 - (embedding <=> $1::vector) AS similarity
FROM rag_chunks
WHERE org_id = $2
  AND doc_type = $3
ORDER BY embedding <=> $1::vector
LIMIT $4;
```

```python
# pgvector with asyncpg
import asyncpg
import numpy as np

async def vector_search(
    query_embedding: list[float],
    org_id: str,
    doc_type: str = None,
    k: int = 5,
) -> list[dict]:
    conn = await asyncpg.connect("postgresql://user:pass@localhost/rag_db")

    where_clauses = ["org_id = $2"]
    params = [query_embedding, org_id]

    if doc_type:
        where_clauses.append(f"doc_type = ${len(params)+1}")
        params.append(doc_type)

    query = f"""
        SELECT content, source, page_num,
               1 - (embedding <=> $1::vector) AS similarity
        FROM rag_chunks
        WHERE {' AND '.join(where_clauses)}
        ORDER BY embedding <=> $1::vector
        LIMIT {k}
    """

    rows = await conn.fetch(query, *params)
    await conn.close()
    return [dict(r) for r in rows]
```

---

### 🟣 Pinecone — Fully Managed

```python
from pinecone import Pinecone, ServerlessSpec

pc = Pinecone(api_key="PINECONE_API_KEY")

# Create serverless index
pc.create_index(
    name="rag-production",
    dimension=1536,
    metric="cosine",
    spec=ServerlessSpec(cloud="aws", region="us-east-1"),
)

index = pc.Index("rag-production")

# Upsert vectors with metadata
vectors = [
    {
        "id":       f"chunk_{i}",
        "values":   embeddings[i],
        "metadata": {
            "text":    chunks[i].page_content,
            "source":  chunks[i].metadata["source"],
            "org_id":  chunks[i].metadata["org_id"],
            "doc_type":chunks[i].metadata.get("doc_type"),
        }
    }
    for i in range(len(chunks))
]
# Batch upsert in groups of 100
for i in range(0, len(vectors), 100):
    index.upsert(vectors=vectors[i:i+100], namespace=org_id)

# Query with filter
results = index.query(
    vector=query_embedding,
    top_k=10,
    filter={"org_id": {"$eq": "acme"}, "doc_type": {"$eq": "policy"}},
    include_metadata=True,
    namespace=org_id,
)
```

---

### ⚔️ Vector Database Selection Guide

| DB | Type | Scale | Hybrid | Multi-tenancy | When to Choose |
|----|------|-------|--------|---------------|----------------|
| **ChromaDB** | Embedded | <5M | No | Basic | Dev, prototyping, single-node |
| **Qdrant** | Client-server | 100M+ | ✅ Native | ✅ | **Production default** |
| **Pinecone** | Managed SaaS | 1B+ | Partial | Namespaces | Zero-ops, variable traffic |
| **pgvector** | In Postgres | <10M | BM25 plugin | SQL filters | Existing Postgres stack |
| **FAISS** | Library | 100M+ | No | No | Batch processing, research |
| **Weaviate** | Client-server | 100M+ | ✅ Native | ✅ | GraphQL-first workflows |

**Decision framework:**
```
Already on Postgres AND < 5M vectors? → pgvector
Need zero-ops managed solution?        → Pinecone
Production, need hybrid + filtering?   → Qdrant  ← most common choice
Prototyping / local dev?               → ChromaDB
Batch ML pipeline, no persistence?     → FAISS
```

---

### 💼 Interview Questions — Vector Databases

**Q1: What is HNSW and why is it the dominant indexing algorithm?**
> HNSW (Hierarchical Navigable Small World) builds a multi-layer navigable graph. Upper layers have sparse long-range edges for fast navigation; the bottom layer contains all vectors densely connected. Search starts at the top, navigates toward the query, descends to find exact neighbours. Time complexity is O(log n) vs O(n) brute force — feasible to search millions of vectors in under 10ms on modern hardware. The trade-off is approximate results, but with `ef_search` tuning you can achieve >99% recall.

**Q2: Qdrant vs pgvector — when would you choose each?**
> pgvector wins when: you already run Postgres (no new service to operate), under 5M vectors, you need ACID transactions alongside vector search, or you want to join vector results with relational data in SQL. Qdrant wins when: you need native hybrid search (BM25 + dense in one query), dataset exceeds 5M vectors, you need fine-grained payload filtering at scale, or you need multi-tenancy with collection-level isolation.

**Q3: What is metadata pre-filtering and why does it matter?**
> Pre-filtering constrains the vector search to a specific subset of vectors before computing similarities. In Qdrant, you pass a `query_filter` — only vectors matching the filter are compared to the query embedding. This is dramatically faster than post-filtering (computing all similarities then discarding non-matching results) and is essential for multi-tenant systems where one org must never see another's data.

---

## 2.5 — Advanced Retrieval

### 🧑‍🎓 Explain Like I'm in BTech 2nd Year

Basic vector search is like searching your WhatsApp messages by scrolling — works for recent messages, misses older ones. Advanced retrieval combines multiple search techniques (like both scrolling and the search bar) and then uses a smarter judge to rank results.

The key insight: **basic vector search alone isn't good enough for production.** You need multiple signals and a smarter scoring step.

---

### 🔍 Hybrid Search: BM25 + Dense Vectors + RRF

```
Dense-only: "Apple Q3 2024 revenue"
  Finds: documents semantically about Apple financial performance
  Misses: documents using "Q3" as abbreviation differently

BM25-only: "Apple Q3 2024 revenue"
  Finds: documents with exact words "Apple", "Q3", "2024", "revenue"
  Misses: documents saying "Cupertino company third quarter earnings"

Hybrid (BM25 + Dense + RRF):
  Gets the best of both → both semantic relevance AND keyword precision
  This is why Google uses both signals simultaneously
```

```python
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_qdrant import QdrantVectorStore
from langchain_openai import OpenAIEmbeddings

# 1. Dense retriever
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
qdrant = QdrantVectorStore.from_existing_collection(
    embedding=embeddings,
    collection_name="rag_docs",
    url="http://localhost:6333",
)
dense_retriever = qdrant.as_retriever(search_kwargs={"k": 15})

# 2. Sparse (BM25) retriever
bm25_retriever = BM25Retriever.from_documents(documents)
bm25_retriever.k = 15

# 3. Ensemble with RRF weighting
hybrid_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, dense_retriever],
    weights=[0.4, 0.6],   # 40% keyword, 60% semantic
)

results = hybrid_retriever.invoke("Apple revenue Q3 2024 billion")
print(f"Retrieved {len(results)} chunks via hybrid search")
```

#### RRF Math Explained

```
BM25 results:  [doc_A(rank1), doc_C(rank2), doc_B(rank3), doc_D(rank4)]
Dense results: [doc_B(rank1), doc_A(rank2), doc_D(rank3), doc_C(rank4)]

RRF score = Σ 1/(k + rank)  where k=60

doc_A: 1/(60+1) + 1/(60+2) = 0.01639 + 0.01613 = 0.03252
doc_B: 1/(60+3) + 1/(60+1) = 0.01587 + 0.01639 = 0.03226
doc_C: 1/(60+2) + 1/(60+4) = 0.01613 + 0.01563 = 0.03176
doc_D: 1/(60+4) + 1/(60+3) = 0.01563 + 0.01587 = 0.03150

Merged ranking: [doc_A, doc_B, doc_C, doc_D]
Documents appearing in BOTH lists get the biggest boost.
```

---

### 🏆 Reranking: The Quality Multiplier

```
Bi-encoder (what vector search uses):
  embed(query) → vector_q
  embed(doc)   → vector_d
  score = cosine(vector_q, vector_d)

  ✅ Fast: pre-compute all doc vectors offline
  ❌ Weak: query and doc are encoded independently

Cross-encoder (what rerankers use):
  input: CONCAT(query, doc) → single joint representation
  output: relevance score 0 to 1

  ✅ Accurate: sees query AND doc together — understands context
  ❌ Slow: can't pre-compute, must run per (query, doc) pair

Production pattern: retrieve 20 fast (bi-encoder) → rerank to 5 accurate (cross-encoder)
```

```python
# Option 1: Cohere Rerank (cloud API — best quality)
import cohere
from langchain_cohere import CohereRerank
from langchain.retrievers import ContextualCompressionRetriever

cohere_reranker = CohereRerank(model="rerank-v3.5", top_n=5)

production_retriever = ContextualCompressionRetriever(
    base_compressor=cohere_reranker,
    base_retriever=hybrid_retriever,  # feeds 30 candidates to reranker
)

results = production_retriever.invoke("What are the payment methods accepted?")
# Hybrid retrieves 30 → Cohere reranks → returns top 5 most relevant

# Option 2: Local cross-encoder (free, no API call)
from sentence_transformers import CrossEncoder

cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def local_rerank(query: str, documents: list, top_n: int = 5) -> list:
    pairs = [(query, doc.page_content) for doc in documents]
    scores = cross_encoder.predict(pairs)
    ranked = sorted(zip(scores, documents), key=lambda x: x[0], reverse=True)
    return [doc for _, doc in ranked[:top_n]]

# Get 20 candidates first
candidates = hybrid_retriever.invoke(query)
# Rerank to top 5
top_results = local_rerank(query, candidates, top_n=5)
```

---

### 🔄 Query Rewriting

```
User query: "what's apple money q3"
→ Too colloquial, too short, abbreviations

Rewritten: "Apple Inc third quarter Q3 2024 financial results revenue earnings"
→ More terms → better BM25 recall → better dense coverage
```

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

rewrite_prompt = ChatPromptTemplate.from_template("""
You are a search query optimizer. Convert the user's question into an
optimised search query that will retrieve the most relevant documents
from a vector database. Expand abbreviations, add synonyms of key terms.

Original: {question}
Optimised query (output ONLY the query, no explanation):
""")

query_rewriter = rewrite_prompt | llm | StrOutputParser()

original = "apple money q3"
rewritten = query_rewriter.invoke({"question": original})
print(rewritten)
# "Apple Inc third quarter Q3 2024 financial results quarterly revenue earnings report"
```

---

### 🧪 HyDE — Hypothetical Document Embeddings

```
Standard approach:
  Query: "What causes attention to scale quadratically?"
  → embed question → find similar docs
  Problem: questions and documents are in different parts of the embedding space

HyDE approach:
  Query: "What causes attention to scale quadratically?"
  Step 1: LLM generates hypothetical answer:
          "Attention scales quadratically because the attention matrix
           is n×n where n is sequence length. Each of the n tokens
           must compute similarity with all other n tokens..."
  Step 2: Embed the hypothetical answer (not the question)
  Step 3: Retrieve using hypothetical-answer embedding
  → Now we're comparing answer-space embeddings to answer-space documents
  → Much better retrieval quality
```

```python
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.5)

hyde_prompt = ChatPromptTemplate.from_template("""
Write a detailed technical paragraph that would directly answer this question.
Write as if extracted from an expert document. Be specific and factual.

Question: {question}
Answer paragraph:
""")

embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")

async def hyde_retrieve(query: str, vectorstore, k: int = 5) -> list:
    # Generate hypothetical answer
    chain = hyde_prompt | llm | StrOutputParser()
    hyp_answer = await chain.ainvoke({"question": query})

    # Embed hypothetical answer
    hyp_embedding = await embeddings_model.aembed_query(hyp_answer)

    # Retrieve using hypothetical embedding
    results = vectorstore.similarity_search_by_vector(hyp_embedding, k=k)
    return results
```

---

### 🎯 Multi-Query Retrieval

```python
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_openai import ChatOpenAI

# LLM generates multiple reformulations of the query
# Retrieves for each → union of results → better coverage
retriever = MultiQueryRetriever.from_llm(
    retriever=qdrant.as_retriever(search_kwargs={"k": 5}),
    llm=ChatOpenAI(model="gpt-4o-mini", temperature=0.5),
)

# For query: "electronics return policy"
# Generates and retrieves for:
#   "procedure for returning electronic items"
#   "how to return electronics, refund process"
#   "electronics purchase return conditions and timeframe"
results = retriever.invoke("electronics return policy")
print(f"{len(results)} unique chunks retrieved (deduplicated)")
```

---

### 🗜️ Context Compression

Remove irrelevant sentences from retrieved chunks before feeding to LLM.

```python
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
compressor = LLMChainExtractor.from_llm(llm)

compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=hybrid_retriever,
)

# Original chunk (500 chars) might contain many sentences
# After compression: only the sentences directly relevant to the query remain (~100 chars)
# Saves tokens, improves signal-to-noise ratio in LLM context
compressed_results = compression_retriever.invoke(
    "What is the electronics return window?"
)
```

---

### 🔙 Contextual Retrieval — Anthropic's Technique

Before embedding, prepend a document-level context summary to each chunk.

```python
import anthropic

client = anthropic.Anthropic()

CONTEXT_PROMPT = """
<document>
{full_document}
</document>

Here is the chunk we want to situate within the whole document:
<chunk>
{chunk}
</chunk>

Give a short succinct context (1-2 sentences) to situate this chunk 
within the overall document for improved search retrieval.
Output ONLY the context sentence, nothing else.
"""

async def create_contextual_chunk(
    full_doc_text: str,
    chunk_text: str
) -> str:
    """Add document context to a chunk before embedding"""
    response = await client.messages.create(
        model="claude-haiku-4-20250514",   # cheap fast model for this
        max_tokens=150,
        system=[{
            "type": "text",
            "text": "Generate brief document context. Be concise.",
            "cache_control": {"type": "ephemeral"}  # cache full_doc_text
        }],
        messages=[{"role": "user", "content": CONTEXT_PROMPT.format(
            full_document=full_doc_text[:8000],
            chunk=chunk_text
        )}]
    )
    context = response.content[0].text.strip()
    return f"{context}\n\n{chunk_text}"

# Before:
# "Items can be returned within 30 days with proof of purchase."

# After contextual retrieval:
# "This chunk is from Chapter 4 of the TechMart Customer Policy document,
#  covering the Returns & Exchanges section."
# "Items can be returned within 30 days with proof of purchase."
```

**Why it helps:** The original chunk alone has no idea it's from a returns policy document. The context sentence dramatically improves retrieval for queries like "return policy" or "refund process."

---

### 🔎 Step-Back Prompting

Abstract the specific query to a more general principle, retrieve for both.

```python
step_back_prompt = ChatPromptTemplate.from_template("""
Given the specific question, derive a more general question that covers
the underlying principle needed to answer it.

Specific: {question}
General (output only the general question):
""")

chain = step_back_prompt | llm | StrOutputParser()

specific = "What is the deadline for returning a MacBook Pro bought on Black Friday?"
general = chain.invoke({"question": specific})
# "What is the electronics return policy and timeframe?"

# Retrieve for BOTH queries → union → better coverage
specific_results = retriever.invoke(specific)
general_results  = retriever.invoke(general)
combined = list({d.page_content: d for d in specific_results + general_results}.values())
```

---

## 2.6 — RAG Architectures

### 📐 Evolution: Naive → Advanced → Agentic

```
NAIVE RAG (2023 — demo quality):
  query → embed → top-K vector search → LLM
  Problems: single-shot retrieval, no quality check, misses keyword queries

ADVANCED RAG (2024 — production quality):
  query → [rewrite] → hybrid search → [rerank] → [compress] → LLM
  Improvements: better retrieval quality, noise reduction

MODULAR RAG (2024 — flexible production):
  Any component is independently swappable: routing, fusion, reranking
  Can send queries to multiple specialised retrievers and merge results

AGENTIC RAG (2025-2026 — intelligent production):
  LLM decides WHAT to retrieve, HOW MANY times, and WHETHER to web search
  Self-checks retrieval quality and retries if insufficient
```

---

### 🔨 Naive RAG — Full Implementation

```python
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
llm = ChatOpenAI(model="gpt-4o", temperature=0)

RAG_PROMPT = ChatPromptTemplate.from_template("""
Answer the question using ONLY the context provided below.
If the answer is not in the context, say "I don't have that information."
Always cite your source at the end using [Source: filename, Page: N].
Do NOT use your own knowledge or training data.

Context:
{context}

Question: {question}

Answer:
""")

def format_docs(docs: list) -> str:
    return "\n\n---\n\n".join(
        f"[Source: {d.metadata.get('source','?')}, Page: {d.metadata.get('page','?')}]\n"
        f"{d.page_content}"
        for d in docs
    )

naive_rag = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | RAG_PROMPT
    | llm
    | StrOutputParser()
)

answer = naive_rag.invoke("What is the refund policy for electronics?")
print(answer)
```

---

### 🔨 Production Hybrid RAG — Full Stack

```python
from langchain.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain_cohere import CohereRerank
from langchain_community.retrievers import BM25Retriever
from langchain_qdrant import QdrantVectorStore

# Build production retriever: hybrid + rerank
qdrant_vs = QdrantVectorStore.from_existing_collection(
    embedding=OpenAIEmbeddings(model="text-embedding-3-small"),
    collection_name="rag_production",
    url="http://localhost:6333",
)
dense_r  = qdrant_vs.as_retriever(search_kwargs={"k": 20})
sparse_r = BM25Retriever.from_documents(documents, k=20)

hybrid_r = EnsembleRetriever(
    retrievers=[sparse_r, dense_r],
    weights=[0.4, 0.6],
)
reranking_r = ContextualCompressionRetriever(
    base_compressor=CohereRerank(model="rerank-v3.5", top_n=5),
    base_retriever=hybrid_r,
)

# Full RAG chain
production_rag = (
    {"context": reranking_r | format_docs, "question": RunnablePassthrough()}
    | RAG_PROMPT
    | llm
    | StrOutputParser()
)

answer = production_rag.invoke("Apple quarterly revenue breakdown by segment")
```

---

### 🔨 Agentic RAG with Self-Correction (LangGraph)

```python
from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults

llm = ChatOpenAI(model="gpt-4o", temperature=0)
web_search = TavilySearchResults(max_results=3)

class RAGState(TypedDict):
    question:           str
    retrieved_docs:     List[str]
    retrieval_quality:  str       # "sufficient" | "insufficient"
    answer:             str
    retry_count:        int
    source:             str       # "internal" | "web"

# Node 1: Retrieve from internal knowledge base
def retrieve_internal(state: RAGState) -> dict:
    docs = production_retriever.invoke(state["question"])
    return {"retrieved_docs": [d.page_content for d in docs]}

# Node 2: Grade retrieval quality
def grade_retrieval(state: RAGState) -> dict:
    context = "\n".join(state["retrieved_docs"][:3])
    grade_prompt = f"""
    Question: {state['question']}

    Retrieved context:
    {context}

    Can this context sufficiently answer the question?
    Reply with ONLY: sufficient  OR  insufficient
    """
    verdict = llm.invoke(grade_prompt).content.strip().lower()
    return {"retrieval_quality": verdict}

# Node 3: Web search fallback
def web_search_fallback(state: RAGState) -> dict:
    results = web_search.invoke(state["question"])
    web_docs = [r["content"] for r in results]
    return {
        "retrieved_docs": state["retrieved_docs"] + web_docs,
        "source":         "web",
        "retry_count":    state["retry_count"] + 1,
    }

# Node 4: Generate answer
def generate_answer(state: RAGState) -> dict:
    context = "\n\n---\n\n".join(state["retrieved_docs"])
    source_note = " (supplemented with web search)" if state.get("source") == "web" else ""
    prompt = f"""
    Answer the question based on the context provided{source_note}.
    If context is still insufficient, state what information is missing.

    Context:
    {context}

    Question: {state['question']}
    Answer:
    """
    answer = llm.invoke(prompt).content
    return {"answer": answer}

# Routing logic
def route_after_grading(state: RAGState) -> str:
    if state["retrieval_quality"] == "sufficient":
        return "generate"
    if state["retry_count"] >= 1:      # only one web search attempt
        return "generate"
    return "web_search"

# Build graph
graph = StateGraph(RAGState)
graph.add_node("retrieve",    retrieve_internal)
graph.add_node("grade",       grade_retrieval)
graph.add_node("web_search",  web_search_fallback)
graph.add_node("generate",    generate_answer)

graph.set_entry_point("retrieve")
graph.add_edge("retrieve",   "grade")
graph.add_conditional_edges("grade", route_after_grading, {
    "generate":   "generate",
    "web_search": "web_search",
})
graph.add_edge("web_search", "generate")
graph.add_edge("generate",   END)

agentic_rag = graph.compile()

result = agentic_rag.invoke({
    "question":          "What is the current GST rate on electronics in India?",
    "retrieved_docs":    [],
    "retrieval_quality": "",
    "answer":            "",
    "retry_count":       0,
    "source":            "internal",
})
print(result["answer"])
print(f"Source used: {result['source']}")
```

---

### 🕸️ GraphRAG — Multi-Hop Reasoning

```
Standard RAG problem (multi-hop question):
  Q: "Who founded the company that acquired Sarvam AI?"
  
  Doc A: "Sarvam AI was acquired by TataGroup in 2025."
  Doc B: "The Tata Group was founded by Jamsetji Tata in 1868."
  
  Vector search for "Sarvam AI founder": retrieves Doc A or Doc B, not both
  Cannot connect the dots: Sarvam AI → acquired by Tata → founded by Jamsetji Tata

GraphRAG solution:
  Build knowledge graph:
    [Sarvam AI] --acquired_by--> [Tata Group]
    [Tata Group] --founded_by--> [Jamsetji Tata]
  
  Multi-hop traversal:
    Query → find Sarvam AI node → follow acquired_by edge → Tata Group
    → follow founded_by edge → Jamsetji Tata → Answer ✅
```

```python
# Microsoft's GraphRAG
# pip install graphrag

# 1. Set up settings.yml and run indexing pipeline
# graphrag index --root ./graphrag_workspace

# 2. Query with local or global search
from graphrag.query.structured_search.local_search.search import LocalSearch

# LocalSearch: entity-level, precise, good for specific questions
# GlobalSearch: community-level, broad, good for thematic questions
```

---

### 📏 Long-Context RAG vs Selective Retrieval

```
Question: "Summarise all key decisions from our 200-page board meeting transcript"

Option A — Selective RAG (retrieve 5 chunks):
  Problem: Misses decisions spread across the document
  Use when: Specific factual question with a localised answer

Option B — Full-context stuffing (send all 200 pages):
  Gemini 2.0 Flash: 1M context window = ~750,000 words
  Cost: ~$0.50 per query for 200 pages (vs $0.02 for selective)
  Quality: Higher (sees everything)
  Use when: Summarisation, analysis of entire document, when answer could be anywhere

Decision rule:
  Needle-in-haystack question → selective RAG (cheaper, faster)
  Document-wide analysis      → full context (more expensive, more complete)
  Long regulatory doc + many users → chunk it (cost control)
```

---

### 💼 Interview Questions — RAG Architectures

**Q1: What is the difference between Naive RAG, Advanced RAG, and Agentic RAG?**
> Naive RAG does single-shot vector retrieval and immediately generates — simple but breaks on keyword queries, complex questions, and retrieval quality issues. Advanced RAG adds pre-retrieval steps (query rewriting, HyDE) and post-retrieval steps (reranking, context compression) for higher quality. Agentic RAG uses an LLM to orchestrate the retrieval loop — it can evaluate its own retrieval quality, decide to retry with a different strategy, route to different knowledge bases, or fall back to web search. Each level adds quality but also cost and latency.

**Q2: What is CRAG and when would you use it?**
> Corrective RAG is an agentic RAG pattern where the system evaluates the quality of retrieved documents before generation. If retrieval quality is below a threshold, it triggers a corrective action — either refining the query and re-retrieving, or falling back to web search via Tavily/SerpAPI. Use it when your internal knowledge base is incomplete for some query types and you want the system to gracefully fall back to web data rather than hallucinating.

**Q3: When would you use GraphRAG instead of standard RAG?**
> GraphRAG is designed for multi-hop questions — questions that require connecting information across multiple documents or entities. Standard RAG retrieves the top-K most similar chunks, but if the answer requires chaining: "Company A was acquired by Company B, which was founded by Person C," vector similarity alone can't make that connection. GraphRAG builds a knowledge graph of entities and relationships, enabling multi-hop traversal. It's significantly more expensive to build and query than standard RAG — use it when your use case explicitly requires relational reasoning across documents.

---

## 2.7 — RAG Evaluation

### 🧑‍🎓 Explain Like I'm in BTech 2nd Year

You've built a RAG chatbot and it "seems to work." But how do you know it works for all query types? What if it's hallucinating 20% of the time on edge cases? What if after a prompt change it gets 10% worse?

RAG evaluation is unit testing for your AI. Without it, you're flying blind.

```
Without evaluation:
  Deploy RAG → Users complain about wrong answers → You have no idea why or how often
  
With evaluation:
  Deploy RAG → RAGAS scores: faithfulness 0.82, recall 0.74
  Change prompt → RAGAS scores: faithfulness 0.79 → REGRESSION DETECTED
  Revert prompt → Scores recover → Safe to ship ✅
```

---

### 📊 RAGAS — The Standard Framework

RAGAS measures 4 core metrics, split into retrieval and generation:

```
RETRIEVAL METRICS
─────────────────
Context Precision:
  "Are the chunks I retrieved actually relevant?"
  = relevant_chunks_retrieved / total_chunks_retrieved
  Low precision: retrieved 10 chunks, only 3 are useful → noise in LLM context

Context Recall:
  "Did I retrieve ALL the information needed to answer?"
  = info_covered_by_retrieved_context / info_in_ground_truth_answer
  Low recall: answer needs facts from 5 chunks, retrieved only 3 → incomplete answer

GENERATION METRICS
──────────────────
Faithfulness:
  "Is the generated answer supported by the retrieved context?" (no hallucination)
  = claims_in_answer_supported_by_context / total_claims_in_answer
  Score 1.0 = every claim is grounded; 0.7 = 30% hallucination rate

Answer Relevancy:
  "Does the answer actually address the question?"
  = semantic_similarity(answer, question)
  Low relevancy: question asked about refunds, answer talks about shipping
```

```python
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
)
from datasets import Dataset
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# Build evaluation dataset
eval_data = {
    "question": [
        "What is the return policy for electronics?",
        "How long does standard shipping take?",
        "What payment methods are accepted?",
        "Can I return a product without a receipt?",
        "Is there a restocking fee for returns?",
    ],
    "answer": [
        # These are your RAG system's generated answers
        "Electronics can be returned within 30 days with original receipt.",
        "Standard shipping takes 3-5 business days.",
        "We accept Visa, Mastercard, UPI, and net banking.",
        "Returns without receipt are accepted for exchanges only, not refunds.",
        "No restocking fee for items returned within the policy window.",
    ],
    "contexts": [
        # These are the chunks your RAG retrieved for each question
        ["Electronics purchased online may be returned within 30 days of delivery..."],
        ["Standard delivery typically takes 3-5 working days for most PIN codes..."],
        ["Accepted payment methods include Visa, Mastercard, RuPay, UPI and net banking..."],
        ["Customers without a purchase receipt may request an exchange within 15 days..."],
        ["TechMart does not charge any restocking fees for returns made within the policy window..."],
    ],
    "ground_truth": [
        # Human-written correct answers (your golden dataset)
        "Electronics can be returned within 30 days with proof of purchase.",
        "Standard shipping takes 3-5 business days.",
        "Visa, Mastercard, UPI, RuPay, and net banking are accepted.",
        "Without receipt, only exchanges are allowed, not cash refunds.",
        "There is no restocking fee for policy-compliant returns.",
    ],
}

dataset = Dataset.from_dict(eval_data)

results = evaluate(
    dataset=dataset,
    metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
    llm=ChatOpenAI(model="gpt-4o"),
    embeddings=OpenAIEmbeddings(),
    raise_exceptions=False,
)

print(results)
# {'faithfulness': 0.87, 'answer_relevancy': 0.94, 'context_recall': 0.83, 'context_precision': 0.79}
```

---

### 🏗️ Golden Dataset Construction

```python
# Method 1: LLM-generated questions from chunks (fast, needs review)
async def generate_qa_from_chunk(chunk_text: str, llm) -> list[dict]:
    prompt = f"""
    Given this document chunk, generate 3 specific factual questions that 
    can be answered using ONLY information in this chunk.
    Return as JSON: [{{"question": "...", "answer": "..."}}]
    
    Chunk:
    {chunk_text}
    
    JSON:
    """
    response = await llm.ainvoke(prompt)
    try:
        import json, re
        json_str = re.search(r'\[.*\]', response.content, re.DOTALL).group()
        return json.loads(json_str)
    except:
        return []

# Generate from all chunks
import asyncio
all_qa_pairs = []
tasks = [generate_qa_from_chunk(c.page_content, llm) for c in chunks[:50]]
results = await asyncio.gather(*tasks)
for qa_list in results:
    all_qa_pairs.extend(qa_list)

print(f"Generated {len(all_qa_pairs)} QA pairs")
# Human reviews and keeps good ones → golden dataset

# Method 2: Real user queries (best quality, needs data collection)
# Log actual user queries + correct answers from your helpdesk → ground truth
```

---

### 🔁 Automated Eval in CI/CD

```python
# scripts/run_rag_eval.py
import json, sys
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_recall, context_precision
from datasets import Dataset

def run_eval_pipeline() -> dict:
    # Load golden dataset
    with open("eval_dataset.json") as f:
        golden = json.load(f)

    # Run RAG system on all questions
    results = []
    for item in golden:
        retrieved_docs = production_retriever.invoke(item["question"])
        answer = production_rag.invoke(item["question"])
        results.append({
            "question":    item["question"],
            "answer":      answer,
            "contexts":    [d.page_content for d in retrieved_docs],
            "ground_truth": item["ground_truth"],
        })

    dataset = Dataset.from_list(results)
    scores = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
    )
    return scores

scores = run_eval_pipeline()

# Quality gates
THRESHOLDS = {
    "faithfulness":    0.75,
    "answer_relevancy":0.80,
    "context_recall":  0.70,
    "context_precision":0.65,
}

failed = []
for metric, threshold in THRESHOLDS.items():
    if scores[metric] < threshold:
        failed.append(f"{metric}: {scores[metric]:.3f} < {threshold}")

if failed:
    print("QUALITY GATE FAILED:")
    for f in failed:
        print(f"  ❌ {f}")
    sys.exit(1)
else:
    print("All quality gates passed ✅")
    for k, v in scores.items():
        print(f"  {k}: {v:.3f}")
    sys.exit(0)
```

```yaml
# .github/workflows/rag_eval.yml
name: RAG Quality Gate

on:
  push:
    branches: [main, develop]
  pull_request:

jobs:
  rag-quality:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: pip install -r requirements.txt
      - name: Run RAG Evaluation
        run: python scripts/run_rag_eval.py
        env:
          OPENAI_API_KEY:  ${{ secrets.OPENAI_API_KEY }}
          QDRANT_URL:      ${{ secrets.QDRANT_URL }}
          COHERE_API_KEY:  ${{ secrets.COHERE_API_KEY }}
```

---

### 🧪 DeepEval — Flexible Alternative

```python
from deepeval import evaluate as deepeval_evaluate
from deepeval.metrics import (
    FaithfulnessMetric,
    AnswerRelevancyMetric,
    ContextualRecallMetric,
    HallucinationMetric,
)
from deepeval.test_case import LLMTestCase

# Define custom threshold per metric
faithfulness_metric   = FaithfulnessMetric(threshold=0.75, model="gpt-4o")
relevancy_metric      = AnswerRelevancyMetric(threshold=0.80, model="gpt-4o")
hallucination_metric  = HallucinationMetric(threshold=0.1,  model="gpt-4o")  # <10% hallucination

# Build test cases
test_cases = [
    LLMTestCase(
        input=item["question"],
        actual_output=rag_answers[i],
        retrieval_context=retrieved_contexts[i],
        expected_output=item["ground_truth"],
    )
    for i, item in enumerate(golden_dataset)
]

# Run evaluation
results = deepeval_evaluate(
    test_cases=test_cases,
    metrics=[faithfulness_metric, relevancy_metric, hallucination_metric],
)
```

---

### 🎯 Metric Targets & What Low Scores Mean

| Metric | Target | Low Score Means | How to Fix |
|--------|--------|----------------|-----------|
| **Faithfulness** | > 0.75 | LLM is hallucinating facts not in context | Stricter RAG prompt, output validation |
| **Answer Relevancy** | > 0.80 | Answers are off-topic or vague | Better retrieval, clearer prompt |
| **Context Recall** | > 0.70 | Missing needed information in retrieved docs | Better chunking, larger K, hybrid search |
| **Context Precision** | > 0.65 | Retrieving too much irrelevant content | Add reranking, tighten metadata filters |

---

### 💼 Interview Questions — RAG Evaluation

**Q1: What is faithfulness and why is it the most critical RAGAS metric?**
> Faithfulness measures what fraction of claims in the generated answer are actually supported by the retrieved context. Score 1.0 = every claim grounded; 0.7 = 30% hallucination rate. It's the most critical because a RAG system with low faithfulness is arguably worse than no RAG — it creates false confidence in fabricated information. If faithfulness is below 0.75, it means the LLM is pulling facts from its training data instead of the retrieved documents, defeating the purpose of RAG.

**Q2: What is the difference between context precision and context recall?**
> Context precision asks: "Of the chunks I retrieved, how many were actually relevant?" Low precision = retrieving noise (wastes context window, confuses LLM). Context recall asks: "Of all the information needed to answer, how much did I retrieve?" Low recall = missing key facts (answer will be incomplete or wrong). You need both: high-precision high-recall retrieval. Typically, hybrid search + reranking improves precision; larger K + multiple query strategies improve recall.

**Q3: How do you build a golden evaluation dataset when you have no labelled data?**
> Three approaches in practice: (1) LLM-generated — use GPT-4o to generate question-answer pairs for each chunk, then human spot-check a 20% sample. (2) Production logs — collect real user questions that got validated correct answers from your helpdesk or support team. (3) Domain expert curation — have subject matter experts write 50-100 representative Q&A pairs covering edge cases and common patterns. Hybrid approach (LLM generates 200, experts review and keep 80 good ones) gives best quality-to-time ratio.

---

## 🏗️ Phase 2 Projects

### Project 2A: Basic RAG — Baseline (8 hrs)

**Goal:** Build the simplest working RAG and establish a RAGAS baseline score.

```
Stack:   LangChain + ChromaDB + OpenAI + RAGAS
Dataset: Any PDF collection (company docs, product manuals, policies)

Build:
  1. PyMuPDF loader for PDFs
  2. Recursive character splitting (1000 chars, 200 overlap)
  3. text-embedding-3-small embeddings → ChromaDB
  4. GPT-4o answering with source citations
  5. RAGAS eval on 20 handcrafted Q&A pairs

Measure and record these baseline scores:
  - faithfulness: _____
  - answer_relevancy: _____
  - context_recall: _____
  - context_precision: _____

These are your benchmark to beat in Project 2B.
```

---

### Project 2B: Production Hybrid RAG (12 hrs)

**Goal:** Beat baseline scores significantly with production-grade techniques.

```
Stack:   Qdrant + LangChain + Cohere Rerank + FastAPI + React + RAGAS

Improvements over 2A:
  - Qdrant (replaces ChromaDB)
  - Parent-child chunking (400/2000 chars)
  - Hybrid search: BM25 + dense vector + RRF
  - Cohere reranker (retrieve 20, rerank to 5)
  - Contextual retrieval (Anthropic-style chunk context)
  - FastAPI streaming backend
  - React streaming UI with source display
  - Prompt caching on Anthropic
  - RAGAS eval pipeline: faithfulness target > 0.80

Expected improvement: faithfulness +15-20pts vs 2A baseline
GitHub: README with before/after RAGAS comparison table
```

---

### Project 2C: Agentic RAG with Self-Correction (15 hrs)

**Goal:** LangGraph agent that judges its own retrieval and falls back to web search.

```
Stack:   LangGraph + Qdrant + Tavily + Cohere + FastAPI + Streamlit

Features:
  - Retrieval quality grader node (LLM grades retrieved context)
  - Corrective RAG: if quality < threshold → Tavily web search fallback
  - Multi-source routing: separate Qdrant collections for (policy, product, FAQ)
  - Query classifier routes to correct collection
  - Full RAGAS + DeepEval evaluation report
  - Streamlit UI showing agent reasoning steps in real-time

Questions your eval should include edge cases for:
  - Questions answered by internal docs only
  - Questions requiring web search (live prices, news)
  - Ambiguous questions that need routing
```

---

### Project 2D: Multi-Tenant Enterprise Knowledge Base (15 hrs)

**Goal:** Production platform supporting multiple organisations with isolated data.

```
Stack:   Qdrant + FastAPI + Celery + Redis + PostgreSQL + React + Docker

Features:
  - Per-org Qdrant payload filtering (org_id on every chunk)
  - Async ingestion: PDF, DOCX, web URLs, REST API sources
  - Celery + Redis job queue with real-time progress via WebSocket
  - Role-based access: admin (ingest+query), user (query only)
  - Per-org RAGAS quality dashboard
  - REST API for programmatic document management
  - Docker Compose: FastAPI + Celery worker + Redis + Qdrant + Postgres

Architecture diagram required in README showing all components.
```

---

## 📝 Interview Cheat Sheet

### Top 15 Phase 2 Questions

| # | Question | One-Line Answer |
|---|----------|----------------|
| 1 | What is RAG? | LLM answers grounded in retrieved external knowledge — reduces hallucination |
| 2 | RAG vs fine-tuning? | RAG for dynamic knowledge (cheap update); fine-tune for style/format |
| 3 | Walk through the RAG pipeline | Ingest: load→clean→chunk→embed→store. Query: embed→search→rerank→generate |
| 4 | Best production chunking? | Parent-child: small chunks for retrieval, large parents for LLM context |
| 5 | Why chunk overlap? | Facts at boundaries appear in both adjacent chunks — retrieval succeeds |
| 6 | What metadata on each chunk? | source, page, section, doc_type, org_id, date — enables pre-filtering |
| 7 | Embedding model choice? | text-embedding-3-small (OpenAI default); BGE-M3 (free, multilingual) |
| 8 | Matryoshka embeddings? | Truncate dims without retraining; tiered search (small dims → full dims) |
| 9 | What is hybrid search? | BM25 keyword + dense vector, merged with RRF |
| 10 | What is RRF? | 1/(k+rank) scored across lists; docs in both get biggest boost |
| 11 | Bi-encoder vs cross-encoder? | Bi: pre-computable, fast. Cross: reads both together, accurate but slow |
| 12 | When to use pgvector vs Qdrant? | pgvector: existing Postgres, <5M. Qdrant: scale, hybrid, filtering |
| 13 | Faithfulness in RAGAS? | % of answer claims supported by context; primary hallucination metric |
| 14 | Context recall vs precision? | Recall: retrieved enough? Precision: retrieved only relevant? |
| 15 | What is HyDE? | Generate hypothetical answer, embed it instead of question — better retrieval |

---

### Speed Round

- **Naive RAG failure** → single-shot retrieval, no quality check, keyword queries missed
- **Semantic chunking** → split at embedding-similarity drops (topic shifts)
- **Late chunking** → embed full doc first, pool to chunk level — retains doc context
- **HNSW** → graph-based ANN; O(log n) search; approximate but tunable recall
- **Contextual retrieval** → prepend document context to chunk before embedding (Anthropic)
- **Step-back prompting** → abstract specific query to general principle, retrieve for both
- **CRAG** → evaluate retrieval quality; web search fallback if insufficient
- **GraphRAG** → knowledge graph + LLM; multi-hop reasoning across entities
- **DeepEval** → flexible eval framework; custom metrics; hallucination metric
- **RAGAS target** → faithfulness > 0.75, answer_relevancy > 0.80 before shipping

---

## 🃏 Quick Revision Cards

**Card 1: RAG Pipeline**
```
INGEST: load → clean → chunk → embed → store
QUERY:  embed_query → search → rerank → build_context → generate
Grounding prompt: "Answer ONLY using the context below."
```

**Card 2: Chunking**
```
Fixed-size      → prototype only (cuts sentences)
Recursive char  → general production default
Semantic        → quality-critical (expensive)
Parent-child    → PRODUCTION DEFAULT (precision + context)
Always: overlap 10-20%, metadata on every chunk
```

**Card 3: Embeddings**
```
text-embedding-3-small → OpenAI default (1536d, $0.02/1M)
BAAI/bge-m3            → Free, local, multilingual (1024d)
embed-multilingual-v3  → Cohere, production multilingual
Matryoshka: truncate dims → smaller index, ~90% quality
Re-embed when model version changes (vector spaces differ)
```

**Card 4: Vector DBs**
```
ChromaDB → local dev
Qdrant   → production (hybrid, filtering, 100M+) ← DEFAULT
Pinecone → managed SaaS, zero-ops
pgvector → existing Postgres, <5M vectors
HNSW: O(log n), approximate, >99% recall tunable
```

**Card 5: Advanced Retrieval**
```
Hybrid search:  BM25 + dense → RRF → best coverage
Reranking:      retrieve 20, cross-encoder to top 5
HyDE:           embed hypothetical answer, not question
Multi-query:    N reformulations → union → dedup
Contextual:     prepend doc context before embedding
```

**Card 6: RAGAS Metrics**
```
Faithfulness    > 0.75 — no hallucination
Answer Relevancy > 0.80 — answer addresses question
Context Recall  > 0.70 — retrieved all needed info
Context Precision > 0.65 — retrieved only relevant chunks
Run eval in CI/CD — fail build on regression
```

---

## 📚 Resources for Phase 2

| Resource | Type | Time | Priority |
|---|---|---|---|
| [LangChain RAG Tutorial](https://python.langchain.com/docs/tutorials/rag/) | Docs | 2 hrs | 🔥 Must-do |
| [RAGAS Docs](https://docs.ragas.io) | Docs | 1 hr | 🔥 Must-read |
| [Qdrant Docs + Tutorials](https://qdrant.tech/documentation/) | Docs | 2 hrs | 🔥 Must-read |
| [DeepLearning.AI — Building + Evaluating Advanced RAG](https://learn.deeplearning.ai/courses/building-evaluating-advanced-rag) | Course | 4 hrs | 🔥 Must-do |
| [Anthropic Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval) | Blog | 30 min | 🔥 Must-read |
| [DeepEval Docs](https://docs.confident-ai.com) | Docs | 1 hr | ⭐ Recommended |
| [Pinecone Learning Centre](https://www.pinecone.io/learn/) | Guides | 2 hrs | ⭐ Recommended |
| [RAG Evaluation Guide — Qdrant](https://qdrant.tech/blog/rag-evaluation-guide/) | Blog | 30 min | ⭐ Recommended |
| [Microsoft GraphRAG](https://github.com/microsoft/graphrag) | GitHub | 1 hr | ⭐ Advanced |
| [ChromaDB Quickstart](https://docs.trychroma.com/getting-started) | Docs | 30 min | ✅ Quick start |
| [RAG Playground](https://rag-play.vercel.app) | Playground Link | 30 min | 🔥 Must-check

---

## ✅ Phase 2 Completion Checklist

```
[ ] Can explain the full RAG pipeline (both ingestion + query) without notes
[ ] Loaded PDFs with PyMuPDF and Unstructured.io — know when to use each
[ ] Implemented hash-based deduplication in ingestion pipeline
[ ] Implemented parent-child chunking and compared quality to fixed-size
[ ] Used BGE-M3 or nomic-embed locally (not just OpenAI)
[ ] Know the difference between bi-encoder and cross-encoder — implemented both
[ ] Built hybrid search (BM25 + Qdrant dense + RRF) — verified quality improvement
[ ] Added Cohere reranking + compared with/without scores
[ ] Used Qdrant in production with payload filtering for multi-tenancy
[ ] Implemented contextual retrieval (Anthropic-style)
[ ] Built RAGAS evaluation pipeline with golden dataset (20+ QA pairs)
[ ] Achieved: faithfulness > 0.75, answer_relevancy > 0.80 on Project 2B
[ ] Built agentic RAG with self-correction (LangGraph) — Project 2C
[ ] Set up eval in CI/CD with quality gate (GitHub Actions)
[ ] Projects 2A, 2B, 2C pushed to GitHub with README + RAGAS report
[ ] Project 2B has: before/after RAGAS comparison (2A vs 2B scores)
```

---

*Phase 2 of GenAI + LLMOps Engineering Roadmap 2026*  
*→ Next: Phase 3 — FastAPI AI Backend (Async Patterns, Streaming, Production)*
