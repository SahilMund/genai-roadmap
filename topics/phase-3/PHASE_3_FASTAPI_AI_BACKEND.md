# ⚡ Phase 3 — FastAPI AI Backend
> **Complete Study Notes | Interview Prep | Code Examples | Production Patterns**
> Part of: GenAI + LLMOps Engineering Roadmap 2026
> Estimated time: 35–45 hrs over 2–3 weeks
> Prerequisites: Phase 1 (LLM APIs, Streaming basics) + Phase 2 (RAG, Embeddings)

---

## 📑 Table of Contents

1. [3.1 — AI-Specific API Design](#31--ai-specific-api-design)
2. [3.2 — Async AI Systems](#32--async-ai-systems)
3. [3.3 — Streaming AI APIs](#33--streaming-ai-apis)
4. [3.4 — Production Backend Patterns](#34--production-backend-patterns)
5. [Phase 3 Project](#-phase-3-project--production-ai-api)
6. [Interview Cheat Sheet](#-interview-cheat-sheet)
7. [Quick Revision Cards](#-quick-revision-cards)

---

## 3.1 — AI-Specific API Design

### 🧑‍🎓 Explain Like I'm in BTech 2nd Year

You already know how to build REST APIs — you've done it with FastAPI. But when LLMs enter the picture, everything changes.

A normal API endpoint is like a vending machine: you press a button, it gives you a product in 200ms. Done.

An LLM endpoint is like ordering custom food at a restaurant: the chef (LLM) takes time, can give different results each time, costs money per dish, and sometimes the kitchen (provider) is on fire.

This section covers what breaks when you plug an LLM into a standard REST API, and how to fix it.

---

### LLM-Backed Endpoint Contracts — Versioning When Prompts Change

**The problem nobody talks about:**

```
Normal API v1:  GET /users → returns {id, name, email}
Normal API v2:  GET /users → returns {id, name, email, avatar_url}
  → client breaks, you version the URL: /v2/users
  → clear contract, predictable

LLM API v1: POST /summarise → returns "The document discusses..."
LLM API:    you change the system prompt to improve quality
            → response format changes subtly
            → client parsing breaks
            → WHO CHANGED IT? Git doesn't track prompt changes clearly
```

**Why LLM APIs need explicit versioning:**

```python
# Bad — no versioning, silent prompt changes break clients
@router.post("/summarise")
async def summarise(doc: str) -> dict:
    response = await llm.complete(
        system="Summarise this document.",   # changed yesterday, nobody noticed
        user=doc,
    )
    return {"summary": response}

# Good — prompt version tracked, contract explicit
from enum import Enum

class SummaryFormat(str, Enum):
    BULLET_POINTS = "bullet_points"   # v1
    PARAGRAPH     = "paragraph"       # v2 (new format)
    JSON_STRUCT   = "json_structured" # v3

@router.post("/v1/summarise")
async def summarise_v1(body: SummariseRequest) -> SummariseResponseV1:
    """
    V1: Returns plain text summary.
    Prompt version: prompts/summarise/v1.txt
    Deprecated: use /v2/summarise for structured output.
    """
    ...

@router.post("/v2/summarise")
async def summarise_v2(body: SummariseRequestV2) -> SummariseResponseV2:
    """
    V2: Returns structured summary with sections and key points.
    Prompt version: prompts/summarise/v2.txt
    """
    ...
```

**Prompt file management:**
```
prompts/
  summarise/
    v1.txt     ← "Summarise this document in 3 sentences."
    v2.txt     ← "Summarise with sections: TL;DR, Key Points, Action Items."
    v3.txt     ← current dev version
  rag/
    grounding/
      v1.txt
      v2.txt
```

```python
# backend/core/prompts.py
from pathlib import Path
from functools import lru_cache

PROMPTS_DIR = Path(__file__).parent.parent / "prompts"

@lru_cache(maxsize=50)
def load_prompt(name: str, version: str = "v1") -> str:
    """
    Load prompt from file with caching.
    Never hardcode prompts in Python files.
    """
    path = PROMPTS_DIR / name / f"{version}.txt"
    if not path.exists():
        raise FileNotFoundError(f"Prompt not found: {name}/{version}")
    return path.read_text(encoding="utf-8")

# Usage
system_prompt = load_prompt("summarise", version="v2")
```

**The contract rule:**
```
Every time you change a prompt that affects output structure → bump the version.
Every breaking change to response schema → bump the API version.
Always keep at least 1 previous version active for 30 days.
```

---

### Idempotency for LLM Requests — No Double Charging

**The problem:**

```
Client sends: POST /generate-report (expensive LLM call, costs $0.50)
Network timeout after 29 seconds
Client thinks it failed
Client retries: POST /generate-report
Server runs AGAIN → charges $1.00 total, report generated twice

This is not theoretical. At scale it happens constantly.
```

**Solution — Idempotency Keys:**

```python
# backend/api/middleware/idempotency.py
import hashlib
import json
from fastapi import Request, HTTPException
import redis.asyncio as aioredis

class IdempotencyMiddleware:
    """
    Client sends: X-Idempotency-Key: <unique-key>
    First request:  run LLM, store result in Redis (TTL 24hr), return result
    Second request: return cached result immediately, no LLM call
    """

    def __init__(self, redis_client: aioredis.Redis, ttl: int = 86400):
        self.redis = redis_client
        self.ttl   = ttl   # 24 hours

    async def get_or_execute(
        self,
        idempotency_key: str,
        operation,         # async callable
    ):
        cache_key = f"idempotent:{idempotency_key}"

        # Check if we already ran this
        cached = await self.redis.get(cache_key)
        if cached:
            return json.loads(cached), True  # (result, is_cached)

        # First time — run the operation
        result = await operation()

        # Store result (idempotency window: 24 hours)
        await self.redis.setex(
            cache_key,
            self.ttl,
            json.dumps(result, default=str),
        )
        return result, False

# Usage in route
@router.post("/generate-report")
async def generate_report(
    body:              ReportRequest,
    x_idempotency_key: str | None = Header(default=None),
    current_user:      AuthUser    = Depends(get_current_user),
    redis:             Redis       = Depends(get_redis),
):
    if not x_idempotency_key:
        raise HTTPException(400, "X-Idempotency-Key header required for this endpoint")

    # Include user context in key (prevent cross-user cache collisions)
    full_key = f"{current_user.user_id}:{x_idempotency_key}"

    idempotency = IdempotencyMiddleware(redis)
    result, was_cached = await idempotency.get_or_execute(
        full_key,
        lambda: _run_report_generation(body, current_user),
    )

    return {
        "data":      result,
        "cached":    was_cached,
        "cache_key": x_idempotency_key,
    }
```

**Client-side pattern (frontend):**
```typescript
// Generate idempotency key client-side (UUID tied to this specific action)
// Same key = same result, no double execution
const generateReport = async (config: ReportConfig) => {
  const idempotencyKey = `report-${userId}-${Date.now()}`

  try {
    const result = await api.post('/generate-report', config, {
      headers: { 'X-Idempotency-Key': idempotencyKey },
    })
    return result.data
  } catch (error) {
    if (isNetworkError(error)) {
      // Safe to retry with SAME key — server handles deduplication
      return api.post('/generate-report', config, {
        headers: { 'X-Idempotency-Key': idempotencyKey },
      })
    }
    throw error
  }
}
```

---

### Webhook Patterns for Long-Running LLM Jobs

**The problem:**

```
POST /ingest-document (500 pages, takes 3 minutes)
↓
HTTP times out at 30 seconds
↓
Client: "Did it work?"

HTTP is not designed for 3-minute operations.
```

**Solution — Async Job + Webhook or Polling:**

```python
# backend/api/routes/data.py

# Pattern 1: Return job ID immediately, client polls
@router.post("/ingest")
async def ingest_document(
    file:         UploadFile,
    webhook_url:  str | None = Body(default=None),  # optional webhook
    current_user: AuthUser   = Depends(get_current_user),
) -> IngestJobResponse:
    """
    Returns immediately with a job_id.
    Client either polls GET /jobs/{job_id} or receives webhook callback.
    """
    # Store file in S3
    s3_key = await s3.upload(file, current_user.org_id)

    # Create job record
    job = await JobRepository.create({
        "org_id":      current_user.org_id,
        "user_id":     current_user.user_id,
        "type":        "document_ingestion",
        "status":      "queued",
        "s3_key":      s3_key,
        "webhook_url": webhook_url,
    })

    # Push to Celery queue
    ingest_document_task.delay(
        job_id=str(job.id),
        s3_key=s3_key,
        webhook_url=webhook_url,
    )

    return IngestJobResponse(
        job_id=str(job.id),
        status="queued",
        poll_url=f"/api/jobs/{job.id}",
        estimated_seconds=120,
    )

# Pattern 2: Polling endpoint
@router.get("/jobs/{job_id}")
async def get_job_status(
    job_id:       str,
    current_user: AuthUser = Depends(get_current_user),
) -> JobStatusResponse:
    job = await JobRepository.get(job_id, current_user.org_id)
    return JobStatusResponse(
        job_id=job_id,
        status=job.status,       # queued | running | done | error
        progress=job.progress,   # 0-100
        result=job.result,       # populated when done
        error=job.error,         # populated when failed
    )

# Pattern 3: Celery task sends webhook on completion
@celery.task
def ingest_document_task(job_id: str, s3_key: str, webhook_url: str | None):
    try:
        # ... do the work ...
        JobRepository.update_sync(job_id, {"status": "done", "progress": 100})

        if webhook_url:
            import httpx
            httpx.post(webhook_url, json={
                "job_id": job_id,
                "status": "done",
                "event":  "document.ingested",
            }, headers={"X-Webhook-Secret": settings.webhook_secret})

    except Exception as e:
        JobRepository.update_sync(job_id, {"status": "error", "error": str(e)})
        if webhook_url:
            httpx.post(webhook_url, json={"job_id": job_id, "status": "error"})
```

---

### Request/Response Schema Evolution

```python
# How to handle schema changes without breaking existing clients

# V1 response
class SummariseResponseV1(BaseModel):
    summary: str                    # plain text only

# V2 response — added fields (backward compatible: V1 clients ignore new fields)
class SummariseResponseV2(BaseModel):
    summary:     str
    key_points:  list[str] = []    # new field, default = safe for V1 clients
    word_count:  int       = 0     # new field, default = safe for V1 clients

# V3 response — changed field type (BREAKING — must version the endpoint)
class SummariseResponseV3(BaseModel):
    summary:    SummarySection     # was str, now object — BREAKING
    key_points: list[KeyPoint]     # was list[str], now list[object] — BREAKING
    word_count: int

# Rule: if existing clients break when they receive the new response → new version
# Rule: if existing clients can safely ignore new fields → backward compatible
```

---

## 3.2 — Async AI Systems

### 🧑‍🎓 Explain Like I'm in BTech 2nd Year

Think of async like a restaurant kitchen vs a fast food counter.

**Sync (blocking):** One chef. Takes one order. Makes the food. Serves it. Takes next order. Everyone waits in line. If one dish takes 20 minutes — everyone waits 20 minutes.

**Async:** One chef. Takes all orders simultaneously. Starts all dishes at once. Serves each as it's ready. 10 orders, still finishes in 20 minutes total if the bottleneck is waiting (LLM API call), not cooking (CPU work).

LLM API calls are 90% *waiting*. Async is non-negotiable.

---

### Async LLM Calls — Never Block the Event Loop

```python
# ❌ WRONG — blocks the entire event loop during LLM call
# While this runs, NO other requests can be handled
@router.post("/chat")
def chat_sync(body: ChatRequest):
    import openai
    client = openai.OpenAI()                    # sync client
    response = client.chat.completions.create(  # BLOCKS event loop
        model="gpt-4o",
        messages=body.messages,
    )
    return {"answer": response.choices[0].message.content}

# ✅ CORRECT — yields control to event loop during LLM call
# Other requests handled while waiting for LLM response
@router.post("/chat")
async def chat_async(body: ChatRequest):
    from openai import AsyncOpenAI
    client = AsyncOpenAI()                            # async client
    response = await client.chat.completions.create(  # yields to event loop
        model="gpt-4o",
        messages=body.messages,
    )
    return {"answer": response.choices[0].message.content}
```

**Why it matters in numbers:**

```
Server: 10 concurrent requests, each LLM call takes 3 seconds

Sync:   request 1 takes 3s, request 2 takes 6s, ..., request 10 takes 30s
        Total: ~30 seconds for all requests

Async:  all 10 requests start simultaneously, all finish around 3-4s
        Total: ~4 seconds for all requests

At 100 req/s: sync server needs 100 workers; async needs ~5 workers
```

---

### `asyncio.gather` — Parallel LLM Calls

```python
import asyncio
from openai import AsyncOpenAI

client = AsyncOpenAI()

# Pattern: fan-out — run N LLM calls in parallel, wait for all
async def multi_perspective_analysis(document: str) -> dict:
    """
    Instead of sequential:
      summary     = await summarise(document)    # 2 seconds
      sentiment   = await analyse_sentiment(doc) # 2 seconds
      key_topics  = await extract_topics(doc)    # 2 seconds
      Total: 6 seconds

    Run in parallel:
      All 3 start simultaneously → Total: ~2 seconds (fastest one)
    """

    summary_task    = summarise(document)
    sentiment_task  = analyse_sentiment(document)
    topics_task     = extract_topics(document)

    # All 3 run in parallel — total time = max(2, 2, 2) = 2 seconds
    summary, sentiment, topics = await asyncio.gather(
        summary_task,
        sentiment_task,
        topics_task,
    )

    return {
        "summary":   summary,
        "sentiment": sentiment,
        "topics":    topics,
    }

# Pattern: fan-out with error handling — some can fail
async def multi_source_retrieval(
    query: str,
    source_ids: list[str],
) -> list[list[Chunk]]:
    """Retrieve from all sources in parallel. If one fails, others continue."""

    tasks = [
        retrieve_from_source(query, source_id)
        for source_id in source_ids
    ]

    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Filter out exceptions, keep successful results
    successful = []
    for source_id, result in zip(source_ids, results):
        if isinstance(result, Exception):
            logger.warning(f"Retrieval failed for source {source_id}: {result}")
        else:
            successful.extend(result)

    return successful

# Pattern: fan-out with timeout — don't wait forever
async def parallel_with_timeout(tasks: list, timeout_seconds: float = 10.0):
    try:
        results = await asyncio.wait_for(
            asyncio.gather(*tasks, return_exceptions=True),
            timeout=timeout_seconds,
        )
        return results
    except asyncio.TimeoutError:
        logger.error(f"Parallel tasks timed out after {timeout_seconds}s")
        raise
```

---

### Background Tasks — FastAPI BackgroundTasks

```python
from fastapi import BackgroundTasks

# Use for: lightweight async work that doesn't need a result
# Don't use for: heavy CPU work, tasks that need to survive server restart

@router.post("/chat")
async def chat(
    body:             ChatRequest,
    background_tasks: BackgroundTasks,
    current_user:     AuthUser = Depends(get_current_user),
) -> ChatResponse:

    # Main response — fast
    answer = await rag_pipeline.run(body.question, body.source_ids)

    # Background work — fires after response sent, doesn't block client
    background_tasks.add_task(
        store_conversation,         # function to call
        question=body.question,     # kwargs
        answer=answer,
        user_id=current_user.user_id,
        org_id=current_user.org_id,
    )

    background_tasks.add_task(
        maybe_run_ragas_eval,
        conversation_id=conversation_id,
    )

    return ChatResponse(answer=answer)

# Limitation: BackgroundTasks dies if server restarts
# For durability: use Celery instead
```

---

### Celery + Redis — Heavy LLM Batch Jobs

```python
# backend/core/celery_app.py
from celery import Celery
from core.config import settings

celery_app = Celery(
    "synapseiq",
    broker=settings.celery_broker_url,    # redis:// locally, sqs:// in prod
    backend=settings.celery_result_url,   # redis:// for result storage
)

celery_app.conf.update(
    task_serializer   = "json",
    result_serializer = "json",
    accept_content    = ["json"],
    task_track_started= True,             # task.status = "STARTED" immediately
    task_acks_late    = True,             # don't ack until task completes (safer)
    worker_prefetch_multiplier = 1,       # one task at a time per worker
    task_routes = {
        "workers.ingestion.*": {"queue": "ingestion"},
        "workers.eval.*":      {"queue": "eval"},
        "workers.reports.*":   {"queue": "reports"},
    },
)

# backend/workers/ingestion.py
from core.celery_app import celery_app

@celery_app.task(
    bind=True,
    name="workers.ingestion.embed_chunks",
    max_retries=3,
    autoretry_for=(Exception,),
    retry_backoff=True,        # 60s → 120s → 240s
    retry_jitter=True,         # add random jitter to avoid thundering herd
    rate_limit="10/m",         # max 10 batch embeds per minute
    queue="ingestion",
)
def embed_chunks_task(self, source_id: str, chunk_ids: list[str], org_id: str):
    """
    Embed chunks in batches of 100.
    Runs in background, doesn't block the API.
    """
    from openai import OpenAI
    client = OpenAI()

    chunks  = ChunkRepository.get_many_sync(chunk_ids)
    batches = [chunks[i:i+100] for i in range(0, len(chunks), 100)]

    for i, batch in enumerate(batches):
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=[c.content for c in batch],
        )
        embeddings = [d.embedding for d in response.data]
        QdrantStore.upsert_sync(source_id, batch, embeddings)

        # Update progress in Redis
        progress = int((i+1) / len(batches) * 100)
        redis_client.set(f"progress:{source_id}", progress, ex=3600)

    SourceRepository.update_sync(source_id, {"status": "ready"})
```

---

### Job Status Polling API with SSE Progress

```python
# backend/api/routes/data.py — SSE progress stream
from fastapi.responses import StreamingResponse
import asyncio, json

@router.get("/sources/{source_id}/status")
async def source_status_sse(
    source_id:    str,
    current_user: AuthUser = Depends(get_current_user),
    redis:        Redis    = Depends(get_redis),
):
    """
    Server-Sent Events endpoint.
    Streams progress updates as the ingestion pipeline runs.
    Client receives: data: {"stage":"embedding","pct":67}\n\n
    """
    # Verify ownership before streaming
    await SourceRepository.get(source_id, current_user.org_id)

    async def generate():
        prev_data = None
        for _ in range(180):   # max 3 minutes
            raw = await redis.get(f"progress:{source_id}")

            if raw:
                data = json.loads(raw)

                # Only send if changed (avoid spam)
                if data != prev_data:
                    yield f"data: {json.dumps(data)}\n\n"
                    prev_data = data

                if data.get("pct") == 100 or data.get("stage") == "error":
                    yield "data: {\"done\": true}\n\n"
                    return

            await asyncio.sleep(1)

        yield "data: {\"error\": \"timeout\"}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":    "no-cache",
            "X-Accel-Buffering":"no",    # disable nginx buffering
        },
    )
```

---

## 3.3 — Streaming AI APIs

### 🧑‍🎓 Explain Like I'm in BTech 2nd Year

Without streaming:
```
User: "Summarise this 50-page document"
UI:   [spinner for 8 seconds]
UI:   [suddenly shows full 500-word answer]
UX:   terrible — user thinks it's broken
```

With streaming:
```
User: "Summarise this 50-page document"
UI:   [text starts appearing immediately, word by word]
UI:   [answer builds over 8 seconds]
UX:   feels fast and responsive even though same total time
```

Streaming is the difference between "this AI feels broken" and "this AI feels alive."

---

### `StreamingResponse` in FastAPI

```python
from fastapi.responses import StreamingResponse
from openai import AsyncOpenAI

client = AsyncOpenAI()

@router.post("/chat/stream")
async def chat_stream(body: ChatRequest) -> StreamingResponse:

    async def generate():
        """
        Generator that yields tokens as they arrive from the LLM.
        FastAPI streams each yielded chunk to the client immediately.
        """
        stream = await client.chat.completions.create(
            model="gpt-4o",
            messages=body.messages,
            stream=True,             # ← key: enables streaming
        )

        async for chunk in stream:
            delta = chunk.choices[0].delta
            if delta.content:
                # Yield each token as SSE event
                yield f"data: {json.dumps({'token': delta.content})}\n\n"

        # Signal stream is complete
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
```

---

### Full SSE Pipeline: LLM → FastAPI → React

```python
# Backend: RAG pipeline with streaming
# backend/services/rag/pipeline.py

async def stream_rag_response(
    question:   str,
    source_ids: list[str],
    org_id:     str,
    config:     RAGConfig,
) -> AsyncIterator[str]:
    """
    Full RAG pipeline that streams tokens to the client.
    Yields SSE-formatted strings.
    """

    # 1. Retrieve chunks (non-streaming, fast)
    embedding = await embed(question)
    chunks    = await retriever.retrieve(embedding, org_id, source_ids)
    reranked  = await reranker.rerank(question, chunks, top_n=5)
    context   = build_context(reranked, config)

    # 2. Yield metadata before streaming starts
    yield f"data: {json.dumps({'type': 'metadata', 'chunk_count': len(reranked)})}\n\n"

    # 3. Stream LLM generation token by token
    from groq import AsyncGroq
    groq_client = AsyncGroq()

    stream = await groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": RAG_SYSTEM_PROMPT},
            {"role": "user",   "content": f"Context:\n{context}\n\nQuestion: {question}"},
        ],
        stream=True,
    )

    full_response = ""
    async for chunk in stream:
        delta = chunk.choices[0].delta
        if delta.content:
            full_response += delta.content
            yield f"data: {json.dumps({'type': 'token', 'content': delta.content})}\n\n"

    # 4. Yield citations after generation complete
    citations = extract_citations(reranked, full_response)
    yield f"data: {json.dumps({'type': 'citations', 'data': citations})}\n\n"

    # 5. Signal done
    yield "data: [DONE]\n\n"
```

```typescript
// Frontend: consuming the SSE stream in React
// hooks/useChat.ts

export function useChatStream() {
  const [tokens, setTokens]     = useState('')
  const [citations, setCitations] = useState<Citation[]>([])
  const [isStreaming, setIsStreaming] = useState(false)

  const sendMessage = async (question: string, sourceIds: string[]) => {
    setTokens('')
    setCitations([])
    setIsStreaming(true)

    const response = await fetch('/api/chat/rag/stream', {
      method: 'POST',
      headers: {
        'Content-Type':  'application/json',
        'Authorization': `Bearer ${getToken()}`,
      },
      body: JSON.stringify({ question, source_ids: sourceIds }),
    })

    const reader = response.body!.getReader()
    const decoder = new TextDecoder()

    while (true) {
      const { done, value } = await reader.read()
      if (done) break

      const text = decoder.decode(value)
      const lines = text.split('\n')

      for (const line of lines) {
        if (!line.startsWith('data: ')) continue
        const data = line.slice(6)   // remove "data: " prefix

        if (data === '[DONE]') {
          setIsStreaming(false)
          break
        }

        const parsed = JSON.parse(data)

        if (parsed.type === 'token') {
          // Append each token as it arrives
          setTokens(prev => prev + parsed.content)
        } else if (parsed.type === 'citations') {
          setCitations(parsed.data)
        }
      }
    }
  }

  return { tokens, citations, isStreaming, sendMessage }
}
```

---

### Streaming with Tool Use — Partial Tool Call Assembly

```python
# When LLM uses tools mid-stream, tool calls come in fragments
# You need to assemble them before you can execute the tool

async def stream_with_tools(messages: list[dict]) -> AsyncIterator[str]:
    client  = AsyncOpenAI()
    stream  = await client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        tools=TOOL_SCHEMAS,
        stream=True,
    )

    # Accumulate tool call fragments
    tool_calls: dict[int, dict] = {}
    content_buffer = ""

    async for chunk in stream:
        delta = chunk.choices[0].delta

        # Text content — stream immediately
        if delta.content:
            content_buffer += delta.content
            yield f"data: {json.dumps({'type': 'token', 'content': delta.content})}\n\n"

        # Tool calls — accumulate fragments, don't execute yet
        if delta.tool_calls:
            for tc in delta.tool_calls:
                idx = tc.index
                if idx not in tool_calls:
                    tool_calls[idx] = {
                        "id":       tc.id or "",
                        "name":     tc.function.name or "" if tc.function else "",
                        "args_str": "",
                    }
                # Append argument fragment (arrives as partial JSON string)
                if tc.function and tc.function.arguments:
                    tool_calls[idx]["args_str"] += tc.function.arguments

        # Stream finished — now execute accumulated tool calls
        if chunk.choices[0].finish_reason == "tool_calls":
            for tc_data in tool_calls.values():
                args = json.loads(tc_data["args_str"])  # now complete JSON

                # Notify frontend a tool is running
                yield f"data: {json.dumps({'type': 'tool_start', 'tool': tc_data['name']})}\n\n"

                # Execute the tool
                result = await execute_tool(tc_data["name"], args)

                yield f"data: {json.dumps({'type': 'tool_result', 'result': result})}\n\n"
```

---

### Backpressure — Client Reads Slower Than LLM Generates

```python
# Problem: LLM generates 50 tokens/second, client processes 10 tokens/second
# Without backpressure: buffer overflows, connection drops, data lost

# FastAPI's StreamingResponse handles basic backpressure via Python's async generator
# When client is slow, `yield` blocks until client catches up
# This is automatic — but set appropriate timeouts

@router.post("/chat/stream")
async def chat_stream(body: ChatRequest) -> StreamingResponse:

    async def generate():
        stream = await client.chat.completions.create(
            model="gpt-4o", messages=body.messages, stream=True,
        )
        async for chunk in stream:
            if chunk.choices[0].delta.content:
                # If client is slow, this yield will wait
                # Python async generator handles backpressure
                yield f"data: {json.dumps({'token': chunk.choices[0].delta.content})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":     "no-cache",
            "X-Accel-Buffering": "no",
            "Connection":        "keep-alive",
        },
    )

# Nginx config for streaming (no buffering)
# proxy_buffering off;
# proxy_cache off;
# proxy_read_timeout 300s;  # 5 minutes max stream duration
```

---

### Mid-Stream Error Recovery

```python
async def resilient_stream(question: str, context: str) -> AsyncIterator[str]:
    """
    Streams tokens. If LLM fails mid-stream, sends an error event
    instead of leaving the client hanging with a broken connection.
    """
    partial_response = ""

    try:
        stream = await groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": f"{context}\n\n{question}"}],
            stream=True,
        )

        async for chunk in stream:
            if chunk.choices[0].delta.content:
                token = chunk.choices[0].delta.content
                partial_response += token
                yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"

        yield "data: [DONE]\n\n"

    except Exception as e:
        logger.error(f"Stream error mid-generation: {e}")

        # Partial response was sent — tell client it's incomplete
        yield f"data: {json.dumps({'type': 'error', 'message': 'Generation interrupted. Partial response shown.', 'partial': partial_response != ''})}\n\n"

        # Try fallback: generate a short non-streamed response
        try:
            fallback = await groq_client.chat.completions.create(
                model="llama-3.1-8b-instant",   # faster, smaller fallback
                messages=[{"role": "user", "content": f"Brief answer only: {question}"}],
                max_tokens=200,
            )
            fallback_text = fallback.choices[0].message.content
            yield f"data: {json.dumps({'type': 'fallback', 'content': fallback_text})}\n\n"
        except Exception:
            pass   # give up gracefully

        yield "data: [DONE]\n\n"
```

---

## 3.4 — Production Backend Patterns

### 🧑‍🎓 Explain Like I'm in BTech 2nd Year

This section is the gap between "it works on localhost" and "it works for 1000 users simultaneously."

Every pattern here exists because someone ran it without it, something broke in production, and this pattern was the fix.

---

### Conversation Session Management — Redis-Backed Multi-Turn Chat

```python
# Problem: HTTP is stateless. Each request knows nothing about previous ones.
# Solution: Store conversation history in Redis. Load on each request.

# backend/services/memory/session_manager.py
import json
from dataclasses import dataclass, asdict
from datetime import datetime

@dataclass
class Message:
    role:      str         # "user" | "assistant" | "system"
    content:   str
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.utcnow().isoformat()

class ConversationSessionManager:
    """
    Stores conversation history in Redis.
    Key: session:{user_id}:{conversation_id}
    TTL: 24 hours (reset on each access)
    """

    SESSION_TTL = 86400    # 24 hours
    MAX_MESSAGES = 50      # hard cap to prevent unbounded growth

    def __init__(self, redis):
        self.redis = redis

    def _key(self, user_id: str, conversation_id: str) -> str:
        return f"session:{user_id}:{conversation_id}"

    async def get_history(
        self, user_id: str, conversation_id: str
    ) -> list[Message]:
        raw = await self.redis.get(self._key(user_id, conversation_id))
        if not raw:
            return []
        messages = json.loads(raw)
        return [Message(**m) for m in messages]

    async def add_message(
        self, user_id: str, conversation_id: str, message: Message
    ) -> list[Message]:
        history = await self.get_history(user_id, conversation_id)
        history.append(message)

        # Hard cap — remove oldest messages first
        if len(history) > self.MAX_MESSAGES:
            history = history[-self.MAX_MESSAGES:]

        await self.redis.setex(
            self._key(user_id, conversation_id),
            self.SESSION_TTL,
            json.dumps([asdict(m) for m in history]),
        )
        return history

    async def clear(self, user_id: str, conversation_id: str) -> None:
        await self.redis.delete(self._key(user_id, conversation_id))

# Usage in chat endpoint
@router.post("/chat")
async def chat(
    body:         ChatRequest,
    current_user: AuthUser = Depends(get_current_user),
    session_mgr:  ConversationSessionManager = Depends(get_session_manager),
):
    # Load existing history
    history = await session_mgr.get_history(
        current_user.user_id,
        body.conversation_id,
    )

    # Add user's message
    history = await session_mgr.add_message(
        current_user.user_id,
        body.conversation_id,
        Message(role="user", content=body.question),
    )

    # Build messages for LLM (history → current context)
    llm_messages = [{"role": m.role, "content": m.content} for m in history]

    # Get LLM response
    answer = await llm_complete(messages=llm_messages)

    # Store assistant response
    await session_mgr.add_message(
        current_user.user_id,
        body.conversation_id,
        Message(role="assistant", content=answer),
    )

    return {"answer": answer, "conversation_id": body.conversation_id}
```

---

### Context Window Management — Trim Old Turns When Context Fills

```python
# backend/services/memory/context_manager.py

import tiktoken

ENCODER = tiktoken.encoding_for_model("gpt-4o")

def count_tokens(text: str) -> int:
    return len(ENCODER.encode(text))

def count_messages_tokens(messages: list[dict]) -> int:
    return sum(count_tokens(m["content"]) for m in messages)

class ContextWindowManager:
    """
    Keeps conversation history within model's context window.
    Strategy: summarise oldest messages, keep recent ones in full.
    """

    def __init__(
        self,
        max_context_tokens: int = 6000,  # leave room for response
        keep_recent:        int = 5,     # always keep last N messages
    ):
        self.max_tokens  = max_context_tokens
        self.keep_recent = keep_recent

    async def trim(
        self,
        messages:       list[dict],
        system_prompt:  str = "",
        llm_client      = None,
    ) -> list[dict]:
        """
        Returns messages trimmed to fit within token budget.
        1. Always keep system prompt
        2. Always keep last N messages in full
        3. Summarise everything older if needed
        """
        system_tokens = count_tokens(system_prompt)
        budget        = self.max_tokens - system_tokens

        # Split: recent (always keep) vs older (may summarise)
        recent = messages[-self.keep_recent:] if len(messages) > self.keep_recent else messages
        older  = messages[:-self.keep_recent] if len(messages) > self.keep_recent else []

        recent_tokens = count_messages_tokens(recent)

        if not older:
            return recent   # no trimming needed

        remaining_budget = budget - recent_tokens

        if remaining_budget <= 0:
            # Even recent messages are too long — just use them (truncated by model)
            return recent

        if remaining_budget > count_messages_tokens(older):
            # Older messages fit — include all
            return older + recent

        # Need to summarise older messages
        if llm_client:
            older_text = "\n".join(
                f"{m['role'].upper()}: {m['content']}" for m in older
            )
            summary = await llm_client.chat.completions.create(
                model="llama-3.1-8b-instant",   # cheap model for summarisation
                messages=[{
                    "role":    "user",
                    "content": f"Summarise this conversation in 2-3 sentences, preserving key facts:\n\n{older_text}",
                }],
                max_tokens=200,
            )
            summary_text = summary.choices[0].message.content
            summary_message = {
                "role":    "system",
                "content": f"[Earlier conversation summary]: {summary_text}",
            }
            return [summary_message] + recent

        # No LLM for summarisation — just drop oldest messages
        trimmed = older[:]
        while trimmed and count_messages_tokens(trimmed) > remaining_budget:
            trimmed = trimmed[1:]   # drop oldest

        return trimmed + recent
```

---

### Multi-Model Fallback — LiteLLM Router

```python
# backend/providers/llm_factory.py
# Use litellm for automatic multi-provider fallback

import litellm

# Configure fallback chain: if Groq fails → try OpenAI → try Bedrock
litellm.set_verbose = False

FALLBACK_CHAIN = [
    "groq/llama-3.3-70b-versatile",     # primary: fast + cheap
    "openai/gpt-4o-mini",               # fallback 1: reliable
    "bedrock/anthropic.claude-haiku-4", # fallback 2: last resort
]

async def llm_complete_with_fallback(
    messages:    list[dict],
    task:        str = "general",
    max_retries: int = 2,
    **kwargs,
) -> str:
    """
    Tries each provider in order.
    Automatically falls back on RateLimitError, APIError, TimeoutError.
    """
    last_error = None

    for model in FALLBACK_CHAIN:
        try:
            response = await litellm.acompletion(
                model=model,
                messages=messages,
                num_retries=max_retries,   # retries within same provider
                timeout=30,
                **kwargs,
            )
            return response.choices[0].message.content

        except litellm.RateLimitError as e:
            logger.warning(f"Rate limit on {model}, trying next provider")
            last_error = e
            continue

        except litellm.APIError as e:
            logger.warning(f"API error on {model}: {e}, trying next provider")
            last_error = e
            continue

        except litellm.Timeout:
            logger.warning(f"Timeout on {model}, trying next provider")
            last_error = litellm.Timeout(model)
            continue

    raise RuntimeError(f"All providers failed. Last error: {last_error}")
```

---

### Circuit Breaker — Stop Sending to Failing Provider

```python
# backend/providers/circuit_breaker.py
import time
from enum import Enum

class CircuitState(Enum):
    CLOSED   = "closed"    # normal — requests flow through
    OPEN     = "open"      # failing — block all requests immediately
    HALF_OPEN= "half_open" # testing recovery — allow one request through

class CircuitBreaker:
    """
    Stops sending requests to a provider that keeps failing.
    After FAILURE_THRESHOLD failures → OPEN (fast fail).
    After RECOVERY_TIMEOUT seconds → HALF_OPEN (test one request).
    If test succeeds → CLOSED. If fails → OPEN again.
    """

    def __init__(
        self,
        failure_threshold: int   = 5,
        recovery_timeout:  float = 60.0,  # seconds before retry
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout  = recovery_timeout
        self.failures          = 0
        self.last_failure_time = 0.0
        self.state             = CircuitState.CLOSED

    def call_succeeded(self):
        self.failures = 0
        self.state    = CircuitState.CLOSED

    def call_failed(self):
        self.failures         += 1
        self.last_failure_time = time.time()
        if self.failures >= self.failure_threshold:
            self.state = CircuitState.OPEN
            logger.warning(f"Circuit OPEN after {self.failures} failures")

    def can_attempt(self) -> bool:
        if self.state == CircuitState.CLOSED:
            return True

        if self.state == CircuitState.OPEN:
            # Check if recovery window has passed
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = CircuitState.HALF_OPEN
                logger.info("Circuit HALF_OPEN — testing recovery")
                return True   # allow one test request
            return False   # still OPEN — fast fail

        return True   # HALF_OPEN — allow the test

    async def execute(self, operation):
        if not self.can_attempt():
            raise RuntimeError(
                f"Circuit OPEN — provider unavailable. "
                f"Retry after {self.recovery_timeout}s"
            )
        try:
            result = await operation()
            self.call_succeeded()
            return result
        except Exception as e:
            self.call_failed()
            raise

# Global circuit breakers per provider
_circuit_breakers: dict[str, CircuitBreaker] = {
    "groq":    CircuitBreaker(failure_threshold=5, recovery_timeout=60),
    "openai":  CircuitBreaker(failure_threshold=5, recovery_timeout=60),
    "bedrock": CircuitBreaker(failure_threshold=3, recovery_timeout=120),
}

async def protected_llm_call(provider: str, operation) -> str:
    breaker = _circuit_breakers.get(provider, CircuitBreaker())
    return await breaker.execute(operation)
```

---

### Request Deduplication — Prevent Duplicate LLM Charges

```python
# Different from idempotency keys (client-controlled).
# This is server-side deduplication based on request fingerprint.

import hashlib

def fingerprint_request(messages: list[dict], model: str) -> str:
    """
    Create a unique fingerprint for a deterministic LLM request.
    Same messages + same model = same fingerprint.
    Only works for temperature=0 (deterministic) requests.
    """
    content = json.dumps({
        "messages": messages,
        "model":    model,
    }, sort_keys=True)
    return hashlib.sha256(content.encode()).hexdigest()

async def deduped_llm_call(
    messages: list[dict],
    model:    str,
    redis:    Redis,
    ttl:      int = 3600,   # cache for 1 hour
) -> str:
    """
    For temperature=0 calls: cache result by fingerprint.
    Same input → same output guaranteed → safe to cache.
    """
    key = f"llm_cache:{fingerprint_request(messages, model)}"

    cached = await redis.get(key)
    if cached:
        logger.debug("LLM cache hit", extra={"extra_fields": {"key": key[:16]}})
        return cached.decode()

    response = await litellm.acompletion(
        model=model,
        messages=messages,
        temperature=0,   # deterministic — required for caching
    )
    result = response.choices[0].message.content

    await redis.setex(key, ttl, result)
    return result
```

---

### Cost Tracking Middleware

```python
# backend/api/middleware/cost_middleware.py
# Track LLM token costs per request, per user, per feature

COST_PER_1K_TOKENS = {
    "groq/llama-3.3-70b-versatile": {"input": 0.00059, "output": 0.00079},
    "groq/llama-3.1-8b-instant":    {"input": 0.00005, "output": 0.00008},
    "openai/gpt-4o":                {"input": 0.0025,  "output": 0.010},
    "openai/gpt-4o-mini":           {"input": 0.00015, "output": 0.00060},
    "openai/text-embedding-3-small": {"input": 0.00002, "output": 0},
}

def calculate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    rates = COST_PER_1K_TOKENS.get(model, {"input": 0.001, "output": 0.002})
    return (
        (input_tokens  / 1000) * rates["input"] +
        (output_tokens / 1000) * rates["output"]
    )

async def track_llm_cost(
    org_id:        str,
    user_id:       str,
    feature:       str,   # "rag_query" | "sql_agent" | "embedding" | "report"
    model:         str,
    input_tokens:  int,
    output_tokens: int,
    redis:         Redis,
    db:            AsyncSession,
):
    cost = calculate_cost(model, input_tokens, output_tokens)

    # Update daily Redis counter (for real-time rate limiting)
    today = datetime.utcnow().strftime("%Y-%m-%d")
    await redis.incrbyfloat(f"daily_cost:{org_id}:{today}", cost)
    await redis.incrbyfloat(f"daily_cost:{org_id}:{feature}:{today}", cost)

    # Write to DB for historical analytics
    await CostRepository.create({
        "org_id":        org_id,
        "user_id":       user_id,
        "feature":       feature,
        "model":         model,
        "input_tokens":  input_tokens,
        "output_tokens": output_tokens,
        "cost_usd":      cost,
    })

    logger.info("LLM cost tracked", extra={"extra_fields": {
        "cost_usd":      round(cost, 6),
        "model":         model,
        "input_tokens":  input_tokens,
        "output_tokens": output_tokens,
        "feature":       feature,
    }})
```

---

### Rate Limiting Per User — slowapi + Redis Sliding Window

```python
# backend/api/middleware/rate_limiter.py
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from fastapi import Request

# Global rate limiter
limiter = Limiter(
    key_func=get_remote_address,    # rate limit per IP (anonymous)
    default_limits=["100/minute"],  # default for all endpoints
)

# Register error handler
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Per-endpoint limits
@router.post("/chat/rag")
@limiter.limit("20/minute")   # 20 RAG queries per minute per IP
async def rag_chat(request: Request, body: ChatRequest):
    ...

@router.post("/chat/sql")
@limiter.limit("10/minute")   # SQL queries are more expensive
async def sql_chat(request: Request, body: ChatRequest):
    ...

# Per-user limits (authenticated — stricter, more precise)
async def check_user_rate_limit(
    user_id:    str,
    org_id:     str,
    endpoint:   str,
    redis:      Redis,
    limit:      int = 100,    # queries per day
) -> None:
    today = datetime.utcnow().strftime("%Y-%m-%d")
    key   = f"rate:{org_id}:{user_id}:{endpoint}:{today}"

    count = int(await redis.incr(key))
    await redis.expireat(key, end_of_day_unix())   # expires at midnight

    if count > limit:
        raise HTTPException(
            status_code=429,
            detail={
                "error":   "rate_limit_exceeded",
                "message": f"Daily limit of {limit} {endpoint} queries reached",
                "resets":  "midnight UTC",
                "count":   count,
                "limit":   limit,
            }
        )
```

---

### Exact Response Cache — Redis for Deterministic Prompts

```python
# backend/services/cache/response_cache.py
# For temperature=0 LLM calls with stable context — same input = same output

class ExactResponseCache:
    """
    L1 cache: exact match by SHA256(question + source_ids + model)
    Hit rate typically 25-40% in production
    """

    def __init__(self, redis: Redis, ttl: int = 3600):
        self.redis = redis
        self.ttl   = ttl

    def _cache_key(
        self, question: str, source_ids: list[str], model: str
    ) -> str:
        content = f"{question}|{'|'.join(sorted(source_ids))}|{model}"
        return f"cache:exact:{hashlib.sha256(content.encode()).hexdigest()}"

    async def get(
        self, question: str, source_ids: list[str], model: str
    ) -> dict | None:
        key    = self._cache_key(question, source_ids, model)
        cached = await self.redis.get(key)
        if cached:
            logger.info("Exact cache hit")
            return json.loads(cached)
        return None

    async def set(
        self,
        question:   str,
        source_ids: list[str],
        model:      str,
        response:   dict,
    ) -> None:
        key = self._cache_key(question, source_ids, model)
        await self.redis.setex(key, self.ttl, json.dumps(response))

    async def invalidate_source(self, source_id: str) -> None:
        """Call when source is updated — clear all related caches"""
        async for key in self.redis.scan_iter(f"cache:exact:*"):
            # Can't selectively invalidate by source_id with pure hash
            # Solution: include source_id in cache key structure
            await self.redis.delete(key)
```

---

## 🏗️ Phase 3 Project — Production AI API

**What you build:**

```
A production-ready AI API server that demonstrates all Phase 3 patterns.

Endpoints:
  POST /api/v1/chat/stream     → RAG + streaming SSE
  POST /api/v1/chat/sql        → SQL Agent (non-streaming)
  POST /api/v1/ingest          → Async document ingestion with webhook
  GET  /api/v1/jobs/{id}       → Job status polling
  GET  /api/v1/sources/{id}/status → SSE progress stream

Infrastructure:
  FastAPI + uvicorn
  Redis (sessions + cache + rate limiting)
  Celery worker (document ingestion)
  Docker Compose (all services)

Patterns demonstrated:
  ✅ Async throughout — no sync LLM calls
  ✅ Multi-model fallback (Groq → OpenAI via LiteLLM)
  ✅ Circuit breaker on each provider
  ✅ Redis conversation sessions
  ✅ Context window trimming with summarisation
  ✅ Streaming SSE end-to-end (LLM → FastAPI → React)
  ✅ Request deduplication (idempotency keys)
  ✅ Rate limiting per user (sliding window)
  ✅ Cost tracking per feature per org
  ✅ Structured JSON logging with request_id
  ✅ Background ingestion job with SSE progress
  ✅ Webhook callback on job completion
```

**Day-by-day breakdown:**

```
Day 1 (3 hrs): Async foundation
  - FastAPI project with async LLM call
  - Middleware: logging + request_id
  - Health check endpoint

Day 2 (3 hrs): Streaming
  - SSE streaming endpoint
  - Client-side EventSource consumption
  - Mid-stream error recovery

Day 3 (3 hrs): Session + Context
  - Redis conversation sessions
  - Context window trimming
  - Multi-turn chat working

Day 4 (3 hrs): Resilience
  - LiteLLM multi-model fallback
  - Circuit breaker implementation
  - Rate limiting with slowapi

Day 5 (3 hrs): Background jobs
  - Celery task for async ingestion
  - SSE progress updates
  - Webhook on completion
  - Idempotency keys

Day 6-7 (6 hrs): Production patterns
  - Cost tracking middleware
  - Response cache
  - Request deduplication
  - Docker Compose everything

Day 8 (3 hrs): Testing + cleanup
  - pytest for all services
  - Load test with locust (10 concurrent users)
  - Document what you built
```

---

## 📋 Interview Cheat Sheet

### Most Asked Questions

**Q: "Why async for LLM APIs?"**
```
LLM calls are I/O bound — 90% waiting, 10% compute.
Async yields control during the wait, serving other requests.
Sync: 10 concurrent users wait in line (total 30s for all)
Async: 10 concurrent users served simultaneously (total 3s)
Rule: never use sync LLM clients (openai.OpenAI) in FastAPI.
Always use AsyncOpenAI, AsyncGroq, etc.
```

**Q: "How do you handle LLM provider failures in production?"**
```
Three-layer resilience:
1. Retry within provider (3 attempts, exponential backoff)
2. Circuit breaker — after 5 failures, stop sending to that provider
   for 60 seconds (prevents cascading failures)
3. Multi-provider fallback — Groq → OpenAI → Bedrock via LiteLLM
Result: provider outages are transparent to users.
We had a Groq outage in production — auto-routed to OpenAI,
zero user impact.
```

**Q: "How do you stream LLM responses to the frontend?"**
```
Three components:
1. LLM provider: AsyncOpenAI/AsyncGroq with stream=True
   → returns AsyncIterator yielding chunks
2. FastAPI: StreamingResponse with async generator
   → yields SSE events (data: {...}\n\n)
3. React: EventSource or fetch with ReadableStream
   → appends tokens to state as they arrive
Challenge: partial tool calls — must accumulate fragments
before executing the tool. Buffer tool_calls by index.
```

**Q: "How do you prevent double-charging on LLM retries?"**
```
Two mechanisms:
1. Idempotency keys (client-controlled):
   Client sends X-Idempotency-Key header.
   Server checks Redis — if key exists, return cached result.
   Client retries with same key → same result, no re-execution.
2. Request fingerprinting (server-controlled):
   For temperature=0 calls: SHA256(messages + model) → cache key.
   Same deterministic input → cached output. Never re-runs.
```

**Q: "How do you manage context window across multi-turn conversations?"**
```
Sliding window with summarisation:
1. Always keep last 5 messages in full
2. For older messages: summarise with fast cheap model (llama-3.1-8b)
3. Include summary as system message prefix
This keeps tokens predictable regardless of conversation length.
Without this: long conversations hit context limit and crash.
With this: context stays within budget, older context preserved.
```

**Q: "How do you handle streaming with tool use?"**
```
Tool calls arrive fragmented in streaming mode:
  chunk 1: tool_call.function.name = "get_"
  chunk 2: tool_call.function.name = "schema"  ← incomplete!
  chunk 3: tool_call.function.arguments = '{"source'
  chunk 4: tool_call.function.arguments = '_id": "abc"}'

Must buffer by tool_calls[index].args_str
Only execute when finish_reason == "tool_calls"
Then parse args_str as complete JSON
```

---

## 🃏 Quick Revision Cards

```
CARD 1: Async Rule
  Every LLM call must be async.
  FastAPI is async. Sync LLM client + async FastAPI = event loop blocked.
  Test: replace await with blocking call → response time for all users spikes.

CARD 2: Streaming Components
  LLM provider → stream=True → yields chunks
  FastAPI → StreamingResponse → yields SSE strings
  React → EventSource → appends tokens to state
  SSE format: "data: {json}\n\n"

CARD 3: asyncio.gather
  Run N independent async tasks in parallel.
  Total time = max(task times), not sum.
  Use return_exceptions=True to handle partial failures.

CARD 4: Circuit Breaker States
  CLOSED → normal operation
  OPEN → all requests fail fast (provider down)
  HALF_OPEN → send one test request after recovery_timeout
  Success → back to CLOSED. Failure → back to OPEN.

CARD 5: Idempotency Key Flow
  Client generates UUID for this action.
  Sends X-Idempotency-Key: <uuid> on every attempt.
  Server: if key in Redis → return cached. Else → run + cache.
  Retry safely with same key. No double charges.

CARD 6: Context Window Strategy
  Problem: unlimited history hits token limit.
  Solution: keep last 5 full + summarise older.
  Budget: max_context_tokens = 6000 (leave room for response).
  Cheap model (llama-3.1-8b) for summarisation.

CARD 7: Cost Tracking Dimensions
  Track by: org_id + user_id + feature + model + date
  Redis: real-time daily counter per org (for rate limiting)
  Postgres: historical record per request (for analytics)
  Langfuse: per-trace breakdown (for debugging)

CARD 8: Rate Limiting Layers
  Layer 1: IP-based (slowapi) → protects from abuse
  Layer 2: User-based (Redis counter) → daily budget per user
  Layer 3: Tenant-based (cost guard) → daily spend limit per org
  Layer 4: Circuit breaker → provider-level protection

CARD 9: SSE vs WebSocket
  SSE: server → client only (unidirectional)
       HTTP/1.1 compatible, auto-reconnects, simpler
       Use for: LLM token streaming, progress updates
  WebSocket: bidirectional
             Better for: real-time chat, collaborative editing
             More complex: manual reconnection, separate protocol
  LLM streaming: SSE is sufficient and simpler.

CARD 10: Background Tasks vs Celery
  BackgroundTasks:
    Tied to web process lifetime.
    Killed if server restarts.
    Use for: lightweight, fire-and-forget.
  Celery:
    Separate worker process.
    Survives server restarts.
    Has retry, scheduling, monitoring.
    Use for: document ingestion, report generation, eval.
```

---

## ✅ Phase 3 Completion Checklist

```
CONCEPTS
[ ] Can explain why async is non-negotiable for LLM APIs
[ ] Know difference between BackgroundTasks and Celery (when to use each)
[ ] Understand SSE vs WebSocket — and when to use which
[ ] Can explain circuit breaker 3 states with real scenario
[ ] Know idempotency key pattern end-to-end (client + server)
[ ] Can explain context window trimming strategy
[ ] Understand multi-model fallback chain and LiteLLM role
[ ] Know how partial tool calls work in streaming mode

CODE
[ ] Built streaming SSE endpoint (LLM → FastAPI → React end-to-end)
[ ] Implemented Redis conversation sessions (multi-turn chat works)
[ ] Implemented context window manager with summarisation
[ ] Implemented LiteLLM multi-provider fallback (Groq → OpenAI)
[ ] Implemented circuit breaker (3 states: closed/open/half-open)
[ ] Implemented idempotency key middleware (duplicate requests handled)
[ ] Implemented per-user rate limiting with Redis counter
[ ] Implemented cost tracking per feature per org
[ ] Implemented async job + polling API + SSE progress
[ ] Full Docker Compose with FastAPI + Redis + Celery

PRODUCTION TESTED
[ ] 10 concurrent users → no blocking (verify with load test)
[ ] Kill Groq API key → fallback to OpenAI works transparently
[ ] Send same request twice → second returns cached result
[ ] Long conversation (10+ turns) → context stays within budget
[ ] Send request with slow client → backpressure handled gracefully
[ ] Mid-stream exception → error event sent, not silent hang
[ ] Rate limit exceeded → 429 with clear message (not 500)
```

---

*Phase 3 Complete Study Notes | GenAI + LLMOps Engineering Roadmap 2026*
*Next: Phase 4 — Agentic AI (LangGraph, HITL, Multi-agent, MCP)*
