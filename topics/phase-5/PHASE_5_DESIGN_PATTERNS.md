# 🏗️ Phase 5 — Design Patterns for Python GenAI
> **Complete Study Notes | Theory + Code + Use Cases | Interview Prep**
> Part of: GenAI + LLMOps Engineering Roadmap 2026
> Estimated time: 30–40 hrs over 2 weeks
> Prerequisites: Phase 1–4 (LLM APIs, RAG, FastAPI, LangGraph basics)

---

## 📑 Table of Contents

1. [Why Design Patterns Matter for AI Systems](#1-why-design-patterns-matter-for-ai-systems)
2. [5.1 — Python Design Patterns for AI Systems](#51--python-design-patterns-for-ai-systems)
3. [5.2 — GenAI-Specific Architecture Patterns](#52--genai-specific-architecture-patterns)
4. [5.3 — Semantic Kernel](#53--semantic-kernel)
5. [5.4 — Agent Communication Protocols (A2A, ACP)](#54--agent-communication-protocols)
6. [Phase 5 Project](#-phase-5-project)
7. [Interview Cheat Sheet](#-interview-cheat-sheet)
8. [Quick Revision Cards](#-quick-revision-cards)

---

## 1. Why Design Patterns Matter for AI Systems

### 🧑‍🎓 Explain Like I'm in BTech 2nd Year

You've probably written code that "works" — the function does the right thing, tests pass, feature ships. But three months later someone else opens it (or you do, after forgetting everything) and it's a mess to modify.

Design patterns are solutions to RECURRING problems that engineers kept running into and eventually wrote down as named strategies. They're not new code libraries — they're ways of thinking about how to organise code so it's easy to change, test, and understand.

In normal web development you might not need them urgently. But AI systems have unique problems:

```
Problem 1: LLM providers change APIs, go down, or get expensive
  → You need to swap providers without rewriting your app

Problem 2: Prompts behave like code — changing them breaks things
  → You need versioning and a registry, just like API versioning

Problem 3: RAG pipelines have 6+ components — any can fail
  → You need clean interfaces so each part is testable in isolation

Problem 4: LLM calls cost money — every bad design decision costs $
  → Caching, routing, compression all need the right patterns

Problem 5: AI systems have unpredictable latency (1s to 30s)
  → You need async, circuit breakers, and fallback chains

These problems don't exist in typical CRUD apps.
Design patterns are how senior AI engineers solve them.
```

### The Difference Between Junior and Senior AI Code

```python
# ❌ Junior AI engineer code:
import openai

def answer_question(question: str) -> str:
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": question}]
    )
    return response.choices[0].message.content

# Problems:
# - OpenAI hardcoded everywhere (can't switch to Groq/Claude without grep+replace)
# - No retry on rate limit
# - No cost tracking
# - No caching
# - No logging
# - Can't test without calling the real API (expensive, slow, flaky tests)

# ✅ Senior AI engineer code (with patterns):
class LLMGateway:
    def __init__(self, provider: LLMProvider, cache: Cache, logger: Logger):
        self.provider = provider  # Strategy Pattern: swap any provider
        self.cache    = cache
        self.logger   = logger

    async def complete(self, prompt: str, **kwargs) -> str:
        # Check cache first (Decorator-like pattern)
        cached = await self.cache.get(prompt)
        if cached:
            self.logger.info("cache_hit", prompt_hash=hash(prompt))
            return cached

        # Call provider with retry (Circuit Breaker)
        with self.logger.trace("llm_call"):
            response = await self.provider.complete(prompt, **kwargs)

        await self.cache.set(prompt, response)
        return response

# Testable, swappable, observable, cost-controlled
```

---

## 5.1 — Python Design Patterns for AI Systems

### Pattern 1 — Factory Pattern

#### Theory

The Factory pattern is about centralising object creation. Instead of creating objects directly (`openai.ChatOpenAI()`), you have one place (`LLMFactory.create("openai")`) that knows how to build each variant.

**Why AI systems need this:**
```
You have 5 different LLMs:
  Groq (fast, cheap, for simple queries)
  GPT-4o (expensive, for complex reasoning)
  Claude Sonnet (for long documents)
  Bedrock (for AWS-native deployments)
  Ollama (for local, no-cost inference)

Without Factory:
  Every file has: from langchain_openai import ChatOpenAI
                  llm = ChatOpenAI(model="gpt-4o", api_key=...)
  Switch to Claude → grep+replace across 20 files
  Miss one → silent bug in production

With Factory:
  Every file has: llm = LLMFactory.create(settings.llm_provider)
  Switch to Claude → change one env var
```

#### Code

```python
# backend/providers/llm_factory.py
from abc import ABC, abstractmethod
from pydantic import BaseModel
from enum import Enum

class LLMProvider(str, Enum):
    GROQ      = "groq"
    OPENAI    = "openai"
    ANTHROPIC = "anthropic"
    BEDROCK   = "bedrock"
    OLLAMA    = "ollama"

class LLMConfig(BaseModel):
    provider:    LLMProvider
    model:       str
    temperature: float = 0.0
    max_tokens:  int   = 2000

class BaseLLMClient(ABC):
    """Every LLM client must implement this interface"""

    @abstractmethod
    async def complete(self, messages: list[dict], **kwargs) -> str: ...

    @abstractmethod
    async def complete_structured(
        self, messages: list[dict], response_model, **kwargs
    ): ...

    @abstractmethod
    async def embed(self, texts: list[str]) -> list[list[float]]: ...

class GroqClient(BaseLLMClient):
    def __init__(self, config: LLMConfig):
        from groq import AsyncGroq
        import instructor
        self._client = instructor.from_groq(AsyncGroq())
        self._model  = config.model

    async def complete(self, messages: list[dict], **kwargs) -> str:
        resp = await self._client.chat.completions.create(
            model=self._model, messages=messages, **kwargs
        )
        return resp.choices[0].message.content

    async def complete_structured(self, messages, response_model, **kwargs):
        return await self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            response_model=response_model,
            **kwargs,
        )

    async def embed(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError("Groq doesn't support embeddings yet")

class OpenAIClient(BaseLLMClient):
    def __init__(self, config: LLMConfig):
        from openai import AsyncOpenAI
        import instructor
        self._client = instructor.from_openai(AsyncOpenAI())
        self._model  = config.model

    async def complete(self, messages: list[dict], **kwargs) -> str:
        resp = await self._client.chat.completions.create(
            model=self._model, messages=messages, **kwargs
        )
        return resp.choices[0].message.content

    async def complete_structured(self, messages, response_model, **kwargs):
        return await self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            response_model=response_model,
            **kwargs,
        )

    async def embed(self, texts: list[str]) -> list[list[float]]:
        from openai import AsyncOpenAI
        client = AsyncOpenAI()
        resp = await client.embeddings.create(
            model="text-embedding-3-small", input=texts
        )
        return [d.embedding for d in sorted(resp.data, key=lambda x: x.index)]

class AnthropicClient(BaseLLMClient):
    def __init__(self, config: LLMConfig):
        import anthropic, instructor
        self._client = instructor.from_anthropic(anthropic.AsyncAnthropic())
        self._model  = config.model

    async def complete(self, messages: list[dict], **kwargs) -> str:
        # Anthropic API uses system separately
        system = next((m["content"] for m in messages if m["role"] == "system"), "")
        user_msgs = [m for m in messages if m["role"] != "system"]
        resp = await self._client.messages.create(
            model=self._model,
            system=system,
            messages=user_msgs,
            max_tokens=kwargs.get("max_tokens", 2000),
        )
        return resp.content[0].text

    async def complete_structured(self, messages, response_model, **kwargs):
        system = next((m["content"] for m in messages if m["role"] == "system"), "")
        user_msgs = [m for m in messages if m["role"] != "system"]
        return await self._client.messages.create(
            model=self._model,
            system=system,
            messages=user_msgs,
            response_model=response_model,
            max_tokens=kwargs.get("max_tokens", 2000),
        )

    async def embed(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError("Use OpenAI or Cohere for embeddings")

# ─── The Factory ────────────────────────────────────────────────────────────
class LLMFactory:
    """
    Single point of LLM client creation.
    Change provider = change env var, not code.
    """
    _clients: dict[str, BaseLLMClient] = {}   # cache created clients (Singleton)

    @classmethod
    def create(cls, config: LLMConfig) -> BaseLLMClient:
        cache_key = f"{config.provider}:{config.model}"
        if cache_key not in cls._clients:
            cls._clients[cache_key] = cls._build(config)
        return cls._clients[cache_key]

    @classmethod
    def _build(cls, config: LLMConfig) -> BaseLLMClient:
        match config.provider:
            case LLMProvider.GROQ:      return GroqClient(config)
            case LLMProvider.OPENAI:    return OpenAIClient(config)
            case LLMProvider.ANTHROPIC: return AnthropicClient(config)
            case _:
                raise ValueError(f"Unknown provider: {config.provider}")

# Usage — same everywhere, provider comes from config
from core.config import settings

llm = LLMFactory.create(LLMConfig(
    provider=LLMProvider(settings.llm_provider),
    model=settings.llm_model,
))
answer = await llm.complete([{"role": "user", "content": "What is RAG?"}])
```

**Testing with Factory:**
```python
# tests/test_rag_service.py
class MockLLMClient(BaseLLMClient):
    """No real API calls in tests — fast, free, deterministic"""
    async def complete(self, messages, **kwargs) -> str:
        return "Mocked answer for: " + messages[-1]["content"]

    async def complete_structured(self, messages, response_model, **kwargs):
        return response_model(**{f: "mock" for f in response_model.model_fields})

    async def embed(self, texts):
        return [[0.1] * 1536 for _ in texts]

async def test_rag_generates_answer():
    mock_llm = MockLLMClient()
    service  = RAGService(llm=mock_llm)             # inject mock
    answer   = await service.answer("What is RAG?")
    assert len(answer) > 0
    # Tests pass instantly — no API calls, no cost, no flakiness
```

---

### Pattern 2 — Strategy Pattern

#### Theory

Strategy defines a family of algorithms behind a common interface, making them interchangeable at runtime. The client doesn't know which specific implementation is running.

```
Without Strategy:
  if self.retrieval_method == "vector":
      chunks = await vector_search(query)
  elif self.retrieval_method == "bm25":
      chunks = await bm25_search(query)
  elif self.retrieval_method == "hybrid":
      chunks = await hybrid_search(query)
  # Add new method → modify this class (violates Open/Closed Principle)

With Strategy:
  chunks = await self.retriever.retrieve(query)
  # self.retriever is set once, then just called
  # Add new method → new class, no changes to existing code
```

#### Code

```python
# backend/services/rag/retrieval_strategy.py
from abc import ABC, abstractmethod

class RetrievalStrategy(ABC):
    @abstractmethod
    async def retrieve(
        self, query: str, query_embedding: list[float],
        source_ids: list[str], org_id: str, k: int = 5
    ) -> list[Chunk]: ...

class VectorRetrievalStrategy(RetrievalStrategy):
    """Pure dense vector search — fast, semantic"""
    def __init__(self, vector_store: VectorStoreProtocol):
        self._vs = vector_store

    async def retrieve(self, query, query_embedding, source_ids, org_id, k=5) -> list[Chunk]:
        return await self._vs.search(
            query_vector=query_embedding, org_id=org_id,
            source_ids=source_ids, k=k,
        )

class BM25RetrievalStrategy(RetrievalStrategy):
    """Keyword search — exact match, handles rare terms well"""
    def __init__(self, corpus: list[str], chunk_map: dict[str, Chunk]):
        from rank_bm25 import BM25Okapi
        self._bm25      = BM25Okapi([doc.lower().split() for doc in corpus])
        self._corpus    = corpus
        self._chunk_map = chunk_map

    async def retrieve(self, query, query_embedding, source_ids, org_id, k=5) -> list[Chunk]:
        scores   = self._bm25.get_scores(query.lower().split())
        top_idxs = scores.argsort()[::-1][:k]
        return [self._chunk_map[self._corpus[i]] for i in top_idxs if scores[i] > 0]

class HybridRetrievalStrategy(RetrievalStrategy):
    """BM25 + dense vector, fused with RRF — best of both worlds"""
    def __init__(self, vector: VectorRetrievalStrategy, bm25: BM25RetrievalStrategy):
        self._vector = vector
        self._bm25   = bm25

    async def retrieve(self, query, query_embedding, source_ids, org_id, k=5) -> list[Chunk]:
        import asyncio
        dense_results, bm25_results = await asyncio.gather(
            self._vector.retrieve(query, query_embedding, source_ids, org_id, k * 2),
            self._bm25.retrieve(query, query_embedding, source_ids, org_id, k * 2),
        )
        return self._rrf_merge(dense_results, bm25_results, k)

    def _rrf_merge(self, a, b, k, rrf_k=60) -> list[Chunk]:
        scores: dict[str, float] = {}
        chunks: dict[str, Chunk] = {}
        for rank, c in enumerate(a):
            scores[c.id] = scores.get(c.id, 0) + 1 / (rrf_k + rank)
            chunks[c.id] = c
        for rank, c in enumerate(b):
            scores[c.id] = scores.get(c.id, 0) + 1 / (rrf_k + rank)
            chunks.setdefault(c.id, c)
        return [chunks[cid] for cid in sorted(scores, key=scores.get, reverse=True)][:k]

# RAG Pipeline uses the strategy — doesn't care which one
class RAGPipeline:
    def __init__(self, retriever: RetrievalStrategy, llm: BaseLLMClient):
        self._retriever = retriever  # injected strategy
        self._llm       = llm

    async def run(self, question: str, source_ids: list[str], org_id: str) -> str:
        embedding = await embed(question)
        chunks    = await self._retriever.retrieve(question, embedding, source_ids, org_id)
        context   = "\n\n".join(c.content for c in chunks)
        return await self._llm.complete([
            {"role": "system", "content": f"Answer using only this context:\n{context}"},
            {"role": "user",   "content": question},
        ])

# Runtime strategy selection — change with config, not code
def build_rag_pipeline(config: AppConfig) -> RAGPipeline:
    llm = LLMFactory.create(LLMConfig(provider=config.llm_provider, model=config.llm_model))
    vs  = QdrantVectorStore(qdrant_client)

    match config.retrieval_strategy:
        case "vector":  retriever = VectorRetrievalStrategy(vs)
        case "bm25":    retriever = BM25RetrievalStrategy(corpus, chunk_map)
        case "hybrid":  retriever = HybridRetrievalStrategy(
                            VectorRetrievalStrategy(vs),
                            BM25RetrievalStrategy(corpus, chunk_map),
                        )
        case _:         retriever = HybridRetrievalStrategy(...)  # default

    return RAGPipeline(retriever=retriever, llm=llm)
```

---

### Pattern 3 — Chain of Responsibility

#### Theory

Passes a request through a chain of handlers. Each handler either processes the request or passes it to the next. This is the perfect pattern for guardrails, middleware, and validation pipelines.

```
Without Chain of Responsibility:
  async def process_query(query: str) -> str:
      # 1. Check injection
      if re.search(INJECTION_PATTERN, query):
          raise ValueError("Injection detected")
      # 2. Mask PII
      query = mask_pii(query)
      # 3. Check cost limit
      if get_daily_cost() > DAILY_LIMIT:
          raise ValueError("Cost limit exceeded")
      # 4. Check rate limit
      if get_request_count() > RATE_LIMIT:
          raise ValueError("Rate limit exceeded")
      # 5. Finally call LLM
      return await llm.complete(query)
  # Adding a new check → modify this function every time

With Chain of Responsibility:
  pipeline = (InjectionGuard() | PIIMaskingGuard() | CostGuard() | RateLimitGuard())
  result   = await pipeline.handle(GuardRequest(query=query))
```

#### Code

```python
# backend/services/guardrails/pipeline.py
from abc import ABC, abstractmethod
from dataclasses import dataclass

@dataclass
class GuardRequest:
    query:    str
    org_id:   str
    user_id:  str
    metadata: dict = None

@dataclass
class GuardResult:
    allowed:   bool
    query:     str           # possibly modified (PII masked)
    reason:    str  = ""

class GuardHandler(ABC):
    def __init__(self):
        self._next: "GuardHandler | None" = None

    def set_next(self, handler: "GuardHandler") -> "GuardHandler":
        self._next = handler
        return handler   # enables chaining: a.set_next(b).set_next(c)

    @abstractmethod
    async def handle(self, request: GuardRequest) -> GuardResult: ...

    async def _pass_to_next(self, request: GuardRequest) -> GuardResult:
        if self._next:
            return await self._next.handle(request)
        return GuardResult(allowed=True, query=request.query)

class InjectionGuard(GuardHandler):
    """Detect prompt injection attempts"""
    PATTERNS = [
        r"ignore (all |previous |prior )?instructions",
        r"you are now",
        r"forget (everything|all)",
        r"jailbreak",
        r"pretend (you|to be)",
    ]

    async def handle(self, request: GuardRequest) -> GuardResult:
        import re
        for pattern in self.PATTERNS:
            if re.search(pattern, request.query.lower()):
                return GuardResult(
                    allowed=False,
                    query=request.query,
                    reason="Potential prompt injection detected"
                )
        return await self._pass_to_next(request)

class PIIMaskingGuard(GuardHandler):
    """Mask PII before it reaches the LLM"""
    PII_PATTERNS = {
        "email":  r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
        "phone":  r"\b(\+91|0)?[6-9]\d{9}\b",
        "pan":    r"\b[A-Z]{5}[0-9]{4}[A-Z]\b",
    }

    async def handle(self, request: GuardRequest) -> GuardResult:
        import re
        masked = request.query
        for pii_type, pattern in self.PII_PATTERNS.items():
            masked = re.sub(pattern, f"[{pii_type.upper()}_REDACTED]", masked)
        request = GuardRequest(query=masked, org_id=request.org_id, user_id=request.user_id)
        return await self._pass_to_next(request)

class CostGuard(GuardHandler):
    """Block requests when daily cost limit is exceeded"""
    def __init__(self, redis, daily_limit_usd: float = 10.0):
        super().__init__()
        self._redis = redis
        self._limit = daily_limit_usd

    async def handle(self, request: GuardRequest) -> GuardResult:
        from datetime import datetime
        today = datetime.utcnow().strftime("%Y-%m-%d")
        spent = float(await self._redis.get(f"cost:{request.org_id}:{today}") or 0)
        if spent >= self._limit:
            return GuardResult(
                allowed=False, query=request.query,
                reason=f"Daily limit of ${self._limit:.2f} reached"
            )
        return await self._pass_to_next(request)

class RateLimitGuard(GuardHandler):
    """Per-user daily query limit"""
    def __init__(self, redis, limit: int = 100):
        super().__init__()
        self._redis = redis
        self._limit = limit

    async def handle(self, request: GuardRequest) -> GuardResult:
        from datetime import datetime
        today = datetime.utcnow().strftime("%Y-%m-%d")
        key   = f"rl:{request.org_id}:{request.user_id}:{today}"
        count = int(await self._redis.incr(key))
        if count == 1:
            await self._redis.expireat(key, end_of_day_unix())
        if count > self._limit:
            return GuardResult(
                allowed=False, query=request.query,
                reason=f"Rate limit of {self._limit} queries/day exceeded"
            )
        return await self._pass_to_next(request)

# Build the chain once at startup — order matters!
def build_guardrail_pipeline(redis) -> GuardHandler:
    injection = InjectionGuard()
    pii       = PIIMaskingGuard()
    cost      = CostGuard(redis, daily_limit_usd=10.0)
    rate      = RateLimitGuard(redis, limit=100)

    injection.set_next(pii).set_next(cost).set_next(rate)
    return injection   # entry point

GUARDRAILS = build_guardrail_pipeline(redis_client)

# Usage — one call, all checks run in order
async def process_query(query: str, org_id: str, user_id: str) -> str:
    req    = GuardRequest(query=query, org_id=org_id, user_id=user_id)
    result = await GUARDRAILS.handle(req)

    if not result.allowed:
        raise HTTPException(status_code=429, detail=result.reason)

    # Use the possibly-modified query (PII masked)
    return await llm.complete([{"role": "user", "content": result.query}])
```

---

### Pattern 4 — Observer Pattern

#### Theory

Observer defines a subscription mechanism — when something happens (an event), all subscribers are automatically notified. Perfect for logging, monitoring, cost tracking, and SSE streaming.

```
Problem without Observer:
  async def run_agent_step(state):
      start   = time.time()
      result  = await call_llm(state)
      elapsed = time.time() - start

      # Scattered tracking code:
      await db.log_step(...)          # ← coupled to DB
      await redis.incr("token_count") # ← coupled to Redis
      langfuse.trace(...)             # ← coupled to Langfuse
      await ws.send({"step": result}) # ← coupled to WebSocket
      # Adding new monitoring = modify this function

With Observer:
  async def run_agent_step(state):
      result = await call_llm(state)
      await event_bus.emit(AgentStepEvent(step="llm_call", result=result))
      # emit is all we do — observers handle the rest independently
```

#### Code

```python
# backend/services/observability/event_bus.py
from dataclasses import dataclass, field
from datetime import datetime
from abc import ABC, abstractmethod
import asyncio

@dataclass
class AgentStepEvent:
    query_id:    str
    org_id:      str
    user_id:     str
    step:        str          # "retrieval" | "reranking" | "llm_generation" etc.
    status:      str          # "started" | "done" | "error"
    duration_ms: int  = 0
    tokens_in:   int  = 0
    tokens_out:  int  = 0
    cost_usd:    float= 0.0
    metadata:    dict = field(default_factory=dict)
    timestamp:   datetime = field(default_factory=datetime.utcnow)

class EventObserver(ABC):
    @abstractmethod
    async def on_event(self, event: AgentStepEvent) -> None: ...

class DBEventObserver(EventObserver):
    """Writes every step to execution_events table"""
    async def on_event(self, event: AgentStepEvent) -> None:
        await ExecutionEventRepository.create(event)

class CostEventObserver(EventObserver):
    """Tracks LLM spend in Redis for real-time guardrails"""
    def __init__(self, redis): self._redis = redis

    async def on_event(self, event: AgentStepEvent) -> None:
        if event.cost_usd > 0:
            today = event.timestamp.strftime("%Y-%m-%d")
            key   = f"cost:{event.org_id}:{today}"
            await self._redis.incrbyfloat(key, event.cost_usd)

class LangfuseEventObserver(EventObserver):
    """Sends span updates to Langfuse for LLM observability"""
    async def on_event(self, event: AgentStepEvent) -> None:
        if event.status == "done" and event.tokens_in > 0:
            langfuse_client.score(
                trace_id=event.query_id,
                name=event.step,
                value=event.duration_ms,
                comment=f"tokens={event.tokens_in+event.tokens_out} cost=${event.cost_usd:.6f}",
            )

class SSEEventObserver(EventObserver):
    """Pushes step events to connected frontend via SSE"""
    def __init__(self, sse_manager): self._sse = sse_manager

    async def on_event(self, event: AgentStepEvent) -> None:
        await self._sse.send(event.query_id, {
            "type":        "step_update",
            "step":        event.step,
            "status":      event.status,
            "duration_ms": event.duration_ms,
        })

class EventBus:
    def __init__(self):
        self._observers: list[EventObserver] = []

    def subscribe(self, observer: EventObserver) -> None:
        self._observers.append(observer)

    async def emit(self, event: AgentStepEvent) -> None:
        """Notify all observers — run concurrently, don't block the pipeline"""
        await asyncio.gather(
            *[obs.on_event(event) for obs in self._observers],
            return_exceptions=True,   # observer failure doesn't crash the pipeline
        )

# Setup once at startup
event_bus = EventBus()
event_bus.subscribe(DBEventObserver())
event_bus.subscribe(CostEventObserver(redis_client))
event_bus.subscribe(LangfuseEventObserver())
event_bus.subscribe(SSEEventObserver(sse_manager))

# Elegant context manager for tracking steps
import time
class track_step:
    def __init__(self, step: str, query_id: str, org_id: str, user_id: str):
        self.step, self.query_id = step, query_id
        self.org_id, self.user_id = org_id, user_id
        self._start = None

    async def __aenter__(self):
        self._start = time.perf_counter()
        await event_bus.emit(AgentStepEvent(
            query_id=self.query_id, org_id=self.org_id, user_id=self.user_id,
            step=self.step, status="started",
        ))
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        duration = int((time.perf_counter() - self._start) * 1000)
        await event_bus.emit(AgentStepEvent(
            query_id=self.query_id, org_id=self.org_id, user_id=self.user_id,
            step=self.step, status="error" if exc_type else "done",
            duration_ms=duration,
        ))

# Usage — clean pipeline, zero scattered tracking code
async def run_rag_pipeline(question: str, query_id: str, org_id: str, user_id: str):
    async with track_step("retrieval", query_id, org_id, user_id):
        chunks = await retriever.retrieve(question, ...)

    async with track_step("reranking", query_id, org_id, user_id):
        reranked = await reranker.rerank(question, chunks)

    async with track_step("llm_generation", query_id, org_id, user_id):
        answer = await llm.complete(...)

    return answer
    # All 3 steps tracked, logged, cost-counted, and pushed to SSE automatically
```

---

### Pattern 5 — Repository Pattern

#### Theory

Repository provides an abstraction layer over data storage. Services never write raw SQL or Qdrant queries — they call repository methods. This makes testing trivial and switching storage simple.

```
Without Repository:
  # In service layer (wrong):
  result = await db.execute(
      "SELECT * FROM chunks WHERE org_id=$1 AND source_id=ANY($2)",
      org_id, source_ids
  )
  # Change DB schema → grep every service file
  # Test this code → need a real DB running

With Repository:
  # In service layer (correct):
  chunks = await chunk_repo.get_by_sources(org_id, source_ids)
  # Change DB schema → only update ChunkRepository
  # Test → inject FakeChunkRepository, no DB needed
```

#### Code

```python
# backend/repositories/chunk_repository.py
from typing import Protocol

class ChunkRepositoryProtocol(Protocol):
    """Interface — services depend on this, not the implementation"""
    async def get_by_sources(self, org_id: str, source_ids: list[str]) -> list[Chunk]: ...
    async def create_many(self, chunks: list[CreateChunkDTO]) -> list[Chunk]: ...
    async def deactivate_by_source(self, source_id: str, org_id: str) -> int: ...
    async def get_without_embedding(self, org_id: str, limit: int) -> list[Chunk]: ...

class PostgresChunkRepository:
    """Real implementation — talks to Postgres"""
    def __init__(self, pool: asyncpg.Pool):
        self._pool = pool

    async def get_by_sources(self, org_id: str, source_ids: list[str]) -> list[Chunk]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """SELECT id, content, metadata, embedding, is_active
                   FROM chunks
                   WHERE org_id = $1 AND source_id = ANY($2) AND is_active = TRUE
                   ORDER BY created_at DESC""",
                org_id, source_ids,
            )
        return [Chunk(**r) for r in rows]

    async def create_many(self, chunks: list[CreateChunkDTO]) -> list[Chunk]:
        async with self._pool.acquire() as conn:
            rows = await conn.executemany(
                """INSERT INTO chunks (source_id, org_id, user_id, content, metadata)
                   VALUES ($1, $2, $3, $4, $5) RETURNING *""",
                [(c.source_id, c.org_id, c.user_id, c.content, c.metadata) for c in chunks],
            )
        return [Chunk(**r) for r in rows]

    async def deactivate_by_source(self, source_id: str, org_id: str) -> int:
        async with self._pool.acquire() as conn:
            result = await conn.execute(
                "UPDATE chunks SET is_active=FALSE WHERE source_id=$1 AND org_id=$2",
                source_id, org_id,
            )
        return int(result.split()[-1])   # "UPDATE N" → N

    async def get_without_embedding(self, org_id: str, limit: int = 100) -> list[Chunk]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT * FROM chunks WHERE org_id=$1 AND embedding IS NULL LIMIT $2",
                org_id, limit,
            )
        return [Chunk(**r) for r in rows]

class FakeChunkRepository:
    """In-memory implementation for tests — fast, no DB needed"""
    def __init__(self):
        self._store: list[Chunk] = []

    async def get_by_sources(self, org_id, source_ids) -> list[Chunk]:
        return [c for c in self._store if c.org_id == org_id and c.source_id in source_ids]

    async def create_many(self, chunks: list[CreateChunkDTO]) -> list[Chunk]:
        created = [Chunk(id=str(uuid4()), **c.model_dump()) for c in chunks]
        self._store.extend(created)
        return created

    async def deactivate_by_source(self, source_id, org_id) -> int:
        count = 0
        for c in self._store:
            if c.source_id == source_id and c.org_id == org_id:
                c.is_active = False
                count += 1
        return count

    async def get_without_embedding(self, org_id, limit=100) -> list[Chunk]:
        return [c for c in self._store if c.org_id == org_id and c.embedding is None][:limit]

# Test — zero DB setup, instant, free
async def test_ingestion_creates_chunks():
    repo    = FakeChunkRepository()
    service = IngestionService(chunk_repo=repo)
    await service.ingest("source-123", "org-abc", "text content here")
    chunks = await repo.get_by_sources("org-abc", ["source-123"])
    assert len(chunks) > 0
    assert chunks[0].is_active is True
```

---

### Pattern 6 — Decorator Pattern

#### Theory

Decorator wraps a function/object to add behaviour without changing its code. For AI systems, this means adding retry, caching, logging, and cost tracking as decorators that stack cleanly.

```python
# backend/core/decorators.py
import functools, time, asyncio, hashlib, json
from typing import Callable

def with_retry(max_retries: int = 3, delay: float = 1.0, backoff: float = 2.0):
    """Retry any async LLM call on failure with exponential backoff"""
    def decorator(func: Callable):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            last_error = None
            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except (RateLimitError, APIConnectionError) as e:
                    last_error = e
                    wait = delay * (backoff ** attempt)
                    logger.warning(f"Retry {attempt+1}/{max_retries} after {wait:.1f}s: {e}")
                    await asyncio.sleep(wait)
                except APIError as e:
                    if e.status_code in (400, 401, 403):
                        raise   # don't retry on client errors
                    last_error = e
                    await asyncio.sleep(delay)
            raise last_error
        return wrapper
    return decorator

def with_cache(ttl_seconds: int = 3600):
    """Cache async function results in Redis by input hash"""
    def decorator(func: Callable):
        @functools.wraps(func)
        async def wrapper(self, *args, **kwargs):
            # Create cache key from function name + args
            key_data = f"{func.__name__}:{json.dumps(args, default=str)}"
            key      = f"cache:{hashlib.sha256(key_data.encode()).hexdigest()}"

            cached = await self._redis.get(key)
            if cached:
                logger.debug(f"Cache HIT: {func.__name__}")
                return json.loads(cached)

            result = await func(self, *args, **kwargs)
            await self._redis.setex(key, ttl_seconds, json.dumps(result, default=str))
            return result
        return wrapper
    return decorator

def with_cost_tracking(task: str):
    """Track token cost of any LLM call automatically"""
    def decorator(func: Callable):
        @functools.wraps(func)
        async def wrapper(self, *args, **kwargs):
            start    = time.perf_counter()
            response = await func(self, *args, **kwargs)
            elapsed  = int((time.perf_counter() - start) * 1000)

            # Calculate and record cost
            usage   = getattr(response, "usage", None)
            if usage:
                cost = calculate_cost(
                    model=self._model,
                    input_tokens=usage.prompt_tokens,
                    output_tokens=usage.completion_tokens,
                )
                await record_cost(org_id=self._org_id, task=task, cost_usd=cost)
                logger.info("llm_cost_tracked", extra={
                    "extra_fields": {"task": task, "cost_usd": cost, "latency_ms": elapsed}
                })
            return response
        return wrapper
    return decorator

# Usage — stack decorators cleanly
class LLMService:
    @with_retry(max_retries=3, delay=2.0)
    @with_cache(ttl_seconds=1800)
    @with_cost_tracking(task="rag_generation")
    async def generate_answer(self, question: str, context: str) -> str:
        return await self._llm.complete([
            {"role": "system", "content": f"Context:\n{context}"},
            {"role": "user",   "content": question},
        ])
    # Retry + Cache + Cost tracking, all with zero boilerplate
```

---

### Pattern 7 — Singleton Pattern

#### Theory

Ensures only one instance of an object exists. In AI systems, this prevents creating a new LLM client connection on every request.

```python
# backend/core/singletons.py
from threading import Lock

class LLMClientSingleton:
    """One client instance per process — not per request"""
    _instance = None
    _lock     = Lock()

    @classmethod
    def get(cls) -> BaseLLMClient:
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:   # double-checked locking
                    cls._instance = LLMFactory.create(settings.llm_config)
                    logger.info("LLM client initialised (singleton)")
        return cls._instance

# FastAPI dependency — reuses the same client for all requests
async def get_llm() -> BaseLLMClient:
    return LLMClientSingleton.get()

# ❌ WRONG — new client created for every single request
@router.post("/chat")
async def chat(body: ChatRequest):
    llm = OpenAIClient(LLMConfig(...))   # ← new connection every request!
    return await llm.complete(body.messages)

# ✅ CORRECT — one client, injected via Depends
@router.post("/chat")
async def chat(body: ChatRequest, llm: BaseLLMClient = Depends(get_llm)):
    return await llm.complete(body.messages)
```

---

### Pattern 8 — Circuit Breaker

#### Theory

Stops sending requests to a failing service, preventing cascade failures. Three states: CLOSED (normal), OPEN (failing — fast fail all), HALF_OPEN (testing recovery).

```python
# backend/providers/circuit_breaker.py
import time
from enum import Enum

class CircuitState(Enum):
    CLOSED    = "closed"    # normal — let requests through
    OPEN      = "open"      # failing — block all requests immediately
    HALF_OPEN = "half_open" # testing — allow one request to test recovery

class CircuitBreaker:
    def __init__(self, failure_threshold=5, recovery_timeout=60.0):
        self.threshold       = failure_threshold
        self.recovery_timeout= recovery_timeout
        self.failures        = 0
        self.last_failure    = 0.0
        self.state           = CircuitState.CLOSED

    def call_succeeded(self):
        self.failures = 0
        self.state    = CircuitState.CLOSED

    def call_failed(self):
        self.failures      += 1
        self.last_failure   = time.time()
        if self.failures >= self.threshold:
            self.state = CircuitState.OPEN
            logger.warning(f"Circuit OPEN after {self.failures} failures")

    def can_attempt(self) -> bool:
        if self.state == CircuitState.CLOSED:
            return True
        if self.state == CircuitState.OPEN:
            if time.time() - self.last_failure > self.recovery_timeout:
                self.state = CircuitState.HALF_OPEN
                logger.info("Circuit HALF_OPEN — testing recovery")
                return True
            return False
        return True   # HALF_OPEN — allow test request

    async def execute(self, operation):
        if not self.can_attempt():
            raise RuntimeError("Circuit OPEN — provider unavailable")
        try:
            result = await operation()
            self.call_succeeded()
            return result
        except Exception as e:
            self.call_failed()
            raise

# Global breakers per provider
_breakers = {
    "groq":    CircuitBreaker(failure_threshold=5, recovery_timeout=60),
    "openai":  CircuitBreaker(failure_threshold=5, recovery_timeout=60),
    "bedrock": CircuitBreaker(failure_threshold=3, recovery_timeout=120),
}

async def protected_llm_call(provider: str, operation) -> str:
    return await _breakers.get(provider, CircuitBreaker()).execute(operation)
```

---

## 5.2 — GenAI-Specific Architecture Patterns

### 🧑‍🎓 Explain Like I'm in BTech 2nd Year

The previous section was about code-level patterns (how to structure classes and functions). This section is about system-level patterns — how to architect your entire AI service so it's reliable, observable, and safe to update in production.

---

### Pattern 1 — Prompt Registry

#### Theory

Prompts are code. They must be versioned, tested, and deployed just like API endpoints. A Prompt Registry is a centralised store for all prompt templates with versioning.

```
Without Prompt Registry:
  # Scattered across 15 files:
  SYSTEM_PROMPT = "You are a helpful assistant. Answer from context."
  # Duplicate in another file:
  SYSTEM = "Answer only from the provided context. Don't hallucinate."
  # Third variation somewhere else
  
  Who owns what? Which is correct? What changed? When?
  → Chaos in production.

With Prompt Registry:
  prompt = await PromptRegistry.get("rag_system_prompt", version="v2")
  # Single source of truth. Versioned. Testable. Auditable.
```

#### Code

```python
# backend/core/prompt_registry.py
from pathlib import Path
from functools import lru_cache
import json

PROMPTS_DIR = Path(__file__).parent.parent.parent / "prompts"

class PromptRegistry:
    """
    File-based prompt registry.
    Structure: prompts/{prompt_name}/{version}.txt
    Metadata:  prompts/{prompt_name}/meta.json

    Example:
      prompts/
        rag_system/
          v1.txt     — original
          v2.txt     — added citation instruction
          v3.txt     — added language constraint
          meta.json  — {"current": "v3", "deprecated": ["v1"]}
        sql_agent/
          v1.txt
    """

    @classmethod
    @lru_cache(maxsize=100)
    def get(cls, name: str, version: str = "current") -> str:
        """Load a prompt template. Cached after first load."""
        prompt_dir = PROMPTS_DIR / name
        if not prompt_dir.exists():
            raise FileNotFoundError(f"Prompt '{name}' not found in registry")

        if version == "current":
            meta_path = prompt_dir / "meta.json"
            if meta_path.exists():
                meta    = json.loads(meta_path.read_text())
                version = meta["current"]
            else:
                # Fall back to highest numbered version
                versions = sorted(prompt_dir.glob("v*.txt"), reverse=True)
                if not versions:
                    raise FileNotFoundError(f"No versions found for '{name}'")
                version = versions[0].stem

        prompt_path = prompt_dir / f"{version}.txt"
        if not prompt_path.exists():
            raise FileNotFoundError(f"Prompt '{name}/{version}' not found")

        return prompt_path.read_text(encoding="utf-8")

    @classmethod
    def render(cls, name: str, version: str = "current", **kwargs) -> str:
        """Load and render a prompt template with variables"""
        template = cls.get(name, version)
        return template.format(**kwargs)

    @classmethod
    def list_versions(cls, name: str) -> list[str]:
        prompt_dir = PROMPTS_DIR / name
        return sorted(p.stem for p in prompt_dir.glob("v*.txt"))

    @classmethod
    def get_meta(cls, name: str) -> dict:
        meta_path = PROMPTS_DIR / name / "meta.json"
        if meta_path.exists():
            return json.loads(meta_path.read_text())
        return {}

# prompts/rag_system/v3.txt:
"""
You are a precise information assistant.

Answer ONLY based on the provided context.
If the answer is not in the context, say exactly:
"I don't have information about this in the provided documents."

Never add information not in the context.
Always cite your source: [document_name, page X]
Answer in the same language as the question.
Keep answers under 300 words unless asked for more.
"""

# prompts/rag_system/meta.json:
# {"current": "v3", "deprecated": ["v1"], "changelog": {"v2": "Added citations", "v3": "Added multilingual support"}}

# Usage
system_prompt = PromptRegistry.get("rag_system")
system_prompt = PromptRegistry.render("sql_agent", schema=schema_str, dialect="postgres")
```

---

### Pattern 2 — LLM Gateway Pattern

#### Theory

All LLM calls in your system go through a single internal gateway. The gateway handles: authentication, logging, rate limiting, cost tracking, caching, and failover. Services never call LLM providers directly.

```
Without Gateway:
  Service A → calls OpenAI directly
  Service B → calls Groq directly
  Worker    → calls Anthropic directly
  
  Result: scattered auth, no unified logging, no cost control,
          provider failover must be implemented 3 times

With Gateway:
  Service A ──→
  Service B ──→  LLM Gateway → (routes to) → OpenAI | Groq | Anthropic
  Worker    ──→
  
  Auth, logging, rate limiting, cost — all in one place.
  Provider change = update gateway, not every service.
```

```python
# backend/core/llm_gateway.py
class LLMGateway:
    """
    Single internal proxy for all LLM calls.
    Services call this — never providers directly.
    """

    def __init__(
        self,
        providers: dict[str, BaseLLMClient],
        cache:     Cache,
        breakers:  dict[str, CircuitBreaker],
        fallback:  list[str],   # ordered fallback chain
    ):
        self._providers = providers
        self._cache     = cache
        self._breakers  = breakers
        self._fallback  = fallback   # ["groq", "openai", "bedrock"]

    async def complete(
        self,
        messages:  list[dict],
        task:      str = "general",
        org_id:    str = "",
        cacheable: bool = False,
        **kwargs,
    ) -> str:
        # 1. Check cache for deterministic requests
        if cacheable:
            cache_key = self._make_key(messages, task)
            if cached := await self._cache.get(cache_key):
                logger.info("gateway_cache_hit", extra={"extra_fields": {"task": task}})
                return cached

        # 2. Try providers in fallback order
        last_error = None
        for provider_name in self._fallback:
            breaker  = self._breakers.get(provider_name, CircuitBreaker())
            provider = self._providers[provider_name]
            try:
                result = await breaker.execute(
                    lambda: provider.complete(messages, **kwargs)
                )
                if cacheable:
                    await self._cache.set(cache_key, result, ttl=3600)
                return result
            except Exception as e:
                logger.warning(f"Provider {provider_name} failed: {e}. Trying next.")
                last_error = e

        raise RuntimeError(f"All providers failed. Last: {last_error}")

    def _make_key(self, messages: list[dict], task: str) -> str:
        import hashlib, json
        content = json.dumps({"messages": messages, "task": task}, sort_keys=True)
        return f"gw:{hashlib.sha256(content.encode()).hexdigest()}"

# FastAPI DI — gateway is a singleton, injected everywhere
@lru_cache
def get_llm_gateway() -> LLMGateway:
    return LLMGateway(
        providers={
            "groq":    LLMFactory.create(LLMConfig(provider="groq", model="llama-3.3-70b")),
            "openai":  LLMFactory.create(LLMConfig(provider="openai", model="gpt-4o")),
        },
        cache    = RedisCache(redis_client),
        breakers = {"groq": CircuitBreaker(), "openai": CircuitBreaker()},
        fallback = ["groq", "openai"],   # try Groq first (cheaper), OpenAI as fallback
    )
```

---

### Pattern 3 — Shadow Mode

#### Theory

Run a new prompt/model in parallel with the current one — silently, without affecting users. Compare outputs. Only switch when you have evidence the new version is better.

```
Without Shadow Mode:
  Deploy new prompt → immediately affects 100% of users
  If worse → users complain → emergency rollback
  Risk: high. Evidence: zero.

With Shadow Mode:
  Run old + new in parallel for 1 week
  Log both outputs, run eval metrics
  "New prompt: faithfulness 0.84 vs old: 0.72"
  Deploy with confidence. Users never knew.
  Risk: zero. Evidence: strong.
```

```python
# backend/services/rag/shadow_mode.py
import asyncio

class ShadowModeRAG:
    """
    Runs current + shadow pipeline in parallel.
    Shadow result logged but never returned to user.
    """

    def __init__(
        self,
        current_pipeline: RAGPipeline,   # what users see
        shadow_pipeline:  RAGPipeline,   # what you're testing
        shadow_ratio:     float = 0.1,   # run shadow on 10% of queries
    ):
        self._current = current_pipeline
        self._shadow  = shadow_pipeline
        self._ratio   = shadow_ratio

    async def run(self, question: str, **kwargs) -> str:
        import random

        # Always run current pipeline
        current_result = await self._current.run(question, **kwargs)

        # Randomly run shadow pipeline (don't slow down every request)
        if random.random() < self._ratio:
            asyncio.create_task(
                self._run_shadow_eval(question, current_result, **kwargs)
            )

        return current_result   # user always gets current result

    async def _run_shadow_eval(
        self, question: str, current_answer: str, **kwargs
    ) -> None:
        """Run shadow pipeline, log comparison — never raises, never blocks user"""
        try:
            shadow_answer = await self._shadow.run(question, **kwargs)
            await ShadowComparisonRepository.log({
                "question":       question,
                "current_answer": current_answer,
                "shadow_answer":  shadow_answer,
                "timestamp":      datetime.utcnow(),
            })
            # Optional: run LLM-as-judge comparison
            winner = await self._judge(question, current_answer, shadow_answer)
            logger.info("shadow_comparison", extra={
                "extra_fields": {"winner": winner, "question": question[:100]}
            })
        except Exception as e:
            logger.warning(f"Shadow eval failed (non-fatal): {e}")
```

---

### Pattern 4 — Eval-Driven Development

#### Theory

Write evaluation tests BEFORE writing the feature. Just like TDD (Test-Driven Development) but for AI. You define what "good" looks like first — then build until your evals pass.

```
Traditional TDD:
  1. Write test: assert add(2, 3) == 5
  2. Test fails
  3. Write code to make it pass
  4. Refactor

Eval-Driven Development (EDD):
  1. Write eval: assert faithfulness(rag_answer, context) > 0.80
  2. Eval fails (nothing built yet)
  3. Build RAG pipeline until eval passes
  4. Improve with RAGAS metric as your north star

Why this matters:
  Without EDD → "the AI seems to work" (how do you know? you don't)
  With EDD → "faithfulness 0.84 on 50-question golden set" (you know exactly)
```

```python
# eval/golden_dataset.py — define quality BEFORE building
GOLDEN_QUESTIONS = [
    {
        "question":      "What is the return policy for electronics?",
        "ground_truth":  "Electronics can be returned within 30 days with receipt.",
        "source_file":   "policy_2026.pdf",
    },
    {
        "question":      "What are the warranty terms for appliances?",
        "ground_truth":  "Appliances have a 2-year manufacturer warranty.",
        "source_file":   "warranty_terms.pdf",
    },
    # ... 50+ questions covering all edge cases
]

# eval/run_eval.py
async def run_eval_suite(rag_pipeline: RAGPipeline) -> EvalReport:
    results = []
    for item in GOLDEN_QUESTIONS:
        answer   = await rag_pipeline.run(item["question"])
        contexts = await retriever.retrieve(item["question"])

        results.append({
            "question":     item["question"],
            "answer":       answer,
            "contexts":     [c.content for c in contexts],
            "ground_truth": item["ground_truth"],
        })

    from ragas import evaluate
    from ragas.metrics import faithfulness, answer_relevancy, context_precision
    from datasets import Dataset

    scores = evaluate(Dataset.from_list(results),
                      metrics=[faithfulness, answer_relevancy, context_precision])

    return EvalReport(
        faithfulness     = scores["faithfulness"],
        answer_relevancy = scores["answer_relevancy"],
        context_precision= scores["context_precision"],
        passed = (
            scores["faithfulness"]      > 0.75 and
            scores["answer_relevancy"]  > 0.80 and
            scores["context_precision"] > 0.65
        )
    )

# CI/CD gate — eval must pass before merge
# .github/workflows/eval.yml:
# - run: python eval/run_eval.py && echo "Eval passed" || exit 1
```

---

### Pattern 5 — Idempotent Ingestion

#### Theory

When the same document is ingested multiple times (re-upload, scheduled re-crawl), you should detect it and skip re-embedding. SHA256 hash of content = fingerprint.

```
Without Idempotent Ingestion:
  User uploads policy.pdf
  System: chunk → embed → store (500 chunks created)
  
  User uploads same policy.pdf (typo fix — only 1 page changed)
  System: chunk → embed → store (500 MORE chunks created)
  Now 1,000 chunks — 500 duplicates degrading retrieval quality
  Cost: paid for 500 unnecessary embeddings (~$0.01 wasted per re-upload)

With Idempotent Ingestion:
  Hash file content → check if hash exists in DB
  Same hash → skip (already indexed, nothing changed)
  Different hash → detect changed pages, re-embed only those
  Cost: $0 for re-uploads, better retrieval quality
```

```python
# backend/services/ingestion/idempotent_ingestion.py
import hashlib

async def ingest_document(
    file_path: str,
    source_id: str,
    org_id: str,
    user_id: str,
) -> IngestResult:
    # 1. Hash the entire document content
    with open(file_path, "rb") as f:
        content_hash = hashlib.sha256(f.read()).hexdigest()

    # 2. Check if already ingested
    existing = await SourceRepository.find_by_hash(content_hash, org_id)
    if existing:
        return IngestResult(
            status="skipped",
            message="Document already indexed (hash match)",
            source_id=existing.id,
        )

    # 3. Chunk-level deduplication (if source exists but content changed)
    pages        = extract_pages(file_path)
    new_chunks   = []
    for page in pages:
        page_hash = hashlib.sha256(page.content.encode()).hexdigest()
        if not await ChunkRepository.hash_exists(page_hash, org_id):
            new_chunks.append(CreateChunkDTO(
                content      = page.content,
                content_hash = page_hash,
                source_id    = source_id,
                org_id       = org_id,
                user_id      = user_id,
                metadata     = page.metadata,
            ))

    if not new_chunks:
        return IngestResult(status="skipped", message="All pages already indexed")

    # 4. Embed and store only new chunks
    embeddings = await embed_batch([c.content for c in new_chunks])
    await ChunkRepository.create_many_with_embeddings(new_chunks, embeddings)
    await SourceRepository.update(source_id, {"content_hash": content_hash})

    return IngestResult(
        status="indexed",
        message=f"Indexed {len(new_chunks)} new chunks",
        source_id=source_id,
    )
```

---

### Pattern 6 — Fallback Chain

#### Theory

Define a sequence of fallback options when the primary fails. For AI systems: primary LLM → cheaper LLM → cached answer → degraded mode (simple template response).

```python
# backend/providers/fallback_chain.py
from typing import Callable, Any

class FallbackChain:
    """
    Try each option in order, return first success.
    Each fallback degrades gracefully.
    """
    def __init__(self, options: list[tuple[str, Callable]]):
        self._options = options   # [(name, async_callable)]

    async def execute(self, *args, **kwargs) -> Any:
        last_error = None
        for name, option in self._options:
            try:
                result = await option(*args, **kwargs)
                if name != self._options[0][0]:
                    logger.warning(f"Used fallback: {name}")
                return result
            except Exception as e:
                logger.warning(f"Option '{name}' failed: {e}. Trying next.")
                last_error = e

        raise RuntimeError(f"All fallbacks failed. Last error: {last_error}")

# Build the chain for RAG answers
rag_fallback = FallbackChain([
    ("gpt4o_full_rag",    lambda q, ctx: gpt4o_rag.answer(q, ctx)),      # primary
    ("groq_full_rag",     lambda q, ctx: groq_rag.answer(q, ctx)),        # cheaper
    ("cached_answer",     lambda q, ctx: cache.get_closest(q)),           # from cache
    ("degraded_mode",     lambda q, ctx: f"I cannot answer '{q}' right now. Try again shortly."),  # last resort
])

answer = await rag_fallback.execute(question, context)
```

---

## 5.3 — Semantic Kernel

### 🧑‍🎓 Explain Like I'm in BTech 2nd Year

If LangChain is the Python-native AI framework, Semantic Kernel (SK) is Microsoft's answer — built for enterprise, with first-class .NET support, tight Azure/Microsoft integration, and a plugin architecture that feels like building an AI operating system.

Think of SK's "Kernel" as the central command console of your AI app. You plug in capabilities (plugins), memory systems, and LLM providers. The AI can then plan and execute complex tasks by combining plugins automatically.

**When you'd encounter SK in the real world:**
- Client is a Microsoft/Azure shop (90% of enterprises)
- .NET backend that needs AI capabilities
- Azure OpenAI is the mandated LLM (compliance requirements)
- Enterprise SSO + Entra ID authentication required

---

### What Semantic Kernel Is

```
Semantic Kernel (SK):
  Creator:    Microsoft (open-source)
  Languages:  Python, C#, Java
  Focus:      Enterprise AI orchestration + Azure integration
  Philosophy: "AI as a brain that uses plugins as tools"

Core mental model:
  Kernel        = the AI "brain" (orchestrates everything)
  Plugins       = collections of capabilities (tools the AI can use)
  Planner       = automatic plan generation from high-level goals
  Memory        = semantic memory (vector DB integration)
  Filters       = pre/post processing hooks (like middleware)
```

### SK vs LangChain vs LangGraph

```
                Semantic Kernel     LangChain          LangGraph
                ───────────────     ─────────          ─────────
Primary lang    Python + C# + Java  Python + JS        Python
Best for        Enterprise/Azure    Quick prototypes   Complex agents
Agent control   Planner (auto)      AgentExecutor      Full graph control
State mgmt      Basic               Basic              ✅ Advanced (checkpointer)
HITL            Plugin-based        Limited            ✅ Native (interrupt)
Enterprise SSO  ✅ Azure Entra      Limited            Limited
Azure OpenAI    ✅ First-class       Supported          Supported
Observability   ✅ AI Events        Callbacks           Stream events
Learning curve  Medium              Low                High

Pick SK when:   Microsoft/Azure ecosystem, .NET teams, enterprise clients
Pick LangGraph: Complex agents, HITL needed, state persistence required
Pick LangChain: Quick RAG prototypes, many integrations needed
```

### Installation and Setup

```bash
pip install semantic-kernel

# Azure OpenAI (most common in enterprise)
pip install semantic-kernel[azure]

# With Qdrant memory
pip install semantic-kernel[qdrant]
```

### Core Concept 1 — The Kernel

```python
# The Kernel is the central object — everything else is attached to it
import asyncio
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion, AzureChatCompletion
from semantic_kernel.connectors.ai.open_ai import OpenAITextEmbedding

# Option A: Standard OpenAI
kernel = Kernel()
kernel.add_service(OpenAIChatCompletion(
    service_id="chat",
    ai_model_id="gpt-4o",
    api_key=settings.openai_api_key,
))

# Option B: Azure OpenAI (enterprise — most common)
kernel = Kernel()
kernel.add_service(AzureChatCompletion(
    service_id="chat",
    deployment_name="gpt-4o",
    endpoint=settings.azure_openai_endpoint,
    api_key=settings.azure_openai_key,
))

# Add embedding model
kernel.add_service(OpenAITextEmbedding(
    service_id="embedding",
    ai_model_id="text-embedding-3-small",
))
```

### Core Concept 2 — Semantic Functions (Prompt-as-Function)

```python
# In SK, prompts are first-class functions — stored in files, callable like Python functions

# plugins/CustomerSupport/AnswerQuery/skprompt.txt:
"""
You are a helpful customer support agent for {{$company_name}}.
Answer the following customer query based on the provided context.
If you don't know the answer, say so clearly.

Context:
{{$context}}

Customer Query: {{$query}}

Response:
"""

# plugins/CustomerSupport/AnswerQuery/config.json:
# {
#   "schema": 1,
#   "description": "Answer customer support queries from context",
#   "execution_settings": {"default": {"max_tokens": 500, "temperature": 0.1}},
#   "input_variables": [
#     {"name": "query",        "description": "Customer question", "is_required": true},
#     {"name": "context",      "description": "Support context",   "is_required": true},
#     {"name": "company_name", "description": "Company name",      "default": "SynapseIQ"}
#   ]
# }

# Load the plugin
from semantic_kernel.functions import KernelPlugin

plugin = kernel.add_plugin(
    plugin_name="CustomerSupport",
    parent_directory="./plugins",
)

# Call like a Python function
result = await kernel.invoke(
    plugin["AnswerQuery"],
    query="How do I reset my password?",
    context="Password reset instructions: Go to Settings → Security → Reset Password.",
    company_name="SynapseIQ",
)
print(result)
```

### Core Concept 3 — Native Functions (Python-as-Tool)

```python
# Python functions exposed as AI-callable tools
from semantic_kernel.functions import kernel_function
from semantic_kernel.functions.kernel_plugin import KernelPlugin

class SupportPlugin:
    """Native plugin — Python functions the AI can call"""

    @kernel_function(name="get_order_status", description="Get the status of a customer order")
    async def get_order_status(self, order_id: str) -> str:
        """
        Retrieve current order status from database.
        Args:
            order_id: The order ID in format ORD-XXXXX
        """
        order = await OrderRepository.get(order_id)
        if not order:
            return f"Order {order_id} not found"
        return f"Order {order_id}: {order.status} — Expected delivery: {order.eta}"

    @kernel_function(name="create_ticket", description="Create a support ticket for issues needing human review")
    async def create_ticket(self, issue: str, priority: str = "normal") -> str:
        """
        Create a support ticket.
        Args:
            issue:    Description of the customer's issue
            priority: 'urgent' for payment/account issues, 'normal' for general issues
        """
        ticket_id = await TicketRepository.create(issue=issue, priority=priority)
        return f"Ticket {ticket_id} created. Our team will respond within {'2 hours' if priority == 'urgent' else '24 hours'}."

# Register native plugin
kernel.add_plugin(SupportPlugin(), plugin_name="Support")
```

### Core Concept 4 — Planner (Auto Plan Generation)

```python
# The Planner is SK's agent — it automatically creates a plan to achieve a goal
# by selecting and combining the right plugins
from semantic_kernel.planners.function_calling_stepwise_planner import (
    FunctionCallingStepwisePlanner,
    FunctionCallingStepwisePlannerOptions,
)

# Create planner with available plugins
planner = FunctionCallingStepwisePlanner(
    service_id="chat",
    options=FunctionCallingStepwisePlannerOptions(max_iterations=10),
)

# Give it a high-level goal — it figures out the steps
result = await planner.invoke(
    kernel=kernel,
    question="Check the status of order ORD-12345 and if it's delayed, create an urgent ticket.",
)
print(result.final_answer)

# Planner automatically:
# 1. Calls get_order_status("ORD-12345")
# 2. Evaluates if it's delayed (yes — expected 3 days ago)
# 3. Calls create_ticket(issue="Order delayed", priority="urgent")
# 4. Returns combined answer to user
```

### Core Concept 5 — Memory (Semantic Search)

```python
# SK Memory: store and retrieve text by semantic similarity
from semantic_kernel.memory.semantic_text_memory import SemanticTextMemory
from semantic_kernel.connectors.memory.qdrant import QdrantMemoryStore

# Setup Qdrant-backed semantic memory
memory_store = QdrantMemoryStore(
    vector_size=1536,
    url="http://localhost:6333",
)
memory = SemanticTextMemory(
    storage=memory_store,
    embeddings_generator=kernel.get_service("embedding"),
)
kernel.add_plugin(TextMemoryPlugin(memory), plugin_name="Memory")

# Store information
await memory.save_information(
    collection="support_docs",
    id="policy_returns",
    text="Products can be returned within 30 days with original receipt.",
    description="Return policy",
)

# Retrieve by semantic similarity
results = await memory.search(
    collection="support_docs",
    query="How long can I keep the product before returning?",
    limit=3,
    min_relevance_score=0.75,
)
for result in results:
    print(f"Score: {result.relevance:.3f} | {result.text}")
```

### Full SK Project — Customer Service Bot

```python
# Complete customer service bot using Semantic Kernel
import asyncio
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
from semantic_kernel.planners.function_calling_stepwise_planner import FunctionCallingStepwisePlanner

async def build_customer_service_bot():
    kernel = Kernel()
    kernel.add_service(OpenAIChatCompletion(
        service_id="chat",
        ai_model_id="gpt-4o",
        api_key=settings.openai_api_key,
    ))

    # Add plugins
    kernel.add_plugin(SupportPlugin(),    plugin_name="Support")
    kernel.add_plugin(OrderPlugin(),      plugin_name="Orders")
    kernel.add_plugin(KnowledgePlugin(),  plugin_name="Knowledge")

    # Add filters (like middleware)
    from semantic_kernel.filters.functions.function_invocation_context import FunctionInvocationContext
    
    @kernel.filter("function_invocation")
    async def log_tool_calls(ctx: FunctionInvocationContext, next):
        logger.info(f"SK calling: {ctx.function.plugin_name}.{ctx.function.name}")
        await next(ctx)
        logger.info(f"SK completed: {ctx.function.name}")

    planner = FunctionCallingStepwisePlanner(service_id="chat")

    async def handle_customer_query(customer_message: str) -> str:
        result = await planner.invoke(kernel=kernel, question=customer_message)
        return result.final_answer

    return handle_customer_query

# Run
bot = await build_customer_service_bot()
answer = await bot("My order ORD-9876 hasn't arrived. It's been 10 days!")
print(answer)
# "I checked your order ORD-9876 — it shows as delayed in transit.
#  I've created an urgent support ticket (TKT-54321) and our team
#  will reach out within 2 hours with an update."
```

---

## 5.4 — Agent Communication Protocols

### 🧑‍🎓 Explain Like I'm in BTech 2nd Year

You've learned how agents USE tools (MCP). Now imagine two AI agents from different companies that need to talk to each other. Agent A (built with LangGraph) needs to delegate a task to Agent B (built with Semantic Kernel). How do they communicate?

Without a standard: you'd need to write a custom API bridge every time.
With a protocol: A2A defines the standard language agents speak to each other.

```
MCP (Model Context Protocol):
  Agent ↔ Tool
  "Fetch this file", "Query this database", "Send this email"
  Tool is passive — it just executes what the agent asks

A2A (Agent-to-Agent Protocol):
  Agent ↔ Agent
  "Agent, please research this topic and give me a summary"
  The other agent is active — it reasons, plans, and executes
```

### A2A Protocol (Google's Standard)

```
Released:    April 2025 by Google DeepMind
Purpose:     Standard protocol for agent-to-agent task delegation
Transport:   HTTP/JSON (familiar to all developers)
Key concept: Agent Cards — each agent publishes what it can do
             (like a business card for AI agents)
```

```python
# A2A Agent Card — what your agent advertises
AGENT_CARD = {
    "name":        "SynapseIQ SQL Analyst",
    "description": "Analyses structured data and answers questions via SQL",
    "url":         "https://api.synapseiq.com/a2a",
    "version":     "1.0.0",
    "capabilities": {
        "streaming":   True,
        "pushNotifications": False,
    },
    "skills": [
        {
            "id":          "data_analysis",
            "name":        "Data Analysis",
            "description": "Analyse CSV, Excel, or database data to answer business questions",
            "examples": [
                "What is the revenue by region for Q3?",
                "Show me the top 10 customers by order value",
            ],
        }
    ]
}

# Serving the A2A endpoint
from fastapi import FastAPI
from pydantic import BaseModel

class A2ATask(BaseModel):
    id:      str
    message: dict   # A2A message format

class A2AResponse(BaseModel):
    id:     str
    result: dict

@router.get("/.well-known/agent.json")
async def get_agent_card():
    """A2A Agent Discovery — other agents call this to learn what we can do"""
    return AGENT_CARD

@router.post("/a2a")
async def handle_a2a_task(task: A2ATask) -> A2AResponse:
    """Handle a task delegated from another agent"""
    user_message = task.message["parts"][0]["text"]

    # Route to our SQL agent
    result = await sql_agent.ainvoke({
        "question": user_message,
        "source_ids": settings.default_source_ids,
    })

    return A2AResponse(
        id=task.id,
        result={
            "parts": [{"type": "text", "text": result["final_answer"]}],
            "status": {"state": "completed"},
        }
    )

# Calling another A2A agent from your agent
import httpx

async def delegate_to_a2a_agent(agent_url: str, task: str) -> str:
    """Call a remote A2A-compatible agent"""
    async with httpx.AsyncClient() as client:
        # 1. Discover agent capabilities
        card = await client.get(f"{agent_url}/.well-known/agent.json")
        agent_info = card.json()

        # 2. Send task
        response = await client.post(
            f"{agent_url}/a2a",
            json={
                "id":      str(uuid4()),
                "message": {"parts": [{"type": "text", "text": task}]},
            },
            timeout=60.0,
        )
        result = response.json()
        return result["result"]["parts"][0]["text"]

# Example: Planner agent delegates SQL queries to SQL agent via A2A
analysis = await delegate_to_a2a_agent(
    agent_url="https://api.client-a.com/sql-agent",
    task="What was last quarter's revenue by region?"
)
```

### MCP vs A2A — Clear Comparison

```
MCP (Tool Protocol):           A2A (Agent Protocol):
  Agent asks Tool to act         Agent asks Agent to act
  Tool is stateless              Agent is stateful (has memory, plans)
  Tool returns raw data          Agent returns reasoned answer
  Tool can't refuse              Agent can ask clarifying questions
  Tool doesn't know context      Agent has its own context
  
  Example:                       Example:
  "Query this database"          "Research this company and summarise"
  Tool executes SQL, returns rows Agent decides HOW to research,
                                 uses its own tools, returns summary
  
When to use which:
  MCP: connecting to databases, APIs, file systems, web search
  A2A: cross-platform agent collaboration, delegating complex tasks
```

### Protocol-Agnostic Agent Design

```python
# Build agents that support BOTH MCP and A2A from day one
# by abstracting the communication protocol

from abc import ABC, abstractmethod

class AgentCapability(ABC):
    """Abstract interface — works whether called via MCP or A2A"""

    @abstractmethod
    async def handle(self, request: str, metadata: dict) -> str: ...

class SQLAnalysisCapability(AgentCapability):
    """The actual capability — doesn't know about the transport protocol"""

    async def handle(self, request: str, metadata: dict) -> str:
        result = await sql_agent.ainvoke({"question": request})
        return result["final_answer"]

# Adapters for each protocol
class MCPAdapter:
    """Exposes capability as MCP tool"""
    def __init__(self, capability: AgentCapability):
        self._cap = capability

    def as_mcp_tool(self):
        @tool(description=f"Execute {self._cap.__class__.__name__}")
        async def tool_fn(request: str) -> str:
            return await self._cap.handle(request, {})
        return tool_fn

class A2AAdapter:
    """Exposes capability as A2A endpoint"""
    def __init__(self, capability: AgentCapability):
        self._cap = capability

    async def handle_a2a_request(self, task: A2ATask) -> A2AResponse:
        text   = task.message["parts"][0]["text"]
        result = await self._cap.handle(text, task.metadata)
        return A2AResponse(id=task.id, result={"parts": [{"type":"text","text":result}]})

# One capability, served via both protocols
capability = SQLAnalysisCapability()
mcp_tool   = MCPAdapter(capability).as_mcp_tool()    # available via MCP
a2a_adapter= A2AAdapter(capability)                   # available via A2A
```

---

## 🏗️ Phase 5 Project

### Project — SynapseIQ Architecture Refactor

**Goal:** Apply all Phase 5 patterns to the SynapseIQ codebase built in Phases 1-4. This is a refactor project, not a new build — just as valuable in the real world.

**What you build:**

```
Task 1 — LLM Factory (2 hrs):
  Replace all direct OpenAI/Groq calls with LLMFactory
  All providers behind BaseLLMClient interface
  Tests use MockLLMClient — zero API calls in tests

Task 2 — Strategy Pattern for Retrieval (2 hrs):
  VectorRetrievalStrategy, BM25Strategy, HybridStrategy
  RAGPipeline receives injected strategy
  Config switches strategy without code change

Task 3 — Guardrail Chain (2 hrs):
  InjectionGuard → PIIMaskingGuard → CostGuard → RateLimitGuard
  Applied to every chat endpoint
  Add new guard without touching existing ones

Task 4 — Prompt Registry (2 hrs):
  Move all hardcoded prompts to prompts/ directory
  Version v1.txt, v2.txt with meta.json
  All code uses PromptRegistry.get("name")

Task 5 — Observer for Tracking (2 hrs):
  EventBus with DBObserver, CostObserver, LangfuseObserver
  track_step context manager wraps every pipeline step
  Remove all scattered log/track calls

Task 6 — Semantic Kernel Bot (3 hrs):
  Rebuild customer support use case using SK
  Plugins: OrderPlugin, KnowledgePlugin
  Planner routes between them automatically
  Compare with LangGraph implementation

Task 7 — A2A Endpoint (2 hrs):
  Add /.well-known/agent.json to SynapseIQ
  POST /a2a handles delegated queries
  Test by calling from a second FastAPI app

Total: ~15 hrs
```

---

## 📋 Interview Cheat Sheet

**Q: Why do you need design patterns for AI systems specifically?**
```
AI systems have unique problems regular web apps don't:
1. Provider lock-in: LLM APIs change, go down, get expensive → Factory + Strategy
2. Prompt versioning: changing a prompt breaks things silently → Prompt Registry
3. Cost control: every bad design decision costs money → Gateway + Decorator
4. Non-determinism: AI output varies → Observer for tracking quality
5. Guardrails: AI can be misused or produce harmful output → Chain of Responsibility

Each pattern solves a specific AI engineering problem.
Not using them → technical debt that costs real money and reliability.
```

**Q: Explain the difference between Factory and Strategy pattern**
```
Factory: about CREATING objects
  "How do I create the right LLM client?"
  LLMFactory.create("groq") → GroqClient
  LLMFactory.create("openai") → OpenAIClient
  Used once at startup.

Strategy: about USING objects interchangeably
  "Which retrieval algorithm should I use right now?"
  RAGPipeline(retriever=HybridStrategy())
  Can be swapped at runtime.
  The pipeline doesn't know which strategy it's using.

Both together:
  Factory creates the Strategy objects
  Strategy plugged into the Service
  LLMFactory.create(provider) → BaseLLMClient (Strategy)
  RAGPipeline(llm=client)
```

**Q: What is the Chain of Responsibility and when do you use it for AI?**
```
A pipeline where each handler either processes or passes to the next.
Can stop the chain at any point.

For AI guardrails:
  InjectionGuard → PIIMaskingGuard → CostGuard → RateLimitGuard
  Each is an independent class with a single responsibility.
  Adding a new guard = new class, no changes to existing ones.

Benefits over if/elif chain:
  Each guard independently testable
  Order is explicit and configurable
  Adding/removing guards doesn't touch other guards
  Open/Closed Principle: open to extension, closed to modification
```

**Q: What is Semantic Kernel and when would you choose it over LangChain/LangGraph?**
```
Semantic Kernel = Microsoft's AI orchestration framework
  Python + C# + Java
  First-class Azure OpenAI integration
  Plugin architecture (semantic + native functions)
  Built-in planner for automatic plan generation

Choose SK when:
  Client is a Microsoft/Azure shop
  .NET team needs AI capabilities (C# SK is excellent)
  Azure OpenAI is mandated (compliance, data residency)
  Enterprise SSO via Azure Entra ID required

Choose LangGraph over SK when:
  Need precise agent state control
  HITL with complex approval workflows
  State persistence across sessions (Postgres checkpointer)
  Non-Azure infrastructure

In practice: many enterprise clients use SK for Azure-native apps.
You should know SK so you can work with those codebases.
```

**Q: What is A2A and how is it different from MCP?**
```
MCP (Model Context Protocol):
  Agent ↔ Tool communication
  Tools are passive executors (query DB, read file, call API)
  Agent calls tool, tool returns data, agent reasons about it

A2A (Agent-to-Agent Protocol):
  Agent ↔ Agent communication
  Both sides are intelligent agents
  Agent A delegates a complex TASK to Agent B
  Agent B plans, uses its own tools, returns reasoned output

Analogy:
  MCP = calling a database (passive — returns what you ask)
  A2A = calling a consultant (active — figures out what to do)

Use case for A2A:
  SynapseIQ SQL Agent (Agent A) needs market research
  It calls a web research agent (Agent B) via A2A
  Research agent searches web, reads articles, returns summary
  Both are agents — A2A is their communication protocol
```

---

## 🃏 Quick Revision Cards

```
CARD 1: Factory Pattern
  Problem: provider hardcoded everywhere
  Solution: LLMFactory.create(provider) → BaseLLMClient
  When:     startup, creating providers/strategies
  Benefit:  swap provider = change env var, not code
  + Singleton: reuse created clients, don't recreate per request

CARD 2: Strategy Pattern
  Problem: if/elif for algorithm selection scattered in code
  Solution: interface + swap implementation at runtime
  When:     retrieval algo, embedding model, output format
  Benefit:  add new algorithm = new class, no existing code changes
  Example:  HybridRetrievalStrategy, VectorRetrievalStrategy

CARD 3: Chain of Responsibility
  Problem: nested validation/guardrail logic in one function
  Solution: handler chain, each passes or stops
  When:     guardrails, middleware, validation pipelines
  Benefit:  add/remove step = one class, no other changes
  Order:    InjectionGuard → PIIMask → CostGuard → RateLimit

CARD 4: Observer Pattern
  Problem: tracking code (log, cost, Langfuse) scattered in pipeline
  Solution: emit events, observers subscribe independently
  When:     monitoring, cost tracking, SSE streaming, logging
  Benefit:  pipeline has zero tracking code; observers handle it
  Tool:     EventBus + track_step context manager

CARD 5: Repository Pattern
  Problem: raw DB queries in service layer, untestable
  Solution: abstract interface (Protocol), fake for tests
  When:     any data access (Postgres, Qdrant, Redis)
  Benefit:  test with FakeRepository, no DB needed
  Rule:     services NEVER write SQL/queries directly

CARD 6: Decorator Pattern
  Problem: retry, cache, cost tracking duplicated per function
  Solution: @with_retry @with_cache @with_cost_tracking decorators
  When:     cross-cutting concerns (retry, logging, caching)
  Benefit:  stack behaviours without changing the function

CARD 7: Circuit Breaker
  CLOSED:    normal — let requests through
  OPEN:      failing — fast fail all (don't hammer dead provider)
  HALF_OPEN: test one request to check recovery
  Recovery:  after timeout, allow one test request
  Use:       wrap every external LLM provider call

CARD 8: Prompt Registry
  Problem:  prompt strings hardcoded across 15 files
  Solution: prompts/{name}/{version}.txt + meta.json
  Usage:    PromptRegistry.get("rag_system") → current version
  Benefit:  version control, A/B test, rollback prompts
  Rule:     NEVER hardcode prompts in Python code

CARD 9: Shadow Mode
  Problem:  deploying new prompt risks production quality
  Solution: run new version in parallel, compare silently
  Flow:     User request → current (returned) + shadow (logged)
  Output:   quality comparison after N requests → deploy with data
  Ratio:    10% of traffic → shadow (to not slow every request)

CARD 10: Semantic Kernel
  Creator:  Microsoft | Languages: Python, C#, Java
  Kernel:   central orchestration object
  Plugin:   collection of callable functions (semantic + native)
  Planner:  auto-generates execution plan from high-level goal
  Memory:   semantic search over stored text (Qdrant, Chroma)
  Use when: Azure/Microsoft shop, .NET team, enterprise client
  vs LG:    SK for Azure-native; LangGraph for complex state/HITL

CARD 11: A2A Protocol
  A2A = Agent-to-Agent (Google, April 2025)
  vs MCP: MCP = agent↔tool | A2A = agent↔agent
  Agent Card: JSON at /.well-known/agent.json (agent discovery)
  Task:   POST /a2a with message → get back reasoned response
  When:   cross-platform agent collaboration
  Example: Planner delegates research to external Research Agent

CARD 12: LLM Gateway Pattern
  All LLM calls route through one internal gateway
  Gateway handles: auth, logging, cost, cache, failover
  Services never call providers directly
  Benefit: change provider = update gateway, not every service
  Implementation: LiteLLM Proxy (Phase 12) or custom gateway class
```

---

## ✅ Phase 5 Completion Checklist

```
PYTHON DESIGN PATTERNS
[ ] Implemented LLMFactory with 3+ providers (Groq, OpenAI, Anthropic)
[ ] All LLM clients implement BaseLLMClient interface
[ ] Tests use MockLLMClient (zero real API calls)
[ ] Implemented Strategy pattern for retrieval (vector/BM25/hybrid)
[ ] Implemented Chain of Responsibility for guardrails (4 handlers)
[ ] Implemented Observer + EventBus + track_step context manager
[ ] All DB access goes through Repository classes
[ ] Tests use FakeRepository (zero real DB)
[ ] Implemented Decorator pattern (retry + cache + cost tracking)
[ ] Circuit Breaker wraps all external LLM calls
[ ] LLM clients are Singletons (not created per request)

GENAI ARCHITECTURE PATTERNS
[ ] Prompt Registry: all prompts in prompts/ directory with versions
[ ] PromptRegistry.get() used everywhere (no hardcoded strings)
[ ] LLM Gateway pattern implemented (all calls route through it)
[ ] Fallback chain: primary → fallback → cached → degraded mode
[ ] Shadow Mode: implemented and tested with 10% traffic ratio
[ ] Idempotent ingestion: hash-based dedup, no re-embedding unchanged docs
[ ] Eval-Driven: golden dataset exists before pipeline was built
[ ] RAGAS suite runs on CI/CD with quality gates

SEMANTIC KERNEL
[ ] Kernel created with OpenAI/Azure service
[ ] At least 2 native plugins implemented (Python functions)
[ ] Semantic function (prompt template) loaded from file
[ ] Planner invoked with high-level goal (auto-plans steps)
[ ] Memory integration with vector store (search by similarity)
[ ] SK vs LangChain vs LangGraph decision made and documented
[ ] Customer service bot built with SK

AGENT COMMUNICATION PROTOCOLS
[ ] Can explain MCP vs A2A difference clearly
[ ] A2A endpoint built: GET /.well-known/agent.json
[ ] A2A POST handler receives delegated tasks
[ ] Tested A2A by calling from a second service
[ ] Protocol-agnostic capability design implemented
[ ] Can explain when to use A2A in production systems

INTERVIEW READINESS
[ ] Can explain all 8 patterns with a real example from SynapseIQ
[ ] Can explain Chain of Responsibility without looking at notes
[ ] Can explain Shadow Mode and why it reduces deployment risk
[ ] Can explain SK Planner and when you'd choose it over LangGraph
[ ] Know the A2A 2-minute explanation cold
```

---

*Phase 5 — Design Patterns for Python GenAI | GenAI + LLMOps Engineering Roadmap 2026*
*Next: Phase 6 — Token Optimization & Cost Reduction*
