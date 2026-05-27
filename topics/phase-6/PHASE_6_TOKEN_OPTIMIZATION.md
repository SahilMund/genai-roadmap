# 💰 Phase 6 — Token Optimization & Cost Reduction
> **Complete Study Notes | Theory + Code + Use Cases | Interview Prep**
> Part of: GenAI + LLMOps Engineering Roadmap 2026
> Estimated time: 20–25 hrs over 1.5 weeks
> Prerequisites: Phase 1 (LLM APIs) + Phase 3 (FastAPI) + Phase 5 (Design Patterns)

---

## 📑 Table of Contents

1. [Why Cost Optimization Matters](#1-why-cost-optimization-matters)
2. [6.1 — Token Economics](#61--token-economics)
3. [6.2 — Prompt Compression](#62--prompt-compression)
4. [6.3 — Semantic Caching](#63--semantic-caching)
5. [6.4 — Model Selection & Routing Strategy](#64--model-selection--routing-strategy)
6. [6.5 — Output Efficiency](#65--output-efficiency)
7. [6.6 — Cost Monitoring Dashboard](#66--cost-monitoring-dashboard)
8. [Phase 6 Project](#-phase-6-project)
9. [Interview Cheat Sheet](#-interview-cheat-sheet)
10. [Quick Revision Cards](#-quick-revision-cards)

---

## 1. Why Cost Optimization Matters

### 🧑‍🎓 Explain Like I'm in BTech 2nd Year

Imagine you started a small restaurant. Your food is great and customers love it. As more customers come, your ingredient costs go up — but you haven't raised prices. You're running at a loss.

LLM APIs are exactly like ingredients in this analogy. At small scale (100 queries/day), cost is irrelevant. At production scale (10,000 queries/day), the same code can cost ₹50,000/month. The architecture decisions you make upfront determine whether your AI product is financially sustainable.

```
What "not caring about cost" looks like at scale:

SynapseIQ use case:
  100 enterprise clients × 50 queries/day × ₹1.20 per query
  = ₹6,000/day = ₹1,80,000/month

  GPT-4o for every query (current approach): ₹1,80,000/month
  Smart routing + caching (this phase):      ₹22,000/month
  Savings:                                   ₹1,58,000/month

At that scale, cost optimization is not an afterthought.
It's the difference between a profitable product and a money pit.
```

### The 5 Levers of LLM Cost Control

```
LEVER 1: CACHE          → Don't compute what you already know
LEVER 2: COMPRESS       → Send fewer tokens to the LLM
LEVER 3: ROUTE          → Use cheaper models for simpler tasks
LEVER 4: CONSTRAIN      → Limit output length
LEVER 5: MONITOR        → Know where your money goes before you run out

This phase covers all 5.
```

---

## 6.1 — Token Economics

### 🧑‍🎓 Explain Like I'm in BTech 2nd Year

A token is roughly 3/4 of a word. "Hello world" = 2 tokens. Every single token you send to or receive from an LLM costs money. Understanding the pricing model is the foundation of cost control.

Think of it like a phone plan where you pay per word — both for what you say AND what you hear back. And it turns out hearing (output) is 3-5x more expensive than speaking (input).

---

### Token Pricing — Real Numbers (2026)

```
Provider          Model                  Input/1M tokens  Output/1M tokens
─────────────────────────────────────────────────────────────────────────
Groq              llama-3.3-70b          $0.59            $0.79
OpenAI            gpt-4o-mini            $0.15            $0.60
OpenAI            gpt-4o                 $2.50            $10.00
Anthropic         claude-haiku-4         $0.80            $4.00
Anthropic         claude-sonnet-4        $3.00            $15.00
Google            gemini-2.0-flash       $0.10            $0.40
Google            gemini-2.5-pro         $1.25            $10.00

KEY INSIGHT: Output is always 3–5× more expensive than input.
A 200-word answer costs more than sending a 2000-word document as context.

CACHED INPUT (Anthropic + OpenAI):
  Anthropic:   cache hit costs 10% of normal input price
  OpenAI:      cache hit costs 50% of normal input price
  → Repeated system prompts should ALWAYS use caching
```

### Cost Anatomy of a Typical RAG Query

```python
# backend/core/cost_calculator.py

COST_PER_1M_TOKENS = {
    "groq/llama-3.3-70b-versatile":  {"input": 0.59,   "output": 0.79,   "cached_input": 0.0},
    "openai/gpt-4o-mini":            {"input": 0.15,   "output": 0.60,   "cached_input": 0.075},
    "openai/gpt-4o":                 {"input": 2.50,   "output": 10.00,  "cached_input": 1.25},
    "anthropic/claude-haiku-4":      {"input": 0.80,   "output": 4.00,   "cached_input": 0.08},
    "anthropic/claude-sonnet-4":     {"input": 3.00,   "output": 15.00,  "cached_input": 0.30},
    "google/gemini-2.0-flash":       {"input": 0.10,   "output": 0.40,   "cached_input": 0.0},
}

def calculate_query_cost(
    model:          str,
    input_tokens:   int,
    output_tokens:  int,
    cached_tokens:  int = 0,
) -> float:
    """Calculate exact cost in USD for one LLM call"""
    rates = COST_PER_1M_TOKENS.get(model, {"input": 0.002, "output": 0.002, "cached_input": 0.001})
    cost = (
        ((input_tokens - cached_tokens) / 1_000_000) * rates["input"]  +
        (cached_tokens                  / 1_000_000) * rates["cached_input"] +
        (output_tokens                  / 1_000_000) * rates["output"]
    )
    return round(cost, 8)

# Example breakdown of one RAG query
def log_rag_query_cost():
    """
    Typical RAG query token breakdown:
    
    System prompt:     500 tokens  (same every request — can be cached)
    Retrieved context: 3000 tokens (5 chunks × 600 tokens each)
    User question:     50 tokens
    ─────────────────────────────
    Total input:       3550 tokens
    
    Generated answer:  250 tokens
    
    Cost on GPT-4o:        (3550/1M × $2.50) + (250/1M × $10.00) = $0.0089 + $0.0025 = $0.011
    Cost on gpt-4o-mini:   (3550/1M × $0.15) + (250/1M × $0.60)  = $0.0005 + $0.00015 = $0.00065
    Cost on Groq Llama:    (3550/1M × $0.59) + (250/1M × $0.79)  = $0.0021 + $0.00020 = $0.0023
    
    100 such queries/day:
      GPT-4o:       $1.10/day  = $33/month
      gpt-4o-mini:  $0.065/day = $1.95/month   ← 17x cheaper for similar quality
      Groq Llama:   $0.23/day  = $6.90/month
    
    At 10,000 queries/day across 100 clients:
      GPT-4o:       $110/day   = $3,300/month
      gpt-4o-mini:  $6.50/day  = $195/month    ← 17x savings, ~$3,000/month
      Groq Llama:   $23/day    = $690/month
    """
```

### Prompt Caching — The Easiest Win

```python
# backend/providers/anthropic_cached.py
import anthropic

client = anthropic.AsyncAnthropic()

async def rag_with_prompt_caching(
    system_prompt: str,
    retrieved_context: str,
    question: str,
) -> str:
    """
    Anthropic prompt caching:
    - Mark the system prompt as cacheable (TTL: 5 minutes)
    - First call: full price
    - Subsequent calls with SAME system prompt: 90% discount on cached tokens

    For RAG, the system prompt is identical on every call.
    Cache hit saves ~90% of the system prompt cost.
    """
    response = await client.messages.create(
        model="claude-haiku-4-20250514",
        max_tokens=1000,
        system=[
            {
                "type": "text",
                "text": system_prompt,
                "cache_control": {"type": "ephemeral"},  # ← mark as cacheable
            }
        ],
        messages=[
            {
                "role": "user",
                "content": f"Context:\n{retrieved_context}\n\nQuestion: {question}",
            }
        ],
    )

    # Log cache performance
    usage = response.usage
    cache_tokens   = getattr(usage, "cache_read_input_tokens", 0)
    uncached_tokens= usage.input_tokens - cache_tokens
    savings = (cache_tokens / max(usage.input_tokens, 1)) * 90  # 90% discount on cached
    logger.info("cache_performance", extra={"extra_fields": {
        "cached_tokens":   cache_tokens,
        "uncached_tokens": uncached_tokens,
        "estimated_savings_pct": round(savings, 1),
    }})

    return response.content[0].text

# OpenAI automatic prompt caching (no explicit setup needed)
async def rag_with_openai_caching(
    system_prompt: str,
    retrieved_context: str,
    question: str,
) -> str:
    """
    OpenAI automatically caches prompts longer than 1024 tokens.
    No code change needed — just ensure your system prompt is >1024 tokens.
    50% discount on cached portions.
    Check: response.usage.prompt_tokens_details.cached_tokens
    """
    from openai import AsyncOpenAI
    client = AsyncOpenAI()

    response = await client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": f"Context:\n{retrieved_context}\n\nQuestion: {question}"},
        ],
        max_tokens=500,
    )

    # Check cache hit
    cached = getattr(response.usage.prompt_tokens_details, "cached_tokens", 0)
    if cached:
        logger.info(f"OpenAI cache hit: {cached} tokens at 50% discount")

    return response.choices[0].message.content
```

### Cost-Per-Feature Calculation

```python
# backend/analytics/cost_estimator.py

class FeatureCostEstimator:
    """
    Estimate monthly LLM cost per product feature before building.
    Avoids expensive surprises in production.
    """

    def estimate_monthly(
        self,
        queries_per_day:    int,
        avg_input_tokens:   int,
        avg_output_tokens:  int,
        model:              str,
        cache_hit_rate:     float = 0.0,   # 0.0 to 1.0
    ) -> dict:
        daily_cost    = self._daily_cost(queries_per_day, avg_input_tokens,
                                          avg_output_tokens, model, cache_hit_rate)
        monthly_cost  = daily_cost * 30
        yearly_cost   = daily_cost * 365

        return {
            "model":          model,
            "queries_per_day":queries_per_day,
            "daily_cost_usd": round(daily_cost, 4),
            "monthly_cost_usd": round(monthly_cost, 2),
            "yearly_cost_usd":  round(yearly_cost, 2),
            "cost_per_query":   round(daily_cost / max(queries_per_day, 1), 6),
        }

    def compare_models(
        self,
        queries_per_day: int,
        avg_input_tokens: int,
        avg_output_tokens: int,
    ) -> list[dict]:
        """Compare cost across all major models for a given usage pattern"""
        results = []
        for model in COST_PER_1M_TOKENS:
            est = self.estimate_monthly(
                queries_per_day, avg_input_tokens, avg_output_tokens, model
            )
            results.append(est)
        return sorted(results, key=lambda x: x["monthly_cost_usd"])

# Usage
estimator = FeatureCostEstimator()
comparison = estimator.compare_models(
    queries_per_day=1000,
    avg_input_tokens=3500,
    avg_output_tokens=300,
)
# Output (sorted cheapest first):
# gemini-2.0-flash:   $3.96/month
# gpt-4o-mini:        $6.30/month
# groq-llama-3.3-70b: $23.10/month
# gpt-4o:             $93.00/month  ← 23x more expensive than Gemini Flash
```

---

## 6.2 — Prompt Compression

### 🧑‍🎓 Explain Like I'm in BTech 2nd Year

You're sending a 5,000-word document to the LLM along with a 300-word system prompt, and the user asked a 20-word question. The LLM only needs maybe 500 words of that document to answer. You're paying for 4,500 words of context that didn't help.

Prompt compression is like a smart highlighter that reads your document and marks only the parts the LLM actually needs — then sends only those parts.

---

### Why Long Prompts Are Expensive

```
Full RAG context example:
  Retrieved chunks: 5 × 1000 tokens = 5000 tokens
  System prompt:    500 tokens
  User question:    30 tokens
  ─────────────────────────────
  Total input:      5530 tokens
  
  On GPT-4o: 5530 × $2.50/1M = $0.0138 per query
  
  After compression (remove irrelevant sentences):
  Compressed context: 5 × 300 tokens = 1500 tokens
  System prompt:      500 tokens
  User question:      30 tokens
  ─────────────────────────────
  Total:              2030 tokens
  
  On GPT-4o: 2030 × $2.50/1M = $0.0051 per query
  
  Savings: 63% per query.
  At 10,000 queries/day: saves ~$87/day = $2,610/month
```

### LLMLingua — Automatic Prompt Compression

```python
# pip install llmlingua
# backend/services/compression/llmlingua_compressor.py

from llmlingua import PromptCompressor

class LLMLinguaCompressor:
    """
    LLMLingua: uses a small LLM (Llama/Phi) to score token importance.
    Removes low-importance tokens while preserving meaning.
    Achieves 2-5x compression with minimal quality loss.
    """
    def __init__(self, model_name: str = "microsoft/llmlingua-2-xlm-roberta-large-meetingbank"):
        self._compressor = PromptCompressor(
            model_name=model_name,
            use_llmlingua2=True,    # LLMLingua-2: faster, better
            device_map="cpu",
        )

    def compress_context(
        self,
        question:          str,
        retrieved_chunks:  list[str],
        target_ratio:      float = 0.4,   # compress to 40% of original
        force_tokens:      list[str] = None,  # tokens that must never be removed
    ) -> str:
        """
        Compress retrieved context while preserving information needed to answer the question.

        Args:
            question:         The user's question (guides which tokens to keep)
            retrieved_chunks: List of retrieved text chunks to compress
            target_ratio:     Final size as % of original (0.4 = 60% reduction)
            force_tokens:     Tokens to always keep (e.g., numbers, names)

        Returns: Compressed context string
        """
        context = "\n\n---\n\n".join(retrieved_chunks)

        result = self._compressor.compress_prompt(
            context,
            instruction=question,       # question guides what to keep
            question=question,
            target_token=len(context.split()) * target_ratio,
            force_tokens=force_tokens or ["not", "no", "never", "always"],
            condition_in_question="after_condition",
        )

        original_tokens   = len(context.split())
        compressed_tokens = len(result["compressed_prompt"].split())
        reduction_pct     = (1 - compressed_tokens / max(original_tokens, 1)) * 100

        logger.info("context_compressed", extra={"extra_fields": {
            "original_tokens":   original_tokens,
            "compressed_tokens": compressed_tokens,
            "reduction_pct":     round(reduction_pct, 1),
        }})

        return result["compressed_prompt"]

# Usage in RAG pipeline
async def rag_with_compression(
    question:  str,
    chunks:    list[Chunk],
    llm_model: str,
) -> str:
    compressor = LLMLinguaCompressor()

    # Compress retrieved context before sending to expensive LLM
    compressed_context = compressor.compress_context(
        question         = question,
        retrieved_chunks = [c.content for c in chunks],
        target_ratio     = 0.4,
    )

    return await llm.complete([
        {"role": "system", "content": RAG_SYSTEM_PROMPT},
        {"role": "user",   "content": f"Context:\n{compressed_context}\n\nQuestion: {question}"},
    ])
```

### System Prompt Compression — Manual Techniques

```python
# backend/core/prompt_optimizer.py

class PromptOptimizer:
    """Manual techniques to reduce system prompt token count"""

    @staticmethod
    def remove_redundancy(prompt: str) -> str:
        """
        Common wastes in system prompts:
        
        Before (145 tokens):
        'You are a very helpful and knowledgeable AI assistant that is designed
         to provide accurate, detailed, and comprehensive answers to any questions
         that the user might have. You should always be polite and respectful in
         your responses. Please make sure to answer the question completely.'

        After (28 tokens):
        'Answer accurately and completely. Be concise.'
        
        Same LLM behaviour. 80% fewer tokens. $0 quality loss.
        """
        lines = [line.strip() for line in prompt.split('\n') if line.strip()]
        # Remove lines that are pure politeness/filler
        filler_phrases = [
            "you should always be", "please make sure", "it is important that",
            "remember to always", "don't forget to",
        ]
        lines = [
            l for l in lines
            if not any(f in l.lower() for f in filler_phrases)
        ]
        return '\n'.join(lines)

    @staticmethod
    def add_token_budget(prompt: str, max_words: int = 150) -> str:
        """Append a concise output constraint — reduces output tokens"""
        return prompt + f"\n\nAnswer in under {max_words} words unless more detail is explicitly requested."

    @staticmethod
    def estimate_tokens(text: str) -> int:
        """Quick token estimate: words / 0.75"""
        return int(len(text.split()) / 0.75)

# Before vs After — Real Example
BEFORE = """
You are a highly knowledgeable customer support assistant working for SynapseIQ,
an AI data intelligence platform. Your job is to help users with their questions
about the platform. You should always be polite, professional, and helpful.
Please make sure to answer the question based only on the provided context.
If you don't know the answer, it is very important that you say so clearly
rather than making up information. Always cite your sources.
"""
# Tokens: ~105

AFTER = """
Customer support agent for SynapseIQ AI platform.
Answer ONLY from context. Cite sources as [doc, page].
If not in context: "I don't have that information."
Under 200 words.
"""
# Tokens: ~38
# Savings: 64% — identical LLM behaviour
```

### Dynamic Few-Shot Selection

```python
# backend/services/prompts/few_shot_selector.py
from openai import AsyncOpenAI

class DynamicFewShotSelector:
    """
    Instead of including ALL examples in every prompt (expensive),
    select only the 2-3 most relevant examples for each query.

    Example bank with 50 examples × 200 tokens = 10,000 tokens always included
    Dynamic selection: 3 examples × 200 tokens = 600 tokens
    Savings: 94% on few-shot tokens
    """

    def __init__(self, examples: list[dict], embedder):
        self._examples = examples
        self._embedder = embedder
        self._example_embeddings: list[list[float]] = []

    async def build_index(self):
        """Pre-embed all examples once at startup"""
        texts = [f"{ex['input']} {ex['output']}" for ex in self._examples]
        self._example_embeddings = await self._embedder.embed(texts)

    async def select(
        self,
        query: str,
        k: int = 3,
    ) -> list[dict]:
        """Select k most semantically similar examples for this query"""
        import numpy as np

        query_embedding = (await self._embedder.embed([query]))[0]
        query_vec       = np.array(query_embedding)

        # Cosine similarity with all examples
        similarities = []
        for i, ex_emb in enumerate(self._example_embeddings):
            ex_vec = np.array(ex_emb)
            sim    = float(np.dot(query_vec, ex_vec) /
                          (np.linalg.norm(query_vec) * np.linalg.norm(ex_vec) + 1e-8))
            similarities.append((i, sim))

        # Return top-k most similar examples
        top_k = sorted(similarities, key=lambda x: x[1], reverse=True)[:k]
        return [self._examples[i] for i, _ in top_k]

    async def build_few_shot_prompt(self, query: str, k: int = 3) -> str:
        selected = await self.select(query, k)
        examples_text = "\n\n".join(
            f"Q: {ex['input']}\nA: {ex['output']}"
            for ex in selected
        )
        return f"Examples:\n{examples_text}\n\nNow answer:\nQ: {query}\nA:"
```

---

## 6.3 — Semantic Caching

### 🧑‍🎓 Explain Like I'm in BTech 2nd Year

Imagine you work in customer support. 50 customers in a row ask "How do I reset my password?" You answer the first customer in 30 seconds. For customers 2-50, you don't re-think the answer — you just say the same thing faster because you already know it.

Semantic caching is this for AI systems. But smarter: it recognises that "password reset steps" and "how to change my password" and "forgot my password, what do I do?" are all asking the same thing — and returns the cached answer for all of them.

```
Exact caching (what Redis does normally):
  "how do I reset my password?"  → cache HIT ✅
  "How do I reset my password?"  → cache MISS ❌  (capital H)
  "how do I reset my password"   → cache MISS ❌  (no ?)
  
Semantic caching (this section):
  "how do I reset my password?"  → cache HIT ✅
  "steps to reset password"      → cache HIT ✅  (similarity 0.94)
  "forgot password, what to do?" → cache HIT ✅  (similarity 0.91)
  "how to delete my account?"    → cache MISS ❌  (similarity 0.31)
```

### Three Levels of Caching

```python
# backend/services/cache/multi_level_cache.py
import hashlib, json
from openai import AsyncOpenAI
import numpy as np
import redis.asyncio as aioredis

class MultiLevelCache:
    """
    Level 1: Exact hash match      → 0ms retrieval
    Level 2: Semantic similarity   → 10ms retrieval  
    Level 3: SQL result cache      → 5ms retrieval

    In production: L1 hits ~20% of queries, L2 hits ~15% more
    Combined ~35% cache rate = significant cost and latency savings
    """

    def __init__(
        self,
        redis:          aioredis.Redis,
        embedder,
        similarity_threshold: float = 0.92,
        l1_ttl:         int = 3600,    # 1 hour exact cache
        l2_ttl:         int = 7200,    # 2 hours semantic cache
    ):
        self._redis     = redis
        self._embedder  = embedder
        self._threshold = similarity_threshold
        self._l1_ttl    = l1_ttl
        self._l2_ttl    = l2_ttl

    def _exact_key(self, query: str, source_ids: list[str]) -> str:
        content = f"{query.lower().strip()}|{'|'.join(sorted(source_ids))}"
        return f"cache:l1:{hashlib.sha256(content.encode()).hexdigest()}"

    async def get(
        self,
        query:      str,
        source_ids: list[str],
    ) -> tuple[str | None, str]:
        """
        Returns: (cached_answer, cache_level) or (None, "miss")
        """
        # ── Level 1: Exact match ──────────────────────────────────────
        l1_key = self._exact_key(query, source_ids)
        exact  = await self._redis.get(l1_key)
        if exact:
            return json.loads(exact)["answer"], "l1_exact"

        # ── Level 2: Semantic match ───────────────────────────────────
        query_embedding = (await self._embedder.embed([query]))[0]
        semantic_result = await self._semantic_search(query_embedding, source_ids)
        if semantic_result:
            return semantic_result, "l2_semantic"

        return None, "miss"

    async def set(
        self,
        query:          str,
        source_ids:     list[str],
        answer:         str,
        query_embedding: list[float] | None = None,
    ) -> None:
        """Store in both L1 (exact) and L2 (semantic) caches"""
        payload = json.dumps({"answer": answer, "query": query})

        # L1: exact key → answer
        l1_key = self._exact_key(query, source_ids)
        await self._redis.setex(l1_key, self._l1_ttl, payload)

        # L2: store embedding → answer mapping
        if query_embedding is None:
            query_embedding = (await self._embedder.embed([query]))[0]

        l2_key = f"cache:l2:emb:{l1_key}"
        await self._redis.setex(
            l2_key,
            self._l2_ttl,
            json.dumps({
                "embedding":  query_embedding,
                "answer":     answer,
                "source_ids": source_ids,
            }),
        )
        # Add to semantic index
        await self._redis.sadd(f"cache:l2:index:{'|'.join(sorted(source_ids))}", l2_key)

    async def _semantic_search(
        self,
        query_embedding: list[float],
        source_ids: list[str],
    ) -> str | None:
        """Find cached answer with similar meaning"""
        index_key  = f"cache:l2:index:{'|'.join(sorted(source_ids))}"
        cache_keys = await self._redis.smembers(index_key)

        if not cache_keys:
            return None

        query_vec  = np.array(query_embedding)
        best_score = 0.0
        best_answer= None

        for key in cache_keys:
            raw = await self._redis.get(key)
            if not raw:
                continue
            data   = json.loads(raw)
            emb    = np.array(data["embedding"])
            sim    = float(np.dot(query_vec, emb) /
                          (np.linalg.norm(query_vec) * np.linalg.norm(emb) + 1e-8))
            if sim > best_score:
                best_score  = sim
                best_answer = data["answer"]

        if best_score >= self._threshold:
            logger.info("semantic_cache_hit", extra={"extra_fields": {
                "similarity": round(best_score, 4),
                "threshold":  self._threshold,
            }})
            return best_answer

        return None

    async def invalidate_source(self, source_id: str) -> int:
        """
        When a source is updated/deleted, invalidate all related cache entries.
        Call this after any source modification to prevent stale answers.
        """
        pattern = f"cache:l1:*"
        count   = 0
        async for key in self._redis.scan_iter(pattern):
            raw = await self._redis.get(key)
            if raw and source_id in raw.decode():
                await self._redis.delete(key)
                count += 1
        logger.info(f"Cache invalidated {count} entries for source {source_id}")
        return count
```

### Streaming Cached Responses

```python
# Problem: user expects streaming, but cache returns instant full text
# Solution: fake-stream the cached answer for UX consistency

async def stream_with_cache(
    question:   str,
    source_ids: list[str],
    cache:      MultiLevelCache,
    pipeline:   RAGPipeline,
) -> AsyncIterator[str]:
    """
    Check cache first. If hit: fake-stream the cached response.
    If miss: real stream from LLM, then cache the full response.
    """
    cached_answer, cache_level = await cache.get(question, source_ids)

    if cached_answer:
        # Fake stream: yield tokens with small delay to match UX expectation
        # Users don't notice the difference between 50ms fake-stream and real stream
        yield f"data: {json.dumps({'type': 'cache_hit', 'level': cache_level})}\n\n"
        words = cached_answer.split()
        for i, word in enumerate(words):
            space = " " if i < len(words) - 1 else ""
            yield f"data: {json.dumps({'type': 'token', 'content': word + space})}\n\n"
            if i % 10 == 0:   # small pause every 10 words
                await asyncio.sleep(0.02)
        yield "data: [DONE]\n\n"
        return

    # Real stream — collect full response for caching
    full_response = ""
    async for token_event in pipeline.stream(question, source_ids):
        yield token_event
        if '"type": "token"' in token_event:
            data  = json.loads(token_event.split("data: ")[1])
            full_response += data.get("content", "")

    # Cache the complete response for future queries
    if full_response:
        await cache.set(question, source_ids, full_response)
```

---

## 6.4 — Model Selection & Routing Strategy

### 🧑‍🎓 Explain Like I'm in BTech 2nd Year

Not all questions need a genius. "What's 2+2?" doesn't need a Nobel Prize winner. "Explain quantum entanglement and its implications for cryptography" might.

Model routing is about asking the right model for each task — cheap models for simple tasks, expensive models only when the task genuinely needs the extra capability. Like having both an intern and a senior engineer: you don't send every task to the senior.

---

### Model Tiers

```
TIER 1 — Ultra-cheap, fast (classification, routing, simple extraction):
  Groq llama-3.1-8b-instant:   $0.05/$0.08 per 1M tokens  | latency: 200ms
  gpt-4o-mini:                 $0.15/$0.60 per 1M tokens  | latency: 600ms
  claude-haiku-4:              $0.80/$4.00 per 1M tokens  | latency: 500ms
  gemini-2.0-flash:            $0.10/$0.40 per 1M tokens  | latency: 400ms
  Use for: classification, summarisation, simple Q&A, extraction

TIER 2 — Mid-tier (most RAG answers, general chat, analysis):
  Groq llama-3.3-70b:          $0.59/$0.79 per 1M tokens  | latency: 800ms
  gpt-4o:                      $2.50/$10.0 per 1M tokens  | latency: 1.5s
  claude-sonnet-4:             $3.00/$15.0 per 1M tokens  | latency: 1.2s
  Use for: RAG generation, most agent steps, detailed analysis

TIER 3 — Premium (complex reasoning, multi-hop, advanced code):
  o3, o4-mini:                 reasoning tokens + output  | latency: 10-60s
  claude-opus-4:               highest capability         | latency: 3-5s
  gemini-2.5-pro:              $1.25/$10.0 per 1M tokens  | latency: 2s
  Use for: complex multi-step reasoning, architecture decisions, hard math

TIER 4 — Free (internal tools, offline processing):
  Ollama (local):              $0/request               | latency: varies
  Use for: internal tools, classification at scale, testing
```

### Cascade Routing — Try Cheap First

```python
# backend/providers/cascade_router.py
from pydantic import BaseModel
from typing import Literal

class QueryComplexity(BaseModel):
    level:    Literal["simple", "moderate", "complex"]
    reason:   str
    routing:  str   # which model to use

class CascadeRouter:
    """
    Try the cheapest model first.
    Evaluate output quality.
    Escalate to next tier only if quality is insufficient.
    
    Typical flow for a RAG query:
      1. Try Groq llama-3.3-70b (cheap, fast)
      2. Evaluate: is this answer grounded? is it complete?
      3. If yes → return (saved 80% vs GPT-4o)
      4. If no  → retry with GPT-4o (only when actually needed)
    """

    def __init__(self, llm_factory, embedder):
        self._factory  = llm_factory
        self._embedder = embedder

        # Tier ordering: try cheapest first
        self._tiers = [
            LLMConfig(provider="groq",   model="llama-3.3-70b-versatile"),
            LLMConfig(provider="openai", model="gpt-4o-mini"),
            LLMConfig(provider="openai", model="gpt-4o"),
        ]

    async def complete_with_cascade(
        self,
        messages:      list[dict],
        context:       str,
        quality_threshold: float = 0.7,
        max_tiers:     int = 2,
    ) -> tuple[str, str]:
        """
        Returns: (answer, model_used)
        """
        last_answer = ""
        for i, config in enumerate(self._tiers[:max_tiers]):
            llm    = self._llm_factory.create(config)
            answer = await llm.complete(messages)

            # Evaluate quality using grounding check
            quality = await self._check_grounding(answer, context)
            logger.info("cascade_tier_evaluated", extra={"extra_fields": {
                "tier":    i + 1,
                "model":   config.model,
                "quality": round(quality, 3),
                "passed":  quality >= quality_threshold,
            }})

            if quality >= quality_threshold:
                return answer, config.model   # good enough — stop here

            last_answer = answer   # save in case all tiers fail

        return last_answer, self._tiers[min(max_tiers, len(self._tiers)) - 1].model

    async def _check_grounding(self, answer: str, context: str) -> float:
        """
        Quick grounding check: is the answer actually supported by context?
        Uses embedding similarity between answer and context as a proxy.
        For production: use RAGAS faithfulness metric.
        """
        if not answer or not context:
            return 0.0
        embeddings    = await self._embedder.embed([answer, context])
        answer_emb    = np.array(embeddings[0])
        context_emb   = np.array(embeddings[1])
        similarity    = float(np.dot(answer_emb, context_emb) /
                             (np.linalg.norm(answer_emb) * np.linalg.norm(context_emb) + 1e-8))
        return similarity
```

### Task Classification Router

```python
# backend/providers/task_classifier.py
from pydantic import BaseModel
from typing import Literal

class TaskClassification(BaseModel):
    complexity:  Literal["simple", "moderate", "complex"]
    task_type:   Literal["factual_qa", "analysis", "generation", "code", "math", "classification"]
    recommended_model: str
    reasoning:   str

class TaskRouter:
    """
    Classify the query BEFORE choosing the model.
    One cheap classification call saves expensive wrong-tier usage.
    """
    ROUTING_TABLE = {
        ("simple",   "factual_qa"):    "groq/llama-3.3-70b-versatile",
        ("simple",   "classification"):"groq/llama-3.1-8b-instant",
        ("moderate", "factual_qa"):    "groq/llama-3.3-70b-versatile",
        ("moderate", "analysis"):      "openai/gpt-4o-mini",
        ("moderate", "generation"):    "openai/gpt-4o-mini",
        ("complex",  "analysis"):      "anthropic/claude-sonnet-4",
        ("complex",  "code"):          "openai/gpt-4o",
        ("complex",  "math"):          "openai/o4-mini",
        ("complex",  "generation"):    "anthropic/claude-sonnet-4",
    }

    def __init__(self, classifier_llm):
        self._llm = classifier_llm   # always use cheapest model for classification

    async def classify_and_route(self, query: str, context: str = "") -> str:
        """Returns model name to use for this query"""
        classification: TaskClassification = await self._llm.complete_structured(
            messages=[{
                "role": "user",
                "content": f"""Classify this query for LLM routing. Be accurate — routing affects cost.

Query: {query}
Context available: {"yes" if context else "no"}

Complexity guide:
  simple:   factual lookup, yes/no, extraction, classification
  moderate: summarisation, explanation, multi-step factual answer
  complex:  multi-hop reasoning, code generation, mathematical proof, 
            novel analysis requiring deep thinking""",
            }],
            response_model=TaskClassification,
        )

        model = self.ROUTING_TABLE.get(
            (classification.complexity, classification.task_type),
            "groq/llama-3.3-70b-versatile",  # safe default
        )

        logger.info("task_routed", extra={"extra_fields": {
            "query_preview":  query[:100],
            "complexity":     classification.complexity,
            "task_type":      classification.task_type,
            "model_selected": model,
        }})
        return model
```

### LiteLLM Router — Production-Grade Routing

```python
# backend/providers/litellm_router.py
import litellm
from litellm import Router

class ProductionRouter:
    """
    LiteLLM Router: automatic routing, fallback, load balancing.
    Deploy as LiteLLM Proxy server in Phase 12 for full gateway benefits.
    """

    def __init__(self):
        self._router = Router(
            model_list=[
                # Primary: Groq (fastest, cheapest)
                {
                    "model_name": "fast-cheap",
                    "litellm_params": {
                        "model":   "groq/llama-3.3-70b-versatile",
                        "api_key": settings.groq_api_key,
                    },
                    "tpm": 30000,   # tokens per minute limit
                    "rpm": 30,      # requests per minute limit
                },
                # Fallback: OpenAI
                {
                    "model_name": "fast-cheap",
                    "litellm_params": {
                        "model":   "openai/gpt-4o-mini",
                        "api_key": settings.openai_api_key,
                    },
                    "tpm": 200000,
                    "rpm": 500,
                },
                # Premium tier
                {
                    "model_name": "powerful",
                    "litellm_params": {
                        "model":   "openai/gpt-4o",
                        "api_key": settings.openai_api_key,
                    },
                },
                # Embedding
                {
                    "model_name": "embedding",
                    "litellm_params": {
                        "model":   "openai/text-embedding-3-small",
                        "api_key": settings.openai_api_key,
                    },
                },
            ],
            routing_strategy="least-busy",   # distribute across available models
            fallbacks=[
                {"fast-cheap": ["openai/gpt-4o-mini"]},   # Groq down → OpenAI
                {"powerful":   ["anthropic/claude-sonnet-4"]},
            ],
            num_retries=3,
            retry_after=5,
        )

    async def complete(self, model_alias: str, messages: list[dict], **kwargs) -> str:
        response = await self._router.acompletion(
            model=model_alias,
            messages=messages,
            **kwargs,
        )
        return response.choices[0].message.content

    async def embed(self, texts: list[str]) -> list[list[float]]:
        response = await self._router.aembedding(
            model="embedding",
            input=texts,
        )
        return [d.embedding for d in sorted(response.data, key=lambda x: x.index)]

# Usage — clean alias-based routing
router = ProductionRouter()

# Simple query → cheap model
answer = await router.complete("fast-cheap", [{"role": "user", "content": simple_question}])

# Complex query → powerful model
analysis = await router.complete("powerful", [{"role": "user", "content": complex_question}])
```

### Ollama — Zero-Cost Local Models

```python
# backend/providers/ollama_client.py
# pip install ollama
import ollama

class OllamaClient(BaseLLMClient):
    """
    Local model serving — zero API cost.
    Best for: internal tools, development, classification at scale.
    Hardware needed: 8GB RAM for 7B, 16GB for 13B, 32GB+ for 70B.

    Models to use:
      llama3.2:     3B  - classification, simple extraction (fast)
      llama3.1:     8B  - good balance of speed and quality
      phi3:         3.8B- excellent for classification tasks
      qwen2.5:      7B  - strong multilingual performance
    """

    def __init__(self, model: str = "llama3.1"):
        self._model = model
        self._client= ollama.AsyncClient(host=settings.ollama_url)

    async def complete(self, messages: list[dict], **kwargs) -> str:
        response = await self._client.chat(
            model=self._model,
            messages=messages,
        )
        return response["message"]["content"]

    async def complete_structured(self, messages, response_model, **kwargs):
        """Use instructor with Ollama for structured output"""
        import instructor
        client = instructor.from_openai(
            openai.AsyncOpenAI(
                base_url=f"{settings.ollama_url}/v1",
                api_key="ollama",   # dummy key
            ),
            mode=instructor.Mode.JSON,
        )
        return await client.chat.completions.create(
            model=self._model,
            response_model=response_model,
            messages=messages,
        )

    async def embed(self, texts: list[str]) -> list[list[float]]:
        embeddings = []
        for text in texts:
            resp = await self._client.embeddings(model="nomic-embed-text", prompt=text)
            embeddings.append(resp["embedding"])
        return embeddings

# Use Ollama for internal tools — zero cost, full privacy
# Great for: document classification, entity extraction, batch processing
ollama_llm = OllamaClient(model="llama3.1")
doc_type   = await ollama_llm.complete_structured(
    messages=[{"role":"user","content": f"Classify: {doc_text[:500]}"}],
    response_model=DocClassification,
)
```

---

## 6.5 — Output Efficiency

### 🧑‍🎓 Explain Like I'm in BTech 2nd Year

Output tokens are the most expensive part of LLM calls (3-5x input price). An LLM that gives a 500-word answer when 100 words suffices is costing you 5x more than necessary. This section is about preventing verbose outputs.

---

### Constrained Generation with JSON Schema

```python
# Structured output prevents verbose filler text

# ❌ Without constraints — LLM adds padding, explanations, hedging
response = await llm.complete([
    {"role": "user", "content": "Extract the sentiment from: 'Great product!'"}
])
# "The sentiment expressed in this text appears to be positive.
#  The word 'Great' is a clear indicator of a positive emotional response,
#  and combined with 'product', it suggests the user is satisfied with..."
# → 50 tokens for a 1-token answer

# ✅ With structured output — forces concise, parseable response
from pydantic import BaseModel
from typing import Literal

class SentimentResult(BaseModel):
    sentiment:  Literal["positive", "negative", "neutral"]
    confidence: float   # 0.0 to 1.0

result: SentimentResult = await llm.complete_structured(
    messages=[{"role":"user","content":"Extract sentiment: 'Great product!'"}],
    response_model=SentimentResult,
)
# → {"sentiment": "positive", "confidence": 0.95}
# → 3 tokens. 94% cheaper than free text answer.
```

### max_tokens Per Endpoint

```python
# backend/core/token_budgets.py
from pydantic import BaseModel

class EndpointTokenBudget(BaseModel):
    max_input_tokens:  int
    max_output_tokens: int
    model:             str

# Define budgets per endpoint based on what the feature actually needs
TOKEN_BUDGETS = {
    "rag_answer":           EndpointTokenBudget(max_input_tokens=4000,  max_output_tokens=500,  model="groq/llama-3.3-70b"),
    "rag_citation_extract": EndpointTokenBudget(max_input_tokens=2000,  max_output_tokens=100,  model="groq/llama-3.1-8b"),
    "sql_explanation":      EndpointTokenBudget(max_input_tokens=2000,  max_output_tokens=200,  model="groq/llama-3.3-70b"),
    "doc_classification":   EndpointTokenBudget(max_input_tokens=1000,  max_output_tokens=10,   model="groq/llama-3.1-8b"),
    "query_rewriting":      EndpointTokenBudget(max_input_tokens=500,   max_output_tokens=50,   model="groq/llama-3.1-8b"),
    "summarisation":        EndpointTokenBudget(max_input_tokens=8000,  max_output_tokens=300,  model="groq/llama-3.3-70b"),
    "report_generation":    EndpointTokenBudget(max_input_tokens=16000, max_output_tokens=2000, model="openai/gpt-4o"),
    "agent_step":           EndpointTokenBudget(max_input_tokens=6000,  max_output_tokens=300,  model="groq/llama-3.3-70b"),
}

# Apply budget in every LLM call
async def call_with_budget(
    endpoint:  str,
    messages:  list[dict],
    llm_client: BaseLLMClient,
) -> str:
    budget = TOKEN_BUDGETS.get(endpoint)
    if not budget:
        raise ValueError(f"No token budget defined for endpoint: {endpoint}")

    # Trim input if over budget
    total_input_tokens = sum(count_tokens(m["content"]) for m in messages)
    if total_input_tokens > budget.max_input_tokens:
        logger.warning(f"Input truncated for {endpoint}: {total_input_tokens} → {budget.max_input_tokens} tokens")
        messages = _trim_messages(messages, budget.max_input_tokens)

    return await llm_client.complete(
        messages,
        max_tokens=budget.max_output_tokens,
    )
```

### Context Right-Sizing

```python
# backend/services/rag/context_builder.py
import tiktoken

ENCODER = tiktoken.encoding_for_model("gpt-4o")

def count_tokens(text: str) -> int:
    return len(ENCODER.encode(text))

class ContextBuilder:
    """
    Don't send 128K tokens when 4K is enough.
    Measure actual context needed, then stay within it.
    
    Research (Liu et al., 2023):
    LLM attention degrades on long contexts.
    Sweet spot: 2000-8000 tokens of context.
    Beyond 8000: quality often DECREASES.
    → More context ≠ better answers after a point.
    """

    def __init__(
        self,
        max_context_tokens: int = 6000,  # hard cap
        max_chunks:         int = 5,     # never more than 5 chunks
        min_chunk_score:    float = 0.5, # discard low-relevance chunks
    ):
        self.max_context_tokens = max_context_tokens
        self.max_chunks         = max_chunks
        self.min_chunk_score    = min_chunk_score

    def build(self, chunks: list[ScoredChunk]) -> tuple[str, dict]:
        """
        Build optimal context from ranked chunks.
        Returns: (context_string, stats)
        """
        parts       = []
        token_count = 0
        used_chunks = 0

        for chunk in chunks[:self.max_chunks]:
            # Skip low-relevance chunks (even if under token limit)
            if chunk.score < self.min_chunk_score:
                continue

            chunk_tokens = count_tokens(chunk.content)

            # Stop if adding this chunk would exceed budget
            if token_count + chunk_tokens > self.max_context_tokens:
                logger.debug(f"Context budget reached at chunk {used_chunks+1}/{len(chunks)}")
                break

            citation = f"[{chunk.metadata.get('source_name', 'Unknown')}, p{chunk.metadata.get('page', '?')}]"
            parts.append(f"{citation}\n{chunk.content}")
            token_count += chunk_tokens
            used_chunks += 1

        context = "\n\n---\n\n".join(parts)
        stats   = {
            "chunks_used":    used_chunks,
            "chunks_total":   len(chunks),
            "context_tokens": token_count,
            "budget_used_pct":round(token_count / self.max_context_tokens * 100, 1),
        }
        logger.debug("context_built", extra={"extra_fields": stats})
        return context, stats
```

---

## 6.6 — Cost Monitoring Dashboard

### Theory

You can't control what you can't measure. Cost monitoring gives you:
1. Real-time spend (so you don't get a surprise bill)
2. Per-feature breakdown (so you know what's expensive)
3. Trend alerts (so you catch runaway costs before they scale)

```python
# backend/services/analytics/cost_monitor.py
from datetime import datetime, timedelta
import asyncpg

class CostMonitor:
    """
    Tracks LLM spend at every dimension:
      - Per tenant (org_id)
      - Per feature (rag_query, sql_agent, report_gen)
      - Per model (gpt-4o, groq-llama, etc.)
      - Per day, week, month
    """

    async def get_dashboard(
        self, org_id: str, period_days: int = 30
    ) -> dict:
        since = datetime.utcnow() - timedelta(days=period_days)
        async with self._pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT
                    date_trunc('day', created_at)::date AS day,
                    feature,
                    model,
                    SUM(input_tokens)  AS total_input_tokens,
                    SUM(output_tokens) AS total_output_tokens,
                    SUM(cost_usd)      AS total_cost_usd,
                    COUNT(*)           AS query_count
                FROM llm_usage
                WHERE org_id    = $1
                  AND created_at >= $2
                GROUP BY day, feature, model
                ORDER BY day DESC, total_cost_usd DESC
            """, org_id, since)

        # Aggregate for dashboard
        total_cost    = sum(r["total_cost_usd"] for r in rows)
        by_feature    = {}
        by_model      = {}
        daily_trend   = {}

        for row in rows:
            feature = row["feature"]
            model   = row["model"]
            day     = str(row["day"])
            cost    = float(row["total_cost_usd"])

            by_feature[feature] = by_feature.get(feature, 0) + cost
            by_model[model]     = by_model.get(model, 0) + cost
            daily_trend[day]    = daily_trend.get(day, 0) + cost

        return {
            "period_days":       period_days,
            "total_cost_usd":    round(total_cost, 4),
            "cost_per_day_avg":  round(total_cost / period_days, 4),
            "projected_monthly": round(total_cost / period_days * 30, 2),
            "by_feature":        dict(sorted(by_feature.items(), key=lambda x: x[1], reverse=True)),
            "by_model":          dict(sorted(by_model.items(),   key=lambda x: x[1], reverse=True)),
            "daily_trend":       dict(sorted(daily_trend.items())),
            "top_insight":       self._generate_insight(by_feature, by_model, total_cost),
        }

    def _generate_insight(self, by_feature, by_model, total_cost) -> str:
        if not by_feature:
            return "No data yet"
        top_feature = max(by_feature, key=by_feature.get)
        top_model   = max(by_model,   key=by_model.get)
        top_pct     = round(by_feature[top_feature] / max(total_cost, 0.001) * 100, 1)
        return (
            f"{top_feature} uses {top_pct}% of budget ({top_model} is the primary model). "
            f"Consider cascade routing or caching for {top_feature} queries."
        )

# FastAPI endpoint
@router.get("/analytics/cost")
async def get_cost_dashboard(
    period_days:  int = 30,
    current_user: AuthUser = Depends(get_current_user),
) -> dict:
    monitor = CostMonitor(db_pool)
    return await monitor.get_dashboard(current_user.org_id, period_days)
```

### CloudWatch Cost Alarms

```python
# infra/aws/cost_alarms.py
import boto3

def create_cost_alarm(org_id: str, daily_threshold_usd: float):
    """
    Alert when daily LLM spend exceeds threshold.
    Fires at 80% (warning) and 100% (critical).
    """
    cw = boto3.client("cloudwatch")

    for pct, severity in [(0.8, "warning"), (1.0, "critical")]:
        cw.put_metric_alarm(
            AlarmName=f"llm-cost-{severity}-{org_id}",
            MetricName="DailyLLMCost",
            Namespace="SynapseIQ/LLMCosts",
            Statistic="Sum",
            Period=86400,   # 24 hours
            EvaluationPeriods=1,
            Threshold=daily_threshold_usd * pct,
            ComparisonOperator="GreaterThanThreshold",
            AlarmActions=[SNS_TOPIC_ARN],
            Dimensions=[{"Name": "OrgId", "Value": org_id}],
        )

# Publish cost metric from application
def publish_cost_metric(org_id: str, cost_usd: float, feature: str):
    cw = boto3.client("cloudwatch")
    cw.put_metric_data(
        Namespace="SynapseIQ/LLMCosts",
        MetricData=[{
            "MetricName": "DailyLLMCost",
            "Dimensions": [
                {"Name": "OrgId",   "Value": org_id},
                {"Name": "Feature", "Value": feature},
            ],
            "Value": cost_usd,
            "Unit":  "None",
        }]
    )
```

---

## 🏗️ Phase 6 Project

### Project — SynapseIQ Cost Optimisation Sprint

**Goal:** Reduce SynapseIQ's LLM cost by 60%+ through the 5 levers — without reducing quality.

**Baseline measurement first (Day 1):**
```
1. Add llm_usage table: (org_id, feature, model, input_tokens, output_tokens, cost_usd, created_at)
2. Instrument every LLM call to log to this table
3. Run 100 test queries — capture full baseline cost breakdown
4. RAGAS baseline: faithfulness, answer_relevancy on 20 golden questions
5. Document: cost per query, tokens per query, by feature, by model
```

**Task 1 — Multi-level Cache (Days 2-3):**
```
Implement: MultiLevelCache (L1 exact + L2 semantic + L3 SQL result)
Target:    >30% cache hit rate after 200 test queries
Measure:   Hit rate per level, latency saved, cost saved
Add:       ⚡ "Answered from cache" badge in frontend
```

**Task 2 — Prompt Compression (Day 4):**
```
Implement: LLMLinguaCompressor on RAG context
Target:    40% reduction in context tokens
Measure:   RAGAS before/after (quality must not drop >5%)
Add:       Log original vs compressed token counts per query
```

**Task 3 — Model Router (Days 5-6):**
```
Implement: TaskRouter (classify complexity → pick model tier)
Default:   Groq llama-3.3-70b for most queries
Escalate:  GPT-4o only for "complex" classification
Target:    <10% of queries reach GPT-4o
Measure:   Cost per tier, quality per tier (RAGAS)
```

**Task 4 — Prompt Caching (Day 7):**
```
Implement: Anthropic cache_control on system prompt
Target:    System prompt cache hit rate >80% (same prompt every query)
Measure:   cached_input_tokens in Anthropic usage object
Add:       Log cache performance in Langfuse
```

**Task 5 — Output Constraints (Day 8):**
```
Implement: max_tokens per endpoint in TOKEN_BUDGETS dict
Target:    Average output tokens reduced from ~400 to ~200
Measure:   Output token reduction, quality check (user ratings)
Add:       Token budget monitoring in admin dashboard
```

**Task 6 — Cost Dashboard (Days 9-10):**
```
Implement: /analytics/cost endpoint + React admin chart
Display:   Daily spend trend, by feature, by model, cache hit rate
Add:       CloudWatch alarms at 80%/100% of daily budget
Target:    Full cost visibility — no surprise bills
```

**Final Measurement:**
```
Compare baseline vs optimised:
  Cost per query:    target 60% reduction
  RAGAS faithfulness:target within 5% of baseline (quality preserved)
  P95 latency:       target no regression (cache should make it faster)
  Cache hit rate:    target 30%+
```

---

## 📋 Interview Cheat Sheet

**Q: How do you approach LLM cost optimisation in production?**
```
5-lever framework:

1. CACHE (biggest win):
   Semantic caching — cache by meaning, not exact string.
   Hit rate 30%+ = 30% of queries answered for free.
   Invalidate when source data changes.

2. COMPRESS:
   LLMLingua for context compression — 40-60% token reduction.
   Dynamic few-shot selection — 3 relevant examples, not 20.
   Concise system prompts — 30 tokens, not 300.

3. ROUTE:
   Task classification → model tier selection.
   Groq (cheap) for most queries, GPT-4o only for complex ones.
   Cascade: try cheap model, escalate if quality insufficient.

4. CONSTRAIN:
   max_tokens budget per endpoint.
   Structured output (JSON schema) prevents verbose filler.
   Context budget: never send >6000 tokens of context.

5. MONITOR:
   Track cost per feature, per model, per tenant in real-time.
   CloudWatch alarms at 80%/100% daily budget.
   "You can't control what you can't measure."

At SynapseIQ: applied all 5, reduced cost 60%+
while maintaining RAGAS faithfulness within 5% of baseline.
```

**Q: What is semantic caching and how is it different from regular caching?**
```
Regular caching (Redis):
  Exact string match: "password reset" ≠ "reset password"
  Miss rate very high on natural language queries.

Semantic caching:
  Embed the query → cosine similarity against cached queries.
  "how to reset password" ≈ "password reset steps" (similarity 0.93)
  → Cache HIT even though strings are different.

Implementation:
  1. Query arrives → exact hash check (L1, 0ms)
  2. Not found → embed query → cosine search over cached embeddings (L2, 10ms)
  3. Similarity > 0.92 threshold → return cached answer
  4. Below threshold → run full RAG pipeline → cache the result

Production target: >30% hit rate saves 30% of LLM costs.
Pitfall: set threshold too low → wrong cached answer returned.
Set too high → miss rate too high → defeats the purpose.
0.90-0.95 is the sweet spot for factual Q&A.
```

**Q: Explain cascade routing for LLM cost optimisation**
```
Idea: try cheapest model first, escalate only when necessary.

Step 1: Send query to Groq llama-3.3-70b ($0.59/1M tokens)
Step 2: Evaluate output quality (grounding score vs context)
Step 3: If quality >= threshold → return (cheap win)
        If quality < threshold → retry with GPT-4o ($2.50/1M)

Result: 85-90% of queries answered cheaply.
Only 10-15% reach expensive tier.
Overall cost: ~20% of all-expensive-tier approach.

Quality check options:
  Fast: embedding similarity (answer vs context)
  Accurate: RAGAS faithfulness on the individual response
  Practical: structured output with confidence score

Key metric: what % of queries escalate to expensive tier?
Target: <15%. If >30%, your cheap model is too weak for your use case.
```

**Q: How do you implement prompt caching with Anthropic?**
```
Mark the system prompt with cache_control: {"type": "ephemeral"}

First call:   full price for system prompt
Subsequent calls (within 5 min): 10% of system prompt price

Works best for:
  RAG queries (same system prompt every call)
  Agent loops (same system prompt per session)
  High-frequency chatbots (same persona/instructions)

Code:
  messages=[{
      "type": "text",
      "text": SYSTEM_PROMPT,
      "cache_control": {"type": "ephemeral"},  # ← one line
  }]

Verify it worked:
  response.usage.cache_read_input_tokens > 0  → cache hit
  Savings: 90% discount on cached_read tokens

OpenAI: automatic for prompts >1024 tokens, no code change.
50% discount (vs Anthropic's 90%) but zero configuration.
```

**Q: What's the "lost in the middle" problem and how does context right-sizing help?**
```
Research (Liu et al., 2023): LLMs perform best on content at the
START and END of context. Information in the MIDDLE is often ignored
or causes confused answers.

Problem:
  Retrieve 20 chunks, send all 20 to LLM (20,000 tokens)
  Chunks 3-18 are largely ignored by attention mechanism
  Answer quality DECREASES despite MORE context
  Cost: 5x higher than needed

Solution (context right-sizing):
  max_chunks = 5          → never more than 5 chunks
  max_context_tokens = 6000  → hard cap on context size
  min_chunk_score = 0.5   → discard low-relevance chunks

Result: better answers with fewer tokens.
At $2.50/1M input tokens, 20K → 6K tokens = $0.035 → $0.015 per query.
43% cost reduction with BETTER quality.
```

---

## 🃏 Quick Revision Cards

```
CARD 1: Token Pricing Mental Model
  Output = 3-5x more expensive than input
  Groq:   cheapest ($0.59/$0.79 per 1M)
  GPT-4o: expensive ($2.50/$10.00 per 1M)
  Cache:  90% discount (Anthropic), 50% discount (OpenAI)
  Rule:   always benchmark with gpt-4o-mini first, escalate only if quality insufficient

CARD 2: Prompt Caching
  Anthropic: cache_control: {"type": "ephemeral"} on system prompt
             → 90% discount on cached tokens, TTL 5 min
  OpenAI:    automatic for prompts >1024 tokens, 50% discount
             → no code change needed
  Best for:  repeated system prompts (RAG, agent loops, chatbots)
  Check hit: response.usage.cache_read_input_tokens > 0

CARD 3: LLMLingua Compression
  What:    Small LLM scores token importance, removes unimportant ones
  How:     compress_prompt(context, instruction=question, target_token=N)
  Ratio:   target_ratio=0.4 → 60% reduction in context tokens
  Quality: RAGAS faithfulness drop < 5% at 40% compression
  When:    RAG context before expensive LLM generation
  Cost:    LLMLingua call is cheap (small local model)

CARD 4: Dynamic Few-Shot Selection
  Problem:  50 examples × 200 tokens = 10,000 tokens every call
  Solution: embed examples once at startup, select top-3 by similarity
  Code:     FewShotSelector.select(query, k=3) → 3 relevant examples
  Savings:  10,000 → 600 tokens on few-shot alone (94% reduction)

CARD 5: Semantic Cache Levels
  L1 Exact:    hash(query + sources) → Redis lookup (0ms)
  L2 Semantic: embed(query) → cosine search > 0.92 (10ms)
  L3 SQL:      hash(SQL + dataset) → Redis lookup (5ms)
  Hit rate:    target >30% combined across all levels
  Invalidation: source updated → delete related cache entries

CARD 6: Cascade Router
  Tier 1: Groq llama (cheap) → check quality → if ok, return
  Tier 2: gpt-4o-mini (medium) → check quality → if ok, return
  Tier 3: gpt-4o (expensive) → always return
  Target: <15% of queries reach Tier 3
  Quality check: embedding similarity(answer, context) > 0.7

CARD 7: Task Router
  Classify FIRST (cheap): complexity + task_type → model selection
  simple + factual:    groq/llama-3.1-8b-instant ($0.05/1M)
  moderate + analysis: openai/gpt-4o-mini ($0.15/1M)
  complex + code:      openai/gpt-4o ($2.50/1M)
  Classification cost: ~100 tokens = $0.000005 → saves $0.005+ if routed correctly

CARD 8: Context Right-Sizing
  max_chunks = 5           (never more than 5 chunks to LLM)
  max_context_tokens = 6000 (hard cap)
  min_chunk_score = 0.5    (discard low-relevance chunks)
  Why: "lost in the middle" — LLM ignores content in long contexts
  Research: >8K tokens often DECREASES quality (Liu et al., 2023)

CARD 9: max_tokens Budget per Endpoint
  doc_classification:  max_output=10    (it's just a label)
  query_rewriting:     max_output=50    (one sentence)
  rag_answer:          max_output=500   (a paragraph)
  report_generation:   max_output=2000  (detailed report)
  Rule: measure actual output lengths → set budget 20% above p95
  Savings: avg output reduced 50% with no quality loss in most cases

CARD 10: Cost Dashboard Dimensions
  Per org:     which client costs the most?
  Per feature: which feature is the cost driver?
  Per model:   GPT-4o usage unexpectedly high?
  Daily trend: sudden spike = something wrong
  Alert at:    80% (warning), 100% (critical) of daily budget
  Metric:      cost/query by feature (most actionable)

CARD 11: Ollama — Zero Cost Local
  Install:     ollama pull llama3.1
  SDK:         ollama.AsyncClient or OpenAI SDK with base_url
  Best for:    classification, batch processing, dev/testing, internal tools
  Hardware:    8GB RAM for 7B, 16GB for 13B
  Models:      llama3.1 (quality), phi3 (classification), qwen2.5 (multilingual)
  Use with:    instructor for structured output (via OpenAI-compatible API)

CARD 12: The 5-Lever Framework
  LEVER 1: CACHE    → don't compute what you already know (30%+ hit rate)
  LEVER 2: COMPRESS → send fewer tokens (40-60% reduction with LLMLingua)
  LEVER 3: ROUTE    → use cheap models for simple tasks (<15% reach premium)
  LEVER 4: CONSTRAIN → limit output length (max_tokens per endpoint)
  LEVER 5: MONITOR  → know where money goes (dashboard + CloudWatch alerts)
  Combined: 60%+ cost reduction, quality preserved within 5%
```

---

## ✅ Phase 6 Completion Checklist

```
TOKEN ECONOMICS
[ ] Can calculate exact cost of any LLM call (input + output + cached)
[ ] Implemented cost tracking (llm_usage table + middleware)
[ ] Used FeatureCostEstimator to predict monthly cost before building
[ ] Anthropic prompt caching: cache_control on system prompt, verified cache hits
[ ] OpenAI automatic caching: confirmed cached_tokens > 0 in usage

PROMPT COMPRESSION
[ ] LLMLinguaCompressor integrated in RAG pipeline
[ ] Measured: original vs compressed token counts per query
[ ] RAGAS faithfulness drop < 5% at 40% compression (verified)
[ ] Dynamic few-shot selector: select 3 from 50 by embedding similarity
[ ] System prompt audited: removed filler text, added concise constraints

SEMANTIC CACHING
[ ] MultiLevelCache: L1 (exact hash) + L2 (semantic similarity) implemented
[ ] Cache invalidation: source deleted → related cache entries cleared
[ ] Fake-streaming for cached responses (UX consistency)
[ ] Cache hit rate measured: target >30% after 200 queries
[ ] ⚡ Badge in frontend for cached responses

MODEL ROUTING
[ ] TaskRouter: classify complexity + type → model tier
[ ] CascadeRouter: try cheap → evaluate quality → escalate if needed
[ ] LiteLLM Router configured with fallback chains
[ ] Ollama: local model running for classification tasks
[ ] Measured: % of queries per tier, cost per tier, quality per tier

OUTPUT EFFICIENCY
[ ] TOKEN_BUDGETS dict defined for every endpoint
[ ] max_tokens enforced in every LLM call
[ ] ContextBuilder: max_chunks=5, max_context_tokens=6000
[ ] Structured output replaces free-text for classification endpoints
[ ] Average output tokens reduced by at least 30%

MONITORING
[ ] Cost dashboard: total, by feature, by model, daily trend
[ ] CloudWatch cost alarms at 80% and 100% of daily budget
[ ] Langfuse: cost per trace visible
[ ] Alert fired at least once in testing (verify Slack notification works)

OVERALL
[ ] Baseline documented: cost/query before optimisation
[ ] Final measurement: cost/query after optimisation
[ ] Target: 60% cost reduction achieved
[ ] RAGAS quality preserved within 5% of baseline
[ ] Can explain all 5 levers in an interview without notes
```

---

*Phase 6 — Token Optimization & Cost Reduction | GenAI + LLMOps Engineering Roadmap 2026*
*Next: Phase 7 — Cloud AI Platforms (Bedrock + Lambda + SageMaker + GCP ADK + K8s)*
