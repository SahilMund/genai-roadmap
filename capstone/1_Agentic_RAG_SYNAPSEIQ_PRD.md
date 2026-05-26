# SynapseIQ — AI Data Intelligence Platform
> **Type:** Internal AI Service → Multi-tenant SaaS (Agency Origin Story)
> **Built with:** Claude Code + BMAD + Graphify
> **Stack:** React 18 + Vite + Tailwind v4 · FastAPI · PostgreSQL + pgvector · Qdrant · Redis · DuckDB · AWS · Clerk · Langfuse
> **Timeline:** 14–16 weeks (weekday-focused, 2–3 hrs/day)
> **Approach:** Vertical slices — working product from Week 1, progressively enhanced

---

## 📑 Table of Contents

1. [The Origin Story](#1-the-origin-story)
2. [The Build Philosophy](#2-the-build-philosophy)
3. [Claude Code + BMAD + Graphify Setup](#3-claude-code--bmad--graphify-setup)
4. [Token Consumption Strategy](#4-token-consumption-strategy)
5. [CLAUDE.md — Project Constitution](#5-claudemd--project-constitution)
6. [UI Design System](#6-ui-design-system)
7. [Technology Stack](#7-technology-stack)
8. [Design Patterns](#8-design-patterns)
9. [System Architecture](#9-system-architecture)
10. [Feature Coverage Map](#10-feature-coverage-map)
11. [Memory Strategies](#11-memory-strategies)
12. [Guardrails](#12-guardrails)
13. [Phase-by-Phase Build Plan](#13-phase-by-phase-build-plan)
    - [Phase 0 — Foundation](#phase-0--foundation)
    - [Phase 1 — Authentication (Clerk)](#phase-1--authentication-clerk)
    - [Phase 2 — SQL Agent (First Vertical Slice)](#phase-2--sql-agent)
    - [Phase 3 — Basic RAG Pipeline (Second Vertical Slice)](#phase-3--basic-rag-pipeline)
    - [Phase 4 — Multi-Tenancy + Metadata + Live DB](#phase-4--multi-tenancy--metadata--live-db)
    - [Phase 5A — Core Advanced Retrieval](#phase-5a--core-advanced-retrieval)
    - [Phase 5B — Retrieval Refinement](#phase-5b--retrieval-refinement)
    - [Phase 6 — RAG Quality Layer](#phase-6--rag-quality-layer)
    - [Phase 7 — Planner Agent + Memory](#phase-7--planner-agent--memory)
    - [Phase 8 — Observability](#phase-8--observability)
    - [Phase 9 — Ingestion Queue + Caching](#phase-9--ingestion-queue--caching)
    - [Phase 10 — Reports + Scheduling](#phase-10--reports--scheduling)
    - [Phase 11 — Advanced Features](#phase-11--advanced-features)
    - [Phase 12 — Production Deployment](#phase-12--production-deployment)
13. [API Reference](#13-api-reference)
14. [Database Schema](#14-database-schema)
15. [Folder Structure](#15-folder-structure)
16. [Out of Scope v1](#16-out-of-scope-v1)
17. [Resume Deliverables](#17-resume-deliverables)

---

## 1. The Origin Story

### Why This Was Built

We were a full-stack development agency building custom applications for multiple clients. Two clients came to us with the same request — they wanted AI features embedded in their applications:

1. A chatbot that knew their product documentation so customers could self-serve during onboarding instead of raising support tickets
2. A way for their managers and admins to query database tables in plain English without depending on developers or analysts

We built it for the first client. When the second client asked for the same thing, we realised we were about to duplicate the entire AI layer — the ingestion pipeline, vector database setup, RAG system, SQL agent — all of it — for every new client. That's not a business. That's technical debt.

### The Insight

Instead of embedding AI into each client's codebase, we built a centralised AI service that any client application could call via API. Each client gets full data isolation — their own S3 bucket, their own vector collection, their own database credentials — all configured through environment variables. Our code is shared. Their data never is.

```
Before (per-client duplication):
  Client A app → custom RAG code → Client A's vector DB
  Client B app → same RAG code (copy-pasted) → Client B's vector DB
  Client C app → same RAG code again → ...
  
  N clients = N codebases to maintain
  One bug fix = N deployments
  Not scalable.

After (centralised AI service):
  Client A app ──→
  Client B app ──→  SynapseIQ API (one codebase)
  Client C app ──→       ↓
                    Routes to client's isolated resources
                    (their S3, their Qdrant collection, their DB)
  
  One codebase. N clients. Zero duplication.
```

### The Three Use Cases It Solves (Per Client)

**Use Case 1 — Customer Onboarding Bot (RAG)**
```
Problem:  New users of a client's inventory management app
          didn't know how to get started. Every question
          meant raising a support ticket — 1 day response time.

Solution: RAG agent trained on the client's product docs,
          user manuals, and FAQs. Users get instant answers.
          Support ticket volume dropped significantly.
```

**Use Case 2 — Internal Team Knowledge Base (RAG)**
```
Problem:  Client's internal teams (ops, HR, finance) had
          SOPs and policies spread across PDFs. Finding
          the right document took 20+ minutes.

Solution: Same RAG pipeline, different document set.
          Staff ask questions, get answers with citations.
          New employee onboarding time reduced.
```

**Use Case 3 — Manager / Admin Analytics (SQL Agent)**
```
Problem:  Managers needed daily data insights but were
          dependent on developers to write SQL queries.
          "How many orders are pending in warehouse B?"
          took 1-2 days to get an answer.

Solution: SQL agent connected to the client's existing DB.
          Managers type questions in plain English.
          Get answers, charts, and explanations in seconds.
          Developer dependency eliminated for routine queries.
```

### The Technical Decision

```
Why a multi-tenant service instead of per-client deployment?

We evaluated two approaches:
  Option A: Deploy a separate instance per client
            → clean isolation but N deployments to maintain
            → bug fix requires N deployments
            → not scalable past 5 clients

  Option B: One centralised service, per-client data isolation
            → single deployment
            → data isolation via org_id + encrypted credentials
            → new client onboarding = provisioning env vars, not deploying code

We chose Option B.
New client onboarding went from "deploy a new instance" to
"add their credentials to our secrets manager."

The architecture is standard multi-tenant SaaS.
We applied it to AI services.
```

---

## 2. The Build Philosophy

### Vertical Slices, Not Horizontal Layers

```
WRONG (horizontal layers):
  Week 1-3:  Build ALL ingestion infrastructure
  Week 4-6:  Build ALL SQL agent
  Week 7-9:  Build ALL RAG pipeline
  Week 10:   First working query
  Result:    10 weeks of work, nothing to show, architecture
             bugs discovered too late to fix

RIGHT (vertical slices):
  Week 1-2:  CSV → SQL Agent → answer on screen  ← WORKING PRODUCT
  Week 3-4:  PDF → Basic RAG → cited answer      ← SECOND FEATURE WORKING
  Week 5-6:  Multi-tenancy hardened
  Week 7-8:  RAG made accurate (hybrid, reranking)
  Week 9:    Quality measured (RAGAS gate in CI/CD)
  Week 10+:  Advanced features on proven foundation
  Result:    Working product from day 10.
             Every phase ends with something demoable.
             Bugs caught early when they're cheap to fix.
```

### The Phase Rule

```
Every phase ends with something a user can actually USE.
Not "infrastructure is ready." Not "backend is done."
A real user can open the browser and DO something new.

Phase 2:  "Upload a CSV and ask a question"          → works
Phase 3:  "Upload a PDF and ask a question"          → works
Phase 4:  "Multiple orgs, isolated data"             → proven
Phase 5A: "Better answers, measurably"               → RAGAS scores improve
Phase 6:  "We can prove quality with numbers"        → CI/CD gate live
Phase 7:  "One question, two systems, one answer"    → hybrid query works
...
```

### Evidence-Based Iteration

```
Never add a technique without measuring its impact.
Phase 5A establishes RAGAS baseline.
Phase 5B only adds techniques that improve the baseline.

"We added hybrid search — Recall@5 improved from 0.61 to 0.79."
Not: "We added hybrid search because it sounds good."

This is the difference between engineering and cargo-culting.
```

---

## 3. Claude Code + BMAD + Graphify Setup

### BMAD — Why It Matters

```
Without BMAD:
  "Claude, build me a FastAPI auth system"
  → unpredictable output, drifts every session, no spec

With BMAD:
  /pm        → user stories from PRD
  /architect → tech spec from stories
  /dev       → implements from spec
  /qa        → tests against acceptance criteria
  → consistent, accountable, traceable
```

| Agent | Command | Use For |
|-------|---------|---------|
| Product Manager | `/pm` | User stories, requirements |
| Architect | `/architect` | DB schema, system design |
| Developer | `/dev` | Implementation |
| QA | `/qa` | Tests, acceptance criteria |
| Scrum Master | `/sm` | Break phases into daily tasks |
| UX | `/ux` | Component design |
| Analyst | `/analyst` | Research before building |

### Installation

```bash
# Prerequisites: node v20+, python 3.11+

# 1. Create project + BMAD
mkdir synapseiq && cd synapseiq
git init
npx bmad-method install --yes --modules bmm --tools claude-code

# 2. Graphify — knowledge graph (saves tokens as codebase grows)
# PyPI: graphifyy (double-y). CLI: graphify (single y)
pip install graphifyy && graphify install

# Verify
ls .claude/skills/      # pm/ architect/ dev/ qa/ sm/ ux/ analyst/
graphify --help         # version info

# 3. Docs structure
mkdir -p docs/stories docs/architecture docs/decisions eval
cp SYNAPSEIQ_PRD.md docs/prd.md

# 4. Open Claude Code
claude
```

### Graphify — Token Savings

```
Phase 5+: codebase has 80+ files
Without Graphify:
  Claude reads 20 files to understand one feature → 15,000 tokens

With Graphify:
  /graphify query "how does RAG retrieval pipeline work?"
  → reads GRAPH_REPORT.md (~500 tokens)
  → navigates to exact files only
  → 95% less context, same quality

# Build initial graph (run inside Claude Code after Phase 0)
/graphify .

# Auto-update on every commit
graphify hook install

# Manual update after major changes
graphify update . --force
```

### BMAD Workflow Per Phase

```
START OF PHASE:
  /pm        → generate user stories from PRD section
  /architect → tech spec, DB migration, API contracts
  /sm        → break into 3-5 daily tasks

DURING PHASE:
  /dev       → implement ONE task at a time
  /qa        → write tests immediately after
  /dev       → fix failures

END OF PHASE:
  Update docs/architecture.md with decisions made
  /compact   → compress context
  git commit → clean commit per phase
  graphify update .
```

---

## 4. Token Consumption Strategy

### The 8 Rules

```
1. CLAUDE.md stays under 200 lines — hard limit
2. .claudeignore everything that isn't source code
3. /compact after every phase — never carry stale context
4. /clear between unrelated tasks
5. One task per session — never chain unrelated work
6. Disable unused MCP servers — each costs tokens on startup
7. Haiku for routine · Sonnet for features · Opus for architecture
8. /graphify query before reading raw files (Phase 3+)
```

### `.claudeignore`

```gitignore
node_modules/
.venv/
__pycache__/
*.pyc
.pytest_cache/
dist/
build/
*.egg-info/
*.pdf
*.csv
*.xlsx
*.txt
uploads/
data/
alembic/versions/
graphify-out/
*.log
logs/
.vscode/
.idea/
```

### Model Selection

| Task | Model | Reason |
|------|-------|--------|
| System architecture, DB design | Opus 4 | High reasoning |
| Feature implementation | Sonnet 4 | Default |
| Tests, docstrings, simple fixes | Haiku 3.5 | Routine |
| Stuck > 30 minutes | Opus 4 | Worth the cost |

---

## 5. CLAUDE.md — Project Constitution

```markdown
# CLAUDE.md — SynapseIQ

## Project
AI Service Platform — multi-tenant SaaS for agency clients
Stack: React 18 + Vite + Tailwind v4 | FastAPI | PostgreSQL + pgvector
       Qdrant | Redis | DuckDB | AWS | Clerk | Langfuse

## Architecture
Frontend:  React Vite SPA → /frontend
Backend:   FastAPI monolith → /backend
Workers:   Celery tasks → /backend/workers/
Auth:      Clerk (JWT validation via Clerk SDK)
DB:        PostgreSQL + pgvector (MVP) → Qdrant (Phase 5+)
Cache:     Redis (sessions, semantic cache, rate limiting)
Files:     AWS S3 (per-tenant bucket)
Queue:     AWS SQS (prod) / Redis (local)
Tracing:   Langfuse (every LLM call)

## Current Phase
Phase: 0 — Foundation
Active story: docs/stories/story-000-setup.md

## Multi-Tenancy
EVERY database query MUST filter by org_id AND user_id.
No exceptions. This is the primary security invariant.
Clerk provides org_id from JWT claims.

## Code Standards
Python:     type hints on ALL functions, Pydantic v2, no bare except
            formatter: ruff format | linter: ruff check (see pyproject.toml)
TypeScript: strict mode, no `any`, barrel exports
            formatter: prettier | linter: eslint (see .eslintrc.json)
Logs:       always use get_logger(__name__) — never print()
            always pass extra={"extra_fields": {...}} for structured data
            never log full prompts or PII at INFO level (use DEBUG)
Tests:      pytest + Vitest — tests alongside every feature
API:        all responses use {data, error, meta} wrapper
Commits:    conventional (feat: fix: chore: docs:) — enforced by commitlint

## Patterns
Repository pattern:       all DB access through /repositories/ (never raw SQL in services)
Strategy pattern:         LLM + VectorStore swappable via Factory + interface
Factory pattern:          LLMFactory, VectorStoreFactory, build_guardrail_pipeline()
Adapter pattern:          all external API responses adapted to internal types
Dependency Injection:     FastAPI Depends() for all services, DB, LLM, cache
Chain of Responsibility:  guardrail pipeline (injection → PII → cost → rate limit)
Observer pattern:         EventBus for execution tracking (DB + Langfuse + SSE)
Custom Hooks:             all data fetching in hooks, components are pure render
Compound Components:      complex UI split into composable sub-components
No LangChain:             direct SDKs only (groq, openai, anthropic, cohere)
instructor:               structured LLM output (Pydantic + auto-retry)
LangGraph:                all agents (SQL Agent Phase 2, Planner Phase 7)
See Section 8 for full pattern details with code examples.

## File Permissions
READ freely:   /backend/api/ /backend/services/ /frontend/src/
READ on ask:   /backend/workers/ /docs/
NEVER READ:    node_modules/ .venv/ dist/ uploads/ logs/ graphify-out/

## Context Navigation (Phase 3+)
1. ALWAYS query graph first: `/graphify query "your question"`
2. Only read raw files if I say "read the file directly"
3. Trust GRAPH_REPORT.md for architecture overview

## Guardrails (always enforce)
1. Injection check on ALL user inputs before LLM call
2. PII masking before sending to external LLM
3. SQL: SELECT only, max_steps=8, HITL before execution
4. Cost guard: check per-tenant daily limit before every LLM call
```

---

## 6. UI Design System

### Design Identity

**Dark, minimal, data-forward.** Professional analytics tool, not a chatbot.

```
Background:  slate-950  #020817
Surface:     slate-900  #0f172a
Border:      slate-800  #1e293b
Primary:     indigo-500 #6366f1
Success:     emerald-500 #10b981
Warning:     amber-500  #f59e0b
Error:       rose-500   #f43f5e
Text:        slate-50   #f8fafc
Muted:       slate-400  #94a3b8
```

### Setup

```bash
npm create vite@latest frontend -- --template react-ts
cd frontend
npm install tailwindcss@latest @tailwindcss/vite
npm install framer-motion @tanstack/react-query zustand
npm install react-router-dom react-hook-form @hookform/resolvers zod
npm install recharts lucide-react clsx tailwind-merge axios
npm install @clerk/clerk-react
npm install -D vitest @testing-library/react @types/node
```

```typescript
// vite.config.ts
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'
import path from 'path'
export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: { alias: { '@': path.resolve(__dirname, './src') } },
  server: {
    port: 3000,
    proxy: { '/api': { target: 'http://localhost:8000', changeOrigin: true } },
  },
})
```

```css
/* src/index.css */
@import "tailwindcss";
@theme {
  --color-brand-500: #6366f1;
  --color-brand-600: #4f46e5;
  --font-sans: 'Inter', system-ui, sans-serif;
  --font-mono: 'JetBrains Mono', monospace;
}
body { @apply bg-slate-950 text-slate-50 font-sans antialiased; }
```

### Core Components

```tsx
// Card
const Card = ({ children, className }: CardProps) => (
  <div className={cn("bg-slate-900 border border-slate-800 rounded-xl p-6", className)}>
    {children}
  </div>
)

// StatusBadge
const STATUS = {
  ready:      { color: 'emerald', dot: true },
  processing: { color: 'amber',   dot: true,  pulse: true },
  error:      { color: 'rose',    dot: true },
  pending:    { color: 'slate',   dot: true },
}
const StatusBadge = ({ status }: { status: keyof typeof STATUS }) => {
  const s = STATUS[status]
  return (
    <span className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full
      text-xs font-medium bg-${s.color}-500/10 text-${s.color}-400
      border border-${s.color}-500/20`}>
      <span className={`w-1.5 h-1.5 rounded-full bg-${s.color}-400
        ${s.pulse ? 'animate-pulse' : ''}`} />
      {status.charAt(0).toUpperCase() + status.slice(1)}
    </span>
  )
}

// ExecutionStep (thinking UI)
const ExecutionStep = ({ label, status, ms, tokens }: StepProps) => (
  <div className="flex items-center gap-3 py-1.5 text-sm">
    <StepIcon status={status} />
    <span className={status === 'done' ? 'text-slate-300' : 'text-slate-500'}>
      {label}
    </span>
    {status === 'done' && (
      <span className="ml-auto text-xs text-slate-600">{ms}ms · {tokens}t</span>
    )}
    {status === 'running' && (
      <span className="ml-auto text-xs text-indigo-400 animate-pulse">Running…</span>
    )}
  </div>
)
```

### Page Routes

```
/              → Dashboard
/login         → Clerk hosted UI (or embedded)
/sources       → Source grid with status badges
/sources/:id   → Detail, preview, health check
/chat          → Unified chat (auto-routes SQL or RAG)
/chat/:id      → Conversation with execution timeline
/reports       → Report cards with schedule badges
/reports/new   → Create report wizard
/admin         → Metrics: latency, cost, quality, cache
```

---

## 7. Technology Stack

### No LangChain. Direct SDKs Only.

```
LangChain:  ❌ abstraction tax, breaks constantly, heavy
LangGraph:  ✅ state machine for agents (NOT LangChain)
litellm:    ✅ thin routing layer (one function call, all providers)
instructor: ✅ structured output via Pydantic (does one thing)
```

### Full Stack

**Frontend**
| Tech | Purpose |
|------|---------|
| React 18 + TypeScript | UI |
| Vite 5 | Build |
| Tailwind v4 | Styling |
| Framer Motion | Animations |
| React Query 5 | Server state |
| Zustand 4 | Client state |
| Clerk React SDK | Auth UI |
| Recharts | Charts |
| Lucide React | Icons |
| React Hook Form + Zod | Forms |

**Backend**
| Tech | Purpose |
|------|---------|
| FastAPI | API framework |
| SQLAlchemy 2 + Alembic | ORM + migrations |
| Pydantic v2 | Validation |
| Clerk Backend SDK | JWT validation |
| Celery + Redis/SQS | Workers + queue |
| litellm | Multi-provider LLM routing |
| instructor | Structured LLM output |
| groq / openai / anthropic SDK | Direct LLM calls |
| LangGraph + langgraph-checkpoint-postgres | Agent orchestration + state |
| PyMuPDF + pdfplumber | PDF extraction (fast path) |
| unstructured[pdf] | PDF extraction (OCR fallback) |
| httpx + BeautifulSoup4 | Web scraping |
| rank_bm25 | BM25 keyword retrieval |
| cohere | Reranking |
| DuckDB | SQL execution + multi-source federation |
| cryptography (Fernet) | Encrypt DB connection URLs |
| ragas | RAG evaluation |

**AI / ML**
| Tech | Purpose |
|------|---------|
| Groq (default) | Fast cheap LLM (llama-3.3-70b) |
| OpenAI | GPT-4o option |
| AWS Bedrock | Claude/Llama option |
| Ollama | Local models |
| text-embedding-3-small | Embeddings |
| BAAI/bge-m3 | Local embedding fallback |
| Cohere rerank-v3.5 | Cross-encoder reranking |
| Langfuse | LLM observability |
| Sentry | Error tracking |

**Infrastructure**
| Tech | Purpose |
|------|---------|
| Clerk | Auth (10K MAU free) |
| PostgreSQL 16 + pgvector | DB + vectors (MVP) |
| Qdrant | Vector DB with hybrid (Phase 5+) |
| Redis 7 | Cache + sessions + queue (local) |
| AWS SQS | Queue (production) |
| AWS S3 | File storage (per-tenant bucket) |
| AWS SES | Email delivery |
| AWS Lambda | Scheduled reports |
| AWS ECS Fargate | Hosting |
| AWS CloudFront | CDN |
| GitHub Actions | CI/CD |

---

## 8. Design Patterns

Every pattern here has a reason. We don't use patterns for academic correctness — we use them when they solve a real problem.

### Backend Patterns

---

#### 1. Repository Pattern — All DB Access Centralised

**Problem:** If every service writes its own SQL, a schema change breaks 20 files. Testing requires a real DB.

**Solution:** All database access goes through Repository classes. Services never touch the DB directly.

```python
# backend/repositories/source_repository.py
from typing import Protocol
from uuid import UUID

class SourceRepositoryProtocol(Protocol):
    """Interface — makes testing easy (swap with FakeSourceRepository in tests)"""
    async def get(self, source_id: UUID, org_id: str) -> Source: ...
    async def list(self, org_id: str, user_id: str, page: int, limit: int) -> list[Source]: ...
    async def create(self, data: CreateSourceDTO) -> Source: ...
    async def update(self, source_id: UUID, org_id: str, data: UpdateSourceDTO) -> Source: ...
    async def soft_delete(self, source_id: UUID, org_id: str) -> None: ...
    async def find_by_hash(self, content_hash: str, org_id: str) -> Source | None: ...

class SourceRepository:
    """Concrete implementation — all SQL lives here"""

    def __init__(self, db: AsyncSession):
        self._db = db

    async def get(self, source_id: UUID, org_id: str) -> Source:
        # org_id ALWAYS in WHERE — security invariant
        result = await self._db.execute(
            select(SourceModel)
            .where(SourceModel.id == source_id)
            .where(SourceModel.org_id == org_id)
            .where(SourceModel.deleted_at.is_(None))
        )
        source = result.scalar_one_or_none()
        if not source:
            raise SourceNotFoundError(source_id)
        return Source.model_validate(source)

    async def list(
        self, org_id: str, user_id: str, page: int = 1, limit: int = 20
    ) -> tuple[list[Source], int]:
        offset = (page - 1) * limit
        query  = (
            select(SourceModel)
            .where(SourceModel.org_id == org_id)
            .where(SourceModel.user_id == user_id)
            .where(SourceModel.deleted_at.is_(None))
            .order_by(SourceModel.created_at.desc())
        )
        total  = await self._db.scalar(select(func.count()).select_from(query.subquery()))
        result = await self._db.execute(query.offset(offset).limit(limit))
        return [Source.model_validate(r) for r in result.scalars().all()], total

# Service layer — never writes SQL
class DataService:
    def __init__(self, source_repo: SourceRepositoryProtocol):
        self._sources = source_repo  # injected

    async def get_source(self, source_id: UUID, org_id: str) -> Source:
        return await self._sources.get(source_id, org_id)

# Test — no real DB needed
class FakeSourceRepository:
    def __init__(self): self._store: dict[UUID, Source] = {}
    async def get(self, source_id: UUID, org_id: str) -> Source:
        s = self._store.get(source_id)
        if not s or s.org_id != org_id: raise SourceNotFoundError(source_id)
        return s
    async def create(self, data: CreateSourceDTO) -> Source:
        s = Source(id=uuid4(), **data.model_dump())
        self._store[s.id] = s
        return s
```

---

#### 2. Strategy Pattern — Swappable LLM Providers

**Problem:** If provider details are hardcoded throughout, switching from Groq to Bedrock requires touching 20 files.

**Solution:** Define an interface. All LLM calls go through it. Swap the implementation without touching callers.

```python
# backend/providers/llm_strategy.py
from abc import ABC, abstractmethod
from pydantic import BaseModel

class LLMResponse(BaseModel):
    content:      str
    input_tokens: int
    output_tokens:int
    cost_usd:     float
    model:        str

class LLMStrategy(ABC):
    """Interface — all providers implement this"""

    @abstractmethod
    async def complete(
        self,
        messages: list[dict],
        task: str,
        **kwargs,
    ) -> LLMResponse: ...

    @abstractmethod
    async def complete_structured(
        self,
        messages: list[dict],
        response_model: type[BaseModel],
        task: str,
        **kwargs,
    ) -> BaseModel: ...

    @abstractmethod
    async def embed(self, texts: list[str]) -> list[list[float]]: ...

class LiteLLMProxyStrategy(LLMStrategy):
    """
    Routes all calls through LiteLLM Proxy (Phase 12).
    Proxy handles: provider selection, fallback, tracing, rate limiting.
    """
    def __init__(self, base_url: str, api_key: str):
        import openai, instructor
        self._client            = openai.AsyncOpenAI(api_key=api_key, base_url=base_url)
        self._structured_client = instructor.from_openai(self._client)
        self._task_aliases      = {
            "sql_agent":      "fast-cheap",
            "rag_generation": "fast-cheap",
            "classification": "fast-small",
            "rewriting":      "fast-small",
            "compression":    "fast-small",
            "summarisation":  "fast-small",
            "synthesis":      "fast-cheap",
            "embedding":      "embedding",
        }

    async def complete(self, messages, task, **kwargs) -> LLMResponse:
        alias    = self._task_aliases.get(task, "fast-cheap")
        response = await self._client.chat.completions.create(
            model=alias, messages=messages, **kwargs
        )
        usage = response.usage
        return LLMResponse(
            content=      response.choices[0].message.content,
            input_tokens= usage.prompt_tokens,
            output_tokens=usage.completion_tokens,
            cost_usd=     self._estimate_cost(alias, usage),
            model=        response.model,
        )

    async def complete_structured(self, messages, response_model, task, **kwargs):
        alias = self._task_aliases.get(task, "fast-cheap")
        return await self._structured_client.chat.completions.create(
            model=alias, response_model=response_model,
            messages=messages, **kwargs,
        )

    async def embed(self, texts: list[str]) -> list[list[float]]:
        response = await self._client.embeddings.create(
            model="embedding", input=texts
        )
        return [d.embedding for d in sorted(response.data, key=lambda x: x.index)]

class DirectGroqStrategy(LLMStrategy):
    """
    Direct Groq SDK — used Phase 2-11 before proxy is set up.
    Same interface, different implementation.
    Swap by changing which strategy is injected at startup.
    """
    def __init__(self, api_key: str):
        from groq import AsyncGroq
        import instructor
        self._client = instructor.from_groq(AsyncGroq(api_key=api_key))

    async def complete(self, messages, task, **kwargs) -> LLMResponse:
        response = await self._client.chat.completions.create(
            model="llama-3.3-70b-versatile", messages=messages, **kwargs
        )
        ...

    async def complete_structured(self, messages, response_model, task, **kwargs):
        return await self._client.chat.completions.create(
            model=self._pick_model(task),
            response_model=response_model,
            messages=messages, **kwargs,
        )

# Factory — single place to decide which strategy to use
class LLMFactory:
    @staticmethod
    def create(settings) -> LLMStrategy:
        if settings.litellm_proxy_url:
            return LiteLLMProxyStrategy(
                base_url=settings.litellm_proxy_url,
                api_key=settings.litellm_master_key,
            )
        # Fallback: direct Groq (Phase 0-11)
        return DirectGroqStrategy(api_key=settings.groq_api_key)

# Dependency injection — services receive interface, not concrete class
async def get_llm(settings: Settings = Depends(get_settings)) -> LLMStrategy:
    return LLMFactory.create(settings)

# Service uses interface — doesn't know or care which provider
class SQLAgentService:
    def __init__(self, llm: LLMStrategy):  # interface injected
        self._llm = llm

    async def run(self, question: str, ...) -> SQLResponse:
        decision = await self._llm.complete_structured(
            messages=messages,
            response_model=AgentDecision,
            task="sql_agent",
        )
        ...
```

---

#### 3. Factory Pattern — Object Creation Centralised

**Problem:** If each service creates its own Qdrant client, DB connection, Redis client — inconsistent config, hard to mock in tests.

**Solution:** Factories create objects with correct config. Services receive already-configured objects.

```python
# backend/providers/vector_store_factory.py
from abc import ABC, abstractmethod

class VectorStoreStrategy(ABC):
    @abstractmethod
    async def search(self, query_vector, org_id, user_id, source_ids, k, **filters) -> list[ScoredChunk]: ...
    @abstractmethod
    async def upsert(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None: ...
    @abstractmethod
    async def delete_by_source(self, source_id: str, org_id: str) -> None: ...

class QdrantVectorStore(VectorStoreStrategy):
    """Production: Qdrant (Phase 5+)"""
    def __init__(self, client: AsyncQdrantClient, collection: str):
        self._client = client
        self._collection = collection
    ...

class PgVectorStore(VectorStoreStrategy):
    """MVP: pgvector (Phase 3-4)"""
    def __init__(self, pool: asyncpg.Pool):
        self._pool = pool
    ...

class VectorStoreFactory:
    @staticmethod
    async def create(settings: Settings) -> VectorStoreStrategy:
        if settings.use_qdrant:
            client = AsyncQdrantClient(url=settings.qdrant_url)
            await client.recreate_collection(...)
            return QdrantVectorStore(client, "chunks")
        # MVP fallback: pgvector
        pool = await asyncpg.create_pool(settings.database_url)
        return PgVectorStore(pool)
```

---

#### 4. Adapter Pattern — Normalise External APIs

**Problem:** Groq, OpenAI, Cohere all return data in slightly different formats. If each caller handles differences, it's scattered and fragile.

**Solution:** Adapters translate external formats to our internal types.

```python
# backend/providers/adapters.py

# Our internal type — what the rest of the app works with
class ChunkResult(BaseModel):
    content:       str
    score:         float
    source_name:   str
    page:          int
    original_rank: int | None = None
    reranked_rank: int | None = None

class CohereRerankAdapter:
    """Adapts Cohere rerank response → our ChunkResult format"""

    @staticmethod
    def adapt(
        cohere_result,           # raw Cohere SDK response
        original_chunks: list[Chunk],
    ) -> list[ChunkResult]:
        return [
            ChunkResult(
                content=       original_chunks[r.index].content,
                score=         r.relevance_score,
                source_name=   original_chunks[r.index].metadata["source_name"],
                page=          original_chunks[r.index].metadata.get("page", 0),
                original_rank= r.index,
                reranked_rank= i,
            )
            for i, r in enumerate(cohere_result.results)
        ]

class QdrantSearchAdapter:
    """Adapts Qdrant search response → our ChunkResult format"""

    @staticmethod
    def adapt(qdrant_results: list) -> list[ChunkResult]:
        return [
            ChunkResult(
                content=    r.payload["content"],
                score=      r.score,
                source_name=r.payload.get("source_name", ""),
                page=       r.payload.get("page", 0),
            )
            for r in qdrant_results
        ]

# Reranker service uses adapter — never touches raw Cohere format
class RerankerService:
    async def rerank(
        self, query: str, chunks: list[Chunk], top_n: int = 5
    ) -> list[ChunkResult]:
        raw_result = await self._cohere_client.rerank(
            query=query,
            documents=[c.content for c in chunks],
            model="rerank-v3.5",
            top_n=top_n,
        )
        # Adapter translates Cohere format → our format
        return CohereRerankAdapter.adapt(raw_result, chunks)
```

---

#### 5. Dependency Injection — FastAPI Style

**Problem:** If services create their own dependencies (DB connections, LLM clients), you can't swap them in tests and can't control lifecycle.

**Solution:** FastAPI's `Depends()` system injects configured dependencies. Services declare what they need, not how to create it.

```python
# backend/api/deps.py — all dependencies defined once

async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with AsyncSessionLocal() as session:
        yield session

async def get_redis() -> Redis:
    return await redis_pool.get_connection()

async def get_llm(settings: Settings = Depends(get_settings)) -> LLMStrategy:
    return LLMFactory.create(settings)

async def get_vector_store(settings: Settings = Depends(get_settings)) -> VectorStoreStrategy:
    return await VectorStoreFactory.create(settings)

async def get_current_user(request: Request) -> AuthUser:
    return await ClerkAuth.verify(request)

# Route uses Depends — receives ready-to-use objects
@router.post("/chat/sql")
async def sql_chat(
    body:         SQLChatRequest,
    current_user: AuthUser          = Depends(get_current_user),
    db:           AsyncSession      = Depends(get_db),
    llm:          LLMStrategy       = Depends(get_llm),
    source_repo:  SourceRepository  = Depends(get_source_repo),
):
    service = SQLAgentService(llm=llm, source_repo=source_repo)
    return await service.run(body.question, current_user.org_id)

# Test — inject fakes, no real services needed
async def test_sql_chat():
    fake_llm    = FakeLLMStrategy(response="SELECT * FROM orders LIMIT 10")
    fake_repo   = FakeSourceRepository()
    fake_repo.add(Source(id=uuid4(), org_id="test-org", ...))

    app.dependency_overrides[get_llm]         = lambda: fake_llm
    app.dependency_overrides[get_source_repo] = lambda: fake_repo

    response = await client.post("/api/chat/sql", json={...})
    assert response.status_code == 200
```

---

#### 6. Chain of Responsibility — Guardrails Pipeline

**Problem:** Multiple guardrail checks need to run in sequence. Each can stop the chain. Hardcoding the order is fragile.

**Solution:** Chain of Responsibility. Each handler either passes to the next or returns an error.

```python
# backend/services/guardrails/pipeline.py
from abc import ABC, abstractmethod

class GuardHandler(ABC):
    def __init__(self):
        self._next: GuardHandler | None = None

    def set_next(self, handler: GuardHandler) -> GuardHandler:
        self._next = handler
        return handler  # allows chaining: a.set_next(b).set_next(c)

    @abstractmethod
    async def handle(self, request: GuardRequest) -> GuardResult: ...

    async def _pass_to_next(self, request: GuardRequest) -> GuardResult:
        if self._next:
            return await self._next.handle(request)
        return GuardResult(allowed=True)  # end of chain — all passed

class InjectionGuard(GuardHandler):
    async def handle(self, request: GuardRequest) -> GuardResult:
        for pattern in INJECTION_PATTERNS:
            if re.search(pattern, request.user_input.lower()):
                return GuardResult(allowed=False, reason="Potential injection detected")
        return await self._pass_to_next(request)

class PIIMaskingGuard(GuardHandler):
    async def handle(self, request: GuardRequest) -> GuardResult:
        sanitised = request.user_input
        for pii_type, pattern in PII_PATTERNS.items():
            sanitised = re.sub(pattern, f"[{pii_type.upper()}_REDACTED]", sanitised)
        request.user_input = sanitised  # mutate request for downstream handlers
        return await self._pass_to_next(request)

class CostGuard(GuardHandler):
    async def handle(self, request: GuardRequest) -> GuardResult:
        result = await check_cost_limit(request.org_id, request.redis)
        if not result.allowed:
            return result
        return await self._pass_to_next(request)

class RateLimitGuard(GuardHandler):
    async def handle(self, request: GuardRequest) -> GuardResult:
        result = await check_query_rate_limit(
            request.user_id, request.org_id, request.query_type, request.redis
        )
        if not result.allowed:
            return result
        return await self._pass_to_next(request)

# Build the chain once at startup
def build_guardrail_pipeline() -> GuardHandler:
    injection  = InjectionGuard()
    pii        = PIIMaskingGuard()
    cost       = CostGuard()
    rate_limit = RateLimitGuard()

    injection.set_next(pii).set_next(cost).set_next(rate_limit)
    return injection  # entry point of chain

GUARDRAIL_PIPELINE = build_guardrail_pipeline()

# Usage in any endpoint — one call, all checks run in order
async def run_guardrails(user_input: str, org_id: str, user_id: str,
                          query_type: str, redis: Redis) -> GuardResult:
    request = GuardRequest(
        user_input=user_input, org_id=org_id,
        user_id=user_id, query_type=query_type, redis=redis,
    )
    return await GUARDRAIL_PIPELINE.handle(request)
```

---

#### 7. Observer Pattern — Execution Event Tracking

**Problem:** Every pipeline step needs to log timing, tokens, cost to `execution_events`. Sprinkling `await log_step(...)` everywhere is messy and easy to forget.

**Solution:** Observer pattern. Pipeline emits events. Observers (DB logger, Langfuse sender, SSE pusher) subscribe and handle independently.

```python
# backend/services/observability/event_bus.py
from dataclasses import dataclass, field
from datetime import datetime

@dataclass
class ExecutionEvent:
    query_id:    str
    org_id:      str
    user_id:     str
    step:        str
    status:      str   # started | done | error
    duration_ms: int   = 0
    tokens:      int   = 0
    cost_usd:    float = 0.0
    metadata:    dict  = field(default_factory=dict)
    timestamp:   datetime = field(default_factory=datetime.utcnow)

class EventObserver(ABC):
    @abstractmethod
    async def on_event(self, event: ExecutionEvent) -> None: ...

class DBEventObserver(EventObserver):
    """Writes to execution_events table"""
    async def on_event(self, event: ExecutionEvent) -> None:
        await ExecutionEventRepository.create(event)

class LangfuseEventObserver(EventObserver):
    """Sends span update to Langfuse"""
    async def on_event(self, event: ExecutionEvent) -> None:
        if event.status == "done":
            langfuse_context.update_current_observation(
                name=event.step,
                metadata={"tokens": event.tokens, "cost": event.cost_usd},
                end_time=event.timestamp,
            )

class EventBus:
    def __init__(self):
        self._observers: list[EventObserver] = []

    def subscribe(self, observer: EventObserver) -> None:
        self._observers.append(observer)

    async def emit(self, event: ExecutionEvent) -> None:
        await asyncio.gather(*[obs.on_event(event) for obs in self._observers])

# Singleton bus — set up at startup
event_bus = EventBus()
event_bus.subscribe(DBEventObserver())
event_bus.subscribe(LangfuseEventObserver())

# Context manager for clean step tracking
class track_step:
    def __init__(self, step: str, query_id: str, org_id: str, user_id: str):
        self.step      = step
        self.query_id  = query_id
        self.org_id    = org_id
        self.user_id   = user_id
        self.start_time = None

    async def __aenter__(self):
        self.start_time = datetime.utcnow()
        await event_bus.emit(ExecutionEvent(
            query_id=self.query_id, org_id=self.org_id,
            user_id=self.user_id, step=self.step, status="started",
        ))
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        duration = int((datetime.utcnow() - self.start_time).total_seconds() * 1000)
        await event_bus.emit(ExecutionEvent(
            query_id=self.query_id, org_id=self.org_id,
            user_id=self.user_id, step=self.step,
            status="error" if exc_type else "done",
            duration_ms=duration,
        ))

# Usage — clean, no manual logging
async def run_rag_pipeline(question: str, query_id: str, ...):
    async with track_step("query_rewriting", query_id, org_id, user_id):
        rewritten = await rewrite_query(question)

    async with track_step("hybrid_retrieval", query_id, org_id, user_id):
        chunks = await retriever.retrieve(rewritten, ...)

    async with track_step("reranking", query_id, org_id, user_id):
        reranked = await reranker.rerank(question, chunks)

    async with track_step("llm_generation", query_id, org_id, user_id):
        async for token in stream_generate(question, reranked):
            yield token
```

---

### Frontend Patterns

---

#### 8. Custom Hooks — Encapsulate Data Fetching Logic

```typescript
// hooks/useSources.ts
// All source-related state, fetching, mutations in one place
// Components just call the hook — no direct API calls

import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { sourcesApi } from '@/services/api'
import type { Source, CreateSourceDTO } from '@/types'

export function useSources() {
  const queryClient = useQueryClient()

  const sources = useQuery({
    queryKey:  ['sources'],
    queryFn:   sourcesApi.list,
    staleTime: 30_000,   // consider fresh for 30 seconds
  })

  const upload = useMutation({
    mutationFn: sourcesApi.upload,
    onSuccess: (newSource) => {
      // Optimistic update — appears in list immediately
      queryClient.setQueryData<Source[]>(['sources'], (old = []) => [newSource, ...old])
    },
    onError: (error) => {
      toast.error(`Upload failed: ${error.message}`)
    },
  })

  const deleteSource = useMutation({
    mutationFn: sourcesApi.delete,
    onMutate: async (sourceId) => {
      // Optimistic: remove from UI immediately, restore on error
      await queryClient.cancelQueries({ queryKey: ['sources'] })
      const previous = queryClient.getQueryData<Source[]>(['sources'])
      queryClient.setQueryData<Source[]>(['sources'],
        old => old?.filter(s => s.id !== sourceId) ?? []
      )
      return { previous }
    },
    onError: (_, __, context) => {
      queryClient.setQueryData(['sources'], context?.previous)
    },
  })

  return {
    sources:      sources.data ?? [],
    isLoading:    sources.isLoading,
    uploadSource: upload.mutate,
    isUploading:  upload.isPending,
    deleteSource: deleteSource.mutate,
  }
}

// Component is clean — zero data fetching logic
function SourcesPage() {
  const { sources, isLoading, uploadSource } = useSources()
  if (isLoading) return <SourcesSkeleton />
  return <SourceGrid sources={sources} onUpload={uploadSource} />
}
```

---

#### 9. Compound Components — Complex UI Split Cleanly

```typescript
// components/chat/ChatMessage.tsx
// Compound component — each part can be used independently
// or composed together

interface ChatMessageProps { message: Message }

function ChatMessage({ message }: ChatMessageProps) {
  return (
    <div className="flex flex-col gap-2">
      <ChatMessage.Bubble message={message} />
      {message.sources && <ChatMessage.Sources sources={message.sources} />}
      {message.sql      && <ChatMessage.SQLCard sql={message.sql} rows={message.rows} />}
      {message.steps    && <ChatMessage.ExecutionTimeline steps={message.steps} />}
      {message.role === 'assistant' && <ChatMessage.Feedback messageId={message.id} />}
    </div>
  )
}

ChatMessage.Bubble = function Bubble({ message }: { message: Message }) {
  return (
    <div className={cn("max-w-[75%] rounded-2xl px-4 py-3 text-sm",
      message.role === 'user'
        ? "bg-indigo-500 text-white self-end rounded-tr-sm"
        : "bg-slate-800 text-slate-200 self-start rounded-tl-sm border border-slate-700"
    )}>
      {message.content}
    </div>
  )
}

ChatMessage.Sources = function Sources({ sources }: { sources: RetrievedSource[] }) {
  const [open, setOpen] = useState(false)
  return (
    <div className="ml-4">
      <button onClick={() => setOpen(!open)}
              className="text-xs text-slate-500 hover:text-slate-300 flex items-center gap-1">
        <BookOpen size={12} />
        {sources.length} sources retrieved
        <ChevronDown size={12} className={cn("transition-transform", open && "rotate-180")} />
      </button>
      {open && (
        <div className="mt-2 space-y-2">
          {sources.map((s, i) => (
            <div key={i} className="text-xs text-slate-400 bg-slate-900 rounded-lg p-3 border border-slate-800">
              <span className="text-indigo-400 font-medium">[{s.source_name}, p{s.page}]</span>
              <span className="ml-2 text-slate-500">score: {s.score.toFixed(3)}</span>
              <p className="mt-1 text-slate-300">{s.content.slice(0, 150)}…</p>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

ChatMessage.ExecutionTimeline = function Timeline({ steps }: { steps: ExecutionStep[] }) {
  const [open, setOpen] = useState(false)
  const total  = steps.reduce((sum, s) => sum + (s.duration_ms ?? 0), 0)
  const tokens = steps.reduce((sum, s) => sum + (s.tokens ?? 0), 0)
  const cost   = steps.reduce((sum, s) => sum + (s.cost_usd ?? 0), 0)

  return (
    <div className="ml-4 border-l-2 border-slate-800 pl-3">
      <button onClick={() => setOpen(!open)}
              className="flex items-center gap-3 text-xs text-slate-500 hover:text-slate-300">
        <span>⏱ {(total/1000).toFixed(1)}s</span>
        <span>💰 ${cost.toFixed(4)}</span>
        <span>📊 {tokens} tokens</span>
        <ChevronDown size={12} className={cn("transition-transform", open && "rotate-180")} />
      </button>
      {open && (
        <div className="mt-2 space-y-1">
          {steps.map((step, i) => (
            <div key={i} className="flex items-center gap-2 text-xs">
              <StepStatusIcon status={step.status} />
              <span className="text-slate-400 w-40">{step.label}</span>
              <span className="text-slate-600">{step.duration_ms}ms</span>
              {step.tokens > 0 && <span className="text-slate-700">· {step.tokens}t</span>}
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

ChatMessage.Feedback = function Feedback({ messageId }: { messageId: string }) {
  const [rating, setRating] = useState<1|-1|null>(null)
  const { submitFeedback }  = useFeedback()

  const handleRate = (r: 1|-1) => {
    setRating(r)
    submitFeedback({ query_id: messageId, rating: r })
  }

  return (
    <div className="ml-4 flex items-center gap-2">
      <button onClick={() => handleRate(1)}
              className={cn("p-1 rounded", rating === 1 ? "text-emerald-400" : "text-slate-600 hover:text-slate-400")}>
        <ThumbsUp size={14} />
      </button>
      <button onClick={() => handleRate(-1)}
              className={cn("p-1 rounded", rating === -1 ? "text-rose-400" : "text-slate-600 hover:text-slate-400")}>
        <ThumbsDown size={14} />
      </button>
    </div>
  )
}
```

---

### Design Patterns Summary

| Pattern | Where Used | Problem It Solves |
|---------|-----------|-------------------|
| **Repository** | All DB access | Schema changes → 1 file, not 20. Easy to mock in tests. |
| **Strategy** | LLM providers, Vector stores | Swap Groq → Bedrock without touching callers |
| **Factory** | LLM, Vector store, Guardrail pipeline | Centralised config, consistent object creation |
| **Adapter** | Cohere, Qdrant, OpenAI responses | External format changes → 1 adapter file, not N callers |
| **Dependency Injection** | FastAPI routes | Services testable without real DB/LLM/Redis |
| **Chain of Responsibility** | Guardrails | Add/remove/reorder checks without changing callers |
| **Observer** | Execution event tracking | Decouple pipeline steps from logging/tracing/SSE |
| **Compound Components** | ChatMessage, SourceCard | Complex UI broken into independently usable parts |
| **Custom Hooks** | useSources, useChat | Data fetching logic isolated from render logic |

### What We Deliberately Do NOT Use

```
Singleton for everything:
  Tempting but makes testing hard. Use DI instead.

God classes (one class that does everything):
  SQLAgentService should NOT also handle embeddings.
  Each class has one job.

Premature abstraction:
  Don't create an interface until you have 2+ implementations.
  ISourceRepository with only one SourceRepository → unnecessary.

Magic strings:
  "sql_agent" task name → MODEL_ALIASES dict, not scattered strings.
  Status values → Enum classes, not raw strings.
```

```
┌─────────────────────────────────────────────────────────────────────┐
│                    FRONTEND  React + Vite                           │
│  Chat · Sources · Dashboard · Reports · Execution Timeline          │
└──────────────────────────────┬──────────────────────────────────────┘
                               │ HTTPS / SSE
                               ▼
                      AWS CloudFront + ALB + WAF
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       FASTAPI  (ECS Fargate)                        │
│                                                                     │
│   Clerk JWT middleware → validates token → extracts org_id+user_id  │
│                                                                     │
│   /api/auth    /api/data    /api/chat    /api/reports               │
│                                                                     │
│   ┌─────────────────────────────────────────────────────────────┐  │
│   │           LANGGRAPH AGENTS                                   │  │
│   │  SQL Agent (Phase 2)    Planner (Phase 7)                   │  │
│   │  Postgres checkpointer   Subgraph composition               │  │
│   │  HITL interrupt_before   Parallel execution                 │  │
│   └─────────────────────────────────────────────────────────────┘  │
│                                                                     │
│   ┌─────────────────────────────────────────────────────────────┐  │
│   │           RAG PIPELINE (Phase 3-5)                           │  │
│   │  Hybrid retrieval · Reranking · Context budget              │  │
│   └─────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
          │                    │                    │
┌─────────▼────────────────────▼────────────────────▼─────────────────┐
│                         DATA LAYER                                   │
│  Postgres+pgvector  Qdrant  Redis  DuckDB  S3  SQS                  │
│  [all scoped by org_id + user_id on every query]                    │
└─────────────────────────────────────────────────────────────────────┘
                               │
              ┌────────────────▼──────────────────────┐
              │           LLM GATEWAY                 │
              │  litellm → Groq · OpenAI · Bedrock    │
              │  Langfuse traces every call           │
              └───────────────────────────────────────┘

WORKERS (Celery — separate ECS Fargate service):
  extract → chunk → dedup → embed(batch 100) → store → notify
  Queues: ingestion | eval | reports
  Local broker: Redis  |  Prod broker: SQS

MONITORING STACK:
  Langfuse   → LLM calls (prompts, tokens, cost, latency, quality)
  CloudWatch → Infrastructure (ECS, RDS, SQS, Redis)
  Sentry     → Exceptions (FastAPI + React)
  Flower     → Celery workers (localhost:5555)
```

---

## 9. Feature Coverage Map

| Feature | Phase | Tier |
|---------|-------|------|
| Clerk auth + Google OAuth | 1 | Core |
| org_id + user_id multi-tenancy | 1 | Core |
| CSV/Excel → NL SQL query | 2 | Core |
| Multi-source SQL (DuckDB federation) | 2 | Core |
| LangGraph SQL Agent (ReAct, HITL) | 2 | Core |
| Langfuse tracing (from Phase 2) | 2 | Core |
| Input guardrails (injection + PII) | 2 | Core |
| PDF/TXT/URL → basic RAG | 3 | Core |
| Tiered PDF extraction (PyMuPDF → Unstructured) | 3 | Core |
| Metadata-rich chunks (hierarchy foundation) | 3 | Core |
| Live DB connection URL (encrypted) | 4 | Core |
| doc_type auto-classification | 4 | Core |
| Hash-based deduplication | 4 | Core |
| Migrate to Qdrant | 5A | Retrieval |
| Hybrid search (BM25 + Qdrant + RRF) | 5A | Retrieval |
| Reranking (Cohere cross-encoder) | 5A | Retrieval |
| Context budget (max_chunks=5, max_tokens=6000) | 5A | Retrieval |
| Domain routing (doc_type filter) | 5A | Retrieval |
| Query rewriting | 5B | Retrieval |
| Context compression | 5B | Retrieval |
| Multi-query retrieval | 5B | Retrieval |
| RAGAS evaluation (background) | 6 | Quality |
| RAGAS CI/CD eval gate | 6 | Quality |
| Retrieval metrics (Recall@5, Precision@5) | 6 | Quality |
| Thumbs up/down feedback | 6 | Quality |
| Cost guardrails (per-tenant daily limit) | 6 | Quality |
| LangGraph Planner (SQL + RAG + hybrid) | 7 | Agent |
| Episodic memory (searchable query history) | 7 | Agent |
| Semantic memory (user preferences) | 7 | Agent |
| Data Health Agent | 7 | Agent |
| Execution timeline UI | 8 | Observability |
| Admin metrics dashboard | 8 | Observability |
| Sentry error tracking | 8 | Observability |
| CloudWatch alarms | 8 | Observability |
| Ingestion queue (Celery chain + batch embed) | 9 | Performance |
| Semantic + exact + SQL cache | 9 | Performance |
| Flower worker monitoring | 9 | Performance |
| PDF reports + scheduling | 10 | Reports |
| AWS Lambda + EventBridge triggers | 10 | Reports |
| SES email delivery | 10 | Reports |
| Self-correction RAG (CRAG) | 11 | Advanced |
| Source versioning + rollback | 11 | Advanced |
| Rate limiting per tenant (Redis counter) | 11 | Advanced |
| Conversation context window management | 11 | Advanced |
| Multi-file upload | 11 | Advanced |
| Production deployment (ECS + CI/CD) | 12 | Production |

---

## 10. Memory Strategies

Four types of memory — each serving a different purpose.

### Type 1 — In-Context Memory (Conversation History)

```
Problem: "Now filter for Q3 only" — what is "now"?
         Without history: system has no context.

Implementation: Sliding window + summarisation
  Keep last 5 messages in full
  Summarise messages 6+ into one paragraph
  Include summary + recent messages in every LLM call
  Saves 90% tokens on long conversations

LangGraph handles this automatically via Postgres checkpointer.
State persists across requests, server restarts, browser closes.
```

### Type 2 — External Memory (RAG + SQL)

```
This IS the system.
Documents → Qdrant (semantic knowledge)
Structured data → DuckDB (factual knowledge)
Nothing extra to build here.
```

### Type 3 — Episodic Memory (Query History)

```
Problem: "Run that churn analysis from last month again"
         System has no idea what query they mean.

Implementation:
  Store embedding of every query in vector store
  Tag: user_id + org_id + query_type + created_at
  
  When user asks → check if similar past query exists (>0.92)
  If found → surface it: "Did you mean this analysis?"
  User clicks → re-runs exact same query
  
  Also enables: "Show my recent analyses" as a feature
```

```python
# backend/services/memory/episodic_memory.py
class EpisodicMemory:
    async def find_similar_past_query(
        self, query: str, user_id: str, org_id: str
    ) -> PastQuery | None:
        embedding = await embed(query)
        results = await qdrant.search(
            collection="query_history",
            query_vector=embedding,
            query_filter=Filter(must=[
                FieldCondition(key="user_id", match=MatchValue(value=user_id)),
                FieldCondition(key="org_id",  match=MatchValue(value=org_id)),
            ]),
            limit=1,
        )
        if results and results[0].score > 0.92:
            return PastQuery(**results[0].payload)
        return None
```

### Type 4 — Semantic Memory (User Preferences)

```
Problem: System treats every user as a stranger every time.
         Manager always filters by "South India warehouse."
         System makes them specify it every query.

Implementation: User preference profile in Redis
  Learn from each interaction:
    Which chart type they use most → preferred_chart
    Which filters they apply most → common_filters
    Their role → inferred from query patterns
  
  Inject into agent system prompt:
    "User context: prefers bar charts, commonly filters
     by region=South India, role=warehouse manager"
  
  TTL: 30 days (refreshed on each interaction)
```

```python
# backend/services/memory/user_memory.py
class UserMemory:
    async def get_context(self, user_id: str) -> str:
        profile = json.loads(await redis.get(f"profile:{user_id}") or "{}")
        parts = []
        if profile.get("preferred_chart"):
            parts.append(f"Prefers {profile['preferred_chart']} charts")
        if profile.get("common_filters"):
            parts.append(f"Commonly filters by: {profile['common_filters']}")
        return "User context: " + ". ".join(parts) if parts else ""

    async def update(self, user_id: str, chart_type: str, filters: dict):
        profile = json.loads(await redis.get(f"profile:{user_id}") or "{}")
        # Update rolling counts → derive preferences
        await redis.set(f"profile:{user_id}", json.dumps(profile), ex=86400*30)
```

| Memory Type | Storage | Phase | Benefit |
|---|---|---|---|
| In-context (conversation) | LangGraph + Postgres | 2 | Follow-up questions work |
| External (RAG + SQL) | Qdrant + DuckDB | 2-3 | Core knowledge retrieval |
| Episodic (query history) | Qdrant (separate collection) | 7 | "Run that again" works |
| Semantic (user preferences) | Redis (JSON per user) | 7 | Personalised responses |

---

## 11. Guardrails

Four layers — applied in order on every request.

### Layer 1 — Input Guardrails (before LLM)

```python
# backend/services/guardrails/input_guard.py
import re
from pydantic import BaseModel

INJECTION_PATTERNS = [
    r"ignore (all |previous |prior )?instructions",
    r"you are now",
    r"forget (everything|all|your instructions)",
    r"jailbreak",
    r"DAN mode",
    r"pretend (you|to be)",
    r"act as (if|a|an)",
]

PII_PATTERNS = {
    "email":       r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
    "phone":       r"\b(\+91|0)?[6-9]\d{9}\b",
    "credit_card": r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b",
    "aadhar":      r"\b\d{4}\s?\d{4}\s?\d{4}\b",
    "pan":         r"\b[A-Z]{5}[0-9]{4}[A-Z]\b",
}

class GuardResult(BaseModel):
    allowed:   bool
    reason:    str | None = None
    sanitised: str | None = None

def check_input(text: str) -> GuardResult:
    lower = text.lower()
    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, lower):
            return GuardResult(
                allowed=False,
                reason="Input contains potentially harmful instructions"
            )
    sanitised = text
    for pii_type, pattern in PII_PATTERNS.items():
        sanitised = re.sub(pattern, f"[{pii_type.upper()}_REDACTED]", sanitised)
    return GuardResult(allowed=True, sanitised=sanitised)
```

### Layer 2 — SQL Guardrails

```python
# backend/services/guardrails/sql_guard.py
BLOCKED_KEYWORDS = [
    "DROP","DELETE","INSERT","UPDATE","TRUNCATE",
    "ALTER","CREATE","EXEC","EXECUTE","GRANT","REVOKE"
]

class SQLGuard:
    MAX_STEPS = 8
    MAX_RESULT_ROWS = 10_000
    QUERY_TIMEOUT_SECONDS = 30

    @staticmethod
    def check_sql(query: str) -> GuardResult:
        q = query.strip().upper()
        if not q.startswith("SELECT"):
            return GuardResult(allowed=False, reason="Only SELECT queries allowed")
        for kw in BLOCKED_KEYWORDS:
            if re.search(rf'\b{kw}\b', q):
                return GuardResult(allowed=False, reason=f"Keyword '{kw}' not permitted")
        return GuardResult(allowed=True)
```

### Layer 3 — Cost Guardrails

```python
# backend/services/guardrails/cost_guard.py
async def check_cost_limit(org_id: str, redis: Redis) -> GuardResult:
    today = datetime.utcnow().strftime("%Y-%m-%d")
    key   = f"daily_cost:{org_id}:{today}"
    spent = float(await redis.get(key) or 0)
    limit = await get_tenant_daily_limit(org_id)  # from DB

    if spent >= limit:
        return GuardResult(
            allowed=False,
            reason=f"Daily limit of ${limit:.2f} reached. Resets at midnight UTC."
        )
    return GuardResult(allowed=True)

async def record_cost(org_id: str, cost_usd: float, redis: Redis):
    today = datetime.utcnow().strftime("%Y-%m-%d")
    key   = f"daily_cost:{org_id}:{today}"
    await redis.incrbyfloat(key, cost_usd)
    await redis.expireat(key, end_of_day_timestamp())
```

### Layer 4 — Output Guardrails

```python
# backend/services/guardrails/output_guard.py
async def check_output(text: str) -> GuardResult:
    # PII detection on output (don't return customer PII in answer)
    for pii_type, pattern in PII_PATTERNS.items():
        if re.search(pattern, text):
            text = re.sub(pattern, f"[{pii_type.upper()}_REDACTED]", text)
    # Faithfulness checked async via RAGAS (Phase 6)
    return GuardResult(allowed=True, sanitised=text)
```

### Guardrail Application Order

```
Every request:
  1. check_input(user_message)          → injection + PII mask
  2. check_cost_limit(org_id)           → daily spend limit
  [LLM call happens here]
  3. check_output(llm_response)         → PII mask output
  4. record_cost(org_id, actual_cost)   → update Redis counter

SQL only:
  Between steps 1 and 2:
    check_sql(generated_sql)            → SELECT only + blocked keywords
    HITL interrupt (LangGraph)          → manager approves before execute
```

---

## 12. Phase-by-Phase Build Plan

> **How to use with Claude Code:**
> Start each phase: `/pm` → stories → `/architect` → spec → `/sm` → daily tasks
> Each task: `/dev` → implement → `/qa` → tests → fix failures
> End phase: `/compact` → commit → `graphify update .` → update CLAUDE.md Current Phase

---

### Phase 0 — Foundation
**Duration:** 4–5 days | **Model:** Sonnet 4

#### Story
```
Why needed:
  Every subsequent phase builds on this foundation.
  Getting infrastructure right once is cheaper than
  fixing it across 12 phases.

Client need:
  Not directly visible to clients — but if this is wrong,
  every client-facing feature is wrong.

Approach chosen:
  Docker Compose for local dev — single command to start everything.
  pgvector/pgvector:pg16 image — Postgres with vector extension pre-built.
  Qdrant in Docker — same service locally as in production (no switching cost).
  Langfuse in Docker — LLM observability from day one, not retrofitted.
  Flower in Docker — Celery monitoring from day one.
  Redis with AOF persistence — queue durability even in local dev.

  Ruff (Python) + ESLint + Prettier (TypeScript) — code style enforced
  from commit 1, not "we'll add linting later" (we never do).

  Husky + lint-staged — every commit is auto-formatted and lint-checked.
  Formatters run on staged files only (fast). Bad commits blocked.
  commitlint enforces conventional commits — readable git history.

  Centralised JSON logging with request_id and correlation_id — every
  log line is queryable in CloudWatch. Celery worker logs carry the
  same correlation_id as the API request that triggered them. No more
  "which log line belongs to which request?" debugging.

Benefit:
  Any developer can clone the repo and be running in 5 minutes.
  No "works on my machine" issues.
  Production services mirrored locally → no surprises in deployment.
  Code style debates eliminated — formatter is the arbiter.
  Every commit is guaranteed clean — no "fix formatting" commits.
  Every production incident has traceable logs from API → worker → LLM.
```

#### Tasks

```bash
# Project + BMAD + Graphify
mkdir synapseiq && cd synapseiq && git init
npx bmad-method install --yes --modules bmm --tools claude-code
pip install graphifyy && graphify install

# Backend deps
python -m venv .venv && source .venv/bin/activate
pip install fastapi "uvicorn[standard]" sqlalchemy alembic psycopg2-binary \
    redis celery pydantic-settings langgraph \
    "langgraph-checkpoint-postgres" litellm instructor \
    groq openai anthropic cohere clerk-backend-api \
    pymupdf pdfplumber "unstructured[pdf]" httpx beautifulsoup4 \
    "python-multipart" boto3 duckdb rank_bm25 qdrant-client \
    cryptography ragas langfuse sentry-sdk pytest pytest-asyncio

# Frontend deps
npm create vite@latest frontend -- --template react-ts && cd frontend
npm install tailwindcss@latest @tailwindcss/vite framer-motion \
    @tanstack/react-query zustand react-router-dom \
    react-hook-form @hookform/resolvers zod \
    recharts lucide-react clsx tailwind-merge axios \
    @clerk/clerk-react
npm install -D vitest @testing-library/react @types/node
```

```yaml
# docker-compose.yml
services:
  postgres:
    image: pgvector/pgvector:pg16
    environment:
      POSTGRES_DB: synapseiq
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: postgres
    ports: ["5432:5432"]
    volumes: ["postgres_data:/var/lib/postgresql/data"]
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U postgres"]
      interval: 5s
      retries: 5

  qdrant:
    image: qdrant/qdrant:latest
    ports: ["6333:6333", "6334:6334"]
    volumes: ["qdrant_data:/qdrant/storage"]

  redis:
    image: redis:7-alpine
    ports: ["6379:6379"]
    command: >
      redis-server
      --maxmemory 256mb
      --maxmemory-policy allkeys-lru
      --appendonly yes
      --appendfsync everysec

  backend:
    build: ./backend
    ports: ["8000:8000"]
    env_file: .env
    depends_on:
      postgres: { condition: service_healthy }
      redis:    { condition: service_started }
    volumes: ["./backend:/app"]
    command: uvicorn main:app --reload --host 0.0.0.0 --port 8000

  worker:
    build: ./backend
    env_file: .env
    depends_on: [postgres, redis]
    volumes: ["./backend:/app"]
    command: celery -A core.celery_app worker --loglevel=info -Q ingestion,eval,reports -c 4

  flower:
    build: ./backend
    ports: ["5555:5555"]
    env_file: .env
    depends_on: [redis]
    command: celery -A core.celery_app flower --port=5555

  langfuse-db:
    image: postgres:16-alpine
    environment:
      POSTGRES_DB: langfuse
      POSTGRES_USER: langfuse
      POSTGRES_PASSWORD: langfuse
    volumes: ["langfuse_db:/var/lib/postgresql/data"]

  langfuse:
    image: langfuse/langfuse:latest
    ports: ["3001:3000"]
    environment:
      DATABASE_URL: postgresql://langfuse:langfuse@langfuse-db:5432/langfuse
      NEXTAUTH_SECRET: dev-secret-change-in-prod
      NEXTAUTH_URL: http://localhost:3001
    depends_on: [langfuse-db]

  frontend:
    build: ./frontend
    ports: ["3000:3000"]
    depends_on: [backend]

volumes:
  postgres_data:
  qdrant_data:
  langfuse_db:
```

---

#### Day 3 — Formatting, Linting + Pre-commit Hooks

##### Backend — Ruff + pyproject.toml

```bash
pip install ruff --dev
```

```toml
# pyproject.toml — single config file for all Python tooling
[tool.ruff]
target-version = "py311"
line-length    = 100
src            = ["backend"]

[tool.ruff.lint]
select = [
    "E",    # pycodestyle errors
    "W",    # pycodestyle warnings
    "F",    # pyflakes (unused imports, undefined names)
    "I",    # isort (import ordering)
    "B",    # flake8-bugbear (likely bugs)
    "C4",   # flake8-comprehensions
    "UP",   # pyupgrade (use modern Python syntax)
    "N",    # pep8-naming
    "SIM",  # flake8-simplify
    "TCH",  # flake8-type-checking (TYPE_CHECKING guard)
]
ignore = [
    "E501",  # line length (ruff formatter handles this)
    "B008",  # do not perform function calls in default args (FastAPI Depends)
]

[tool.ruff.lint.isort]
known-first-party = ["core", "api", "services", "repositories", "providers", "workers"]
force-sort-within-sections = true

[tool.ruff.format]
quote-style              = "double"
indent-style             = "space"
skip-magic-trailing-comma = false

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths    = ["tests"]
```

```bash
# Usage
ruff check backend/             # lint check
ruff check backend/ --fix       # auto-fix safe issues
ruff format backend/            # format (replaces black)
ruff format backend/ --check    # format check only (for CI)
```

---

##### Frontend — ESLint + Prettier

```bash
cd frontend
npm install -D \
    eslint \
    @typescript-eslint/eslint-plugin \
    @typescript-eslint/parser \
    eslint-plugin-react \
    eslint-plugin-react-hooks \
    eslint-plugin-react-refresh \
    eslint-plugin-import \
    eslint-import-resolver-typescript \
    prettier \
    eslint-config-prettier \
    eslint-plugin-prettier
```

```json
// frontend/.eslintrc.json
{
  "root": true,
  "env": { "browser": true, "es2022": true },
  "parser": "@typescript-eslint/parser",
  "parserOptions": {
    "ecmaVersion": "latest",
    "sourceType": "module",
    "project": "./tsconfig.json"
  },
  "plugins": [
    "@typescript-eslint",
    "react",
    "react-hooks",
    "react-refresh",
    "import",
    "prettier"
  ],
  "extends": [
    "eslint:recommended",
    "plugin:@typescript-eslint/recommended-type-checked",
    "plugin:react/recommended",
    "plugin:react-hooks/recommended",
    "plugin:import/recommended",
    "plugin:import/typescript",
    "prettier"
  ],
  "rules": {
    // TypeScript
    "@typescript-eslint/no-explicit-any":          "error",
    "@typescript-eslint/no-unused-vars":           ["error", { "argsIgnorePattern": "^_" }],
    "@typescript-eslint/consistent-type-imports":  ["error", { "prefer": "type-imports" }],
    "@typescript-eslint/no-floating-promises":     "error",
    "@typescript-eslint/no-misused-promises":      "error",

    // React
    "react/react-in-jsx-scope":   "off",        // not needed in React 18
    "react/prop-types":           "off",        // we use TypeScript
    "react-hooks/rules-of-hooks": "error",
    "react-hooks/exhaustive-deps":"warn",
    "react-refresh/only-export-components": "warn",

    // Imports
    "import/order": ["error", {
      "groups": [
        "builtin", "external", "internal", "parent", "sibling", "index"
      ],
      "newlines-between": "always",
      "alphabetize": { "order": "asc" }
    }],
    "import/no-duplicates": "error",

    // Prettier (surfaces format issues as ESLint errors)
    "prettier/prettier": "error"
  },
  "settings": {
    "react": { "version": "detect" },
    "import/resolver": {
      "typescript": { "alwaysTryTypes": true, "project": "./tsconfig.json" }
    }
  }
}
```

```json
// frontend/.prettierrc
{
  "semi":          false,
  "singleQuote":   true,
  "trailingComma": "all",
  "printWidth":    100,
  "tabWidth":      2,
  "bracketSpacing":true,
  "arrowParens":   "always",
  "endOfLine":     "lf",
  "plugins":       ["prettier-plugin-tailwindcss"]
}
```

```bash
# Install Tailwind Prettier plugin (sorts class names automatically)
npm install -D prettier-plugin-tailwindcss
```

```json
// frontend/package.json — add scripts
{
  "scripts": {
    "dev":          "vite",
    "build":        "tsc && vite build",
    "lint":         "eslint src --ext .ts,.tsx --report-unused-disable-directives",
    "lint:fix":     "eslint src --ext .ts,.tsx --fix",
    "format":       "prettier --write src/",
    "format:check": "prettier --check src/",
    "type-check":   "tsc --noEmit",
    "test":         "vitest run",
    "test:watch":   "vitest"
  }
}
```

---

##### Pre-commit Hooks — lint-staged + Husky

```
Why Husky + lint-staged over pre-commit:
  Husky:       manages git hooks for the Node.js ecosystem
  lint-staged: runs formatters/linters ONLY on staged files
               (not the whole codebase — fast even in large repos)
  pre-commit:  Python-based, good but heavier for mixed JS/Python repos

Together: on every git commit →
  Backend staged .py files → ruff format + ruff check --fix
  Frontend staged .ts/.tsx files → prettier --write + eslint --fix
  If any check fails → commit blocked
  If formatters changed files → changes auto-staged and commit continues
```

```bash
# Install at project root (not inside frontend/)
npm init -y                        # creates root package.json
npm install -D husky lint-staged

# Initialise Husky
npx husky init
# Creates .husky/ directory and sets up git hooks automatically
```

```bash
# .husky/pre-commit
#!/usr/bin/env sh
. "$(dirname -- "$0")/_/husky.sh"

echo "🔍 Running pre-commit checks..."
npx lint-staged
```

```bash
# .husky/commit-msg  (conventional commit format enforcement)
#!/usr/bin/env sh
. "$(dirname -- "$0")/_/husky.sh"

npx --no -- commitlint --edit "$1"
```

```bash
# Install commitlint
npm install -D @commitlint/cli @commitlint/config-conventional
```

```javascript
// commitlint.config.js
module.exports = {
  extends: ['@commitlint/config-conventional'],
  rules: {
    'type-enum': [2, 'always', [
      'feat',     // new feature
      'fix',      // bug fix
      'chore',    // maintenance
      'docs',     // documentation
      'style',    // formatting, no logic change
      'refactor', // code change, no feature/fix
      'test',     // adding tests
      'perf',     // performance improvement
      'ci',       // CI/CD changes
      'revert',   // revert commit
    ]],
    'subject-case': [2, 'always', 'lower-case'],
    'subject-max-length': [2, 'always', 72],
  },
}
```

```json
// root package.json — lint-staged config
{
  "scripts": {
    "prepare": "husky"
  },
  "lint-staged": {
    "backend/**/*.py": [
      "ruff format",
      "ruff check --fix --unsafe-fixes"
    ],
    "frontend/src/**/*.{ts,tsx}": [
      "prettier --write",
      "eslint --fix"
    ],
    "frontend/src/**/*.{json,css,md}": [
      "prettier --write"
    ],
    "*.{json,yaml,yml,md}": [
      "prettier --write"
    ]
  }
}
```

```
How it works on git commit:

  git add backend/services/rag/pipeline.py
  git add frontend/src/components/chat/ChatMessage.tsx
  git commit -m "feat: add hybrid retrieval to RAG pipeline"

  Husky fires pre-commit hook →
    lint-staged runs on ONLY staged files:
      pipeline.py    → ruff format (auto-formats) → ruff check --fix
      ChatMessage.tsx → prettier --write (auto-formats) → eslint --fix

    commit-msg hook →
      commitlint checks: "feat: add hybrid retrieval..." → ✅ valid format

  If any file fails lint after auto-fix → commit blocked with clear error
  If files were modified by formatter → auto-restaged → commit continues

  Result: every commit in the repo is guaranteed formatted + lint-clean
```

```bash
# Verify hooks are active after setup
cat .git/hooks/pre-commit     # should show husky script
cat .git/hooks/commit-msg     # should show husky script

# Test it manually
git add .
git commit -m "bad commit message"   # → blocked by commitlint
git commit -m "feat: proper message" # → passes, lint-staged runs
```

```bash
# Skip hooks temporarily (for WIP commits — use sparingly)
git commit -m "wip: debugging" --no-verify
```

---

#### Day 4 — Centralised Logging

##### Why Centralised Logging Matters

```
Without it (print statements, ad-hoc logging):
  print(f"User {user_id} asked: {question}")
  logger.info("ingestion done")
  print("ERROR:", e)

  Problems:
  - No structure → can't filter by org_id in CloudWatch
  - No request_id → can't trace one request across multiple log lines
  - No correlation_id → can't link a Celery worker log to the API log that triggered it
  - Different format in every file → impossible to parse programmatically
  - No log level control per environment (debug logs in prod)

With centralised structured logging:
  Every log line is JSON → CloudWatch Logs Insights can query it
  Every request has request_id → filter all logs for one request
  Every log has org_id, user_id → filter logs by tenant
  Correlation IDs link API → Celery worker → Langfuse trace
```

##### Backend — Structured JSON Logger

```python
# backend/core/logging.py
import logging
import sys
import json
from datetime import datetime, timezone
from typing import Any
from contextvars import ContextVar

# Context variables — automatically propagated through async calls
request_id_ctx:      ContextVar[str] = ContextVar("request_id",      default="")
org_id_ctx:          ContextVar[str] = ContextVar("org_id",           default="")
user_id_ctx:         ContextVar[str] = ContextVar("user_id",          default="")
correlation_id_ctx:  ContextVar[str] = ContextVar("correlation_id",   default="")

class JSONFormatter(logging.Formatter):
    """
    Formats every log record as a single JSON line.
    CloudWatch Logs Insights can query any field directly.

    Example output:
    {
      "ts":             "2026-05-25T10:30:00.123Z",
      "level":          "INFO",
      "logger":         "synapseiq.rag.pipeline",
      "message":        "Hybrid retrieval complete",
      "request_id":     "req-abc-123",
      "org_id":         "org_abc",
      "user_id":        "user_xyz",
      "correlation_id": "corr-789",
      "duration_ms":    145,
      "chunk_count":    20
    }
    """

    def format(self, record: logging.LogRecord) -> str:
        log_entry: dict[str, Any] = {
            "ts":             datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "level":          record.levelname,
            "logger":         record.name,
            "message":        record.getMessage(),
            "request_id":     request_id_ctx.get(""),
            "org_id":         org_id_ctx.get(""),
            "user_id":        user_id_ctx.get(""),
            "correlation_id": correlation_id_ctx.get(""),
        }

        # Attach any extra fields passed to the log call
        # e.g. logger.info("done", extra={"duration_ms": 145, "chunk_count": 20})
        if hasattr(record, "extra_fields"):
            log_entry.update(record.extra_fields)

        # Attach exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_entry, default=str)


def setup_logging(level: str = "INFO") -> None:
    """
    Call once at application startup.
    Replaces all handlers on the root logger with our JSON formatter.
    """
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JSONFormatter())

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Silence noisy third-party loggers
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("groq").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """
    Get a named logger. Use module __name__ as the name.

    Usage:
        from core.logging import get_logger
        logger = get_logger(__name__)
        logger.info("Retrieval complete", extra={"chunk_count": 5})
    """
    return logging.getLogger(name)
```

##### Request ID Middleware

```python
# backend/api/middleware/logging_middleware.py
import time
import uuid
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from core.logging import (
    get_logger, request_id_ctx,
    org_id_ctx, user_id_ctx, correlation_id_ctx,
)

logger = get_logger(__name__)

class LoggingMiddleware(BaseHTTPMiddleware):
    """
    Runs on every request.
    Sets context vars so every log line in this request
    automatically carries request_id, org_id, user_id.
    """

    async def dispatch(self, request: Request, call_next) -> Response:
        # Generate or propagate request ID
        # Frontend can pass X-Request-ID for tracing across services
        request_id = (
            request.headers.get("X-Request-ID")
            or str(uuid.uuid4())
        )
        # Correlation ID links this request to a Celery job or Langfuse trace
        correlation_id = (
            request.headers.get("X-Correlation-ID")
            or request_id
        )

        # Set context vars — propagated to all async calls in this request
        request_id_token     = request_id_ctx.set(request_id)
        correlation_id_token = correlation_id_ctx.set(correlation_id)

        # org_id and user_id set after auth middleware runs
        # (auth middleware calls org_id_ctx.set() and user_id_ctx.set())

        start = time.perf_counter()

        logger.info(
            f"→ {request.method} {request.url.path}",
            extra={"extra_fields": {
                "method":         request.method,
                "path":           request.url.path,
                "query":          str(request.query_params),
                "client_ip":      request.client.host if request.client else "",
            }}
        )

        response = await call_next(request)

        duration_ms = int((time.perf_counter() - start) * 1000)

        logger.info(
            f"← {response.status_code} {request.method} {request.url.path}",
            extra={"extra_fields": {
                "status_code": response.status_code,
                "duration_ms": duration_ms,
                "method":      request.method,
                "path":        request.url.path,
            }}
        )

        # Pass request_id back to client (useful for support tickets)
        response.headers["X-Request-ID"]     = request_id
        response.headers["X-Correlation-ID"] = correlation_id

        # Reset context vars
        request_id_ctx.reset(request_id_token)
        correlation_id_ctx.reset(correlation_id_token)

        return response
```

##### Auth Middleware Sets User Context

```python
# backend/api/deps.py — add to get_current_user
from core.logging import org_id_ctx, user_id_ctx

async def get_current_user(request: Request) -> AuthUser:
    state = authenticate_request(request, ...)
    claims = state.payload

    user = AuthUser(
        user_id=claims["sub"],
        org_id=claims.get("org_id", f"personal_{claims['sub']}"),
    )

    # Set context vars → all subsequent log lines in this request
    # will automatically include org_id and user_id
    org_id_ctx.set(user.org_id)
    user_id_ctx.set(user.user_id)

    return user
```

##### Registering Middleware in FastAPI

```python
# backend/main.py
import sentry_sdk
from fastapi import FastAPI
from core.logging import setup_logging, get_logger
from core.config import settings
from api.middleware.logging_middleware import LoggingMiddleware

# Setup logging FIRST — before anything else
setup_logging(level=settings.log_level)  # "DEBUG" in dev, "INFO" in prod
logger = get_logger(__name__)

sentry_sdk.init(dsn=settings.sentry_dsn, environment=settings.environment)

app = FastAPI(title="SynapseIQ API", version="1.0.0")

# Middleware order matters — logging wraps everything
app.add_middleware(LoggingMiddleware)

logger.info("SynapseIQ API starting", extra={"extra_fields": {
    "environment": settings.environment,
    "debug":       settings.debug,
}})
```

##### Celery Worker Logging — Correlation IDs

```python
# backend/workers/ingestion_worker.py
from core.logging import get_logger, org_id_ctx, user_id_ctx, correlation_id_ctx

logger = get_logger(__name__)

@celery.task(bind=True, ...)
def extract_worker(self, source_id: str, s3_key: str, org_id: str,
                   user_id: str, correlation_id: str) -> dict:
    """
    Pass correlation_id from API → Celery task.
    All worker logs carry the same correlation_id as the API request that triggered them.
    In CloudWatch: filter by correlation_id to see API + worker logs together.
    """
    # Set context for this worker task
    org_id_ctx.set(org_id)
    user_id_ctx.set(user_id)
    correlation_id_ctx.set(correlation_id)

    logger.info("extract_worker started", extra={"extra_fields": {
        "source_id": source_id,
        "task_id":   self.request.id,
    }})

    try:
        pages = extract_pdf(s3_key)
        logger.info("extraction complete", extra={"extra_fields": {
            "source_id":  source_id,
            "page_count": len(pages),
        }})
        return {"source_id": source_id, "pages": pages}

    except Exception as e:
        logger.error("extraction failed", extra={"extra_fields": {
            "source_id": source_id,
            "error":     str(e),
            "retry":     self.request.retries,
        }})
        raise
```

##### Logging Usage Throughout Codebase

```python
# Standard usage pattern — every service file
from core.logging import get_logger

logger = get_logger(__name__)   # name = "synapseiq.services.rag.pipeline"

class AdvancedRAGPipeline:

    async def run(self, question: str, source_ids: list, org_id: str, ...) -> ...:
        logger.info("RAG pipeline started", extra={"extra_fields": {
            "question_length": len(question),
            "source_count":    len(source_ids),
        }})

        chunks = await self.retriever.retrieve(question, ...)
        logger.debug("retrieval complete", extra={"extra_fields": {
            "chunk_count": len(chunks),
            "top_score":   chunks[0].score if chunks else 0,
        }})

        reranked = await self.reranker.rerank(question, chunks)
        logger.info("reranking complete", extra={"extra_fields": {
            "before_rerank_top": chunks[0].content[:50] if chunks else "",
            "after_rerank_top":  reranked[0].content[:50] if reranked else "",
        }})

        # NEVER log full prompts or user content at INFO level
        # Use DEBUG only — disabled in production via LOG_LEVEL=INFO
        logger.debug("generating answer", extra={"extra_fields": {
            "context_tokens": count_tokens(context),
            "model":          config.llm_model,
        }})
```

##### Log Levels — What Goes Where

```
DEBUG:   detailed diagnostic info (full query, schema details, chunk content)
         OFF in production (LOG_LEVEL=INFO)
         ON in development (LOG_LEVEL=DEBUG)

INFO:    normal operation milestones
         "extraction complete" "retrieval done" "answer generated"
         Always on in all environments

WARNING: unexpected but recoverable
         "cache miss for frequently asked question"
         "Groq rate limit approaching"
         "PDF page extraction yielded < 100 chars — using fallback"

ERROR:   operation failed, affects user
         "embedding API call failed after 3 retries"
         "SQL execution error"
         Always on, also sent to Sentry

CRITICAL: system-level failures
          "database connection pool exhausted"
          "Redis unavailable"
          Pages on-call
```

##### `.env` — Log Level Config

```bash
# .env.example additions
LOG_LEVEL=DEBUG        # local dev → see everything
# LOG_LEVEL=INFO       # production → only meaningful events
```

##### CloudWatch Logs Insights Queries

```sql
-- Find all logs for one request
fields @timestamp, level, message, duration_ms
| filter request_id = "req-abc-123"
| sort @timestamp asc

-- Find all errors for a specific org in last 1 hour
fields @timestamp, message, exception
| filter level = "ERROR" and org_id = "org_xyz"
| sort @timestamp desc
| limit 50

-- P95 latency by endpoint
fields @timestamp, path, duration_ms
| filter ispresent(duration_ms)
| stats percentile(duration_ms, 95) as p95 by path
| sort p95 desc

-- Ingestion failures by org
fields @timestamp, org_id, source_id, error
| filter logger like "ingestion_worker" and level = "ERROR"
| stats count() as failures by org_id
| sort failures desc

-- Correlation: API request → Celery worker
fields @timestamp, logger, level, message
| filter correlation_id = "corr-xyz-789"
| sort @timestamp asc
```

---

#### Acceptance Criteria
```
[ ] docker compose up → all 8 services healthy
[ ] GET /health → {"status":"ok","service":"synapseiq-api"}
[ ] localhost:3000 → dark themed shell (sidebar + layout visible)
[ ] localhost:5555 → Flower (3 queues: ingestion, eval, reports)
[ ] localhost:3001 → Langfuse dashboard (create account, get keys)
[ ] localhost:6333 → Qdrant dashboard
[ ] Alembic migrations run cleanly
[ ] BMAD skills: ls .claude/skills/ shows all agents
[ ] CLAUDE.md at project root (≤200 lines)
[ ] .claudeignore configured
[ ] /graphify . → generates graphify-out/GRAPH_REPORT.md
[ ] graphify hook install → git hooks active

Formatting + Linting:
[ ] ruff format backend/ --check → exits 0 (no formatting issues)
[ ] ruff check backend/ → exits 0 (no lint violations)
[ ] cd frontend && npm run format:check → exits 0
[ ] cd frontend && npm run lint → exits 0
[ ] cd frontend && npm run type-check → exits 0

Pre-commit Hooks:
[ ] Husky installed: .husky/pre-commit and .husky/commit-msg exist
[ ] git commit -m "bad message" → blocked by commitlint
[ ] git commit -m "feat: proper message" → hooks run, lint-staged fires
[ ] Modify a .py file with bad formatting → commit auto-formats it
[ ] Modify a .tsx file → prettier auto-formats it on commit
[ ] Intentionally break lint rule → commit blocked with clear error

Logging:
[ ] Every API request: JSON log line with request_id, method, path, status, duration_ms
[ ] Every API response: JSON log line with same request_id
[ ] X-Request-ID header returned on every response
[ ] Celery worker logs carry correlation_id matching the API request that triggered it
[ ] org_id + user_id appear in every log line after auth (verify with curl + watch logs)
[ ] LOG_LEVEL=DEBUG → detailed logs visible
[ ] LOG_LEVEL=INFO → debug logs suppressed (verify noisy logs gone)
[ ] docker compose logs backend | python -m json.tool → valid JSON (every line)
[ ] CloudWatch Logs Insights: filter by request_id → all related lines appear

[ ] First commit pushed to GitHub
```

---

### Phase 1 — Authentication (Clerk)
**Duration:** 3–4 days | **Model:** Sonnet 4

#### Story
```
Why needed:
  Every data record must be scoped to a user and organisation.
  Without auth, there's no identity → no isolation → no security.
  This is the foundation of the multi-tenant architecture.

Client need:
  Client A's users must never see Client B's documents or data.
  Managers must only query their org's databases.
  Data isolation is a hard requirement, not a nice-to-have.

Why Clerk over custom JWT:
  We evaluated building JWT auth ourselves (signup, login, refresh,
  OAuth, session management) — 4-5 days of work for non-differentiating
  infrastructure. Clerk provides all of this plus MFA, audit logs,
  Google OAuth, and multi-tenant organisation management out of the box.
  Free tier: 10,000 MAU — covers entire portfolio + early production.
  
  As an agency, we don't want to maintain auth code across client
  deployments. Clerk gives us multi-tenant auth that scales with us.
  
  "We chose not to build auth. We chose to build AI."

Benefit:
  Auth done in 3 days instead of 5.
  Google OAuth working in 30 minutes.
  MFA enabled with one Clerk dashboard toggle.
  Audit logs available for compliance without extra code.
  org_id from Clerk JWT → every DB query scoped automatically.
```

#### Tasks

**Clerk Setup**
```bash
# 1. Create account at clerk.com
# 2. Create application → get publishable key + secret key
# 3. Enable Google OAuth in Clerk dashboard (2 clicks)
# 4. Create organisation feature → enable multi-tenant
```

```python
# backend/api/deps.py — Clerk JWT validation
from clerk_backend_api import Clerk
from clerk_backend_api.jwks_helpers import authenticate_request

clerk = Clerk(bearer_auth=settings.clerk_secret_key)

async def get_current_user(request: Request) -> AuthUser:
    """Extract org_id + user_id from Clerk JWT — injected into every endpoint"""
    try:
        state = authenticate_request(
            request,
            AuthenticateRequestOptions(
                authorized_parties=[settings.frontend_url]
            )
        )
        if not state.is_signed_in:
            raise HTTPException(status_code=401, detail="Not authenticated")

        claims = state.payload
        return AuthUser(
            user_id = claims["sub"],
            org_id  = claims.get("org_id", f"personal_{claims['sub']}"),
            email   = claims.get("email", ""),
        )
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid token")
```

```tsx
// frontend/src/main.tsx — Clerk provider wraps everything
import { ClerkProvider } from '@clerk/clerk-react'

const CLERK_KEY = import.meta.env.VITE_CLERK_PUBLISHABLE_KEY

ReactDOM.createRoot(document.getElementById('root')!).render(
  <ClerkProvider publishableKey={CLERK_KEY} afterSignInUrl="/dashboard">
    <App />
  </ClerkProvider>
)

// frontend/src/components/auth/ProtectedRoute.tsx
import { useAuth } from '@clerk/clerk-react'
import { Navigate } from 'react-router-dom'

export function ProtectedRoute({ children }: { children: React.ReactNode }) {
  const { isSignedIn, isLoaded } = useAuth()
  if (!isLoaded) return <LoadingSpinner />
  if (!isSignedIn) return <Navigate to="/login" />
  return <>{children}</>
}
```

**Tenant Provisioning Webhook**
```python
# backend/api/routes/webhooks.py
# When new user/org created in Clerk → provision their resources
from fastapi import APIRouter, Request
from svix import Webhook

router = APIRouter()

@router.post("/webhooks/clerk")
async def clerk_webhook(request: Request):
    payload = await request.body()
    headers = dict(request.headers)
    
    # Verify webhook signature (Svix)
    wh = Webhook(settings.clerk_webhook_secret)
    event = wh.verify(payload, headers)
    
    if event["type"] == "organization.created":
        org_id = event["data"]["id"]
        # Provision: create Qdrant collection, S3 prefix, default settings
        await provision_tenant(org_id)
    
    if event["type"] == "user.created":
        user_id = event["data"]["id"]
        org_id  = event["data"].get("organization_memberships", [{}])[0].get("organization", {}).get("id")
        await create_user_profile(user_id, org_id)
    
    return {"status": "ok"}
```

#### Database
```sql
CREATE TABLE users (
    id         TEXT PRIMARY KEY,           -- Clerk user ID (user_xxx)
    org_id     TEXT NOT NULL,              -- Clerk org ID (org_xxx)
    email      TEXT NOT NULL,
    name       TEXT,
    role       TEXT DEFAULT 'member',      -- member | admin
    created_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX ON users (org_id);

CREATE TABLE organisations (
    id              TEXT PRIMARY KEY,      -- Clerk org ID
    name            TEXT NOT NULL,
    daily_llm_limit NUMERIC DEFAULT 10.0, -- $10/day default
    plan            TEXT DEFAULT 'free',
    created_at      TIMESTAMPTZ DEFAULT NOW()
);
```

#### Acceptance Criteria
```
[ ] /login → Clerk sign-in UI loads (dark theme)
[ ] Sign up with email → user created in Clerk + provisioned in DB
[ ] Google OAuth → completes in < 10 seconds
[ ] JWT contains org_id in claims (verify at jwt.io)
[ ] Protected route /sources → redirects to /login when signed out
[ ] Clerk webhook: new org → Qdrant collection created + S3 prefix set
[ ] Two different org accounts → completely isolated (verify in DB)
[ ] MFA: enable in Clerk dashboard → works without code change
[ ] Rate limit on sign-in attempts (Clerk handles this automatically)
```

---

### Phase 2 — SQL Agent
**Duration:** 7–8 days | **Model:** Sonnet 4 (Opus 4 for LangGraph design)

#### Story
```
Why needed:
  Managers and admins at our clients needed daily data insights —
  pending orders, inventory levels, revenue by region.
  Getting an answer required opening a Jira ticket, waiting for
  a developer to write SQL, waiting for review, waiting for results.
  Average time: 1-2 days for a question that takes 30 seconds to answer.

Client need:
  "I need to know how many orders are pending in Warehouse B.
   I need this every morning. I can't wait for a developer."

Why SQL Agent over simple NL-to-SQL:
  Client schemas have 30-80 tables. Single-shot SQL fails on:
  - Multi-table questions (join across 4 tables)
  - Ambiguous column names ("amt" — rupees or units?)
  - Date questions ("last quarter" — what's today's date again?)
  - Follow-ups ("now filter by region B")
  
  The agent loop solves these by exploring the schema interactively,
  just like a human analyst would before writing a query.
  We capped at 8 steps to control cost — at Groq pricing this costs
  ~$0.004 per query even for complex 8-step queries.

Why LangGraph for the agent:
  State persistence — manager starts a complex query, closes their
  laptop, comes back next day. LangGraph + Postgres checkpointer
  means the agent resumes exactly where it stopped.
  
  HITL checkpoint — before executing any SQL, the agent pauses and
  shows the manager what it's about to run. They approve or reject.
  This was a hard requirement: clients didn't want AI running
  arbitrary queries on production databases without human review.
  LangGraph's interrupt_before makes this one line of code.

Why multi-source (DuckDB federation):
  Clients had data in multiple places — customer list as CSV exports,
  live order data in their Postgres DB. Managers needed to correlate
  across both. DuckDB attaches both sources simultaneously and lets
  the agent write JOINs across them transparently.

Benefit:
  Time-to-insight: 1-2 days → 30 seconds.
  Developer time freed from routine data requests.
  Managers independently answer their own questions.
  Langfuse traces every agent step — full audit trail.
```

#### Frontend
- [ ] `/sources` — upload zone + source cards grid with status badges
- [ ] Upload modal — CSV/Excel drag-drop, file type icon
- [ ] Source card — name, type, status badge (ready/processing/error), row count
- [ ] SSE progress listener — live ingestion status
- [ ] `/chat` — sidebar (conversation history) + main chat area
- [ ] Data mode tab — SQL chat interface
- [ ] SQL response card — answer + syntax-highlighted SQL + table + chart
- [ ] Chart auto-selector — bar/line/pie based on column types
- [ ] LLM selector dropdown — provider (Groq/OpenAI/Bedrock) + model
- [ ] HITL approval modal — "Approve this SQL?" [Run] [Cancel] [Edit]
- [ ] Agent steps display — "Step 1: Reading schema... Step 2: Executing..."
- [ ] Export CSV button on results
- [ ] Cost badge — "$0.004 · 4 steps · 2.1s"

#### Backend — LangGraph SQL Agent

```python
# backend/services/sql/agent.py
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from groq import AsyncGroq
import instructor
from typing import TypedDict, Annotated
import operator

# Direct Groq SDK — no LangChain
groq_client = instructor.from_groq(AsyncGroq())

class SQLAgentState(TypedDict):
    question:     str
    source_ids:   list[str]
    org_id:       str
    user_id:      str
    llm_provider: str
    llm_model:    str
    messages:     Annotated[list, operator.add]
    final_sql:    str | None
    final_answer: str | None
    chart_type:   str | None
    error:        str | None
    step_count:   int

TOOLS = {
    "get_schema":          get_all_sources_schema,     # all attached sources
    "get_sample_rows":     get_sample_rows,
    "execute_sql":         execute_multi_source,        # DuckDB federation
    "check_distinct":      check_distinct_values,
    "get_date_range":      get_date_range,
    "list_sources":        list_available_sources,
}

async def agent_node(state: SQLAgentState) -> dict:
    """LLM decides next action"""
    decision: AgentDecision = await groq_client.chat.completions.create(
        model=state["llm_model"],
        response_model=AgentDecision,
        messages=state["messages"],
        max_retries=2,
    )
    return {
        "messages":   [{"role": "assistant", "content": decision.model_dump_json()}],
        "step_count": state["step_count"] + 1,
    }

async def tool_node(state: SQLAgentState) -> dict:
    """Execute chosen tool"""
    last = json.loads(state["messages"][-1]["content"])
    result = await TOOLS[last["action"]](
        **last["args"],
        source_ids=state["source_ids"],
        org_id=state["org_id"],
    )
    return {"messages": [{"role": "user", "content": f"Tool result: {result}"}]}

def route(state: SQLAgentState) -> str:
    if state["step_count"] >= 8:
        return "end"
    last = json.loads(state["messages"][-1].get("content", "{}"))
    if last.get("action") == "answer":
        return "end"
    return "tools"

# Build graph
builder = StateGraph(SQLAgentState)
builder.add_node("agent", agent_node)
builder.add_node("tools", tool_node)
builder.set_entry_point("agent")
builder.add_conditional_edges("agent", route, {"tools": "tools", "end": END})
builder.add_edge("tools", "agent")

# Postgres checkpointer — state persists across requests, server restarts
checkpointer = AsyncPostgresSaver.from_conn_string(settings.database_url)
sql_agent = builder.compile(
    checkpointer=checkpointer,
    interrupt_before=["tools"],   # HITL: pause before every tool execution
)
```

```python
# DuckDB multi-source federation
# backend/services/sql/executor.py
import duckdb

async def execute_multi_source(
    sql: str,
    source_ids: list[str],
    org_id: str,
) -> list[dict]:
    conn = duckdb.connect()
    
    for source_id in source_ids:
        source = await SourceRepository.get(source_id, org_id)
        
        if source.type in ("csv", "excel"):
            local_path = await s3.download_temp(source.s3_key)
            conn.execute(f"""
                CREATE VIEW {source.table_alias} AS
                SELECT * FROM read_csv_auto('{local_path}')
            """)
        
        elif source.type == "db":
            url = fernet.decrypt(source.connection_url)
            conn.execute(f"""
                ATTACH '{url}' AS {source.db_alias} (TYPE POSTGRES)
            """)
    
    result = conn.execute(sql).df()
    return result.head(10_000).to_dict("records")  # max 10K rows
```

#### Langfuse Integration

```python
# backend/providers/llm_factory.py
import litellm
import os

# Set once at startup — litellm auto-traces every call to Langfuse
os.environ["LANGFUSE_PUBLIC_KEY"] = settings.langfuse_public_key
os.environ["LANGFUSE_SECRET_KEY"] = settings.langfuse_secret_key
os.environ["LANGFUSE_HOST"]       = settings.langfuse_host
litellm.success_callback = ["langfuse"]
litellm.failure_callback = ["langfuse"]

async def llm_complete(
    provider: str,
    model: str,
    messages: list,
    trace_id: str | None = None,
    **kwargs,
):
    model_str = f"{provider}/{model}" if provider != "openai" else model
    return await litellm.acompletion(
        model=model_str,
        messages=messages,
        metadata={"trace_id": trace_id},   # links to parent trace
        **kwargs,
    )
```

#### Database
```sql
CREATE TABLE sources (
    id             UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    org_id         TEXT NOT NULL REFERENCES organisations(id),
    user_id        TEXT NOT NULL REFERENCES users(id),
    name           TEXT NOT NULL,
    type           TEXT NOT NULL,      -- csv|excel|pdf|txt|url|db
    table_alias    TEXT,               -- DuckDB view name for this source
    db_alias       TEXT,               -- DuckDB attached DB alias
    s3_key         TEXT,
    connection_url TEXT,               -- Fernet encrypted, type=db only
    content_hash   TEXT,               -- SHA256 for dedup
    status         TEXT DEFAULT 'pending',
    error_message  TEXT,
    row_count      INTEGER,
    chunk_count    INTEGER DEFAULT 0,
    file_size      BIGINT,
    schema_info    JSONB,
    doc_type       TEXT,               -- auto-classified: policy|financial|technical|general
    version        INTEGER DEFAULT 1,
    is_current     BOOLEAN DEFAULT TRUE,
    parent_id      UUID REFERENCES sources(id),
    created_at     TIMESTAMPTZ DEFAULT NOW(),
    deleted_at     TIMESTAMPTZ
);
CREATE INDEX ON sources (org_id, user_id) WHERE deleted_at IS NULL;

CREATE TABLE sql_conversations (
    id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    org_id        TEXT NOT NULL,
    user_id       TEXT NOT NULL,
    source_ids    TEXT[],
    thread_id     TEXT NOT NULL,       -- LangGraph thread ID for checkpointing
    question      TEXT NOT NULL,
    generated_sql TEXT,
    result_rows   JSONB,
    chart_type    TEXT,
    explanation   TEXT,
    steps_taken   INTEGER,
    llm_provider  TEXT DEFAULT 'groq',
    llm_model     TEXT DEFAULT 'llama-3.3-70b-versatile',
    latency_ms    INTEGER,
    token_count   INTEGER,
    cost_usd      NUMERIC(10,6),
    query_embedding vector(1536),      -- for episodic memory search
    created_at    TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX ON sql_conversations (org_id, user_id);
```

#### Acceptance Criteria
```
[ ] CSV upload → "ready" in < 10 seconds
[ ] "Show top 10 products by revenue" → correct SQL + bar chart
[ ] Multi-table question (needs 3+ agent steps) → correct answer
[ ] DuckDB JOINs across CSV + live Postgres DB simultaneously
[ ] HITL: SQL shown before execution → [Run] executes → [Cancel] stops
[ ] Agent stops at max_steps=8 (test with impossible question)
[ ] DROP TABLE → blocked immediately with clear error
[ ] Manager closes browser mid-query → reopens → resumes from checkpoint
[ ] LLM selector: switch Groq → OpenAI → same question works
[ ] Cost badge accurate (verify against Groq billing dashboard)
[ ] Every agent step visible in Langfuse trace
[ ] Input injection attempt → rejected before LLM call
[ ] All data scoped to org_id (second org cannot see first org's sources)
```

---

### Phase 3 — Basic RAG Pipeline
**Duration:** 7–8 days | **Model:** Sonnet 4

#### Story
```
Why needed:
  SQL Agent handles structured data questions.
  But clients had vast amounts of unstructured knowledge —
  product documentation, onboarding guides, policy documents,
  SOPs — that couldn't be answered by SQL.
  
  New users of a client's inventory app didn't know how to
  get started. Every question meant raising a support ticket.
  Average response time: 1 business day.

Client need:
  "Our users need to onboard themselves. We can't have our
   support team answering the same 50 questions repeatedly.
   Can the AI read our documentation and answer for us?"

Why tiered PDF extraction:
  We thought PyMuPDF would handle all PDFs. It didn't.
  Some clients had scanned manuals from 2015 — PyMuPDF
  returned empty strings. Users uploaded docs that never
  got indexed. Silent failures.
  
  Tiered approach: PyMuPDF for digital PDFs (fast, free),
  automatic fallback to Unstructured (OCR, complex layouts)
  when extraction quality is too low (< 100 chars/page average).
  This handles 100% of real-world PDFs correctly.

Why citations matter:
  Clients asked "how do we know the AI isn't making this up?"
  Every answer includes [source_name, page_number] citations.
  Users can click and verify. Trust built.

Benefit:
  Support ticket volume from onboarding questions dropped.
  New user time-to-first-action reduced.
  Same pipeline serves Use Case 1 (onboarding) and
  Use Case 2 (internal teams) with different document sets.
```

#### Frontend
- [ ] Documents tab in chat (alongside Data tab from Phase 2)
- [ ] Source type selector — PDF, TXT, URL now available
- [ ] Citation chips under answers — `[Policy.pdf, p.3]`
- [ ] Retrieved sources panel — collapsible, chunk text + similarity score
- [ ] Streaming response — tokens appear word by word (SSE)
- [ ] URL input field in upload modal — paste URL to scrape

#### Backend

```python
# backend/services/ingestion/extractor.py
import fitz           # PyMuPDF
import pdfplumber
from unstructured.partition.pdf import partition_pdf

async def extract_pdf(path: str) -> list[dict]:
    """Tiered extraction: fast path → quality check → OCR fallback"""

    # Tier 1: PyMuPDF (fast, works for 80% of PDFs)
    pages = _extract_pymupdf(path)
    avg_chars = sum(len(p["text"]) for p in pages) / max(len(pages), 1)

    if avg_chars >= 100:
        return pages  # good quality — use fast path

    # Tier 2: Unstructured (OCR, complex layouts, scanned docs)
    return _extract_unstructured(path)

def _extract_pymupdf(path: str) -> list[dict]:
    doc = fitz.open(path)
    return [
        {"text": page.get_text(), "page": i+1, "method": "pymupdf"}
        for i, page in enumerate(doc)
        if page.get_text().strip()
    ]

def _extract_unstructured(path: str) -> list[dict]:
    from unstructured.documents.elements import Table, NarrativeText, Title
    elements = partition_pdf(
        filename=path,
        strategy="hi_res",           # OCR-capable
        infer_table_structure=True,
    )
    pages: dict[int, list[str]] = {}
    for el in elements:
        page_num = el.metadata.page_number or 1
        pages.setdefault(page_num, [])
        if isinstance(el, Table):
            pages[page_num].append(_table_to_markdown(el))
        elif isinstance(el, (NarrativeText, Title)):
            pages[page_num].append(str(el))
    return [
        {"text": "\n".join(texts), "page": num, "method": "unstructured"}
        for num, texts in sorted(pages.items())
    ]
```

```python
# backend/services/ingestion/chunker.py
def recursive_split(
    text: str,
    chunk_size: int = 1000,
    overlap: int = 200,
) -> list[str]:
    """Pure Python recursive character splitting. No LangChain."""
    separators = ["\n\n", "\n", ". ", " ", ""]
    for sep in separators:
        if sep == "":
            chunks, start = [], 0
            while start < len(text):
                chunks.append(text[start:start+chunk_size])
                start += chunk_size - overlap
            return [c for c in chunks if c.strip()]
        parts = text.split(sep)
        if all(len(p) <= chunk_size for p in parts):
            return _merge_splits(parts, sep, chunk_size, overlap)
    return [text]
```

```python
# backend/services/rag/pipeline.py
from langfuse.decorators import observe, langfuse_context
from openai import AsyncOpenAI

embed_client = AsyncOpenAI()

class BasicRAGPipeline:

    @observe(name="rag_pipeline")
    async def run(
        self,
        question: str,
        source_ids: list[str],
        org_id: str,
        user_id: str,
        config: RAGConfig,
    ) -> AsyncIterator[str]:

        # 1. Embed question
        embedding = await self._embed(question)

        # 2. Vector search (pgvector — Phase 3 uses pgvector, Qdrant from Phase 5)
        chunks = await self._vector_search(
            embedding=embedding,
            org_id=org_id,
            user_id=user_id,
            source_ids=source_ids,
            k=5,
        )

        # 3. Build context with citations
        context = self._build_context(chunks)

        # Link chunks to Langfuse trace
        langfuse_context.update_current_observation(
            metadata={"chunks_retrieved": len(chunks),
                      "top_score": chunks[0].score if chunks else 0}
        )

        # 4. Stream generation
        async for token in self._stream(question, context, config):
            yield token

    @observe(name="vector_search")
    async def _vector_search(
        self, embedding, org_id, user_id, source_ids, k=5
    ) -> list[Chunk]:
        """Direct asyncpg — no ORM overhead on hot path"""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT id, content, metadata,
                       1 - (embedding <=> $1::vector) AS score
                FROM chunks
                WHERE org_id   = $2
                  AND user_id  = $3
                  AND source_id = ANY($4)
                  AND is_active = TRUE
                  AND 1 - (embedding <=> $1::vector) > 0.5
                ORDER BY embedding <=> $1::vector
                LIMIT $5
            """, embedding, org_id, user_id, source_ids, k)
        return [Chunk(**r) for r in rows]
```

#### Database
```sql
-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE sections (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source_id   UUID REFERENCES sources(id) ON DELETE CASCADE,
    org_id      TEXT NOT NULL,
    title       TEXT,
    page_start  INTEGER,
    page_end    INTEGER,
    level       INTEGER DEFAULT 1,   -- heading level (H1=1, H2=2, etc.)
    created_at  TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE chunks (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source_id   UUID REFERENCES sources(id) ON DELETE CASCADE,
    section_id  UUID REFERENCES sections(id),
    org_id      TEXT NOT NULL,
    user_id     TEXT NOT NULL,
    content     TEXT NOT NULL,
    embedding   vector(1536),
    metadata    JSONB NOT NULL DEFAULT '{}',
    -- metadata contains: source_name, page, chunk_index,
    --                     doc_type, method (pymupdf/unstructured),
    --                     section_title, ingested_at
    version     INTEGER DEFAULT 1,
    is_active   BOOLEAN DEFAULT TRUE,
    created_at  TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX ON chunks USING hnsw (embedding vector_cosine_ops)
             WITH (m=16, ef_construction=64);
CREATE INDEX ON chunks (org_id, user_id) WHERE is_active = TRUE;
CREATE INDEX ON chunks (source_id) WHERE is_active = TRUE;

CREATE TABLE rag_conversations (
    id               UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    org_id           TEXT NOT NULL,
    user_id          TEXT NOT NULL,
    question         TEXT NOT NULL,
    rewritten_query  TEXT,
    answer           TEXT,
    retrieved_chunks JSONB,
    confidence       NUMERIC(5,2),
    llm_provider     TEXT DEFAULT 'groq',
    llm_model        TEXT DEFAULT 'llama-3.3-70b-versatile',
    latency_ms       INTEGER,
    token_count      INTEGER,
    cost_usd         NUMERIC(10,6),
    query_embedding  vector(1536),     -- for episodic memory
    created_at       TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX ON rag_conversations (org_id, user_id);
```

#### Acceptance Criteria
```
[ ] PDF upload (digital) → "ready" via PyMuPDF in < 20 seconds
[ ] PDF upload (scanned) → Unstructured fallback triggered automatically
[ ] TXT file → chunked and indexed
[ ] URL scrape → nav/footer stripped → clean content indexed
[ ] "What is the return policy?" → correct answer from PDF
[ ] Citation chips [filename, page] under every factual claim
[ ] Sources panel shows top 5 chunks with similarity scores
[ ] Streaming: first token appears within 1.5 seconds
[ ] Answer grounded in docs — "I don't have that information" for unknown queries
[ ] org_id + user_id on every chunk (verify in DB)
[ ] is_active=TRUE filter working (soft-deleted chunks not retrieved)
[ ] Langfuse: trace shows retrieval + generation spans with token counts
```

---

### Phase 4 — Multi-Tenancy + Metadata + Live DB
**Duration:** 5–6 days | **Model:** Sonnet 4

#### Story
```
Why needed:
  Phase 2-3 had user_id scoping. But our multi-tenant SaaS story
  requires org_id as the primary isolation boundary.
  Client A's staff must be completely isolated from Client B's data,
  even if they somehow share infrastructure.
  
  Also: clients couldn't upload their production databases as CSV files.
  A client's order DB has 50GB of data. We needed live connection support.

Client need:
  "Our managers need to query live order data — not CSV exports
   from two days ago. Real-time. Against our actual database."

Why Fernet encryption for connection URLs:
  A Postgres connection string contains credentials:
  postgresql://admin:password@prod-server:5432/orders
  
  Storing this in plaintext is a critical security vulnerability.
  Fernet symmetric encryption + key in AWS Secrets Manager means
  even if the DB is compromised, connection URLs are unreadable.
  Only the application with the key can decrypt.

Why doc_type auto-classification:
  Before adding this, a question about "delivery charges" was
  returning chunks from financial reports alongside policy docs.
  Cross-domain noise degraded answer quality significantly.
  Auto-classifying doc type (one cheap LLM call at upload time)
  enables filtering that eliminates this noise at query time.

Benefit:
  Iron-clad data isolation (org_id on every single DB query).
  Live DB queries against real production data.
  Metadata filtering reduces retrieval noise measurably.
  Deduplication prevents vector store bloat over time.
```

#### Tasks

**org_id Enforcement**
```python
# backend/api/deps.py
# Middleware enforces org_id on every request
async def verify_org_access(
    resource_org_id: str,
    current_user: AuthUser = Depends(get_current_user)
) -> None:
    if resource_org_id != current_user.org_id:
        raise HTTPException(status_code=403, detail="Access denied")

# Every repository method REQUIRES org_id — it's not optional
class SourceRepository:
    @staticmethod
    async def get(source_id: str, org_id: str) -> Source:
        # org_id in WHERE clause — ALWAYS
        return await db.fetch_one(
            "SELECT * FROM sources WHERE id=$1 AND org_id=$2",
            source_id, org_id
        )
```

**Live DB Connection**
```python
# backend/services/ingestion/live_db.py
from cryptography.fernet import Fernet
import duckdb

fernet = Fernet(settings.fernet_key)  # key from AWS Secrets Manager

class LiveDBHandler:
    async def test_and_store(
        self, connection_url: str, org_id: str, user_id: str
    ) -> Source:
        # Test connection before storing
        try:
            conn = duckdb.connect()
            conn.execute(f"ATTACH '{connection_url}' AS test (TYPE POSTGRES)")
            tables = conn.execute(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema='public'"
            ).fetchall()
            conn.close()
        except Exception as e:
            raise ValueError(f"Could not connect: {e}")

        # Encrypt before storing
        encrypted = fernet.encrypt(connection_url.encode()).decode()

        return await SourceRepository.create({
            "org_id":         org_id,
            "user_id":        user_id,
            "type":           "db",
            "connection_url": encrypted,
            "status":         "ready",
            "schema_info":    {"tables": [t[0] for t in tables]},
        })
```

**doc_type Auto-Classification**
```python
# backend/services/ingestion/classifier.py
from pydantic import BaseModel
from typing import Literal

class DocClassification(BaseModel):
    doc_type: Literal["policy","financial","technical","hr","general"]
    confidence: float
    reasoning: str

async def classify_document(first_page_text: str) -> str:
    """One cheap LLM call at upload time — saves many expensive misses at query time"""
    result: DocClassification = await instructor_client.chat.completions.create(
        model="llama-3.1-8b-instant",   # use cheapest model for classification
        response_model=DocClassification,
        messages=[{"role":"user","content":
            f"Classify this document into one category.\n\n{first_page_text[:1000]}"}]
    )
    return result.doc_type
```

**Hash-Based Deduplication**
```python
async def should_ingest(content: str, org_id: str) -> bool:
    new_hash = hashlib.sha256(content.encode()).hexdigest()
    existing = await SourceRepository.find_by_hash(new_hash, org_id)
    if existing:
        return False  # already indexed — skip
    return True
```

#### Acceptance Criteria
```
[ ] Every DB query has org_id in WHERE clause (grep codebase — verify)
[ ] User from Org A cannot access Org B resources (403 on attempt)
[ ] Live Postgres URL → test connection → tables listed → SQL queries work
[ ] Invalid DB URL → "Could not connect: ..." error in UI
[ ] connection_url stored as ciphertext in DB (never plaintext)
[ ] Same PDF uploaded twice → "Already indexed" toast
[ ] doc_type auto-classified on upload (policy/financial/technical/hr/general)
[ ] Metadata filter "doc_type=policy" → only policy chunks retrieved
[ ] version=1, is_active=TRUE set on all new chunks
[ ] Delete source → is_active=FALSE on all its chunks (soft delete)
```

---

### Phase 5A — Core Advanced Retrieval
**Duration:** 5–6 days | **Model:** Opus 4 for design, Sonnet 4 for impl

#### Story
```
Why needed:
  Phase 3 RAG worked. But "worked" meant it returned answers.
  It didn't mean it returned CORRECT answers consistently.
  
  We measured with RAGAS on 20 golden questions.
  Baseline (Phase 3): faithfulness 0.67, context_precision 0.54.
  Not good enough for production. Clients expected accuracy.

Client need:
  "The AI needs to find the right information, not just
   similar-sounding information. Our policy has very specific
   product codes and SKU references that vector search alone
   can't find reliably."

Why migrate to Qdrant:
  pgvector served us well for MVP. But for hybrid search
  (dense + sparse in one query), native payload filtering,
  and the scale we expected, Qdrant is the right tool.
  It's purpose-built for vector search with production features.

Why hybrid search:
  Pure vector search missed exact matches — product SKUs, policy
  references like "Section 4.2.1", specific employee IDs.
  BM25 catches these keyword matches. RRF fusion combines both.
  Recall@5 improved from 0.61 → 0.79 after adding hybrid.

Why reranking:
  After retrieval, chunk ranked #1 by cosine similarity was often
  not the most relevant for the question. Cross-encoder reads
  query and document together — much better relevance signal.
  Top-5 precision improved from 0.58 → 0.81 after reranking.

Why context budget:
  Research (Liu et al., 2023) shows LLMs perform best on content
  at the START and END of context — middle chunks are often ignored.
  We found 10+ chunks hurt answer quality vs 5 focused chunks.
  max_chunks=5, max_context_tokens=6000 improved faithfulness.

Benefit:
  Recall@5: 0.61 → 0.79 (hybrid search)
  Precision@5: 0.58 → 0.81 (reranking)
  Faithfulness: 0.67 → 0.82 (all improvements combined)
  These are the numbers we can state in interviews.
```

#### Tasks

```python
# backend/services/vector_store/qdrant_store.py
from qdrant_client.models import (
    Distance, VectorParams, PointStruct,
    Filter, FieldCondition, MatchValue, MatchAny
)

async def search(
    collection: str,
    query_vector: list[float],
    org_id: str,
    user_id: str,
    source_ids: list[str],
    doc_type: str | None = None,
    k: int = 20,
) -> list[ScoredChunk]:
    must = [
        FieldCondition(key="org_id",   match=MatchValue(value=org_id)),
        FieldCondition(key="source_id",match=MatchAny(any=source_ids)),
        FieldCondition(key="is_active", match=MatchValue(value=True)),
    ]
    if doc_type:
        must.append(FieldCondition(key="doc_type", match=MatchValue(value=doc_type)))

    results = await qdrant_client.search(
        collection_name=collection,
        query_vector=query_vector,
        query_filter=Filter(must=must),
        limit=k,
        with_payload=True,
    )
    return [ScoredChunk(content=r.payload["content"], score=r.score,
                        metadata=r.payload) for r in results]
```

```python
# backend/services/rag/retriever.py
from rank_bm25 import BM25Okapi

class HybridRetriever:
    def __init__(self, vector_store, corpus: list[str]):
        self.vs   = vector_store
        self.bm25 = BM25Okapi([doc.lower().split() for doc in corpus])
        self.corpus = corpus

    async def retrieve(
        self,
        query: str,
        query_embedding: list[float],
        org_id: str,
        user_id: str,
        source_ids: list[str],
        doc_type: str | None,
        k: int = 20,
    ) -> list[ScoredChunk]:

        # Dense search via Qdrant
        dense = await self.vs.search(
            query_vector=query_embedding,
            org_id=org_id, user_id=user_id,
            source_ids=source_ids, doc_type=doc_type, k=k,
        )

        # BM25 sparse search
        scores = self.bm25.get_scores(query.lower().split())
        top_idx = scores.argsort()[::-1][:k]
        sparse = [ScoredChunk(content=self.corpus[i], score=float(scores[i]))
                  for i in top_idx if scores[i] > 0]

        return self._rrf(dense, sparse, k=k)

    def _rrf(self, dense, sparse, k=60) -> list[ScoredChunk]:
        scores: dict[str, float] = {}
        chunks_map: dict[str, ScoredChunk] = {}
        for rank, c in enumerate(dense):
            scores[c.content] = scores.get(c.content, 0) + 1/(k+rank)
            chunks_map[c.content] = c
        for rank, c in enumerate(sparse):
            scores[c.content] = scores.get(c.content, 0) + 1/(k+rank)
            chunks_map.setdefault(c.content, c)
        return [chunks_map[key] for key in sorted(scores, key=scores.get, reverse=True)]
```

```python
# backend/services/rag/reranker.py
import cohere

co = cohere.AsyncClient(api_key=settings.cohere_api_key)

async def rerank(
    query: str,
    chunks: list[ScoredChunk],
    top_n: int = 5,
) -> list[ScoredChunk]:
    result = await co.rerank(
        query=query,
        documents=[c.content for c in chunks],
        model="rerank-v3.5",
        top_n=top_n,
    )
    return [
        ScoredChunk(
            content=chunks[r.index].content,
            score=r.relevance_score,
            metadata={**chunks[r.index].metadata,
                      "original_rank": r.index,
                      "reranked_rank": i},
        )
        for i, r in enumerate(result.results)
    ]
```

```python
# RAGConfig with context budget
class RAGConfig(BaseModel):
    max_chunks:          int   = 5
    max_context_tokens:  int   = 6000
    min_chunk_score:     float = 0.5
    hybrid_enabled:      bool  = True
    reranking_enabled:   bool  = True
    llm_provider:        str   = "groq"
    llm_model:           str   = "llama-3.3-70b-versatile"

def build_context(chunks: list[ScoredChunk], config: RAGConfig) -> str:
    parts, token_count = [], 0
    for chunk in chunks[:config.max_chunks]:
        if chunk.score < config.min_chunk_score:
            continue
        chunk_tokens = count_tokens(chunk.content)
        if token_count + chunk_tokens > config.max_context_tokens:
            break
        parts.append(
            f"[{chunk.metadata['source_name']}, p{chunk.metadata['page']}]\n"
            f"{chunk.content}"
        )
        token_count += chunk_tokens
    return "\n\n---\n\n".join(parts)
```

#### Migration Script
```python
# scripts/migrate_pgvector_to_qdrant.py
"""Run once to migrate all chunks from pgvector to Qdrant"""
async def migrate():
    chunks = await db.fetch_all("SELECT * FROM chunks WHERE is_active=TRUE")
    batch_size = 100
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i+batch_size]
        points = [PointStruct(
            id=str(c["id"]),
            vector=c["embedding"],
            payload={
                "org_id":      c["org_id"],
                "user_id":     c["user_id"],
                "source_id":   str(c["source_id"]),
                "content":     c["content"],
                "is_active":   c["is_active"],
                **c["metadata"],
            }
        ) for c in batch]
        await qdrant_client.upsert(collection_name="chunks", points=points)
        print(f"Migrated {i+len(batch)}/{len(chunks)}")
```

#### Acceptance Criteria
```
[ ] All pgvector chunks migrated to Qdrant (count matches)
[ ] Hybrid search returns different (better) results than vector-only
[ ] RAGAS Recall@5 > 0.75 on 20-question golden set
[ ] Reranking visibly changes chunk order (log original vs reranked ranks)
[ ] Reranked top-5 precision > 0.70 on golden set
[ ] Context budget enforced: never more than 5 chunks OR 6000 tokens sent to LLM
[ ] doc_type filter correctly narrows search space (verify via Qdrant query stats)
[ ] Retrieval P95 latency < 200ms (logged in execution_events)
[ ] Dedup filter removes chunks with cosine > 0.95 before reranking
```

---

### Phase 5B — Retrieval Refinement
**Duration:** 4–5 days | **Model:** Sonnet 4
**Gate: Only build if Phase 5A RAGAS scores < targets**

#### Story
```
Why needed:
  Phase 5A addressed the biggest retrieval problems.
  Phase 5B addresses the remaining edge cases:
  
  Query rewriting: "what's the rtn policy on electronics?"
  → system didn't understand "rtn" = "return"
  → rewriting expands abbreviations, adds synonyms
  
  Multi-query: "compare our warranty and return policies"
  → single query retrieves warranty OR return chunks
  → multi-query retrieves both separately, deduplicates
  
  Context compression: chunks often had preamble/footer text
  → sending full chunk wasted tokens on irrelevant sentences
  → compression extracts only the relevant sentences

Approach:
  Only add these if Phase 5A RAGAS scores are below targets.
  Evidence-based: measure first, add if needed.
  Each technique is measured independently.
```

```python
# Query rewriting
class RewrittenQuery(BaseModel):
    rewritten: str

async def rewrite_query(question: str) -> str:
    result: RewrittenQuery = await instructor_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        response_model=RewrittenQuery,
        messages=[{"role":"user","content":
            f"Rewrite for document search. Expand abbreviations, add synonyms. "
            f"Keep under 20 words.\nOriginal: {question}"}]
    )
    return result.rewritten

# Multi-query retrieval
class QueryVariants(BaseModel):
    variants: list[str]   # exactly 3 variants

async def multi_query_retrieve(
    question: str, retriever: HybridRetriever, ...
) -> list[ScoredChunk]:
    variants_response: QueryVariants = await instructor_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        response_model=QueryVariants,
        messages=[{"role":"user","content":f"Generate 3 different search queries for: {question}"}]
    )
    all_chunks = []
    for v in variants_response.variants:
        emb = await embed(v)
        chunks = await retriever.retrieve(v, emb, ...)
        all_chunks.extend(chunks)
    # Deduplicate by content hash, keep highest score
    seen, unique = set(), []
    for c in sorted(all_chunks, key=lambda x: x.score, reverse=True):
        key = hashlib.md5(c.content.encode()).hexdigest()
        if key not in seen:
            seen.add(key)
            unique.append(c)
    return unique[:15]

# Context compression
async def compress_context(query: str, chunks: list[ScoredChunk]) -> str:
    compressed = []
    for chunk in chunks:
        class Compressed(BaseModel):
            relevant_sentences: str
        result: Compressed = await instructor_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            response_model=Compressed,
            messages=[{"role":"user","content":
                f"Extract only sentences relevant to: '{query}'\n\nText: {chunk.content}"}]
        )
        if result.relevant_sentences.strip():
            compressed.append(
                f"[{chunk.metadata['source_name']}, p{chunk.metadata['page']}]\n"
                f"{result.relevant_sentences}"
            )
    return "\n\n---\n\n".join(compressed)
```

#### Acceptance Criteria
```
[ ] Query rewriting: "rtn policy" → rewritten → correct chunks found
[ ] Multi-query: compound question retrieves chunks for both sub-topics
[ ] Context compression: avg context length reduced by ≥ 25%
[ ] RAGAS faithfulness after 5B > 0.80 (target)
[ ] RAGAS answer_relevancy > 0.82 (target)
```

---

### Phase 6 — RAG Quality Layer
**Duration:** 5–6 days | **Model:** Sonnet 4

#### Story
```
Why needed:
  "The AI seems to work" is not good enough for clients
  who are using it for business-critical onboarding and
  internal knowledge management.
  
  We needed to answer: "How accurate is it?" with a number,
  not an opinion. We needed to catch quality regressions
  before they reached production.

Client need:
  "If you update the AI system, how do we know the answers
   don't get worse? We need assurance, not trust."

Why RAGAS in CI/CD:
  After adding hybrid search and reranking, we had a situation
  where a prompt change accidentally reduced faithfulness from
  0.82 to 0.61. Without the eval gate, this would have shipped.
  The gate caught it, we reverted, quality maintained.

Why thumbs up/down:
  RAGAS measures technical quality (grounding, precision).
  User feedback measures perceived quality (helpfulness, clarity).
  Both are needed. A technically grounded answer can still
  be poorly worded. User feedback catches what RAGAS misses.

Why cost guardrails here:
  As retrieval improved, LLM calls got larger (more context tokens).
  Without guardrails, a single complex query could cost $0.50.
  Per-tenant daily limits protect us and our clients from runaway costs.

Benefit:
  "We have automated quality measurement on every PR.
   No regression ships without us knowing."
  That's production-grade AI engineering, not vibe-coding.
```

#### Tasks

```python
# eval/golden_dataset.json structure
[
  {
    "question": "What is the return policy for electronics?",
    "ground_truth": "Electronics can be returned within 30 days with original receipt.",
    "ground_truth_chunk_ids": ["chunk-uuid-1", "chunk-uuid-2"],
    "doc_type": "policy"
  },
  ...
]
# 20+ entries covering: edge cases, multi-hop questions,
# date-specific queries, cross-domain questions
```

```python
# scripts/run_eval_gate.py
import json, sys
from ragas import evaluate
from ragas.metrics import (
    faithfulness, answer_relevancy,
    context_precision, context_recall,
)
from datasets import Dataset

def run():
    with open("eval/golden_dataset.json") as f:
        golden = json.load(f)

    results = []
    for item in golden:
        # Run production RAG pipeline
        retrieved = retriever.retrieve(item["question"], ...)
        answer    = rag_pipeline.run(item["question"], ...)
        results.append({
            "question":     item["question"],
            "answer":       answer,
            "contexts":     [c.content for c in retrieved[:5]],
            "ground_truth": item["ground_truth"],
        })

    scores = evaluate(
        Dataset.from_list(results),
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    )

    THRESHOLDS = {
        "faithfulness":     0.75,
        "answer_relevancy": 0.78,
        "context_precision":0.65,
        "context_recall":   0.70,
    }
    failed = [f"{k}: {scores[k]:.3f} < {v}"
              for k, v in THRESHOLDS.items() if scores[k] < v]

    if failed:
        print("❌ EVAL GATE FAILED:")
        for f in failed: print(f"   {f}")
        sys.exit(1)

    print("✅ All quality gates passed")
    for k, v in scores.items(): print(f"   {k}: {v:.3f}")

run()
```

```yaml
# .github/workflows/ci.yml
name: CI
on: [push, pull_request]
jobs:
  quality-gate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.11" }
      - run: pip install ragas datasets groq openai cohere
      - run: python scripts/run_eval_gate.py
        env:
          GROQ_API_KEY: ${{ secrets.GROQ_API_KEY }}
          DATABASE_URL:  ${{ secrets.STAGING_DATABASE_URL }}
          QDRANT_URL:    ${{ secrets.STAGING_QDRANT_URL }}
```

```python
# Background RAGAS eval — non-blocking
@celery.task(queue="eval")
def run_ragas_background(conversation_id: str):
    """Runs async on every 10th RAG query — never blocks user response"""
    conv = RAGConversationRepository.get(conversation_id)
    scores = evaluate(Dataset.from_dict({
        "question":    [conv.question],
        "answer":      [conv.answer],
        "contexts":    [[c["content"] for c in conv.retrieved_chunks[:5]]],
        "ground_truth":[conv.question],
    }), metrics=[faithfulness, answer_relevancy])
    EvaluationRepository.save(conversation_id, scores)
    # Also send to Langfuse as custom scores on the trace
    langfuse_client.score(
        trace_id=conv.langfuse_trace_id,
        name="faithfulness",
        value=scores["faithfulness"],
    )
```

#### Database
```sql
CREATE TABLE feedback (
    id         UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    org_id     TEXT NOT NULL,
    user_id    TEXT NOT NULL,
    query_id   UUID NOT NULL,
    query_type TEXT NOT NULL,   -- rag | sql
    rating     INTEGER CHECK (rating IN (1, -1)),
    comment    TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE evaluations (
    id                UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    query_id          UUID NOT NULL,
    faithfulness      NUMERIC(5,3),
    answer_relevancy  NUMERIC(5,3),
    context_precision NUMERIC(5,3),
    context_recall    NUMERIC(5,3),
    evaluated_at      TIMESTAMPTZ DEFAULT NOW()
);
```

#### Acceptance Criteria
```
[ ] Golden dataset: 20+ Q&A pairs in eval/golden_dataset.json
[ ] CI/CD gate runs on every PR (verify on GitHub Actions tab)
[ ] Gate fails when faithfulness drops: artificially break prompt → PR fails
[ ] Gate passes when quality OK: fix prompt → PR passes
[ ] Background RAGAS runs without blocking query response
[ ] RAGAS scores appear in Langfuse as custom scores on traces
[ ] Admin dashboard: faithfulness trend chart over 30 days
[ ] Thumbs down captured and visible in admin
[ ] Cost guard: hit daily limit → 429 with clear message
[ ] Cost recorded accurately in Redis counter after every LLM call
```

---

### Phase 7 — Planner Agent + Memory
**Duration:** 6–7 days | **Model:** Opus 4 for graph design, Sonnet 4 for impl

#### Story
```
Why needed:
  The SQL Agent and RAG Pipeline were separate.
  Managers started asking questions that required both:
  "Our policy says free shipping over ₹500 — how many
   orders last month qualified?"
  
  This needs: RAG (find the ₹500 threshold) + SQL (count orders).
  Neither system alone could answer it.
  
  Also: "Run that analysis from last month again" —
  system had no memory of past queries.

Client need:
  "I shouldn't have to choose between querying my documents
   and querying my database. One question, one answer."

Why LangGraph for the Planner:
  The Planner orchestrates two LangGraph subgraphs (SQL Agent, RAG)
  and runs them in parallel for hybrid queries. It needs:
  - Parallel execution with result merging
  - State persistence (the conversation context spans multiple turns)
  - The same HITL checkpoint inherited from SQL Agent subgraph
  LangGraph subgraph composition is exactly designed for this.

Why episodic + semantic memory now:
  Usage patterns emerged: managers asked the same 10-15 questions
  repeatedly, just with different date ranges or filters.
  Episodic memory (searchable query history) lets them say
  "run that churn analysis again" without retyping.
  Semantic memory (user preferences) stops the system from
  asking "which region?" every time for a manager who always
  uses South India.

Benefit:
  Hybrid queries answer questions neither system could alone.
  Repeat query time: from 45 seconds to 3 seconds (episodic cache hit).
  User experience: system feels like it knows you.
```

#### LangGraph Planner

```python
# backend/services/agents/planner.py
from langgraph.graph import StateGraph, END, Send
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

class PlannerState(TypedDict):
    question:     str
    org_id:       str
    user_id:      str
    source_ids:   list[str]
    config:       RAGConfig
    intent:       str | None        # rag | sql | hybrid
    sql_result:   SQLResponse | None
    rag_result:   RAGResponse | None
    final_answer: str | None
    user_context: str               # from semantic memory

async def classify_intent(state: PlannerState) -> dict:
    """LLM classifies query intent"""
    class Intent(BaseModel):
        intent: Literal["rag","sql","hybrid"]
        reasoning: str

    result: Intent = await instructor_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        response_model=Intent,
        messages=[{
            "role": "system",
            "content": (
                "Classify the query:\n"
                "rag: needs information from documents/policies\n"
                "sql: needs data analysis from databases/spreadsheets\n"
                "hybrid: needs both documents AND data"
            )
        }, {"role": "user", "content": state["question"]}]
    )
    return {"intent": result.intent}

async def run_sql(state: PlannerState) -> dict:
    result = await sql_agent_subgraph.ainvoke({
        "question": state["question"],
        "org_id":   state["org_id"],
        "user_id":  state["user_id"],
        "source_ids": [s for s in state["source_ids"] if s.type in ("csv","excel","db")],
    })
    return {"sql_result": result}

async def run_rag(state: PlannerState) -> dict:
    result = await rag_pipeline.run(
        question=state["question"],
        org_id=state["org_id"],
        user_id=state["user_id"],
        source_ids=[s for s in state["source_ids"] if s.type in ("pdf","txt","url")],
        config=state["config"],
    )
    return {"rag_result": result}

async def synthesise(state: PlannerState) -> dict:
    """Merge SQL + RAG results into one coherent answer"""
    prompt = SYNTHESIS_PROMPT.format(
        question=state["question"],
        sql_answer=state["sql_result"].explanation if state["sql_result"] else "N/A",
        sql_table=state["sql_result"].rows[:5] if state["sql_result"] else [],
        doc_answer=state["rag_result"].answer if state["rag_result"] else "N/A",
        doc_citations=state["rag_result"].citations if state["rag_result"] else [],
    )
    answer = await llm_complete("groq", "llama-3.3-70b-versatile",
                                 [{"role":"user","content":prompt}])
    return {"final_answer": answer}

def route_intent(state: PlannerState) -> list:
    if state["intent"] == "hybrid":
        return [Send("run_sql", state), Send("run_rag", state)]  # parallel
    elif state["intent"] == "sql":
        return [Send("run_sql", state)]
    else:
        return [Send("run_rag", state)]

# Build planner graph
builder = StateGraph(PlannerState)
builder.add_node("classify",   classify_intent)
builder.add_node("run_sql",    run_sql)
builder.add_node("run_rag",    run_rag)
builder.add_node("synthesise", synthesise)
builder.set_entry_point("classify")
builder.add_conditional_edges("classify", route_intent)
builder.add_edge("run_sql",    "synthesise")
builder.add_edge("run_rag",    "synthesise")
builder.add_edge("synthesise", END)

planner = builder.compile(
    checkpointer=AsyncPostgresSaver.from_conn_string(settings.database_url)
)
```

#### Memory Implementation

```python
# backend/services/memory/episodic_memory.py
async def find_past_query(query: str, user_id: str, org_id: str) -> PastQuery | None:
    embedding = await embed(query)
    results = await qdrant_client.search(
        collection_name="query_history",
        query_vector=embedding,
        query_filter=Filter(must=[
            FieldCondition(key="user_id", match=MatchValue(value=user_id)),
            FieldCondition(key="org_id",  match=MatchValue(value=org_id)),
        ]),
        limit=1,
    )
    if results and results[0].score > 0.92:
        return PastQuery(**results[0].payload)
    return None

async def store_query(query: str, result: QueryResponse, user_id: str, org_id: str):
    embedding = await embed(query)
    await qdrant_client.upsert(
        collection_name="query_history",
        points=[PointStruct(
            id=str(uuid4()),
            vector=embedding,
            payload={
                "question":   query,
                "answer":     result.answer[:500],
                "query_type": result.type,
                "user_id":    user_id,
                "org_id":     org_id,
                "created_at": datetime.utcnow().isoformat(),
            }
        )]
    )
```

**Data Health Agent**
```python
# backend/services/agents/health_agent.py
async def run_health_check(source_id: str, org_id: str) -> HealthReport:
    source = await SourceRepository.get(source_id, org_id)
    path   = await s3.download_temp(source.s3_key)
    conn   = duckdb.connect()
    conn.execute(f"CREATE TABLE data AS SELECT * FROM read_csv_auto('{path}')")

    # Missing values per column
    columns  = [r[0] for r in conn.execute("DESCRIBE data").fetchall()]
    missing  = {}
    for col in columns:
        pct = conn.execute(
            f'SELECT 100.0*COUNT(*) FILTER (WHERE "{col}" IS NULL)/COUNT(*) FROM data'
        ).fetchone()[0]
        missing[col] = round(pct, 2)

    # Duplicates
    total  = conn.execute("SELECT COUNT(*) FROM data").fetchone()[0]
    unique = conn.execute("SELECT COUNT(*) FROM (SELECT DISTINCT * FROM data)").fetchone()[0]

    # LLM summary
    summary = await llm_complete("groq", "llama-3.1-8b-instant", [{
        "role":    "user",
        "content": f"Summarise these data quality issues with actionable fixes:\n"
                   f"Missing values: {missing}\nDuplicate rows: {total-unique}"
    }])
    return HealthReport(
        missing_values=missing,
        duplicate_rows=total - unique,
        llm_summary=summary,
    )
```

#### Acceptance Criteria
```
[ ] "What is our return policy AND how many returns last month?" → hybrid answer
[ ] Hybrid runs SQL + RAG in parallel (latency ≈ max of both, not sum)
[ ] "What is our return policy?" → RAG only (SQL agent not called)
[ ] "Show revenue by region" → SQL only (RAG not called)
[ ] "Run that analysis from last month" → episodic memory surfaces past query
[ ] User preference: manager always uses South India → auto-applied after 3 queries
[ ] Data Health: 50MB CSV → missing values + duplicates → LLM summary in < 15 seconds
[ ] State persists: close browser mid-hybrid query → reopen → can resume
[ ] Langfuse: planner trace shows both SQL and RAG child spans
```

---

### Phase 8 — Observability
**Duration:** 4–5 days | **Model:** Sonnet 4

#### Story
```
Why needed:
  We had a production incident where RAG answers degraded
  for 2 hours before anyone noticed. No alerts. No traces.
  We only found out when a client reported it.
  
  "Something is slow" is not enough information.
  We needed to know: which step is slow, which LLM call
  failed, what prompt was sent, how much it cost.

Client need:
  "We need an audit trail. Which documents did the AI search
   for this answer? What did it actually send to the LLM?"
  (compliance requirement from one enterprise client)

Why 4-layer observability:
  Langfuse alone doesn't tell you if ECS is running out of memory.
  CloudWatch alone doesn't tell you which prompt caused a regression.
  Sentry alone doesn't show you LLM token costs.
  Each layer covers what the others can't.

Why execution timeline in UI:
  Users and managers wanted to understand what the system did.
  "Why did it take 4 seconds?" → they can see: retrieval was fast,
  reranking took 380ms, the LLM took 2.1 seconds.
  Transparency builds trust. Black box erodes it.

Benefit:
  Next incident: identified root cause in 3 minutes (Langfuse trace).
  Compliance audit: full prompt + retrieval log for any conversation.
  Cost optimisation: identified that query rewriting was adding $0.001
  per query unnecessarily — removed for simple queries.
```

#### Four-Layer Stack

```
Langfuse (LLM layer):
  What: every LLM call — prompt, response, tokens, cost, latency, model
  Scope: all agents + RAG pipeline
  Setup: 3 env vars + litellm auto-traces
  Key insight: "why did this answer hallucinate?" → see exact prompt

CloudWatch (infrastructure layer):
  What: ECS task health, RDS connections, SQS queue depth, Redis memory
  Scope: all AWS services
  Alarms → Slack: error rate >5%, P95 >10s, daily cost >₹2000, DLQ >0

Sentry (error layer):
  What: exceptions with stack trace, user context, request details
  Scope: FastAPI backend + React frontend
  Why CloudWatch isn't enough: CW says "500 error". Sentry says exactly
  which line of code, with what state, triggered by which user.

execution_events table (custom trace layer):
  What: step-level timing for every agent/pipeline step
  Scope: intent_detection, query_rewriting, retrieval, reranking, generation
  Drives: execution timeline UI in the frontend
```

#### Frontend — Execution Timeline

```tsx
// ExecutionTimeline component
const steps = [
  { key: "intent_detection", label: "Understanding query" },
  { key: "query_rewriting",  label: "Optimising search query" },
  { key: "bm25_retrieval",   label: "Keyword search (BM25)" },
  { key: "vector_retrieval", label: "Semantic search" },
  { key: "reranking",        label: "Re-ranking results" },
  { key: "llm_generation",   label: "Generating answer" },
]

// Rendered below every answer (collapsible)
// Each step shows: status dot | label | duration ms | token count
// Total line: ⏱ 2.7s  💰 $0.004  📊 519 tokens
```

#### Sentry Setup

```python
# backend/main.py
import sentry_sdk
from sentry_sdk.integrations.fastapi import FastApiIntegration
from sentry_sdk.integrations.sqlalchemy import SqlalchemyIntegration

sentry_sdk.init(
    dsn=settings.sentry_dsn,
    integrations=[FastApiIntegration(), SqlalchemyIntegration()],
    traces_sample_rate=0.2,   # 20% of requests for performance monitoring
    environment=settings.environment,
)
```

```typescript
// frontend/src/main.tsx
import * as Sentry from "@sentry/react"
Sentry.init({
  dsn: import.meta.env.VITE_SENTRY_DSN,
  environment: import.meta.env.MODE,
})
```

#### Database
```sql
CREATE TABLE execution_events (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    query_id    UUID NOT NULL,
    org_id      TEXT NOT NULL,
    user_id     TEXT NOT NULL,
    step        TEXT NOT NULL,
    status      TEXT NOT NULL,   -- started | done | error
    duration_ms INTEGER,
    tokens      INTEGER DEFAULT 0,
    cost_usd    NUMERIC(10,8) DEFAULT 0,
    metadata    JSONB DEFAULT '{}',
    created_at  TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX ON execution_events (query_id);
CREATE INDEX ON execution_events (org_id, created_at DESC);
```

#### Acceptance Criteria
```
[ ] Every query: execution timeline visible in UI with step durations
[ ] Cost badge accurate (verify against Groq billing dashboard ± 5%)
[ ] Langfuse: click any trace → see full prompt + response + tokens
[ ] Langfuse: RAGAS scores appear as custom scores on RAG traces
[ ] Sentry: intentional 500 → appears in Sentry with stack trace + user_id
[ ] CloudWatch: 4 metrics dashboard live (latency, errors, cost, queue depth)
[ ] Slack alert fires on intentional error rate spike (>5%)
[ ] Admin dashboard loads in < 2 seconds
[ ] Flower: Celery workers visible with queue depths
```

---

### Phase 9 — Ingestion Queue + Caching
**Duration:** 5–6 days | **Model:** Sonnet 4

#### Story
```
Why needed (Queue):
  A client uploaded their entire documentation library — 300 PDFs —
  at once. The ingestion system blocked for 45 minutes. During that
  time, the API was slow for all users. One long ingestion job was
  starving everyone else.
  
  The async queue decouples upload from processing. Upload returns
  in 200ms. Worker processes in background. Users see live progress.
  Multiple large uploads don't affect each other.

Why needed (Caching):
  Monitoring revealed: 40% of queries were duplicates or near-duplicates.
  Managers asked "how many pending orders?" every morning.
  Each query cost $0.004 and took 2.7 seconds.
  Same answer. Every day. Same cost. Every day.
  
  Semantic cache: "how many pending orders?" and "what's the count
  of outstanding orders?" → same cached answer in 80ms.

Why Celery canvas (chain pattern):
  One big worker = one failure point, can't retry individual steps.
  Chained workers = each step independently retryable.
  embed_worker fails → only retry embedding, not re-extract the PDF.
  Scales independently: 5 embed workers for bulk import, 1 otherwise.

Why Redis locally, SQS in production:
  Redis: zero extra setup for local dev, already in docker-compose.
  SQS: durable (messages survive Redis crash), dead letter queue
  built in, CloudWatch integration, essentially free at our scale.
  Same Celery code, different CELERY_BROKER_URL env var. Zero code change.

Benefit:
  Upload response time: 45 minutes → 200ms (async).
  Repeat query cost: $0.004 → ~$0.0001 (cache amortised).
  Cache hit rate: 35% after 1 week of usage.
  Bulk import: 300 PDFs processed overnight without user impact.
```

#### Celery Chain

```python
# backend/workers/ingestion_worker.py
from celery import chain
from core.celery_app import celery

@celery.task(bind=True, max_retries=3,
             autoretry_for=(Exception,),
             retry_backoff=True,    # 60s → 120s → 240s
             retry_jitter=True,
             queue="ingestion")
def extract_worker(self, source_id: str, s3_key: str) -> dict:
    """Download from S3, extract text by type, update progress"""
    _update_progress(source_id, "extracting", 10)
    source = SourceRepository.get_sync(source_id)
    path   = s3.download_temp_sync(s3_key)

    if source.type == "pdf":
        pages = extract_pdf(path)
    elif source.type in ("csv","excel"):
        pages = extract_csv(path)
    elif source.type == "txt":
        pages = extract_txt(path)
    elif source.type == "url":
        pages = extract_url_sync(source.name)

    _update_progress(source_id, "extracted", 25)
    return {"source_id": source_id, "pages": pages}

@celery.task(bind=True, max_retries=3, queue="ingestion")
def chunk_worker(self, previous: dict) -> dict:
    source_id = previous["source_id"]
    _update_progress(source_id, "chunking", 40)
    chunks = []
    for page in previous["pages"]:
        page_chunks = recursive_split(page["text"])
        for i, c in enumerate(page_chunks):
            chunks.append({
                "content":  c,
                "metadata": {**page, "chunk_index": i}
            })
    _update_progress(source_id, "chunked", 55)
    return {"source_id": source_id, "chunks": chunks}

@celery.task(bind=True, max_retries=3,
             rate_limit="10/m",          # max 10 batches/min (respect API limits)
             queue="ingestion")
def embed_worker(self, previous: dict) -> dict:
    source_id = previous["source_id"]
    chunks    = previous["chunks"]
    embeddings = []

    for i in range(0, len(chunks), 100):
        batch = chunks[i:i+100]
        resp  = openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=[c["content"] for c in batch],
        )
        embeddings.extend([d.embedding for d in resp.data])
        progress = 55 + int((i+len(batch)) / len(chunks) * 30)
        _update_progress(source_id, "embedding", progress)

    return {"source_id": source_id, "chunks": chunks, "embeddings": embeddings}

@celery.task(bind=True, max_retries=3, queue="ingestion")
def store_worker(self, previous: dict) -> dict:
    source_id  = previous["source_id"]
    chunks     = previous["chunks"]
    embeddings = previous["embeddings"]
    _update_progress(source_id, "storing", 88)
    QdrantStore.upsert_sync(source_id, chunks, embeddings)
    return {"source_id": source_id, "chunk_count": len(chunks)}

@celery.task(queue="ingestion")
def notify_worker(previous: dict):
    source_id   = previous["source_id"]
    chunk_count = previous["chunk_count"]
    SourceRepository.update_sync(source_id, {
        "status":      "ready",
        "chunk_count": chunk_count,
    })
    redis_client.set(f"progress:{source_id}",
                     json.dumps({"stage":"ready","pct":100}), ex=3600)
    _push_sse_to_user(source_id, "ready")

def start_ingestion(source_id: str, s3_key: str):
    """Kick off the full chain"""
    chain(
        extract_worker.s(source_id, s3_key),
        chunk_worker.s(),
        embed_worker.s(),
        store_worker.s(),
        notify_worker.s(),
    ).apply_async(
        queue="ingestion",
        link_error=on_ingestion_failure.s(source_id),
    )

@celery.task
def on_ingestion_failure(request, exc, traceback, source_id: str):
    """Called when all retries exhausted"""
    SourceRepository.update_sync(source_id, {
        "status":        "error",
        "error_message": str(exc),
    })
    redis_client.delete(f"progress:{source_id}")
    _push_sse_to_user(source_id, "error", error=str(exc))
```

#### SSE Progress Endpoint

```python
# backend/api/routes/data.py
@router.get("/sources/{source_id}/status")
async def source_status_stream(
    source_id: str,
    current_user: AuthUser = Depends(get_current_user),
):
    """SSE stream that reads Redis progress key and streams to frontend"""
    async def generate():
        for _ in range(120):  # max 2 minutes of streaming (120 × 1s)
            raw = await redis_client.get(f"progress:{source_id}")
            if raw:
                data = json.loads(raw)
                yield f"data: {json.dumps(data)}\n\n"
                if data.get("pct") == 100 or data.get("stage") == "error":
                    return
            await asyncio.sleep(1)

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control":"no-cache","X-Accel-Buffering":"no"},
    )
```

#### Semantic Cache

```python
# backend/services/cache/semantic_cache.py
class SemanticCache:
    async def get(self, question: str, source_ids: list) -> CacheResult | None:
        # Level 1: Exact match
        key = f"exact:{hashlib.sha256((question+''.join(sorted(source_ids))).encode()).hexdigest()}"
        exact = await redis.get(key)
        if exact:
            return CacheResult(hit_type="exact", response=json.loads(exact))

        # Level 2: Semantic match
        q_emb   = await embed(question)
        similar = await self._find_similar(q_emb, source_ids, threshold=0.92)
        if similar:
            return CacheResult(hit_type="semantic", response=similar)

        return None

    async def set(self, question: str, source_ids: list, response: dict, embedding: list):
        key = f"exact:{hashlib.sha256((question+''.join(sorted(source_ids))).encode()).hexdigest()}"
        await redis.setex(key, 3600, json.dumps(response))
        await self._store_for_semantic(question, source_ids, embedding, response)

    async def invalidate_source(self, source_id: str):
        """Call when source is deleted/updated — clear all related cache"""
        async for key in redis.scan_iter(f"source:{source_id}:*"):
            await redis.delete(key)
```

#### Acceptance Criteria
```
[ ] Upload 100-page PDF → status "processing" in < 200ms
[ ] Progress bar updates live: extracting → chunking → embedding → storing → ready
[ ] embed_worker batches 100 chunks per API call (verify in Flower)
[ ] Worker failure → retries 3x with backoff → source shows "error" after exhaustion
[ ] Flower dashboard shows all 3 queues with task counts
[ ] Same question twice → second answer from cache in < 100ms
[ ] ⚡ "Answered from cache" badge appears on cache hits
[ ] Semantic match: "pending orders count?" = "how many outstanding orders?" → cache hit
[ ] Cache invalidated when source deleted (re-ask → full pipeline runs)
[ ] Local → Redis broker. Prod → SQS (verify via CELERY_BROKER_URL swap)
[ ] DLQ CloudWatch alarm triggers on simulated max-retry failure
```

---

### Phase 10 — Reports + Scheduling
**Duration:** 5–6 days | **Model:** Sonnet 4

#### Story
```
Why needed:
  Managers were getting value from the SQL Agent for ad-hoc queries.
  But they wanted certain analyses every week automatically —
  pending orders by warehouse every Monday, inventory summary
  every Friday — without opening the app.
  
  "I want the report in my email before I start my day.
   I don't want to remember to ask for it."

Client need:
  Predictable, scheduled insights delivered automatically.
  PDF format for sharing with non-technical stakeholders.
  No developer involvement after initial setup.

Why Lambda + EventBridge (not Celery beat):
  Celery beat requires the worker service to be running 24/7.
  Lambda + EventBridge is serverless — zero cost when not running,
  no infrastructure to maintain, handles timezone-aware scheduling,
  fails gracefully and notifies via CloudWatch.

Why WeasyPrint (not Puppeteer/wkhtmltopdf):
  Pure Python. No headless browser. No security concerns.
  Generates professional PDFs from HTML templates.
  Recharts charts are server-side rendered to SVG and embedded.

Benefit:
  Reports arrive in email without any user action.
  Manager's Monday morning starts with data already ready.
  Frees them from remembering to open the app for routine checks.
```

#### Tasks

```python
# backend/services/reports/generator.py
import asyncio
from weasyprint import HTML

async def generate_report(report_id: str) -> str:
    """Run questions, compile HTML, render PDF, upload to S3"""
    report = await ReportRepository.get(report_id)

    sections = []
    for item in report.config["questions"]:
        if item["type"] == "sql":
            result = await sql_agent_subgraph.ainvoke({
                "question":  item["question"],
                "source_ids":item["source_ids"],
                "org_id":    report.org_id,
            })
            sections.append(render_sql_section(item["question"], result))
        else:
            result = await rag_pipeline.run(
                question=item["question"],
                source_ids=item["source_ids"],
                org_id=report.org_id,
            )
            sections.append(render_rag_section(item["question"], result))

    html = REPORT_TEMPLATE.format(
        title=report.title,
        generated_at=datetime.utcnow().strftime("%d %b %Y, %H:%M UTC"),
        sections="\n".join(sections),
    )
    pdf_bytes = HTML(string=html).write_pdf()

    s3_key = f"reports/{report.org_id}/{report_id}/{datetime.utcnow().date()}.pdf"
    await s3.upload(s3_key, pdf_bytes, content_type="application/pdf")

    return s3_key
```

```python
# infra/aws/lambda_report_trigger.py
# AWS Lambda triggered by EventBridge on schedule
import boto3, json

def handler(event, context):
    """Triggered by EventBridge cron — finds due reports and triggers generation"""
    sqs = boto3.client("sqs")
    rds = boto3.client("rds-data")

    # Find reports due to run now
    due_reports = rds.execute_statement(
        resourceArn=...,
        secretArn=...,
        sql="SELECT id FROM reports WHERE next_run_at <= NOW() AND schedule IS NOT NULL"
    )
    for report in due_reports["records"]:
        sqs.send_message(
            QueueUrl=REPORTS_QUEUE_URL,
            MessageBody=json.dumps({"report_id": report[0]["stringValue"]})
        )
```

#### Database
```sql
CREATE TABLE reports (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    org_id      TEXT NOT NULL,
    user_id     TEXT NOT NULL,
    title       TEXT NOT NULL,
    config      JSONB NOT NULL,    -- [{question, source_ids, type: sql|rag}]
    schedule    JSONB,             -- {frequency:"weekly",day:"monday",time:"09:00",tz:"Asia/Kolkata"}
    recipients  TEXT[],
    last_run_at TIMESTAMPTZ,
    next_run_at TIMESTAMPTZ,
    created_at  TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE report_runs (
    id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    report_id    UUID REFERENCES reports(id),
    status       TEXT DEFAULT 'pending',   -- pending|generating|done|error
    s3_key       TEXT,
    error        TEXT,
    retry_count  INTEGER DEFAULT 0,
    generated_at TIMESTAMPTZ DEFAULT NOW()
);
```

#### Acceptance Criteria
```
[ ] Create report with 2 questions (SQL + RAG) → PDF in < 45 seconds
[ ] PDF contains correct answers + chart for SQL question
[ ] Schedule: set to 2 minutes from now → email arrives with download link
[ ] Email has signed S3 download URL (expires in 7 days)
[ ] Failed generation retries 3x before marking error
[ ] Schedule pause/resume toggle works
[ ] Lambda + EventBridge trigger verifiable in AWS console
```

---

### Phase 11 — Advanced Features
**Duration:** 7–8 days | **Model:** Sonnet 4

#### Story
```
Why needed:
  Six months into production, we had patterns:
  
  CRAG: ~15% of queries had poor retrieval quality. The system
  was generating answers from weak context. Users noticed.
  Self-correction (grade retrieval → web fallback) turned 15%
  bad answers into acceptable ones.
  
  Source versioning: clients updated their policies regularly.
  Old policy chunks stayed active alongside new ones. RAG was
  mixing v1 and v2 answers. Confusion. Versioning fixed it.
  
  Rate limiting: one client's user was running automated scripts
  hitting the API 500 times/day. $50/day just from that one user.
  Per-tenant limits with clear messaging solved it.
  
  Context window: long conversations (15+ messages) were sending
  15,000 tokens of history to the LLM. Cost increased 3x for
  heavy users. Sliding window + summarisation brought it back down.

Benefit:
  System handles its own failure modes gracefully.
  Policy updates propagate cleanly.
  Costs predictable and controlled.
  Long conversations don't inflate costs unboundedly.
```

#### Self-Correction RAG (CRAG)

```python
# backend/services/rag/crag.py
class CorrectedRAGPipeline:

    async def run(self, question: str, source_ids: list, org_id: str,
                  user_id: str, config: RAGConfig) -> RAGResponse:

        chunks = await self.retriever.retrieve(question, ...)

        judgement: QualityJudgement = await self._grade_retrieval(question, chunks)

        if judgement.quality == "sufficient":
            return await self._generate(question, chunks, config)

        elif judgement.quality == "insufficient":
            # Fallback: web search via Tavily
            web_chunks = await self.web_search.search(question)
            all_chunks = chunks + web_chunks
            return await self._generate(
                question, all_chunks, config,
                note="[Answer supplemented with web search]"
            )

        else:  # ambiguous
            rewritten = await rewrite_query(question)
            chunks    = await self.retriever.retrieve(rewritten, ...)
            return await self._generate(question, chunks, config)
```

#### Source Versioning

```python
# backend/services/ingestion/versioning.py
async def create_new_version(
    existing_source_id: str, new_file_path: str, org_id: str
) -> Source:
    """Replace current version with new one"""

    # Deactivate old chunks
    await db.execute(
        "UPDATE chunks SET is_active=FALSE WHERE source_id=$1 AND org_id=$2",
        existing_source_id, org_id
    )

    # Mark old source as archived
    await SourceRepository.update(existing_source_id, {"is_current": False})

    # Create new version
    old_source = await SourceRepository.get(existing_source_id, org_id)
    new_source  = await SourceRepository.create({
        **old_source.dict(exclude={"id","version","is_current","created_at"}),
        "version":    old_source.version + 1,
        "is_current": True,
        "parent_id":  existing_source_id,
    })

    # Trigger ingestion for new version
    start_ingestion(new_source.id, new_s3_key)
    return new_source

async def rollback_version(source_id: str, version: int, org_id: str):
    """Reactivate a previous version"""
    target = await SourceRepository.get_version(source_id, version, org_id)
    # Deactivate current
    current = await SourceRepository.get_current(source_id, org_id)
    await db.execute("UPDATE chunks SET is_active=FALSE WHERE source_id=$1", current.id)
    await SourceRepository.update(current.id, {"is_current": False})
    # Reactivate target version
    await db.execute("UPDATE chunks SET is_active=TRUE WHERE source_id=$1", target.id)
    await SourceRepository.update(target.id, {"is_current": True})
```

#### Rate Limiting

```python
# backend/services/guardrails/rate_limiter.py
async def check_query_rate_limit(user_id: str, org_id: str,
                                  query_type: str, redis: Redis) -> GuardResult:
    today = datetime.utcnow().strftime("%Y-%m-%d")
    key   = f"queries:{org_id}:{user_id}:{query_type}:{today}"
    count = int(await redis.get(key) or 0)

    limits = {"rag": 100, "sql": 50}  # free tier
    limit  = await get_tenant_query_limit(org_id, query_type) or limits[query_type]

    if count >= limit:
        return GuardResult(
            allowed=False,
            reason=f"Daily {query_type.upper()} limit of {limit} queries reached. Resets at midnight UTC."
        )
    await redis.incr(key)
    await redis.expireat(key, end_of_day_unix())
    return GuardResult(allowed=True)
```

#### Conversation Context Window

```python
# backend/services/memory/conversation_memory.py
class ConversationMemory:
    FULL_WINDOW = 5
    MAX_SUMMARY_TOKENS = 400

    async def get_messages_for_context(self, conversation_id: str) -> list[dict]:
        messages = await ConversationRepository.get_messages(conversation_id)
        if len(messages) <= self.FULL_WINDOW:
            return messages

        older  = messages[:-self.FULL_WINDOW]
        recent = messages[-self.FULL_WINDOW:]
        summary = await llm_complete("groq", "llama-3.1-8b-instant", [{
            "role":    "user",
            "content": f"Summarise this conversation in 3 sentences:\n"
                       + "\n".join(f"{m['role']}: {m['content']}" for m in older)
        }])
        return [
            {"role":"system","content":f"Previous conversation: {summary}"},
            *recent,
        ]
```

#### Acceptance Criteria
```
[ ] CRAG: ask question where internal docs insufficient → web search triggered
[ ] CRAG: "[supplemented with web search]" note visible in UI
[ ] CRAG: sufficient context → web search NOT called (verify in execution trace)
[ ] Source versioning: upload v2 → old chunks deactivated → new chunks active
[ ] "Roll back to v1" → v1 chunks reactivated → v1 answers returned
[ ] Version history UI shows all versions with [Restore] button
[ ] Rate limit: 101st RAG query → 429 with "Daily limit reached" message
[ ] 5-message conversation → full history sent
[ ] 10-message conversation → summary + last 5 (verify token count reduced)
[ ] Multi-file upload (3 PDFs) → 3 parallel Celery jobs visible in Flower
```

---

### Phase 12 — Production Deployment
**Duration:** 6–8 days | **Model:** Opus 4 for infrastructure decisions

#### Story
```
Why needed:
  Everything built so far runs on localhost.
  Clients need a URL. They need uptime. They need to trust
  the system won't disappear when our laptop is closed.
  
  Production deployment is not an afterthought — it's the
  delivery mechanism for everything we built.

Why ECS Fargate over EC2:
  No servers to manage, patch, or resize. We define containers
  and a task definition. AWS runs them. Auto-scaling built in.
  We pay per second of compute. No idle server costs.

Why blue/green deployment:
  Zero downtime. New version deployed alongside old.
  Traffic switched when health checks pass.
  Rollback is instant (switch traffic back).
  Clients never see a maintenance window.

Why eval gate in CI/CD:
  "Deploy only if quality is maintained."
  We had one incident where a prompt change reduced faithfulness.
  The eval gate would have caught it. It's not optional now.

Benefit:
  "Our AI service is live at a custom domain with 99.9% uptime,
   auto-scales under load, deploys zero-downtime, and won't ship
   quality regressions because the CI/CD gate blocks them."
  That's the closing line of the portfolio story.
```

#### Infrastructure

```
Frontend:  React build → S3 → CloudFront → custom domain (HTTPS, HTTP/2)
Backend:   ECR → ECS Fargate (min 1, max 5, CPU 70% scale trigger)
Workers:   ECR → ECS Fargate service (min 1, max 3, queue depth trigger)
DB:        RDS PostgreSQL 16 + pgvector (Multi-AZ, daily backup, 7d retention)
Qdrant:    ECS Fargate + EBS gp3 volume (persist vector data)
Redis:     ElastiCache Serverless (auto-scales, no sizing needed)
Files:     S3 (per-tenant prefix) + lifecycle → Glacier after 90 days
Reports:   Lambda + EventBridge (serverless scheduling)
Auth:      Clerk (cloud.clerk.com — managed, 10K MAU free)
LLM obs:   cloud.langfuse.com (50K observations/month free)
Errors:    sentry.io (5K errors/month free)
```

#### CI/CD Pipeline

```yaml
# .github/workflows/ci.yml
name: CI/CD
on:
  push:
    branches: [main, develop]
  pull_request:

jobs:
  quality-check:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Lint (ruff + eslint)
        run: |
          pip install ruff && ruff check backend/
          cd frontend && npm ci && npm run lint

      - name: Unit tests
        run: pytest backend/tests/ -x -q --tb=short

      - name: RAG eval gate (PRs only)
        if: github.event_name == 'pull_request'
        run: python scripts/run_eval_gate.py
        env:
          GROQ_API_KEY:    ${{ secrets.GROQ_API_KEY }}
          DATABASE_URL:    ${{ secrets.STAGING_DATABASE_URL }}
          QDRANT_URL:      ${{ secrets.STAGING_QDRANT_URL }}
          COHERE_API_KEY:  ${{ secrets.COHERE_API_KEY }}

  deploy-staging:
    needs: quality-check
    if: github.ref == 'refs/heads/develop'
    runs-on: ubuntu-latest
    steps:
      - name: Build + push to ECR
        run: |
          aws ecr get-login-password | docker login --username AWS --password-stdin $ECR_URI
          docker build -t synapseiq-api ./backend
          docker tag synapseiq-api:latest $ECR_URI/synapseiq-api:$GITHUB_SHA
          docker push $ECR_URI/synapseiq-api:$GITHUB_SHA

      - name: Deploy to ECS staging
        run: |
          aws ecs update-service \
            --cluster synapseiq-staging \
            --service synapseiq-api \
            --force-new-deployment \
            --deployment-configuration "deploymentCircuitBreaker={enable=true,rollback=true}"

  deploy-production:
    needs: quality-check
    if: github.ref == 'refs/heads/main'
    runs-on: ubuntu-latest
    environment: production
    steps:
      - name: Build + push to ECR
        run: |
          docker build -t synapseiq-api ./backend
          docker push $ECR_URI/synapseiq-api:$GITHUB_SHA

      - name: Blue/green deploy to ECS production
        run: |
          aws ecs update-service \
            --cluster synapseiq-prod \
            --service synapseiq-api \
            --force-new-deployment \
            --deployment-configuration "deploymentCircuitBreaker={enable=true,rollback=true}"

      - name: Smoke test
        run: |
          sleep 30
          curl -f https://api.synapseiq.com/health || exit 1

      - name: Notify Slack
        run: |
          curl -X POST $SLACK_WEBHOOK \
            -d '{"text":"✅ Deployed SynapseIQ v${{ github.sha }} to production"}'
```

#### Security Checklist

```
[ ] All secrets in AWS Secrets Manager (grep codebase — no hardcoded values)
[ ] VPC: RDS + ElastiCache in private subnets (no public internet access)
[ ] Security groups: ALB → ECS → RDS chain only (minimal blast radius)
[ ] WAF on CloudFront: rate limiting + SQL injection + XSS rules
[ ] HTTPS enforced everywhere (redirect HTTP → HTTPS at CloudFront)
[ ] S3 buckets private (pre-signed URLs for all access, 15min expiry)
[ ] Fernet key stored in Secrets Manager + rotation procedure documented
[ ] Clerk webhook signature verified on every webhook call
[ ] ECS task role: least privilege (only S3 prefix, only SQS queue it owns)
[ ] RDS: automated backups enabled, 7-day retention, test restore works
```

#### CloudWatch Alarms → Slack

```python
# infra/aws/cloudwatch_alarms.py
ALARMS = [
    {"name": "HighErrorRate",   "metric": "5xxErrors",          "threshold": 5,     "unit": "%"},
    {"name": "SlowP95Latency",  "metric": "TargetResponseTime", "threshold": 10,    "unit": "s"},
    {"name": "HighDailyCost",   "metric": "EstimatedCharges",   "threshold": 25,    "unit": "USD"},
    {"name": "DLQHasMessages",  "metric": "ApproximateNumberOfMessagesVisible", "threshold": 0, "unit": "count"},
    {"name": "LowECSTaskCount", "metric": "RunningTaskCount",   "threshold": 1,     "unit": "count"},
    {"name": "HighRedisMemory", "metric": "DatabaseMemoryUsagePercentage", "threshold": 80, "unit": "%"},
]
# Each alarm → SNS → Lambda → Slack message with context
```

#### LiteLLM Proxy — Centralised LLM Gateway

```
Why add this in Phase 12 and not earlier:

  Phase 0-11: litellm library works fine
    Each service calls litellm.acompletion() directly
    Rate limiting handled per-service
    Good enough at portfolio scale

  Phase 12 (production): multiple ECS services all making LLM calls
    FastAPI (chat endpoints)
    Celery workers (classification, health agent, eval)
    Report generator
    
  Problem without proxy:
    Each service manages its own rate limits
    Groq rate limit hits one service → other services unaffected but degraded
    No centralised budget enforcement per tenant
    If Groq goes down → each service fails independently, no auto-switch
    
  With LiteLLM Proxy:
    One entry point for ALL LLM traffic
    Proxy handles: routing, fallback, rate limiting, budget enforcement
    All services just call: POST http://litellm-proxy:4000/v1/chat/completions
    OpenAI-compatible API — same code, different base_url
```

```yaml
# docker-compose.yml addition (also deployed as ECS Fargate service in prod)
litellm-proxy:
  image: ghcr.io/berriai/litellm:main-latest
  ports: ["4000:4000"]
  volumes: ["./litellm_config.yaml:/app/config.yaml"]
  command: --config /app/config.yaml --detailed_debug
  environment:
    GROQ_API_KEY:        ${GROQ_API_KEY}
    OPENAI_API_KEY:      ${OPENAI_API_KEY}
    LANGFUSE_PUBLIC_KEY: ${LANGFUSE_PUBLIC_KEY}
    LANGFUSE_SECRET_KEY: ${LANGFUSE_SECRET_KEY}
    LANGFUSE_HOST:       ${LANGFUSE_HOST}
  healthcheck:
    test: ["CMD", "curl", "-f", "http://localhost:4000/health"]
    interval: 10s
    retries: 3
```

```yaml
# litellm_config.yaml
model_list:
  # Alias: fast-cheap → Groq primary, OpenAI fallback
  - model_name: fast-cheap
    litellm_params:
      model: groq/llama-3.3-70b-versatile
      api_key: os.environ/GROQ_API_KEY
      rpm: 30                            # requests per minute limit

  - model_name: fast-cheap              # fallback: same alias, different provider
    litellm_params:
      model: openai/gpt-4o-mini
      api_key: os.environ/OPENAI_API_KEY

  # Alias: fast-small → for cheap tasks (classification, compression)
  - model_name: fast-small
    litellm_params:
      model: groq/llama-3.1-8b-instant
      api_key: os.environ/GROQ_API_KEY

  # Alias: powerful → for architecture, complex reasoning
  - model_name: powerful
    litellm_params:
      model: groq/llama-3.3-70b-versatile
      api_key: os.environ/GROQ_API_KEY

  # Alias: embedding
  - model_name: embedding
    litellm_params:
      model: openai/text-embedding-3-small
      api_key: os.environ/OPENAI_API_KEY

general_settings:
  master_key: os.environ/LITELLM_MASTER_KEY  # stored in Secrets Manager
  database_url: os.environ/DATABASE_URL       # LiteLLM uses DB for virtual keys

litellm_settings:
  success_callback: ["langfuse"]   # ALL LLM calls traced centrally at proxy level
  failure_callback: ["langfuse"]   # no per-service Langfuse setup needed
  request_timeout: 30
  num_retries: 3
  drop_params: true                # ignore unsupported params per provider

router_settings:
  routing_strategy: least-busy     # load balance across keys/models
  fallbacks:
    - {"fast-cheap": ["openai/gpt-4o-mini"]}   # Groq down → OpenAI
  context_window_fallbacks:
    - {"fast-cheap": ["openai/gpt-4o"]}         # context too long → GPT-4o

# Virtual keys — budget per service (optional, configure in LiteLLM UI)
# api-service:    $8/day budget
# worker-service: $2/day budget
```

```python
# backend/providers/llm_factory.py — UPDATED for proxy
# All LLM calls now go through LiteLLM Proxy
# Langfuse integration moves to proxy level (remove from litellm library calls)

import openai
from core.config import settings

# Single client pointing to LiteLLM Proxy
# OpenAI-compatible API — exact same interface
_proxy_client = openai.AsyncOpenAI(
    api_key=settings.litellm_master_key,
    base_url=settings.litellm_proxy_url,  # http://litellm-proxy:4000
)

# Model alias mapping — what alias to use for each task type
MODEL_ALIASES = {
    "sql_agent":        "fast-cheap",   # SQL generation steps
    "rag_generation":   "fast-cheap",   # RAG answer generation
    "classification":   "fast-small",   # intent, doc_type classification
    "rewriting":        "fast-small",   # query rewriting
    "compression":      "fast-small",   # context compression
    "summarisation":    "fast-small",   # conversation summary
    "health_summary":   "fast-small",   # data health LLM summary
    "synthesis":        "fast-cheap",   # planner synthesis
    "embedding":        "embedding",    # all embedding calls
}

async def llm_complete(
    task: str,                      # e.g. "sql_agent", "rag_generation"
    messages: list[dict],
    response_model=None,            # Pydantic model for structured output
    stream: bool = False,
    user_id: str | None = None,
    org_id: str | None = None,
    **kwargs,
) -> str | BaseModel:
    """
    Single entry point for ALL LLM calls in the system.
    Routes through LiteLLM Proxy which handles:
      - Provider selection (Groq vs OpenAI vs Bedrock)
      - Automatic fallback if provider fails
      - Rate limiting
      - Langfuse tracing (at proxy level)
      - Cost tracking
    """
    model_alias = MODEL_ALIASES.get(task, "fast-cheap")

    if response_model:
        # Structured output via instructor wrapping the proxy client
        import instructor
        structured_client = instructor.from_openai(_proxy_client)
        return await structured_client.chat.completions.create(
            model=model_alias,
            response_model=response_model,
            messages=messages,
            metadata={                 # passed to Langfuse via proxy
                "user_id": user_id,
                "org_id":  org_id,
                "task":    task,
            },
            **kwargs,
        )

    if stream:
        return await _proxy_client.chat.completions.create(
            model=model_alias,
            messages=messages,
            stream=True,
            **kwargs,
        )

    response = await _proxy_client.chat.completions.create(
        model=model_alias,
        messages=messages,
        **kwargs,
    )
    return response.choices[0].message.content
```

```python
# Usage throughout codebase — clean, consistent, no provider details leaked

# SQL Agent (before proxy):
groq_client = instructor.from_groq(AsyncGroq())
result = await groq_client.chat.completions.create(model="llama-3.3-70b", ...)

# SQL Agent (after proxy):
result = await llm_complete("sql_agent", messages, response_model=AgentDecision)
# Proxy decides: use Groq llama-3.3-70b. If Groq is down: use OpenAI gpt-4o-mini.

# Classification (cheap task):
doc_type = await llm_complete("classification", messages, response_model=DocClassification)
# Proxy uses: llama-3.1-8b-instant (fast-small alias) → $0.00002 per call

# The task name is the only thing the application knows.
# Provider, model, fallback, tracing — all proxy's responsibility.
```

```
Benefits in production:
  Provider switch: change litellm_config.yaml → all services updated instantly
                   no code deployments needed
  Groq outage:     proxy auto-routes to OpenAI → zero downtime
  Budget control:  each virtual key has daily budget → workers can't starve API
  Langfuse:        ALL traces centralised at proxy → one dashboard, full picture
  Model aliases:   "fast-cheap" today = Groq. Tomorrow = Gemini Flash.
                   Application code never changes.

LiteLLM Proxy UI: http://localhost:4000/ui
  → view all virtual keys + budgets
  → see request logs
  → toggle providers on/off
  → set per-model rate limits
```

#### Acceptance Criteria

```
[ ] Frontend at custom domain with HTTPS (A+ score on SSL Labs)
[ ] GET https://api.yourdomain.com/health → {"status":"ok"}
[ ] PR with bad prompt → eval gate fails (GitHub Actions shows ❌)
[ ] PR with good code → eval gate passes → auto-merges to staging
[ ] Merge to main → blue/green deploy → zero downtime
    (test: curl -s api.yourdomain.com/health every 2 seconds during deploy)
[ ] ECS auto-scales: load test 50 concurrent → 1 → 3 tasks (verify in ECS console)
[ ] All secrets in Secrets Manager (grep codebase confirms no hardcoded keys)
[ ] CloudWatch dashboard: 4 panels showing real data
[ ] Slack alert fires when error rate >5% (trigger intentionally)
[ ] Langfuse cloud: traces appear from production API calls
[ ] Sentry: production error appears with full context
[ ] RDS backup: snapshot exists in RDS console
[ ] Full deploy pipeline: < 12 minutes from push to live

LiteLLM Proxy:
[ ] All LLM calls route through proxy (verify: no direct Groq/OpenAI calls in app code)
[ ] Kill Groq env var → proxy auto-routes to OpenAI fallback → queries still work
[ ] LiteLLM UI at /ui shows all requests with cost breakdown
[ ] Langfuse receives traces from proxy (not from application code)
[ ] fast-small alias uses llama-3.1-8b-instant (verify in proxy logs)
[ ] fast-cheap alias uses llama-3.3-70b-versatile (verify in proxy logs)
```

---

## 13. API Reference

```
/api/webhooks
  POST /webhooks/clerk       → Clerk event handler (user/org provisioning)

/api/data
  POST /data/upload          → multipart file → S3 → Celery chain
  POST /data/url             → URL scraping job
  POST /data/connect-db      → test + store encrypted DB URL
  GET  /data/connect-db/:id/tables
  GET  /data/sources         → paginated list (org_id + user_id scoped)
  GET  /data/sources/:id     → detail + text preview
  DELETE /data/sources/:id   → soft delete (is_active=FALSE on chunks)
  GET  /data/sources/:id/status      → SSE progress stream
  POST /data/sources/:id/version     → create new version
  POST /data/sources/:id/rollback    → rollback to previous version
  GET  /data/sources/:id/versions    → version history

/api/chat
  POST /chat/sql             → SQL Agent (LangGraph)
  POST /chat/sql/:thread_id/approve → HITL: approve pending SQL
  POST /chat/rag             → RAG Pipeline (streams SSE tokens)
  POST /chat/query           → Planner (auto-routes, streams)
  GET  /chat/history         → paginated conversations
  GET  /chat/:id/messages    → messages in a conversation

/api/llm
  GET  /llm/providers        → available providers + models + pricing

/api/reports
  POST /reports              → create report + schedule
  GET  /reports              → list org's reports
  GET  /reports/:id          → detail
  POST /reports/:id/generate → one-time generation
  GET  /reports/:id/download → signed S3 URL (7-day expiry)
  PATCH /reports/:id/schedule → pause/resume/update schedule

/api/analysis
  POST /analysis/health/:source_id → trigger health check (Celery eval queue)
  GET  /analysis/health/:source_id → latest health report

/api/feedback
  POST /feedback             → thumbs up/down + optional comment

/api/analytics
  GET  /analytics/dashboard       → aggregated metrics (7/30 day)
  GET  /analytics/executions/:id  → step timeline for one query
  GET  /analytics/evaluation      → RAGAS metric trends

Standard response wrapper:
  { "data": T | null, "error": string | null, "meta": { total?, page?, cost_usd?, latency_ms? } }
```

---

## 14. Database Schema

```sql
-- Auth (Clerk manages passwords/sessions)
organisations  (id[Clerk org_id], name, daily_llm_limit, plan, created_at)
users          (id[Clerk user_id], org_id, email, name, role, created_at)

-- Data Sources
sources        (id, org_id, user_id, name, type, table_alias, db_alias,
                s3_key, connection_url[encrypted], content_hash,
                status, error_message, row_count, chunk_count, file_size,
                schema_info JSONB, doc_type, version, is_current,
                parent_id, created_at, deleted_at)

-- Document Hierarchy
sections       (id, source_id, org_id, title, page_start, page_end, level)

-- Vectors (pgvector MVP → Qdrant Phase 5+)
chunks         (id, source_id, section_id, org_id, user_id, content,
                embedding vector(1536), metadata JSONB,
                version, is_active, created_at)

-- SQL Conversations
sql_conversations  (id, org_id, user_id, source_ids TEXT[], thread_id,
                    question, generated_sql, result_rows JSONB, chart_type,
                    explanation, steps_taken, llm_provider, llm_model,
                    latency_ms, token_count, cost_usd,
                    query_embedding vector(1536), created_at)

-- RAG Conversations
rag_conversations  (id, org_id, user_id, question, rewritten_query,
                    answer, retrieved_chunks JSONB, confidence,
                    llm_provider, llm_model, langfuse_trace_id,
                    latency_ms, token_count, cost_usd,
                    query_embedding vector(1536), created_at)

-- Reports
reports        (id, org_id, user_id, title, config JSONB, schedule JSONB,
                recipients TEXT[], last_run_at, next_run_at, created_at)
report_runs    (id, report_id, status, s3_key, error, retry_count, generated_at)

-- Quality
feedback       (id, org_id, user_id, query_id, query_type, rating, comment, created_at)
evaluations    (id, query_id, faithfulness, answer_relevancy,
                context_precision, context_recall, evaluated_at)
health_reports (id, source_id, org_id, report JSONB, summary, created_at)

-- Observability
execution_events (id, query_id, org_id, user_id, step, status,
                  duration_ms, tokens, cost_usd, metadata JSONB, created_at)
```

---

## 15. Folder Structure

```
synapseiq/
├── CLAUDE.md                         ← ≤200 lines, update Current Phase each phase
├── .claudeignore
├── .env.example
├── docker-compose.yml
├── Makefile                          ← make dev | test | migrate | eval | graphify
│
├── docs/
│   ├── prd.md                        ← this file
│   ├── architecture.md               ← updated after each phase
│   ├── stories/                      ← one .md per Claude Code session
│   └── decisions/                    ← ADRs (why we chose X over Y)
│
├── eval/
│   └── golden_dataset.json           ← 20+ Q&A pairs for CI/CD gate
│
├── scripts/
│   ├── run_eval_gate.py              ← CI/CD quality gate
│   ├── migrate_pgvector_qdrant.py    ← Phase 5A migration
│   └── provision_tenant.py           ← manual tenant provisioning
│
├── frontend/
│   ├── vite.config.ts
│   ├── tailwind.config.ts
│   └── src/
│       ├── pages/
│       │   ├── Dashboard.tsx
│       │   ├── Sources.tsx
│       │   ├── Chat.tsx              ← unified (SQL + RAG tabs)
│       │   ├── Reports.tsx
│       │   ├── admin/
│       │   │   └── Metrics.tsx
│       │   └── auth/
│       │       └── Login.tsx         ← Clerk SignIn component
│       ├── components/
│       │   ├── ui/                   ← Card, Button, Badge, Modal, Skeleton
│       │   ├── layout/               ← Layout, Sidebar, NavItem
│       │   ├── chat/
│       │   │   ├── MessageBubble.tsx
│       │   │   ├── ExecutionTimeline.tsx
│       │   │   ├── SourcePanel.tsx
│       │   │   ├── CitationChip.tsx
│       │   │   ├── SQLCard.tsx       ← SQL result with table + chart
│       │   │   └── HITLModal.tsx     ← SQL approval modal
│       │   ├── charts/               ← BarChart, LineChart, PieChart wrappers
│       │   └── upload/               ← DropZone, ProgressBar, SourceTypeIcon
│       ├── hooks/
│       │   ├── useAuth.ts            ← Clerk useUser wrapper
│       │   ├── useChat.ts
│       │   ├── useSources.ts
│       │   └── useSSE.ts             ← SSE hook for progress streams
│       ├── services/                 ← api.ts (axios + Clerk token), typed methods
│       ├── store/                    ← uiStore.ts (Zustand, UI state only)
│       └── types/                    ← shared TypeScript types
│
├── backend/
│   ├── main.py                       ← FastAPI app, routers, middleware, Sentry
│   ├── core/
│   │   ├── config.py                 ← Pydantic Settings (all env vars)
│   │   ├── database.py               ← SQLAlchemy async engine
│   │   ├── cache.py                  ← Redis async client
│   │   ├── celery_app.py             ← Celery init, queues, beat schedule
│   │   └── logging.py                ← structured JSON logger
│   ├── api/
│   │   ├── deps.py                   ← get_current_user (Clerk), get_db, get_cache
│   │   └── routes/
│   │       ├── webhooks.py           ← Clerk webhook handler
│   │       ├── data.py
│   │       ├── chat.py
│   │       ├── reports.py
│   │       ├── analysis.py
│   │       ├── feedback.py
│   │       └── analytics.py
│   ├── services/
│   │   ├── rag/
│   │   │   ├── pipeline.py           ← BasicRAGPipeline (Phase 3)
│   │   │   ├── advanced_pipeline.py  ← AdvancedRAGPipeline (Phase 5)
│   │   │   ├── crag.py               ← self-correction (Phase 11)
│   │   │   ├── retriever.py          ← HybridRetriever (BM25 + Qdrant)
│   │   │   ├── reranker.py           ← Cohere cross-encoder
│   │   │   ├── query_rewriter.py     ← Phase 5B
│   │   │   ├── multi_query.py        ← Phase 5B
│   │   │   └── compressor.py         ← context compression, Phase 5B
│   │   ├── sql/
│   │   │   ├── agent.py              ← LangGraph SQL Agent
│   │   │   ├── executor.py           ← DuckDB multi-source federation
│   │   │   └── validator.py          ← SQL safety checker
│   │   ├── agents/
│   │   │   ├── planner.py            ← LangGraph Planner (Phase 7)
│   │   │   └── health_agent.py       ← Data Health Agent
│   │   ├── ingestion/
│   │   │   ├── extractor.py          ← tiered PDF + TXT + URL extractors
│   │   │   ├── chunker.py            ← recursive split (pure Python)
│   │   │   ├── embedder.py           ← batch embed via openai SDK
│   │   │   ├── classifier.py         ← doc_type auto-classification
│   │   │   ├── live_db.py            ← Fernet encrypt + DuckDB ATTACH
│   │   │   └── versioning.py         ← source version management (Phase 11)
│   │   ├── memory/
│   │   │   ├── conversation_memory.py← sliding window + summarisation
│   │   │   ├── episodic_memory.py    ← searchable query history (Phase 7)
│   │   │   └── user_memory.py        ← preference profiles in Redis (Phase 7)
│   │   ├── guardrails/
│   │   │   ├── input_guard.py        ← injection detection + PII masking
│   │   │   ├── sql_guard.py          ← SQL safety rules
│   │   │   ├── cost_guard.py         ← per-tenant daily spend limit
│   │   │   ├── rate_limiter.py       ← per-user query limits (Phase 11)
│   │   │   └── output_guard.py       ← PII masking on output
│   │   ├── cache/
│   │   │   └── semantic_cache.py     ← exact + semantic + SQL result cache
│   │   ├── evaluation/
│   │   │   └── ragas_eval.py         ← background + CI/CD eval
│   │   └── reports/
│   │       ├── generator.py          ← WeasyPrint PDF
│   │       └── emailer.py            ← AWS SES
│   ├── providers/
│   │   └── llm_factory.py            ← litellm routing + Langfuse integration
│   ├── repositories/                 ← one file per model, ALL DB access here
│   ├── workers/                      ← Celery tasks (ingestion, eval, reports)
│   ├── models/                       ← SQLAlchemy models
│   └── tests/
│       ├── test_sql_agent.py
│       ├── test_rag_pipeline.py
│       ├── test_ingestion.py
│       ├── test_guardrails.py
│       └── test_auth.py
│
└── infra/
    ├── Dockerfile.backend
    ├── Dockerfile.worker
    └── aws/
        ├── ecs-task-definitions/
        ├── cloudwatch-alarms.py
        ├── cloudwatch-dashboard.json
        └── lambda-report-trigger.py
```

---

## 16. Out of Scope v1

| Feature | Reason |
|---------|--------|
| LangChain / wrappers | Direct SDKs cleaner, more debuggable |
| GraphRAG | Research-level complexity, wrong problem for this user base |
| Webhooks (outbound) | B2B API feature — not needed for current client use cases |
| Domain partitioning (separate Qdrant collections) | Problem at 1M+ chunks/org; metadata filter sufficient now |
| Offline embedding precompute | Needed at enterprise scale; online batch embedding sufficient |
| Kafka | SQS handles queue requirements; Kafka adds operational complexity |
| Kubernetes | ECS Fargate sufficient; K8s adds significant operational overhead |
| RBAC beyond member/admin | Single role system sufficient for v1 |
| Multi-region | Single region sufficient for portfolio + early production |
| Mobile app | Web-first |
| NoSQL support | Out of scope; SQL DBs cover all current client needs |

---

## 17. Resume Deliverables

| Deliverable | Details |
|-------------|---------|
| ✅ Live application | Public URL on custom domain with HTTPS |
| ✅ GitHub repository | Clean commits, BMAD stories, architecture docs |
| ✅ RAGAS eval report | Faithfulness + precision scores, before/after comparison |
| ✅ Architecture diagram | System design drawing for interviews |
| ✅ Cost analysis | Per-query cost breakdown by feature |
| ✅ CI/CD eval gate | Screenshot of failing + passing PR |
| ✅ Langfuse dashboard | Screenshot of real production traces |
| ✅ Video demo | 3-minute walkthrough of all 3 use cases |

### Resume Bullet

> *"Built SynapseIQ — a multi-tenant AI service platform for agency clients, featuring LangGraph SQL Agent with HITL approval and Postgres state persistence, hybrid RAG (BM25 + Qdrant + Cohere reranking, RAGAS faithfulness 0.84), self-correction RAG with web search fallback, DuckDB multi-source federation, event-driven ingestion (SQS + Celery chain with batched embedding), semantic caching (Redis, 35% hit rate), Clerk multi-tenant auth, Langfuse LLM observability, automated RAGAS eval gate in CI/CD, and PDF report scheduling via AWS Lambda + SES — deployed on AWS ECS with zero-downtime blue/green deployments."*

### The Interview Story (2 minutes)

> *"At our agency, we were building custom applications for multiple clients. Two clients asked for AI features — a chatbot that knew their product docs, and a way for managers to query their database in plain English.*
>
> *When the second client asked for the same thing, we realised we were about to duplicate the entire AI layer for every new client. So instead of embedding it in each client's codebase, we built a centralised AI service that any client could call via API. Each client gets full data isolation — their own S3 bucket, vector collection, database credentials — configured via environment variables. One codebase, N clients, zero duplication.*
>
> *The service solves three problems: customer onboarding via RAG over product docs, internal team knowledge management, and manager analytics via SQL Agent.*
>
> *For the SQL Agent, we used LangGraph with a Postgres checkpointer. Managers can start complex multi-table queries, close their laptop, come back the next day — the agent resumes from exactly where it stopped. Before executing any SQL, the agent pauses for human approval — a hard requirement from clients who didn't want AI running arbitrary queries on production databases.*
>
> *For RAG, we implemented hybrid search combining BM25 and Qdrant dense vectors with RRF fusion, Cohere reranking, and context compression. We measure quality with RAGAS — faithfulness 0.84 on our golden dataset — and block any PR that regresses below 0.75 in CI/CD.*
>
> *We use Clerk for auth, Langfuse for LLM observability, and deployed on AWS ECS with zero-downtime blue/green deployments."*

---

*SynapseIQ PRD v4.0 | Multi-tenant AI Service Platform*
*Clerk · LangGraph · Qdrant · Langfuse · No LangChain*
*Claude Code + BMAD + Graphify · 12 Phases · 14–16 weeks*
