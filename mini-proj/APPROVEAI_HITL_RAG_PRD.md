# ✋ ApproveAI — RAG Agent with Human-in-the-Loop
> **Type:** Minor Project · Phase 4 Capstone (HITL + LangGraph)
> **Stack:** FastAPI · LangGraph · Redis · PostgreSQL · Clerk · React 18 · Langfuse
> **Timeline:** 2–3 weeks (25–30 hrs)
> **Teaches:** `interrupt_before` / `interrupt_after` · Async approval queues · Agent state persistence · HITL UI patterns

---

## 📑 Table of Contents

1. [The Idea](#1-the-idea)
2. [Why This Project](#2-why-this-project)
3. [What It Does](#3-what-it-does)
4. [System Architecture](#4-system-architecture)
5. [LangGraph HITL Design](#5-langgraph-hitl-design)
6. [HITL Patterns Implemented](#6-hitl-patterns-implemented)
7. [Core Code](#7-core-code)
8. [API Reference](#8-api-reference)
9. [Database Schema](#9-database-schema)
10. [Folder Structure](#10-folder-structure)
11. [Phase-by-Phase Build Plan](#11-phase-by-phase-build-plan)
12. [Resume Deliverables](#12-resume-deliverables)

---

## 1. The Idea

**ApproveAI** is a customer support RAG agent that handles most queries automatically but pauses for human approval on high-stakes actions — refunds above ₹5,000, account deletions, policy exceptions, or any response the agent itself flags as uncertain.

```
80% of queries: agent answers → user gets response (fully automated)
15% of queries: agent drafts answer → human reviews → approves/edits → sent
 5% of queries: agent escalates → human takes over entirely

The human reviewer sees:
  • The customer's original question
  • What context was retrieved (which chunks, from which docs)
  • The agent's proposed response
  • The agent's confidence score
  • One-click: Approve / Edit / Reject / Escalate
```

This mirrors exactly how production AI systems at real companies work. No serious company ships a fully autonomous agent for consequential actions without a human approval layer.

---

## 2. Why This Project

```
WHAT MOST CANDIDATES BUILD:
  A chatbot that always responds automatically.
  No concept of uncertainty, escalation, or human review.
  "Trust the LLM 100%" is not production thinking.

WHAT THIS PROJECT SHOWS:
  You understand that agents make mistakes on edge cases.
  You know how to design for graceful human handoff.
  You've implemented LangGraph HITL with interrupt_before/after.
  You've built an async approval queue (not just blocking HTTP).
  You've thought about what triggers human review vs auto-approve.

INTERVIEW TALKING POINT:
  "I built a RAG support agent where the agent itself can flag uncertainty
   and suspend execution pending human review. The reviewer sees a full
   diff-style view: the retrieved context, the proposed answer, and the
   agent's reasoning. They can approve, edit, or reject. This pattern
   is how you safely deploy AI in high-stakes support contexts."
```

---

## 3. What It Does

```
CUSTOMER FLOW:
  Customer asks a question via chat widget
  Agent retrieves relevant docs from Qdrant (RAG)
  Agent generates a response

  IF confidence > 0.85 AND no high-stakes action:
    → Response sent immediately (< 2 seconds)

  IF confidence < 0.85 OR high-stakes action detected:
    → Agent suspends
    → "Hang on, let me confirm the details and get back to you."
    → Review request appears in human dashboard
    → Human reviews: approve / edit / reject
    → Agent resumes with human's decision
    → Response sent to customer

HUMAN REVIEWER DASHBOARD:
  Pending reviews queue (sorted by: age, priority, customer tier)
  Per-review view:
    Customer question (with conversation history)
    Retrieved chunks (highlighted relevance score)
    Agent's proposed response
    Agent's reasoning + confidence score
    [Approve] [Edit & Approve] [Reject + Custom Response] [Escalate to Human]
  Metrics: avg review time, approval rate, override rate

TRIGGERS FOR HUMAN REVIEW:
  1. Confidence < 0.85 (agent flags own uncertainty)
  2. Refund amount > ₹5,000 mentioned in query
  3. Account deletion / data erasure request
  4. Legal / compliance topic detected (keyword guard)
  5. Customer tier = VIP (always reviewed, no exceptions)
  6. Customer is angry (sentiment < -0.6)
  7. Previous interaction in same session was escalated
```

---

## 4. System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         ApproveAI                                   │
│                                                                     │
│  Customer Chat             Human Dashboard                          │
│  (React Widget)            (React Admin, Clerk auth)                │
│       │                           │                                 │
│  ┌────▼───────────────────────────▼────────────────────────────┐   │
│  │                    FastAPI Backend                          │   │
│  │                                                             │   │
│  │  POST /chat           GET  /reviews/pending                 │   │
│  │  GET  /chat/{id}      POST /reviews/{id}/approve            │   │
│  │  WS   /chat/stream    POST /reviews/{id}/reject             │   │
│  │                       POST /reviews/{id}/edit               │   │
│  └──────────────────┬────────────────────────────────────────┘   │
│                     │                                              │
│  ┌──────────────────▼────────────────────────────────────────┐   │
│  │              LangGraph Agent (interrupt-capable)           │   │
│  │                                                            │   │
│  │  retrieve_node → evaluate_node → generate_node            │   │
│  │        ↓ (if HITL needed)                                 │   │
│  │  interrupt_before("send_response")                        │   │
│  │        ↓                                                   │   │
│  │  [SUSPENDED — waiting for human input]                    │   │
│  │        ↓ (human approves/edits)                           │   │
│  │  resume → send_response_node                              │   │
│  └────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐             │
│  │ Qdrant   │ │  Redis   │ │Postgres  │ │ Langfuse │             │
│  │(RAG docs)│ │(sessions)│ │(reviews) │ │(traces)  │             │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘             │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 5. LangGraph HITL Design

```python
# app/agent/graph.py

from typing import TypedDict, Annotated, Literal
from operator import add
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.redis import RedisSaver
from langgraph.types import interrupt

class SupportState(TypedDict):
    # ── Conversation ──────────────────────────────────────────────
    session_id:      str
    customer_id:     str
    customer_tier:   Literal["free", "starter", "pro", "vip"]
    messages:        Annotated[list[dict], add]

    # ── RAG ───────────────────────────────────────────────────────
    retrieved_chunks:  list[dict]    # [{content, score, source}]
    context_quality:   float         # avg chunk relevance score

    # ── Agent reasoning ───────────────────────────────────────────
    draft_response:    str | None
    confidence:        float          # 0.0 – 1.0
    reasoning:         str | None     # why agent said this
    detected_intents:  list[str]      # ["refund_request", "angry", ...]

    # ── HITL control ─────────────────────────────────────────────
    requires_review:   bool
    review_id:         str | None
    review_reason:     str | None     # why it needs review
    human_decision:    str | None     # "approved" | "edited" | "rejected"
    human_response:    str | None     # human's edited response (if edited)
    reviewed_by:       str | None     # reviewer user_id

    # ── Output ────────────────────────────────────────────────────
    final_response:    str | None
    escalated:         bool

def build_support_graph(redis_client) -> "CompiledGraph":
    builder = StateGraph(SupportState)

    builder.add_node("retrieve",        retrieve_node)
    builder.add_node("evaluate",        evaluate_node)       # decides: HITL or not
    builder.add_node("generate",        generate_node)
    builder.add_node("review_gate",     review_gate_node)    # interrupt point
    builder.add_node("send_response",   send_response_node)
    builder.add_node("escalate",        escalate_node)

    builder.set_entry_point("retrieve")
    builder.add_edge("retrieve", "evaluate")

    builder.add_conditional_edges(
        "evaluate",
        route_after_evaluate,
        {
            "generate":  "generate",    # high confidence, low stakes
            "escalate":  "escalate",    # definitely can't handle
        },
    )

    builder.add_edge("generate", "review_gate")

    builder.add_conditional_edges(
        "review_gate",
        lambda s: "wait" if s["requires_review"] else "send",
        {
            "wait": END,               # agent suspends here (HITL pending)
            "send": "send_response",   # auto-approve, send immediately
        },
    )

    builder.add_edge("send_response", END)
    builder.add_edge("escalate", END)

    return builder.compile(
        checkpointer=RedisSaver(redis_client),
        interrupt_before=["send_response"],  # always pause before sending
        # In auto-approve mode: review_gate removes from interrupted set
    )
```

---

## 6. HITL Patterns Implemented

### Pattern 1 — `interrupt_before` (Pre-send Gate)

```python
# app/agent/nodes/review_gate.py

async def review_gate_node(state: SupportState) -> dict:
    """
    Decides whether agent can auto-send or must wait for human.
    If human review needed: creates a review record and sets requires_review=True.
    LangGraph then sees interrupt_before=["send_response"] → suspends.
    """
    needs_review, reason = await should_require_review(state)

    if not needs_review:
        return {"requires_review": False}

    # Create review record in Postgres
    review_id = await ReviewRepository.create({
        "session_id":       state["session_id"],
        "customer_id":      state["customer_id"],
        "customer_tier":    state["customer_tier"],
        "question":         state["messages"][-1]["content"],
        "draft_response":   state["draft_response"],
        "retrieved_chunks": state["retrieved_chunks"],
        "confidence":       state["confidence"],
        "reasoning":        state["reasoning"],
        "review_reason":    reason,
        "status":           "pending",
    })

    # Notify reviewers (Slack/email in production)
    await notify_reviewers(review_id, reason, state["customer_tier"])

    return {
        "requires_review": True,
        "review_id":       review_id,
        "review_reason":   reason,
    }

async def should_require_review(state: SupportState) -> tuple[bool, str]:
    """Returns (needs_review, reason_string)"""
    reasons = []

    # Rule 1: Agent confidence below threshold
    if state["confidence"] < 0.85:
        reasons.append(f"low_confidence:{state['confidence']:.2f}")

    # Rule 2: High-stakes intents detected
    HIGH_STAKES = ["refund_request", "account_deletion", "legal_complaint",
                   "data_erasure", "policy_exception"]
    flagged = [i for i in state["detected_intents"] if i in HIGH_STAKES]
    if flagged:
        reasons.append(f"high_stakes:{','.join(flagged)}")

    # Rule 3: VIP customer — always reviewed
    if state["customer_tier"] == "vip":
        reasons.append("vip_customer")

    # Rule 4: Poor retrieval quality
    if state["context_quality"] < 0.60:
        reasons.append(f"weak_context:{state['context_quality']:.2f}")

    # Rule 5: Angry customer
    anger_signals = ["legal", "fraud", "consumer court", "police",
                     "pathetic", "worst", "useless"]
    last_msg = state["messages"][-1]["content"].lower()
    if any(s in last_msg for s in anger_signals):
        reasons.append("angry_customer")

    if reasons:
        return True, " | ".join(reasons)
    return False, ""
```

### Pattern 2 — Resume After Human Decision

```python
# app/api/reviews.py

@router.post("/reviews/{review_id}/approve")
async def approve_review(
    review_id:    str,
    current_user: AuthUser = Depends(get_current_user),
):
    """Human approves agent's draft response — resume the graph"""
    review = await ReviewRepository.get(review_id)
    if not review or review["status"] != "pending":
        raise HTTPException(404, "Review not found or already processed")

    # Update review record
    await ReviewRepository.update(review_id, {
        "status":        "approved",
        "reviewed_by":   current_user.user_id,
        "reviewed_at":   datetime.utcnow(),
        "human_decision":"approved",
    })

    # Resume the LangGraph graph with human's decision
    config = {"configurable": {"thread_id": review["session_id"]}}
    await graph.ainvoke(
        {
            "human_decision": "approved",
            "human_response": None,           # use agent's draft as-is
            "reviewed_by":    current_user.user_id,
        },
        config=config,
    )
    return {"status": "approved", "review_id": review_id}

@router.post("/reviews/{review_id}/edit")
async def edit_and_approve(
    review_id:      str,
    body:           EditReviewRequest,   # {edited_response: str}
    current_user:   AuthUser = Depends(get_current_user),
):
    """Human edits the draft and approves the edited version"""
    await ReviewRepository.update(review_id, {
        "status":         "edited",
        "reviewed_by":    current_user.user_id,
        "human_decision": "edited",
        "human_response": body.edited_response,
    })

    config = {"configurable": {"thread_id": (await ReviewRepository.get(review_id))["session_id"]}}
    await graph.ainvoke(
        {
            "human_decision": "edited",
            "human_response": body.edited_response,
            "reviewed_by":    current_user.user_id,
        },
        config=config,
    )
    return {"status": "edited", "review_id": review_id}

@router.post("/reviews/{review_id}/reject")
async def reject_review(
    review_id:    str,
    body:         RejectReviewRequest,  # {reason: str, custom_response: str}
    current_user: AuthUser = Depends(get_current_user),
):
    """Human rejects agent's response and provides their own"""
    await ReviewRepository.update(review_id, {
        "status":          "rejected",
        "reviewed_by":     current_user.user_id,
        "human_decision":  "rejected",
        "human_response":  body.custom_response,
        "reject_reason":   body.reason,
    })

    config = {"configurable": {"thread_id": (await ReviewRepository.get(review_id))["session_id"]}}
    await graph.ainvoke(
        {
            "human_decision": "rejected",
            "human_response": body.custom_response,
            "reviewed_by":    current_user.user_id,
        },
        config=config,
    )
    return {"status": "rejected", "review_id": review_id}
```

### Pattern 3 — `send_response_node` Handles All Outcomes

```python
# app/agent/nodes/send_response.py

async def send_response_node(state: SupportState) -> dict:
    """
    Called after review_gate clears OR after human decision arrives.
    Selects the right response based on human_decision.
    """
    decision  = state.get("human_decision")
    response  = state.get("human_response") or state.get("draft_response")

    # Always log the outcome
    await ConversationRepository.append_message(
        session_id=state["session_id"],
        role="assistant",
        content=response,
        metadata={
            "auto_approved": decision is None,
            "human_decision": decision,
            "reviewed_by":   state.get("reviewed_by"),
            "confidence":    state["confidence"],
            "review_id":     state.get("review_id"),
        },
    )

    # Track in Langfuse
    langfuse_context.score_current_trace(
        name="human_override", value=1.0 if decision == "rejected" else 0.0
    )
    langfuse_context.score_current_trace(
        name="confidence", value=state["confidence"]
    )

    return {"final_response": response}
```

---

## 7. Core Code

### Retrieve Node

```python
# app/agent/nodes/retrieve.py

from langfuse.decorators import observe

@observe(name="retrieve")
async def retrieve_node(state: SupportState) -> dict:
    query       = state["messages"][-1]["content"]
    org_id      = "support_kb"   # single shared KB for this project

    # Hybrid search: dense + sparse
    results     = await qdrant.search(
        collection_name="support_docs",
        query_vector=await embedder.embed([query])[0],
        query_filter={"must": [{"key": "is_active", "match": {"value": True}}]},
        limit=5,
        with_payload=True,
    )

    chunks = [{"content": r.payload["content"],
                "score":   round(r.score, 3),
                "source":  r.payload["source"],
                "page":    r.payload.get("page")} for r in results]
    avg_score = sum(c["score"] for c in chunks) / max(len(chunks), 1)

    return {
        "retrieved_chunks":  chunks,
        "context_quality":   round(avg_score, 3),
    }
```

### Evaluate Node

```python
# app/agent/nodes/evaluate.py

from pydantic import BaseModel
from typing import Literal
import instructor

class QueryEvaluation(BaseModel):
    intents:      list[Literal[
        "general_qa", "refund_request", "order_status", "return_request",
        "account_deletion", "legal_complaint", "data_erasure",
        "policy_exception", "angry", "compliment",
    ]]
    can_handle:   bool     # can this query be answered from context?
    confidence:   float    # 0.0 – 1.0 how confident the agent is
    reasoning:    str      # one sentence: why this confidence level?

groq_instructor = instructor.from_groq(AsyncGroq())

@observe(name="evaluate")
async def evaluate_node(state: SupportState) -> dict:
    query   = state["messages"][-1]["content"]
    context = "\n\n".join(f"[{c['score']:.2f}] {c['content']}" for c in state["retrieved_chunks"])

    evaluation: QueryEvaluation = await groq_instructor.chat.completions.create(
        model="llama-3.1-8b-instant",   # cheap model for classification
        response_model=QueryEvaluation,
        messages=[{
            "role": "user",
            "content": f"""Evaluate whether the context is sufficient to answer this query.

Query: {query}
Retrieved context:
{context}

Assess: what is the user asking? Can we answer from the context?
Confidence = 1.0 if context has a direct, clear answer.
Confidence = 0.5 if context has partial info.
Confidence = 0.2 if context has nothing useful.""",
        }],
    )

    return {
        "detected_intents": evaluation.intents,
        "confidence":       evaluation.confidence,
        "reasoning":        evaluation.reasoning,
    }

def route_after_evaluate(state: SupportState) -> str:
    if not state.get("retrieved_chunks"):
        return "escalate"   # no context at all
    if "legal_complaint" in state["detected_intents"]:
        return "escalate"   # always escalate legal
    return "generate"
```

### Generate Node

```python
# app/agent/nodes/generate.py

SUPPORT_SYSTEM_PROMPT = """You are a helpful customer support agent.
Answer based ONLY on the context below.
If the answer is not in the context: say "I need to check this with our team."
Cite your source: [document name].
Keep answers under 150 words.
Be warm and professional."""

@observe(as_type="generation", name="generate")
async def generate_node(state: SupportState) -> dict:
    context  = "\n\n".join(f"[{c['source']}]\n{c['content']}" for c in state["retrieved_chunks"])
    messages = [
        {"role": "system", "content": SUPPORT_SYSTEM_PROMPT.format()},
        *state["messages"][:-1],  # conversation history
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {state['messages'][-1]['content']}"},
    ]

    response = await groq.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        max_tokens=300,
        temperature=0.2,
    )
    return {"draft_response": response.choices[0].message.content}
```

---

## 8. API Reference

```
── CUSTOMER CHAT ─────────────────────────────────────────────────────
POST   /chat                   Start or continue conversation
       Body: {session_id, customer_id, customer_tier, message}
       Response: {session_id, status: "answered"|"under_review", response?, review_eta?}

GET    /chat/{session_id}      Get conversation + current status
WS     /chat/{session_id}/ws  Real-time status updates (SSE alternative)

── REVIEW QUEUE (human dashboard) ───────────────────────────────────
GET    /reviews/pending         All pending reviews (paginated, sorted)
GET    /reviews/{review_id}     Full review detail: question, context, draft
POST   /reviews/{review_id}/approve   Approve draft as-is
POST   /reviews/{review_id}/edit      Approve with edits {edited_response}
POST   /reviews/{review_id}/reject    Reject + custom {reason, custom_response}
POST   /reviews/{review_id}/escalate  Mark as needing senior human agent

── ANALYTICS ─────────────────────────────────────────────────────────
GET    /analytics/review-queue  Avg wait time, pending count, SLA breach rate
GET    /analytics/agent         Auto-approval rate, confidence distribution
GET    /analytics/overrides     Override rate by reason, reviewer patterns
```

---

## 9. Database Schema

```sql
CREATE TABLE sessions (
    id              UUID PRIMARY KEY,
    customer_id     TEXT NOT NULL,
    customer_tier   TEXT NOT NULL,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE messages (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id  UUID REFERENCES sessions(id),
    role        TEXT NOT NULL,         -- user | assistant | system
    content     TEXT NOT NULL,
    metadata    JSONB,                 -- {auto_approved, confidence, review_id, ...}
    created_at  TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX ON messages (session_id);

CREATE TABLE reviews (
    id               UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id       UUID REFERENCES sessions(id),
    customer_id      TEXT NOT NULL,
    customer_tier    TEXT NOT NULL,
    question         TEXT NOT NULL,
    draft_response   TEXT NOT NULL,
    retrieved_chunks JSONB NOT NULL,   -- chunks + scores used
    confidence       FLOAT NOT NULL,
    reasoning        TEXT,
    review_reason    TEXT NOT NULL,    -- why HITL triggered
    status           TEXT DEFAULT 'pending',  -- pending|approved|edited|rejected|escalated
    human_decision   TEXT,
    human_response   TEXT,
    reject_reason    TEXT,
    reviewed_by      TEXT,             -- Clerk user_id
    reviewed_at      TIMESTAMPTZ,
    sla_deadline     TIMESTAMPTZ,      -- 30 min for VIP, 4 hr for others
    created_at       TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX ON reviews (status, created_at);
CREATE INDEX ON reviews (customer_tier, status);
```

---

## 10. Folder Structure

```
approveai/
├── app/
│   ├── main.py
│   ├── agent/
│   │   ├── graph.py              ← StateGraph, interrupt config
│   │   ├── state.py              ← SupportState TypedDict
│   │   └── nodes/
│   │       ├── retrieve.py
│   │       ├── evaluate.py       ← intent detection, confidence score
│   │       ├── generate.py
│   │       ├── review_gate.py    ← HITL trigger logic
│   │       ├── send_response.py  ← post-approval handler
│   │       └── escalate.py
│   ├── api/
│   │   ├── chat.py               ← customer endpoints
│   │   └── reviews.py            ← approve/edit/reject
│   ├── services/
│   │   ├── embedder.py
│   │   └── notifier.py           ← Slack/email alert on new review
│   └── repositories/
│       ├── review_repo.py
│       └── conversation_repo.py
├── frontend/
│   ├── customer/                 ← chat widget (React)
│   └── dashboard/                ← review queue UI (React + Clerk)
│       └── src/pages/
│           ├── ReviewQueue.tsx   ← pending reviews list
│           └── ReviewDetail.tsx  ← approve/edit/reject UI
├── tests/
│   ├── test_graph.py             ← graph runs, interrupt fires correctly
│   ├── test_hitl_triggers.py     ← all 5 trigger conditions tested
│   └── test_resume.py            ← agent resumes correctly after approval
├── docker-compose.yml
└── README.md
```

---

## 11. Phase-by-Phase Build Plan

### Phase 0 — Skeleton (2 days)
```
[ ] FastAPI + LangGraph + Redis checkpointer wired up
[ ] SupportState TypedDict defined
[ ] POST /chat returns stub response
[ ] Postgres tables created (sessions, messages, reviews)
Showable: POST /chat → {session_id, status: "answered", response: "stub"}
```

### Phase 1 — RAG Pipeline (3 days)
```
[ ] Qdrant collection with 20+ support docs indexed
[ ] retrieve_node: hybrid search, returns chunks + avg_score
[ ] evaluate_node: intent detection + confidence (Groq 8b)
[ ] generate_node: response from context (Groq 70b)
[ ] Auto-approve path: full pipeline, response returned in < 2s
Showable: "What is the return policy?" → correct answer in < 2s
```

### Phase 2 — HITL (4 days)
```
[ ] review_gate_node: all 5 trigger conditions
[ ] interrupt_before("send_response") in graph
[ ] ReviewRepository: create/get/update
[ ] POST /chat: returns {status: "under_review", review_eta: "~5 min"}
[ ] GET /reviews/pending: list of pending reviews
[ ] POST /reviews/{id}/approve: resumes graph
[ ] POST /reviews/{id}/edit: resumes with edited response
[ ] POST /reviews/{id}/reject: resumes with custom response
Showable:
  "I want to delete my account" → status: under_review
  Human opens dashboard → sees the review → approves
  Customer receives response
```

### Phase 3 — Dashboard UI (3 days)
```
[ ] Review queue page: pending reviews sorted by priority
[ ] Review detail: question + retrieved chunks + draft + confidence
[ ] [Approve] [Edit] [Reject] [Escalate] buttons
[ ] Real-time: new review appears without page refresh (WebSocket)
[ ] Analytics: auto-approval rate, avg review time, override rate
Showable: full end-to-end demo with human reviewer in the loop
```

### Phase 4 — Observability + Polish (2 days)
```
[ ] Langfuse: trace every LLM call, tag with review_id
[ ] SLA tracking: VIP reviews flagged if pending > 30 min
[ ] Slack notification on new high-priority review
[ ] Eval suite: 20 queries tested, correct HITL triggers verified
Showable: Langfuse dashboard showing auto vs human approval distribution
```

---

## 12. Resume Deliverables

```
GitHub repo:     approveai — with architecture diagram + demo GIF
Live demo:       Railway or Fly.io (free tier friendly)
Key numbers:     80% auto-approved, 15% human-reviewed, 5% escalated
               avg review time: 4 minutes (SLA: 30 min VIP, 4 hr free)

Interview story (60 seconds):
"I built a support RAG agent that handles most queries automatically but
 suspends execution for human review on high-stakes actions — refunds,
 account deletions, or when its own confidence drops below 85%.

 The key technical piece is LangGraph's interrupt_before mechanism.
 The graph hits a gate node, evaluates whether human review is needed,
 and if so, creates a review record and suspends. The agent's state is
 persisted in Redis via the checkpointer. When a human approves or edits
 via the dashboard, we call graph.ainvoke() again with the human's
 decision, and the graph resumes from exactly where it paused.

 Every approval or rejection trains us — override rate by reason tells
 us where the agent is systematically wrong. Over 2 weeks of use,
 we added those failure cases to the RAG knowledge base and the
 override rate on 'refund policy' queries dropped from 40% to 8%."
```

---

*ApproveAI — RAG + HITL Minor Project · Phase 4 Capstone*
*LangGraph · interrupt_before · Redis Checkpointer · Async Approval Queue*
