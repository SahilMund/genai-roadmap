# VoiceIQ — AI Voice Agent Platform
> **Type:** Production Voice AI System — Revenue Recovery + Customer Support
> **Stack:** FastAPI · Twilio · Pipecat · LangGraph · Deepgram · OpenAI · Redis · Postgres
> **Based on:** ai_hr prototype (Twilio + Pipecat + LangGraph architecture)
> **Timeline:** 14–16 weeks

---

## 📑 Table of Contents

1. [The Origin Story](#1-the-origin-story)
2. [What This Platform Does](#2-what-this-platform-does)
3. [The Two Agents](#3-the-two-agents)
4. [Architecture Overview](#4-architecture-overview)
5. [Technology Stack](#5-technology-stack)
6. [Core Concepts — Voice AI Specific](#6-core-concepts--voice-ai-specific)
7. [LangGraph Design — Both Agents](#7-langgraph-design--both-agents)
8. [Agent 1 — Revenue Recovery Agent (RRO)](#8-agent-1--revenue-recovery-agent-rro)
9. [Agent 2 — Customer Support Agent (CSA)](#9-agent-2--customer-support-agent-csa)
10. [Shared Infrastructure](#10-shared-infrastructure)
11. [Database Schema](#11-database-schema)
12. [API Reference](#12-api-reference)
13. [Business Rules — Both Agents](#13-business-rules--both-agents)
14. [Guardrails](#14-guardrails)
15. [Observability](#15-observability)
16. [Phase-by-Phase Build Plan](#16-phase-by-phase-build-plan)
17. [Folder Structure](#17-folder-structure)
18. [Resume Deliverables](#18-resume-deliverables)

---

## 1. The Origin Story

### The Problem

E-commerce companies lose 70% of revenue at checkout. A ₹10,000 cart abandoned is not just a lost sale — it's a customer who was interested, who chose your product, but dropped off at the last moment. Email recovery gets 3% open rates after 24 hours. WhatsApp bots feel impersonal. Nothing works as well as a real human calling within 20 minutes.

But human agents are expensive, inconsistent, and can't scale to 1,000 abandoned carts overnight.

The second problem: when customers DO want to buy, ask questions, or need support — they call. Most companies have a menu system from 2010. Customers wait. Customers leave. Revenue leaks in both directions.

### The Solution

VoiceIQ is a voice AI platform with two agents running on the same infrastructure:

**Agent 1 — Revenue Recovery Agent (RRO)**
Calls customers who abandoned checkout. Understands WHY they dropped off. Offers solutions — discounts, fresh payment links, WhatsApp follow-ups — in a natural, intelligent conversation. Stops if the customer says no.

**Agent 2 — Customer Support Agent (CSA)**
Answers inbound customer calls. Checks order status, stock availability, product questions, processes orders, handles returns. A complete commerce assistant over the phone.

One platform. Two directions of value.

---

## 2. What This Platform Does

```
OUTBOUND (Revenue Recovery Agent):
  Trigger:    customer abandons checkout
  Action:     AI calls them within 15 minutes
  Goal:       understand objection → resolve → recover sale
  
  Handles:
    price concerns     → controlled discount offer
    trust concerns     → social proof + guarantee messaging  
    technical failure  → fresh payment link
    confusion          → clear explanation + hand-holding
    competitor mention → comparative value messaging
    payment failure    → alternative payment method
    DND / no          → graceful exit, log for suppression

INBOUND (Customer Support Agent):
  Trigger:    customer calls the support line
  Action:     AI picks up, understands need, resolves it
  Goal:       resolve without human escalation
  
  Handles:
    order status       → real-time lookup + status update
    stock check        → inventory query by SKU/name
    place order        → guided order placement over phone
    return/refund      → initiate return flow
    product Q&A        → RAG over product catalog
    complaint          → empathetic handling + ticket creation
    escalation needed  → warm transfer to human agent
```

---

## 3. The Two Agents

### Side-by-Side Comparison

```
                        RRO Agent              CSA Agent
                        ─────────────          ─────────────
Direction               Outbound               Inbound
Trigger                 Abandoned checkout     Customer dials
Initiator               Your system            Customer
Conversation goal       Recover revenue        Resolve support need
Tone                    Warm, persuasive       Helpful, efficient
Primary tools           Coupon, PayLink, WA    OrderDB, StockDB, CRM
Max call duration       5 minutes              10 minutes
Key guardrail           Discount cap, DND      Never confirm wrong order
LangGraph graphs        preflight + turn       greet + turn
Escalation              End gracefully         Transfer to human
```

---

## 4. Architecture Overview

```
                    ┌──────────────────────────────────────────┐
                    │           VoiceIQ Platform               │
                    │                                          │
    Inbound call ──→│  Twilio Voice ──→ TwiML Router           │
    Outbound API ──→│  POST /calls/outbound                    │
                    │         ↓                                │
                    │  FastAPI WebSocket Handler               │
                    │  /twilio/media (WS)                     │
                    │         ↓                                │
                    │  Pipecat Pipeline                        │
                    │  ┌───────────────────────────────────┐   │
                    │  │ Twilio Transport                  │   │
                    │  │ → Deepgram STT (real-time)        │   │
                    │  │ → VAD (Voice Activity Detection)   │   │
                    │  │ → LangGraph Router                │   │
                    │  │     ↓              ↓              │   │
                    │  │  RRO Graph      CSA Graph         │   │
                    │  │ → OpenAI LLM   → OpenAI LLM      │   │
                    │  │ → Tool Exec    → Tool Exec        │   │
                    │  │ → OpenAI TTS   → OpenAI TTS      │   │
                    │  │ → Audio out    → Audio out        │   │
                    │  └───────────────────────────────────┘   │
                    │         ↓                                │
                    │  Redis  (session state, DND, campaign)   │
                    │  Postgres (calls, orders, analytics)     │
                    │  Langfuse (LLM observability)            │
                    └──────────────────────────────────────────┘
                    
EXTERNAL SERVICES:
  Twilio:    outbound/inbound call management, Media Streams
  Deepgram:  real-time speech-to-text (streaming)
  OpenAI:    LLM reasoning + TTS voice synthesis
  WhatsApp:  Twilio Business API for recovery messages
  Payment:   Razorpay / Stripe for payment link generation
```

### Call Flow — Outbound (RRO)

```
Your API → POST /calls/outbound
  → Build CallState (user, product, cart_value, session_id)
  → Store in Redis
  → Twilio: calls.create(twiml_url, status_callback)
  → Twilio calls customer
  → Customer answers → Twilio POSTs /twilio/twiml
  → TwiML: <Connect><Stream url="wss://.../twilio/media">
  → WebSocket opens
  → Pipecat pipeline starts
  → LangGraph: preflight_graph → turn_graph (loop)
  → Agent speaks, listens, decides, acts
  → Call ends (agent or customer)
  → Status callback → POST /twilio/status
  → Outcome written to Postgres
```

### Call Flow — Inbound (CSA)

```
Customer dials support number
  → Twilio receives call
  → Twilio POSTs /twilio/inbound/twiml
  → TwiML: greet message + <Connect><Stream>
  → WebSocket opens /twilio/media
  → Pipecat pipeline starts
  → LangGraph: greet_graph → support_turn_graph (loop)
  → Agent identifies need, resolves, or escalates
  → Call ends
  → Outcome written to Postgres
```

---

## 5. Technology Stack

```
CORE VOICE PIPELINE:
  Pipecat          → real-time audio pipeline (STT → LLM → TTS in one loop)
  Twilio           → outbound/inbound call management + Media Streams
  Deepgram         → streaming STT (word-by-word, not batch)
  OpenAI TTS       → voice synthesis (alloy/nova/shimmer voices)
  OpenAI GPT-4o    → LLM reasoning for both agents

ORCHESTRATION:
  LangGraph        → agent state machine (preflight + turn graphs)
  FastAPI          → HTTP + WebSocket server
  asyncio          → full async pipeline (critical for low latency)

STORAGE:
  Redis            → session state, DND list, rate limits, campaign queue
  Postgres         → call logs, orders, outcomes, analytics
  
OBSERVABILITY:
  Langfuse         → LLM call tracing (prompt, tokens, cost, latency)
  Sentry           → error tracking
  CloudWatch       → infrastructure metrics

INTEGRATIONS:
  Razorpay/Stripe  → payment link generation
  Twilio WA API    → WhatsApp recovery messages
  Internal API     → order DB, stock DB, CRM

INFRA:
  Docker Compose   → local dev
  AWS ECS Fargate  → production
  AWS ElastiCache  → Redis production
  AWS RDS          → Postgres production
```

---

## 6. Core Concepts — Voice AI Specific

### 6.1 Why Voice AI is Different from Text AI

```
Text chatbot:
  User types → wait 3 seconds for LLM → user reads response
  Latency: doesn't matter much (user is reading)
  Errors: user re-reads, re-sends

Voice agent:
  User speaks → STT → LLM → TTS → user hears response
  Latency: CRITICAL. > 2 seconds = "is anyone there?"
  Errors: user can't re-hear, gets confused, hangs up

Implications for architecture:
  1. Streaming STT — start processing before user finishes speaking
  2. Streaming LLM — start TTS on first sentence, don't wait for full response
  3. Interruption handling — user can interrupt mid-sentence
  4. VAD (Voice Activity Detection) — know when user started/stopped speaking
  5. Filler words — "Let me check that for you..." while tool runs
```

### 6.2 Pipecat Pipeline — How It Works

```
Pipecat is a framework for real-time audio AI pipelines.
Think of it as a conveyor belt where audio flows through processors:

Audio In (Twilio WebSocket)
  ↓
AudioResampler (normalize sample rate)
  ↓
DeepgramSTTService (streaming transcription)
  ↓
LLMUserResponseAggregator (collect full user turn)
  ↓
OpenAILLMService (with tool use)
  ↓
LLMAssistantResponseAggregator (collect full response)
  ↓
OpenAITTSService (stream audio)
  ↓
Audio Out (Twilio WebSocket)

Each "service" is a Pipecat FrameProcessor.
Frames flow downstream: TextFrame, AudioRawFrame, LLMMessagesFrame, etc.
```

### 6.3 LangGraph in a Voice Context

```
Normal LangGraph: one complete run per user message
Voice LangGraph:  one complete run per USER TURN (could be multi-sentence)

Structure:
  preflight_graph → runs ONCE at call start (validation, opening)
  turn_graph      → runs for EACH user turn throughout the call

Key difference from text agents:
  Text: stateless, each session fresh
  Voice: stateful within call, state must persist across turns
  Storage: Redis (fast, call-scoped, TTL = call duration + 5 min)
```

### 6.4 VAD — Voice Activity Detection

```
VAD decides: "has the user finished speaking?"

Without VAD: agent speaks immediately → interrupts user mid-sentence
With VAD:
  User starts speaking → VAD fires (start event)
  User pauses > 700ms → VAD fires (stop event)
  Agent processes what was said

Settings:
  vad_silence_duration_ms: 700   # wait 700ms of silence before processing
  vad_min_speech_duration_ms: 200 # ignore sounds shorter than 200ms (coughs, noise)
```

### 6.5 Latency Budget — Target < 1.5 seconds

```
P2P latency target: customer speaks → agent replies in < 1.5 seconds

Budget breakdown:
  STT streaming:        0ms (starts processing as user speaks)
  STT word complete:    ~100ms (final word recognition)
  LangGraph routing:    ~50ms
  LLM first token:      ~400ms (Groq llama-3.3-70b fastest)
  TTS first audio:      ~150ms
  Network (WebSocket):  ~100ms
  ─────────────────────────────
  Total (fast path):    ~800ms  ✅

  LLM + tool call:      +500ms (tool execution)
  ─────────────────────────────
  Total (with tool):    ~1.3s   ✅ still under budget

Optimizations:
  Use Groq for LLM (fastest inference, ~400ms first token)
  Stream TTS (don't wait for full response)
  Filler phrases during tool execution ("One moment...")
  Pre-load product/user data at call start (preflight graph)
```

---

## 7. LangGraph Design — Both Agents

### 7.1 Shared State Structure

```python
# app/models/state.py

from typing import TypedDict, Annotated, Literal
from operator import add

class CallState(TypedDict):
    # ── Identity ──────────────────────────────────────────────────────
    session_id:      str
    call_sid:        str | None
    agent_type:      Literal["rro", "csa"]
    
    # ── Customer context ──────────────────────────────────────────────
    user_id:         str
    name:            str
    phone:           str
    user_type:       Literal["new", "returning", "vip", "low_intent"]
    
    # ── Product / order context ───────────────────────────────────────
    product_name:    str | None
    product_id:      str | None
    cart_value:      float | None
    order_id:        str | None
    
    # ── Conversation ─────────────────────────────────────────────────
    messages:        Annotated[list[dict], add]   # full conversation history
    current_turn:    int
    
    # ── Agent decisions ───────────────────────────────────────────────
    detected_intent: str | None       # price | trust | confusion | technical | ...
    strategy:        str | None       # what the agent decided to say/do
    tool_calls:      list[dict] | None
    tool_results:    list[dict] | None
    
    # ── Offer state (RRO only) ────────────────────────────────────────
    discounts_offered:     list[float]    # track what was offered
    max_discount_cap:      float          # never exceed this
    coupon_code:           str | None
    payment_link_sent:     bool
    whatsapp_sent:         bool
    
    # ── Support state (CSA only) ─────────────────────────────────────
    order_fetched:         dict | None
    stock_checked:         dict | None
    ticket_created:        str | None
    escalation_requested:  bool
    
    # ── Call control ─────────────────────────────────────────────────
    end_call:          bool
    end_reason:        str | None    # "completed" | "opted_out" | "max_retries" | "escalated"
    retry_count:       int
    max_retries:       int
    is_dnd:            bool
    
    # ── Outcome ───────────────────────────────────────────────────────
    outcome:           str | None    # "recovered" | "failed" | "resolved" | "escalated"
    outcome_details:   dict | None
```

### 7.2 RRO Graph

```python
# app/agent/graph.py — Revenue Recovery Agent

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.redis import RedisSaver

# ── Preflight graph (runs once at call start) ────────────────────────
preflight_builder = StateGraph(CallState)
preflight_builder.add_node("preflight_check", preflight_check_node)
preflight_builder.add_node("execute",         execute_node)
preflight_builder.set_entry_point("preflight_check")
preflight_builder.add_conditional_edges(
    "preflight_check",
    lambda s: "end" if s["end_call"] else "continue",
    {"end": "execute", "continue": END},   # execute=end_call action, continue=open call
)
preflight_builder.add_edge("execute", END)

preflight_graph = preflight_builder.compile(
    checkpointer=RedisSaver(redis_client)
)

# ── Turn graph (runs per user turn) ──────────────────────────────────
turn_builder = StateGraph(CallState)
turn_builder.add_node("run_strategy",  strategy_node)
turn_builder.add_node("execute",       execute_node)
turn_builder.set_entry_point("run_strategy")

# After strategy: did agent decide to end call?
turn_builder.add_conditional_edges(
    "run_strategy",
    lambda s: "end" if s["end_call"] else "execute",
    {"end": "execute", "execute": "execute"},
)

# After execute: need second strategy pass (tool results available)?
turn_builder.add_conditional_edges(
    "execute",
    route_after_execute,   # checks: end_call? tool_results? max_retries?
    {
        "end":          END,
        "second_pass":  "run_strategy_pass2",
        "done":         END,
    },
)

turn_builder.add_node("run_strategy_pass2", strategy_node)
turn_builder.add_conditional_edges(
    "run_strategy_pass2",
    lambda s: "end" if s["end_call"] else "done",
    {"end": "execute", "done": END},
)

rro_turn_graph = turn_builder.compile(
    checkpointer=RedisSaver(redis_client),
    interrupt_before=["execute"],   # optional HITL in admin mode
)
```

### 7.3 CSA Graph

```python
# Customer Support Agent graph

# ── Greet graph (runs once) ───────────────────────────────────────────
greet_builder = StateGraph(CallState)
greet_builder.add_node("identify_customer", identify_customer_node)
greet_builder.add_node("open_support",      open_support_node)
greet_builder.set_entry_point("identify_customer")
greet_builder.add_edge("identify_customer", "open_support")
greet_builder.add_edge("open_support", END)

greet_graph = greet_builder.compile(checkpointer=RedisSaver(redis_client))

# ── Support turn graph (runs per turn) ───────────────────────────────
support_builder = StateGraph(CallState)
support_builder.add_node("classify_intent",  classify_intent_node)
support_builder.add_node("resolve",          resolve_node)
support_builder.add_node("execute_tools",    execute_node)
support_builder.add_node("confirm_action",   confirm_action_node)  # before placing orders
support_builder.add_node("escalate",         escalate_node)

support_builder.set_entry_point("classify_intent")
support_builder.add_conditional_edges(
    "classify_intent",
    route_support_intent,
    {
        "order_status":   "resolve",
        "stock_check":    "resolve",
        "place_order":    "confirm_action",   # always confirm before placing
        "return_refund":  "resolve",
        "product_qa":     "resolve",
        "complaint":      "resolve",
        "escalate":       "escalate",
        "general":        "resolve",
    },
)
support_builder.add_edge("resolve", "execute_tools")
support_builder.add_conditional_edges(
    "confirm_action",
    lambda s: "place" if s.get("customer_confirmed") else "cancel",
    {"place": "execute_tools", "cancel": END},
)
support_builder.add_edge("execute_tools", END)
support_builder.add_edge("escalate", END)

csa_turn_graph = support_builder.compile(checkpointer=RedisSaver(redis_client))
```

---

## 8. Agent 1 — Revenue Recovery Agent (RRO)

### 8.1 Story

```
A customer added ₹12,999 worth of courses to their cart.
They went to checkout. Got to the payment page.
And left.

Without VoiceIQ:
  Email sent 24 hours later → 3% open rate → ~0.5% recovery
  SMS sent → feels spammy → ignored
  Customer never comes back

With VoiceIQ RRO:
  15 minutes after abandonment → AI calls
  "Hi Rahul, I noticed you were checking out our Full Stack Pro course.
   Did you face any issue with payment?"
  
  Rahul: "Actually the price is a bit high for me right now"
  
  AI: "I completely understand. We actually have a limited-time offer
       — I can share a 10% discount code valid for the next 2 hours.
       Would that help?"
  
  Rahul: "That sounds good"
  
  AI: "Perfect! I'm sending you a fresh payment link with the
       discount already applied. You'll get a WhatsApp message
       in the next 30 seconds."
  
  [Sends link, logs outcome: recovered, discount=10%]
```

### 8.2 Preflight Node

```python
# app/agent/nodes/preflight.py

async def preflight_check_node(state: CallState) -> dict:
    """
    Runs before the call opens.
    Validates: DND, retry limits, business hours.
    Loads: user history, product details, campaign rules.
    Returns: end_call=True if should not call, or opening_line to speak.
    """

    # 1. DND check
    is_dnd = await redis_client.sismember("dnd:users", state["user_id"])
    if is_dnd:
        return {
            "end_call":   True,
            "end_reason": "dnd",
            "is_dnd":     True,
        }

    # 2. Retry check — did we already call this user too many times?
    retry_key = f"retries:{state['user_id']}:{state['product_id']}"
    retry_count = int(await redis_client.get(retry_key) or 0)
    if retry_count >= state["max_retries"]:
        return {
            "end_call":    True,
            "end_reason":  "max_retries",
            "retry_count": retry_count,
        }

    # 3. Load campaign rules (seasonality, discount cap)
    campaign = await load_campaign_rules(state["product_id"])
    max_discount = campaign.get("max_discount_pct", 15.0)

    # 4. Load user history (previous purchases, past recovery attempts)
    user_history = await load_user_history(state["user_id"])

    # 5. Build personalised opening line
    opening_line = build_opening_line(
        name=state["name"],
        product_name=state["product_name"],
        user_type=state["user_type"],
        user_history=user_history,
    )

    return {
        "max_discount_cap": max_discount,
        "end_call":         False,
        "messages":         [{"role": "system", "content": build_system_prompt(state)}],
        "opening_line":     opening_line,
    }
```

### 8.3 Strategy Node

```python
# app/agent/nodes/strategy.py

RRO_SYSTEM_PROMPT = """
You are a warm, helpful recovery agent for {company_name}.

Customer context:
  Name:         {name}
  User type:    {user_type}
  Product:      {product_name}
  Cart value:   ₹{cart_value}
  Max discount: {max_discount_cap}%
  Discounts offered so far: {discounts_offered}

Your goal: recover this sale through a natural, empathetic conversation.

Intent detection guide:
  "expensive/price" → price_concern
  "trust/refund/guarantee" → trust_concern  
  "how does/what is" → confusion
  "didn't work/error/crash" → technical_issue
  "competitor/other platform" → competitor_comparison
  "payment failed/card declined" → payment_failure
  "need to think/come back" → delay
  "manager/boss/partner" → approval_dependency
  "no/not interested/remove" → opt_out

Offer rules (STRICT):
  - Start with smallest meaningful discount (5%)
  - Only escalate if customer still hesitant
  - NEVER offer more than {max_discount_cap}%
  - NEVER stack two discounts in one turn
  - NEVER offer best price immediately

Voice rules:
  - Speak in short sentences (max 2 sentences per turn)
  - Never use bullet points or lists (this is voice, not text)
  - Use the customer's name at most twice per call
  - Acknowledge before responding: "I see", "That makes sense"
  - If placing tool call: say filler first ("Let me check that for you")

Exit rules — say goodbye and set end_call=True when:
  - Customer says no / not interested (clearly)
  - Customer is angry or asks to be removed
  - Retry count >= max_retries
  - Outcome is confirmed (sale recovered or genuinely lost)
"""

async def strategy_node(state: CallState) -> dict:
    """LLM decides what to say and what tools to call"""

    response = await openai_client.chat.completions.create(
        model="gpt-4o",
        messages=state["messages"],
        tools=RRO_TOOLS,
        tool_choice="auto",
        max_tokens=200,    # short — this is voice, not email
        temperature=0.3,
    )

    msg = response.choices[0].message
    tool_calls = msg.tool_calls or []

    # Parse end_call signal
    end_call = any(
        tc.function.name == "end_call"
        for tc in tool_calls
    )

    return {
        "messages":    [msg],
        "tool_calls":  [tc.model_dump() for tc in tool_calls],
        "end_call":    end_call,
        "strategy":    msg.content,
    }
```

### 8.4 RRO Tools

```python
# app/agent/tools/rro_tools.py

RRO_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "generate_coupon",
            "description": "Generate a discount coupon for the customer. Use when customer shows price hesitation and a smaller discount hasn't been tried yet.",
            "parameters": {
                "type": "object",
                "properties": {
                    "discount_pct": {
                        "type": "number",
                        "description": "Discount percentage. Must not exceed max_discount_cap. Start small (5-10%), escalate only if needed."
                    },
                    "reason": {
                        "type": "string",
                        "description": "Why this discount is being offered (price_concern, trust_concern, etc.)"
                    }
                },
                "required": ["discount_pct", "reason"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "generate_payment_link",
            "description": "Generate a fresh, personalised payment link. Use when: (1) customer is ready to pay, (2) original link expired, (3) payment failed previously.",
            "parameters": {
                "type": "object",
                "properties": {
                    "coupon_code": {"type": "string", "description": "Include if coupon was already generated"},
                    "send_whatsapp": {"type": "boolean", "description": "True to also send via WhatsApp"},
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "send_whatsapp_message",
            "description": "Send a WhatsApp recovery message with payment link. Use after customer agrees but prefers async channel.",
            "parameters": {
                "type": "object",
                "properties": {
                    "template": {
                        "type": "string",
                        "enum": ["payment_link", "discount_offer", "trust_message", "follow_up"],
                        "description": "Which message template to send"
                    }
                },
                "required": ["template"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "schedule_followup",
            "description": "Schedule a follow-up call or message for later. Use when customer says 'call me tomorrow' or 'remind me next week'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "delay_hours": {"type": "integer", "description": "Hours from now"},
                    "channel": {"type": "string", "enum": ["call", "whatsapp", "sms"]}
                },
                "required": ["delay_hours", "channel"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "add_to_dnd",
            "description": "Add customer to do-not-disturb list. Use when customer says they don't want to be contacted.",
            "parameters": {
                "type": "object",
                "properties": {
                    "reason": {"type": "string", "enum": ["opted_out", "angry", "requested_removal"]}
                },
                "required": ["reason"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "end_call",
            "description": "End the call gracefully. Use when: (1) sale recovered, (2) customer clearly not interested, (3) customer opted out, (4) retry limit reached.",
            "parameters": {
                "type": "object",
                "properties": {
                    "outcome": {
                        "type": "string",
                        "enum": ["recovered", "failed", "scheduled_followup", "opted_out", "dnd_added"]
                    },
                    "farewell_message": {"type": "string", "description": "Final thing to say before hanging up"}
                },
                "required": ["outcome", "farewell_message"]
            }
        }
    },
]

async def execute_rro_tool(tool_name: str, args: dict, state: CallState) -> dict:
    match tool_name:
        case "generate_coupon":
            return await _generate_coupon(args, state)
        case "generate_payment_link":
            return await _generate_payment_link(args, state)
        case "send_whatsapp_message":
            return await _send_whatsapp(args, state)
        case "schedule_followup":
            return await _schedule_followup(args, state)
        case "add_to_dnd":
            return await _add_to_dnd(args, state)
        case "end_call":
            return {"end_call": True, "outcome": args["outcome"],
                    "farewell": args["farewell_message"]}
        case _:
            return {"error": f"Unknown tool: {tool_name}"}

async def _generate_coupon(args: dict, state: CallState) -> dict:
    discount = args["discount_pct"]

    # Enforce discount cap — never exceed
    if discount > state["max_discount_cap"]:
        discount = state["max_discount_cap"]

    # Enforce non-stacking
    if discount in state["discounts_offered"]:
        return {"error": "This discount was already offered. Try a different amount or end call."}

    code = f"RECOVER{int(discount)}-{state['user_id'][-4:].upper()}"
    # In production: call coupon service API
    return {
        "coupon_code":  code,
        "discount_pct": discount,
        "valid_hours":  2,
        "message":      f"Coupon {code} created for {discount}% off, valid 2 hours.",
    }
```

### 8.5 Intent Handling Matrix

```
Intent              First response              Escalation
──────────────────────────────────────────────────────────────────
price_concern       5% discount offer           10% → max_cap%
trust_concern       Guarantee + social proof    Refund policy emphasis
confusion           Clear explanation            Simplified + step-by-step
technical_issue     Fresh payment link          Alternative payment method
payment_failure     New link + alt method       WhatsApp payment guide
competitor_compare  Value comparison            Unique feature highlight
delay               2-hour limited offer        WhatsApp follow-up
approval_need       Schedule follow-up call     Send details to email
opt_out             Acknowledge + DND           End call immediately
dnd                 Skip entirely               —
```

---

## 9. Agent 2 — Customer Support Agent (CSA)

### 9.1 Story

```
Customer calls at 9 PM.
"I ordered 3 days ago and haven't got a delivery update"

Without VoiceIQ CSA:
  IVR menu: "Press 1 for orders, Press 2 for returns..."
  Hold music for 8 minutes
  Human agent: looks up order, gives update
  Customer satisfied — but it took 10 minutes and cost ₹40/minute (human)

With VoiceIQ CSA:
  AI picks up in < 1 second
  "Hi! This is VoiceIQ support. How can I help you today?"
  
  Customer: "I ordered 3 days ago and haven't got a delivery update"
  
  AI: "Of course, let me look that up. Can I have your registered
       mobile number or order ID?"
  
  Customer: "9876543210"
  
  AI: [looks up order in real-time]
      "I found your order #ORD-45231 — it's with the courier and
       expected by tomorrow between 10 AM and 2 PM. You'll get
       an SMS when it's out for delivery. Is there anything else
       I can help you with?"
  
  Customer: "That's great, thanks"
  
  Time: 45 seconds. Cost: < ₹2. Customer satisfied.
```

### 9.2 CSA Capabilities

```
CAPABILITY 1 — Order Status
  Input:  order_id OR phone number
  Action: lookup order in order DB
  Output: status, expected delivery, tracking link
  
  Example:
    "Where is my order?" → lookup by phone → "Order #45231 is out for delivery"

CAPABILITY 2 — Stock Check
  Input:  product name OR SKU
  Action: query inventory service
  Output: in_stock, quantity, next_restock_date
  
  Example:
    "Is the blue XL t-shirt available?" → check stock → "Yes, 3 units left"

CAPABILITY 3 — Place Order
  Input:  product, size/variant, address confirmation
  Action: validate → confirm with customer → create order via API
  Output: order_id, confirmation
  
  Safety rules:
    ALWAYS read back the order before placing
    ALWAYS get verbal confirmation ("Yes, please place it")
    NEVER place order without explicit customer confirmation
    
  Example:
    "I want to order the blue XL t-shirt"
    → "The blue XL t-shirt is ₹899. Should I use your saved
       address at 123 MG Road, Bangalore? Say yes to confirm."
    → Customer: "Yes"
    → [places order, gives order ID]

CAPABILITY 4 — Return / Refund
  Input:  order_id, reason
  Action: check return eligibility → initiate return
  Output: return authorised + pickup date OR rejection reason
  
  Example:
    "I want to return my order" → check policy → initiate pickup

CAPABILITY 5 — Product Q&A (RAG)
  Input:  any product question
  Action: search product knowledge base (RAG)
  Output: answer with citation
  
  Example:
    "Does this laptop support 5G?" → RAG over product specs → answer

CAPABILITY 6 — Complaint Handling
  Input:  customer describes issue
  Action: acknowledge + escalate + create ticket
  Output: ticket ID + expected resolution time
  
  Example:
    "I received a damaged item" → empathise → create ticket → confirm

CAPABILITY 7 — Escalation to Human
  Input:  complex issue or customer requests human
  Action: warm transfer via Twilio
  Output: customer connected to human agent
  
  Triggers:
    - Customer explicitly asks for human
    - Issue not resolved after 2 turns
    - Legal/compliance topics
    - High-value complaints (> ₹5000)
```

### 9.3 CSA Tools

```python
# app/agent/tools/csa_tools.py

CSA_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_order_status",
            "description": "Look up order status by order ID or customer phone number. Use when customer asks about delivery, order tracking, or order details.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id":     {"type": "string", "description": "Order ID like ORD-12345"},
                    "phone_number": {"type": "string", "description": "Customer's registered phone"},
                },
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "check_stock",
            "description": "Check if a product is in stock. Use when customer asks about availability before purchasing.",
            "parameters": {
                "type": "object",
                "properties": {
                    "product_name": {"type": "string"},
                    "sku":          {"type": "string"},
                    "variant":      {"type": "string", "description": "Size, colour, configuration"},
                },
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_product_details",
            "description": "Get detailed product information including specs, price, variants, and availability. Use before placing an order.",
            "parameters": {
                "type": "object",
                "properties": {
                    "product_name": {"type": "string"},
                    "sku":          {"type": "string"},
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "place_order",
            "description": "Place a new order for the customer. ONLY call after reading back full order details and receiving EXPLICIT verbal confirmation from the customer. Never place without confirmation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "product_id":         {"type": "string"},
                    "variant_id":         {"type": "string"},
                    "quantity":           {"type": "integer", "default": 1},
                    "delivery_address_id":{"type": "string", "description": "Use saved address ID if available"},
                    "payment_method":     {"type": "string", "enum": ["cod", "prepaid_saved", "new_link"]},
                    "customer_confirmed": {"type": "boolean", "description": "Must be true — set only after customer says yes"},
                },
                "required": ["product_id", "customer_confirmed"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "initiate_return",
            "description": "Start a return or refund process for an order. Check return eligibility before initiating.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id":   {"type": "string"},
                    "reason":     {"type": "string", "enum": ["damaged", "wrong_item", "not_as_described", "changed_mind", "defective"]},
                    "action":     {"type": "string", "enum": ["return_pickup", "exchange", "refund"]},
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_product_knowledge",
            "description": "Search the product knowledge base for answers to product questions. Use for: specs, compatibility, usage instructions, comparison questions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The customer's question to search for"}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "create_support_ticket",
            "description": "Create a support ticket for complex issues needing human follow-up. Use for: damaged items, missing items, wrong deliveries, payment disputes.",
            "parameters": {
                "type": "object",
                "properties": {
                    "issue_type":   {"type": "string", "enum": ["damaged", "missing", "wrong_item", "payment", "technical", "other"]},
                    "description":  {"type": "string"},
                    "order_id":     {"type": "string"},
                    "priority":     {"type": "string", "enum": ["low", "normal", "high", "urgent"]},
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "escalate_to_human",
            "description": "Transfer the call to a human agent. Use when: customer requests it, issue too complex, legal/compliance topic, or complaint > ₹5000.",
            "parameters": {
                "type": "object",
                "properties": {
                    "reason":         {"type": "string"},
                    "priority_queue": {"type": "string", "enum": ["general", "priority", "vip"]},
                    "summary":        {"type": "string", "description": "Brief summary for human agent context"},
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "end_call",
            "description": "End the support call after issue is resolved.",
            "parameters": {
                "type": "object",
                "properties": {
                    "outcome":         {"type": "string", "enum": ["resolved", "escalated", "ticket_created", "order_placed", "return_initiated"]},
                    "farewell_message":{"type": "string"}
                },
                "required": ["outcome", "farewell_message"]
            }
        }
    },
]

async def execute_csa_tool(tool_name: str, args: dict, state: CallState) -> dict:
    match tool_name:
        case "get_order_status":
            return await order_service.get_status(**args)
        case "check_stock":
            return await inventory_service.check(**args)
        case "get_product_details":
            return await catalog_service.get_details(**args)
        case "place_order":
            if not args.get("customer_confirmed"):
                return {"error": "Cannot place order without customer confirmation"}
            return await order_service.place(**args)
        case "initiate_return":
            return await returns_service.initiate(**args)
        case "search_product_knowledge":
            return await rag_service.search(args["query"])
        case "create_support_ticket":
            return await crm_service.create_ticket(**args)
        case "escalate_to_human":
            return await call_service.transfer_to_human(state["call_sid"], **args)
        case "end_call":
            return {"end_call": True, "outcome": args["outcome"]}
        case _:
            return {"error": f"Unknown tool: {tool_name}"}
```

### 9.4 Order Placement Flow (CSA — Critical Path)

```
Customer: "I want to order the blue XL cotton t-shirt"

Step 1 — Product lookup
  get_product_details(product_name="blue XL cotton t-shirt")
  → product_id, price, stock, variants

Step 2 — Confirm details (agent speaks)
  "The Blue Cotton T-Shirt in XL is ₹899, in stock.
   Should I place it to your saved address at 123 MG Road, Bangalore?
   Payment will be Cash on Delivery."

Step 3 — Wait for confirmation
  Customer: "Yes please"

Step 4 — Place order
  place_order(
    product_id="PRD-1234",
    variant_id="VAR-XL-BLUE",
    quantity=1,
    delivery_address_id="ADDR-789",
    payment_method="cod",
    customer_confirmed=True,    # ← MUST be True
  )

Step 5 — Confirm to customer
  "Your order #ORD-56789 has been placed!
   You'll get a WhatsApp confirmation shortly.
   Expected delivery in 3-5 business days."

SAFETY RULES for order placement:
  ✅ Always read back: product + price + variant + address + payment
  ✅ Always wait for explicit "yes" / "confirm" / "go ahead"
  ❌ Never place if customer says "maybe" / "I think so" / "sure whatever"
  ❌ Never place if stock check failed
  ❌ Never place with wrong product (verify before placing)
```

### 9.5 Product RAG — Knowledge Base

```python
# app/services/rag_service.py
# Product Q&A uses RAG over the product catalog

class ProductRAGService:
    """
    Indexes: product descriptions, specs, FAQs, comparison pages, user manuals
    Updates: on product catalog change (webhook-triggered re-index)
    """

    def __init__(self, qdrant_client, embedder):
        self._qdrant  = qdrant_client
        self._embedder= embedder

    async def search(self, query: str, product_id: str | None = None) -> str:
        embedding = await self._embedder.embed([query])
        filters   = None
        if product_id:
            filters = {"product_id": product_id}

        results = await self._qdrant.search(
            collection_name="product_knowledge",
            query_vector=embedding[0],
            query_filter=filters,
            limit=3,
            score_threshold=0.70,
        )

        if not results:
            return "I don't have specific information about that. Let me create a ticket and our team will follow up."

        # Format for voice (no bullet points, no markdown)
        context = ". ".join(r.payload["text"] for r in results)
        return context   # passed to LLM as tool result for the agent to summarise
```

---

## 10. Shared Infrastructure

### 10.1 Bot Runtime (Pipecat)

```python
# app/agent/bot.py — shared across RRO and CSA

from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask, PipelineParams
from pipecat.services.deepgram import DeepgramSTTService
from pipecat.services.openai import OpenAITTSService
from pipecat.transports.services.twilio import TwilioTransport

async def run_bot(transport: TwilioTransport, state: CallState):
    """
    Shared Pipecat runtime for both RRO and CSA agents.
    The LangGraph frame processor routes to the correct graph based on state.agent_type.
    """

    stt = DeepgramSTTService(
        api_key=settings.deepgram_api_key,
        model="nova-2-phonecall",    # optimised for phone audio quality
        language="hi-en",            # bilingual: Hindi + English
        punctuate=True,
        interim_results=True,        # word-by-word streaming
        vad_events=True,             # voice activity detection
    )

    tts = OpenAITTSService(
        api_key=settings.openai_api_key,
        model="tts-1",
        voice="nova",        # warm, professional voice
        speed=1.0,
    )

    langgraph_processor = LangGraphVoiceProcessor(
        state=state,
        rro_preflight=preflight_graph,
        rro_turn=rro_turn_graph,
        csa_greet=greet_graph,
        csa_turn=csa_turn_graph,
    )

    pipeline = Pipeline([
        transport.input(),             # audio in from Twilio
        stt,                           # speech → text
        langgraph_processor,           # text → LangGraph → text response
        tts,                           # text → speech
        transport.output(),            # audio out to Twilio
    ])

    runner = PipelineRunner()
    task   = PipelineTask(
        pipeline,
        PipelineParams(
            allow_interruptions=True,  # customer can interrupt mid-response
        ),
    )
    await runner.run(task)


class LangGraphVoiceProcessor(FrameProcessor):
    """
    Custom Pipecat processor that:
    1. Receives transcribed text frames
    2. Routes to correct LangGraph (RRO or CSA)
    3. Runs the graph, collects response text
    4. Emits TextFrame for TTS
    5. Emits filler audio during tool calls
    """

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        if isinstance(frame, TranscriptionFrame) and frame.text:
            await self._handle_user_turn(frame.text)
        else:
            await self.push_frame(frame, direction)

    async def _handle_user_turn(self, user_text: str):
        # Emit filler while processing
        await self.push_frame(
            TTSSpeakFrame("One moment..."),  # speaks while LangGraph runs
        )

        config = {"configurable": {"thread_id": self._state["session_id"]}}

        # Update messages
        new_state = {
            "messages": [{"role": "user", "content": user_text}],
            "current_turn": self._state["current_turn"] + 1,
        }

        # Run correct graph
        if self._state["agent_type"] == "rro":
            result = await rro_turn_graph.ainvoke(new_state, config=config)
        else:
            result = await csa_turn_graph.ainvoke(new_state, config=config)

        # Get agent response text
        last_msg = result["messages"][-1]
        response_text = last_msg.get("content", "")

        # Push response text to TTS
        if response_text:
            await self.push_frame(TTSSpeakFrame(response_text))

        # Handle end_call
        if result.get("end_call"):
            await self.push_frame(EndFrame())
```

### 10.2 Session Management

```python
# app/services/session_service.py

class SessionService:
    """
    Manages call sessions in Redis.
    TTL: call duration + 30 minutes (for post-call processing).
    """
    SESSION_TTL = 3600   # 1 hour

    async def create(self, state: CallState) -> str:
        session_id = f"session:{state['user_id']}:{int(time.time())}"
        await redis_client.setex(
            session_id,
            self.SESSION_TTL,
            json.dumps(state),
        )
        return session_id

    async def get(self, session_id: str) -> CallState | None:
        raw = await redis_client.get(session_id)
        return json.loads(raw) if raw else None

    async def update(self, session_id: str, updates: dict) -> None:
        state = await self.get(session_id)
        if state:
            state.update(updates)
            await redis_client.setex(session_id, self.SESSION_TTL, json.dumps(state))

    async def delete(self, session_id: str) -> None:
        await redis_client.delete(session_id)
```

### 10.3 Campaign Queue — Outbound Batch Calling

```python
# app/services/campaign_service.py

class CampaignService:
    """
    Manages outbound call campaigns.
    Rate limits: max N calls per minute per campaign.
    Retry scheduling: exponential backoff between retry attempts.
    """

    async def queue_campaign(
        self,
        users: list[dict],
        campaign_id: str,
        calls_per_minute: int = 10,
        delay_minutes: int = 15,   # wait N min after abandonment before calling
    ) -> dict:
        """Queue a batch of outbound calls with rate limiting"""
        queued = 0
        for i, user in enumerate(users):
            # Rate limit: spread calls over time
            delay_seconds = (i // calls_per_minute) * 60 + delay_minutes * 60

            await redis_client.zadd(
                f"campaign:{campaign_id}:queue",
                {json.dumps(user): time.time() + delay_seconds},
            )
            queued += 1

        return {"campaign_id": campaign_id, "queued": queued}

    async def process_campaign_queue(self, campaign_id: str) -> None:
        """Background worker — runs campaign queue"""
        now = time.time()
        due = await redis_client.zrangebyscore(
            f"campaign:{campaign_id}:queue",
            "-inf", now,
            withscores=False,
            count=10,   # process 10 at a time
        )
        for raw in due:
            user = json.loads(raw)
            await call_service.place_outbound_call(user)
            await redis_client.zrem(f"campaign:{campaign_id}:queue", raw)
```

---

## 11. Database Schema

```sql
-- ── Calls ────────────────────────────────────────────────────────────
CREATE TABLE calls (
    id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id   TEXT NOT NULL UNIQUE,
    call_sid     TEXT,                          -- Twilio call SID
    agent_type   TEXT NOT NULL,                 -- rro | csa
    direction    TEXT NOT NULL,                 -- outbound | inbound
    user_id      TEXT NOT NULL,
    phone        TEXT NOT NULL,
    status       TEXT DEFAULT 'initiated',      -- initiated | ringing | answered | completed | failed
    started_at   TIMESTAMPTZ,
    ended_at     TIMESTAMPTZ,
    duration_sec INTEGER,
    created_at   TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX ON calls (user_id);
CREATE INDEX ON calls (status, agent_type);

-- ── Call Outcomes ─────────────────────────────────────────────────────
CREATE TABLE call_outcomes (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    call_id         UUID REFERENCES calls(id),
    outcome         TEXT,            -- recovered | failed | resolved | escalated | opted_out
    detected_intent TEXT,
    turns_taken     INTEGER,
    end_reason      TEXT,

    -- RRO specific
    discount_offered NUMERIC(5,2),
    coupon_code      TEXT,
    payment_link_sent BOOLEAN DEFAULT FALSE,
    whatsapp_sent     BOOLEAN DEFAULT FALSE,
    cart_value        NUMERIC(10,2),
    recovered_value   NUMERIC(10,2),

    -- CSA specific
    issue_type        TEXT,
    order_id          TEXT,
    ticket_id         TEXT,
    order_placed      BOOLEAN DEFAULT FALSE,
    escalated_to_human BOOLEAN DEFAULT FALSE,

    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- ── Conversation Turns ────────────────────────────────────────────────
CREATE TABLE conversation_turns (
    id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    call_id      UUID REFERENCES calls(id),
    turn_number  INTEGER,
    speaker      TEXT,            -- user | agent
    text         TEXT,
    intent       TEXT,
    tools_called JSONB,
    latency_ms   INTEGER,
    tokens_used  INTEGER,
    cost_usd     NUMERIC(10,6),
    created_at   TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX ON conversation_turns (call_id);

-- ── DND List ─────────────────────────────────────────────────────────
CREATE TABLE dnd_list (
    user_id    TEXT PRIMARY KEY,
    phone      TEXT,
    reason     TEXT,
    added_at   TIMESTAMPTZ DEFAULT NOW()
);

-- ── Campaigns ─────────────────────────────────────────────────────────
CREATE TABLE campaigns (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name            TEXT,
    agent_type      TEXT,
    total_users     INTEGER,
    calls_placed    INTEGER DEFAULT 0,
    calls_answered  INTEGER DEFAULT 0,
    outcomes        JSONB,
    started_at      TIMESTAMPTZ,
    completed_at    TIMESTAMPTZ,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

-- ── Orders (CSA reference) ────────────────────────────────────────────
CREATE TABLE orders (
    id               UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    order_id         TEXT UNIQUE NOT NULL,
    user_id          TEXT,
    phone            TEXT,
    product_id       TEXT,
    product_name     TEXT,
    variant          TEXT,
    quantity         INTEGER DEFAULT 1,
    status           TEXT,
    total_amount     NUMERIC(10,2),
    delivery_address JSONB,
    placed_at        TIMESTAMPTZ,
    placed_via_call  BOOLEAN DEFAULT FALSE,   -- was this order placed by CSA?
    call_id          UUID REFERENCES calls(id),
    created_at       TIMESTAMPTZ DEFAULT NOW()
);

-- ── Product Knowledge (RAG source tracking) ───────────────────────────
CREATE TABLE product_knowledge_sync (
    product_id   TEXT PRIMARY KEY,
    last_synced  TIMESTAMPTZ,
    chunk_count  INTEGER,
    status       TEXT    -- synced | pending | error
);
```

---

## 12. API Reference

```
── CALL MANAGEMENT ───────────────────────────────────────────────────

POST /calls/outbound
  Body: { users: [{user_id, name, phone, user_type, product_name, cart_value}] }
  Response: { call_sid, session_id, estimated_start: "2-5 minutes" }
  Purpose: trigger RRO outbound call for abandoned cart recovery

POST /calls/campaign
  Body: { campaign_name, users: [...], calls_per_minute, delay_minutes }
  Response: { campaign_id, queued_count }
  Purpose: queue a batch outbound campaign

GET /calls/{call_id}
  Response: call details + outcome + turns

GET /calls/{call_id}/transcript
  Response: full conversation transcript with timestamps

GET /calls/analytics
  Query: ?agent_type=rro&from=2026-01-01&to=2026-01-31
  Response: { total_calls, answer_rate, recovery_rate, avg_discount, cost_per_call }

── TWILIO WEBHOOKS (called by Twilio, not by you) ────────────────────

POST /twilio/twiml?session_id={session_id}
  Response: TwiML XML with <Connect><Stream> directive
  Purpose: Twilio calls this to get instructions for the call

POST /twilio/inbound/twiml
  Response: TwiML XML for inbound call handling
  Purpose: Twilio calls this when someone dials your support number

WebSocket /twilio/media
  Protocol: Twilio Media Streams (mulaw audio, JSON events)
  Purpose: live bidirectional audio during call

POST /twilio/status
  Body: Twilio status callback payload
  Purpose: call status updates (initiated, ringing, answered, completed)

── SESSION MANAGEMENT ────────────────────────────────────────────────

GET /sessions/{session_id}
  Response: current call state from Redis

── SUPPORT TOOLS ─────────────────────────────────────────────────────

POST /dnd/add
  Body: { user_id, phone, reason }
  Purpose: add user to DND list

DELETE /dnd/{user_id}
  Purpose: remove from DND

GET /dnd/{user_id}
  Response: { is_dnd: bool }

POST /product-knowledge/sync
  Body: { product_ids: [...] }
  Purpose: re-index product knowledge for RAG

── ANALYTICS ─────────────────────────────────────────────────────────

GET /analytics/rro
  Response: recovery rate, avg discount, conversion by intent type

GET /analytics/csa
  Response: resolution rate, escalation rate, avg handle time, CSAT

GET /analytics/campaigns/{campaign_id}
  Response: per-campaign performance breakdown
```

---

## 13. Business Rules — Both Agents

### 13.1 RRO Rules

```
OFFER RULES:
  ✅ Start with smallest meaningful discount (5%)
  ✅ Escalate only if customer still hesitant after first offer
  ✅ Never exceed max_discount_cap (set per product/campaign)
  ✅ Never stack two discount offers
  ✅ During peak season: prefer urgency messaging over discounts
  ✅ For VIP users: skip small discounts, start at 10%
  ❌ Never offer best price immediately
  ❌ Never offer discount if customer hasn't expressed price concern

TIMING RULES:
  Call delay after abandonment: 15 minutes (default, configurable)
  Business hours only: 9 AM – 9 PM
  Max retries per user per product: 2 (configurable)
  Retry cooldown: 24 hours minimum between retries

EXIT RULES:
  End call immediately if:
    - User says: "no", "not interested", "remove me", "stop calling"
    - User is clearly angry
    - retry_count >= max_retries
    - User added to DND
  End call gracefully if:
    - Sale recovered (confirmed payment initiated)
    - Follow-up scheduled
    - Opt-out received
```

### 13.2 CSA Rules

```
ORDER PLACEMENT RULES:
  ✅ Always read back: product + size + price + address + payment
  ✅ Wait for explicit verbal confirmation before placing
  ✅ Confirm order ID after placing
  ❌ Never place if customer said "maybe" or "I'll think about it"
  ❌ Never place if stock check returned 0
  ❌ Never place wrong product/variant

ESCALATION TRIGGERS:
  Escalate to human if:
    - Customer explicitly asks for human agent
    - Issue not resolved after 2 turns
    - Complaint value > ₹5,000
    - Legal, fraud, or compliance topic
    - Medical or emergency mention
    - Agent confidence < threshold

REFUND/RETURN RULES:
  Check return window first (typically 7-30 days depending on product)
  Check product condition eligibility
  If eligible: initiate pickup, give timeline
  If ineligible: explain why, offer alternatives (exchange, credit)
  If borderline: create ticket for human review

DATA HANDLING:
  Never repeat full card number on call
  Never ask for OTP or CVV over phone
  Always verify customer identity before sharing order details
```

### 13.3 Shared Rules

```
CALL DURATION:
  RRO: hard stop at 5 minutes (voice prompt at 4:45)
  CSA: hard stop at 10 minutes (escalate to human if not resolved)

LANGUAGE:
  Default: English
  Auto-detect Hindi from first user utterance
  Switch seamlessly to Hindi if detected
  Deepgram model: nova-2-phonecall with hi-en bilingual

RECORDING:
  All calls recorded via Twilio
  Stored: 90 days
  Consent: spoken at call start ("This call may be recorded for quality purposes")

COMPLIANCE:
  DND check before every outbound call
  TRAI business hours compliance (9 AM – 9 PM)
  DPDP Act: no PII in logs, masked in transcripts
```

---

## 14. Guardrails

### 14.1 Input Guardrails

```python
# app/agent/guardrails.py

class VoiceGuardrailPipeline:
    """
    Applied to every user utterance before LangGraph processing.
    Voice-specific: checks are simpler (no injection attacks in speech)
    but add call-specific safety rules.
    """

    async def check(self, text: str, state: CallState) -> GuardResult:
        # 1. Opt-out detection (highest priority)
        if self._is_opt_out(text):
            return GuardResult(action="add_dnd_and_end")

        # 2. Anger/distress detection
        if self._is_angry(text):
            return GuardResult(action="de_escalate_or_end")

        # 3. DND reconfirmation
        if state["is_dnd"]:
            return GuardResult(action="end_call_immediately")

        # 4. Max retries (RRO)
        if state["retry_count"] >= state["max_retries"]:
            return GuardResult(action="end_call_gracefully")

        # 5. Sensitive topic detection (CSA)
        if self._is_sensitive_topic(text) and state["agent_type"] == "csa":
            return GuardResult(action="escalate_to_human")

        return GuardResult(action="continue")

    def _is_opt_out(self, text: str) -> bool:
        OPT_OUT_PHRASES = [
            "do not call", "don't call", "stop calling", "remove me",
            "unsubscribe", "not interested", "leave me alone",
            "मुझे कॉल मत करो", "बंद करो",   # Hindi phrases
        ]
        return any(p in text.lower() for p in OPT_OUT_PHRASES)

    def _is_angry(self, text: str) -> bool:
        ANGER_SIGNALS = ["shut up", "idiot", "fraud", "cheating", "police",
                          "consumer court", "file complaint"]
        return any(s in text.lower() for s in ANGER_SIGNALS)

    def _is_sensitive_topic(self, text: str) -> bool:
        SENSITIVE = ["legal", "lawsuit", "lawyer", "police", "fraud", "rbi",
                     "complaint", "media", "press", "journalist"]
        return any(s in text.lower() for s in SENSITIVE)
```

### 14.2 Output Guardrails (LLM Response)

```python
def validate_llm_response(response: str, state: CallState) -> str:
    """
    Before sending response to TTS:
    1. Check response is appropriate for voice (no lists, no markdown)
    2. Check discount not exceeded
    3. Check no PII leaked (card numbers, passwords)
    4. Check length appropriate (< 50 words for voice)
    """
    # PII pattern detection
    if re.search(r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b', response):
        return "I'm sorry, I can't share that information over the phone."

    # Discount cap enforcement
    discount_match = re.search(r'(\d+)%\s*discount', response)
    if discount_match:
        offered = float(discount_match.group(1))
        if offered > state["max_discount_cap"]:
            return response.replace(
                f"{int(offered)}%",
                f"{int(state['max_discount_cap'])}%"
            )

    # Length check (trim if too long for voice)
    words = response.split()
    if len(words) > 60:
        response = " ".join(words[:50]) + "..."

    return response
```

---

## 15. Observability

### 15.1 Stack

```
Langfuse:     every LLM call — prompt, tokens, cost, latency, response
Sentry:       exceptions in bot runtime, WebSocket errors, tool failures
CloudWatch:   call volume, answer rate, avg duration, error rate
Custom DB:    per-call analytics, outcome tracking, funnel analysis
```

### 15.2 Key Metrics

```
RRO Metrics:
  recovery_rate:         recovered / total_answered (target: >15%)
  avg_discount:          average discount offered on recovered calls
  intent_distribution:   which objections are most common
  call_answer_rate:      answered / placed (target: >40%)
  cost_per_recovery:     total LLM + Twilio cost / recovered calls

CSA Metrics:
  first_call_resolution: resolved without human / total calls (target: >70%)
  escalation_rate:        escalated to human / total calls (target: <20%)
  avg_handle_time:        seconds per call (target: <180s)
  order_placement_rate:   orders placed via call / attempted
  csat_score:             post-call satisfaction survey (target: >4/5)

Shared Metrics:
  p50_p95_latency:       STT → LLM → TTS round trip
  tool_success_rate:      tool calls that returned valid response
  dnd_rate:              opted-out / contacted (monitor for spam signal)
  call_completion_rate:   completed normally / started
```

### 15.3 Langfuse Tracing

```python
# Every LLM call traced with context
from langfuse.decorators import observe

@observe(name="rro_strategy")
async def strategy_node(state: CallState) -> dict:
    langfuse_context.update_current_observation(
        metadata={
            "agent_type":    state["agent_type"],
            "turn":          state["current_turn"],
            "intent":        state["detected_intent"],
            "session_id":    state["session_id"],
        }
    )
    # ... LLM call
```

---

## 16. Phase-by-Phase Build Plan

### Phase 0 — Foundation (4 days)
```
Tasks:
  [ ] FastAPI project + Docker Compose
  [ ] Twilio account + phone number
  [ ] Deepgram + OpenAI keys
  [ ] TwiML endpoint (inbound + outbound)
  [ ] Pipecat dependency installed and tested
  [ ] Basic WebSocket /twilio/media endpoint
  [ ] Redis + Postgres running in Docker
  [ ] Sentry + Langfuse connected

Showable:
  POST /calls/outbound → Twilio calls your phone
  You answer → hear silence (pipeline not connected yet)
  Call logged in Postgres
```

### Phase 1 — Working Voice Pipeline (5 days)
```
Tasks:
  [ ] Pipecat pipeline with Deepgram STT + OpenAI TTS
  [ ] Audio flows: phone → STT → text logged → TTS → phone
  [ ] Basic echo bot (repeat what user says)
  [ ] VAD working (end of turn detection)
  [ ] Latency measurement: target < 1.5s P95

Showable:
  Call your phone
  Say something → agent echoes it back
  Latency logged in Langfuse
```

### Phase 2 — RRO Agent (LangGraph) (7 days)
```
Tasks:
  [ ] CallState TypedDict
  [ ] preflight_graph (DND, retry, opening line)
  [ ] turn_graph (strategy → execute loop)
  [ ] OpenAI LLM node with RRO system prompt
  [ ] Tools: generate_coupon, generate_payment_link (stubbed)
  [ ] DND check + add_to_dnd
  [ ] end_call tool + EndFrame
  [ ] Session state in Redis per call
  [ ] Outcome written to Postgres on call end
  [ ] Filler audio during tool execution

Showable:
  Call triggered for abandoned cart
  Agent speaks opening line with customer's name
  Agent detects "too expensive" → offers 10% discount
  Agent sends (stub) payment link
  Agent ends call gracefully
  Outcome in Postgres: recovered / failed
```

### Phase 3 — CSA Agent (8 days)
```
Tasks:
  [ ] Inbound TwiML route
  [ ] greet_graph + support_turn_graph
  [ ] CSA system prompt
  [ ] Tools: get_order_status, check_stock (connect to real/mock DB)
  [ ] place_order tool with confirmation step
  [ ] initiate_return tool
  [ ] search_product_knowledge (basic keyword search first, RAG later)
  [ ] create_support_ticket
  [ ] escalate_to_human (Twilio warm transfer)
  [ ] Customer identity verification (phone → user lookup)

Showable:
  Call support number
  "Where is my order?"
  Agent asks for phone/order ID
  Agent fetches from DB, gives update
  "I want to order the blue XL tshirt"
  Agent confirms details, waits for yes, places order
```

### Phase 4 — Guardrails + Business Rules (4 days)
```
Tasks:
  [ ] VoiceGuardrailPipeline (opt-out, anger, DND, sensitive topics)
  [ ] Discount cap enforcement in execute node
  [ ] Non-stacking offer rule
  [ ] Business hours check before calling
  [ ] Call duration hard stop (5 min RRO, 10 min CSA)
  [ ] PII masking in transcripts
  [ ] Consent recording message at call start

Showable:
  Say "do not call me" → agent apologises, adds to DND, ends call
  RRO: offer 25% → enforced down to max_discount_cap
  Call past 5 min → agent wraps up automatically
```

### Phase 5 — Product RAG for CSA (4 days)
```
Tasks:
  [ ] Qdrant collection: product_knowledge
  [ ] Ingest product catalog (JSON → chunks → embeddings)
  [ ] search_product_knowledge using Qdrant
  [ ] Webhook: product update → re-index
  [ ] Test: "Does this laptop have 5G?" → correct spec answer

Showable:
  Call CSA
  "Does the XPS 15 support WiFi 6E?"
  Agent answers from product knowledge (not hallucinated)
  Response cites chunk source in Langfuse
```

### Phase 6 — Campaign Queue + Analytics (4 days)
```
Tasks:
  [ ] CampaignService: queue + rate limit + retry scheduling
  [ ] POST /calls/campaign endpoint
  [ ] Background worker: drains campaign queue
  [ ] Analytics endpoints (recovery rate, resolution rate, etc.)
  [ ] Admin dashboard: calls table, outcomes, per-campaign metrics
  [ ] Langfuse dashboards: cost per call, latency percentiles

Showable:
  POST /calls/campaign with 50 users
  Watch calls go out at 10/minute
  Dashboard shows recovery rate, avg discount, cost per call
```

### Phase 7 — Production Hardening (5 days)
```
Tasks:
  [ ] Twilio webhook signature validation
  [ ] Redis: move to ElastiCache (persistent, clustered)
  [ ] Postgres: move to RDS (Multi-AZ)
  [ ] ECS Fargate deployment (FastAPI + Celery worker)
  [ ] CloudWatch alarms: call failure rate, latency spikes
  [ ] Real Razorpay/Stripe payment link generation
  [ ] Real WhatsApp Business API integration
  [ ] Load test: 50 simultaneous calls

Showable:
  Live system on real phone numbers
  Real payment links sent
  Real WhatsApp messages delivered
  50 concurrent calls without degradation
```

---

## 17. Folder Structure

```
voiceiq/
├── main.py                           ← browser testing entrypoint
├── twilio_main.py                    ← Twilio FastAPI entrypoint
├── Makefile
├── docker-compose.yml
├── .env.example
│
├── app/
│   ├── agent/
│   │   ├── bot.py                    ← shared Pipecat runtime
│   │   ├── graph.py                  ← all LangGraph graph definitions
│   │   │
│   │   ├── nodes/
│   │   │   ├── preflight.py          ← RRO startup validation
│   │   │   ├── identify.py           ← CSA customer identification
│   │   │   ├── strategy.py           ← shared LLM strategy node
│   │   │   ├── execute.py            ← tool execution node
│   │   │   ├── confirm.py            ← order confirmation node (CSA)
│   │   │   └── escalate.py           ← human transfer node (CSA)
│   │   │
│   │   ├── tools/
│   │   │   ├── rro_tools.py          ← coupon, payment link, WhatsApp
│   │   │   ├── csa_tools.py          ← order, stock, return, RAG, ticket
│   │   │   └── shared_tools.py       ← end_call, dnd, schedule_followup
│   │   │
│   │   ├── prompts/
│   │   │   ├── rro_system.txt        ← RRO system prompt template
│   │   │   ├── csa_system.txt        ← CSA system prompt template
│   │   │   └── builder.py            ← inject state into prompt templates
│   │   │
│   │   └── guardrails.py             ← VoiceGuardrailPipeline
│   │
│   ├── models/
│   │   ├── state.py                  ← CallState TypedDict
│   │   ├── requests.py               ← API request/response models
│   │   └── outcomes.py               ← outcome Enums and models
│   │
│   ├── services/
│   │   ├── call_service.py           ← outbound call placement, Twilio API
│   │   ├── session_service.py        ← Redis session management
│   │   ├── campaign_service.py       ← batch campaign queue
│   │   ├── order_service.py          ← order lookup + placement
│   │   ├── inventory_service.py      ← stock check
│   │   ├── rag_service.py            ← product knowledge search
│   │   ├── returns_service.py        ← return/refund initiation
│   │   ├── crm_service.py            ← ticket creation
│   │   ├── payment_service.py        ← payment link (Razorpay/Stripe)
│   │   └── whatsapp_service.py       ← WhatsApp Business API
│   │
│   ├── api/
│   │   ├── calls.py                  ← /calls/outbound, /calls/campaign
│   │   ├── twilio.py                 ← TwiML, /twilio/media, /twilio/status
│   │   ├── sessions.py               ← session inspection
│   │   ├── dnd.py                    ← DND management
│   │   ├── analytics.py              ← reporting endpoints
│   │   └── product_knowledge.py      ← RAG index management
│   │
│   └── core/
│       ├── config.py                 ← settings (env vars)
│       ├── database.py               ← Postgres + asyncpg
│       ├── cache.py                  ← Redis client
│       └── logging.py                ← structured JSON logging
│
├── test_data/
│   ├── product.json                  ← demo product + user context
│   └── sample_calls.json             ← test scenarios
│
└── tests/
    ├── test_rro_graph.py
    ├── test_csa_graph.py
    ├── test_guardrails.py
    └── test_tools.py
```

---

## 18. Resume Deliverables

| Deliverable | Details |
|---|---|
| Live demo | Real outbound call on real phone number |
| GitHub repo | Clean, documented, with architecture diagram |
| Call recordings | 3-5 sample calls: RRO recovery, CSA order, CSA escalation |
| Analytics dashboard | Screenshot: recovery rate, cost per call, latency |
| Langfuse dashboard | LLM traces showing cost + latency per call |
| Architecture diagram | System design drawing for interviews |

### The Interview Story (2 minutes)

> *"At our company, we had two revenue leakage problems. We were losing 70% of checkout completions and our inbound support was costing ₹40/call with human agents who couldn't scale.*
>
> *We built VoiceIQ — a voice AI platform with two agents on the same infrastructure. The first agent calls customers who abandoned checkout. It uses LangGraph to manage the conversation state — detecting the objection type, deciding whether to offer a discount, and knowing when to stop. It never offers the maximum discount upfront, works through a controlled escalation, and stops immediately if the customer says no.*
>
> *The second agent handles inbound support calls. It checks order status from our DB in real time, answers product questions via RAG over our catalog, and — this is the important one — can actually place orders over the phone. But it always reads back the full order and waits for explicit verbal confirmation before placing. The guardrail is in the tool definition itself: customer_confirmed must be True.*
>
> *Both agents run on Pipecat for real-time audio, Deepgram for streaming STT, and LangGraph with Redis checkpointing for state. The whole conversation fits in under 1.5 seconds P95 latency.*
>
> *Recovery rate for abandoned checkout: 18%. First-call resolution for support: 72%. Cost per support call dropped from ₹40 to ₹2.80."*

---

*VoiceIQ PRD v1.0 | Revenue Recovery Agent + Customer Support Agent*
*Twilio · Pipecat · LangGraph · Deepgram · OpenAI · Redis · Postgres*
