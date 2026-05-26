# 🕸️ LangGraph — Intro to Advanced
> **Complete Study Notes | Theory + Code + Use Cases | Interview Prep**
> Part of: GenAI + LLMOps Engineering Roadmap 2026 — Phase 4
> Estimated time: 20–25 hrs
> This IS used in SynapseIQ — SQL Agent (Phase 2) + Planner Agent (Phase 7)

---

## 📑 Table of Contents

1. [What is LangGraph — Theory](#1-what-is-langgraph--theory)
2. [Core Concepts — State, Nodes, Edges](#2-core-concepts--state-nodes-edges)
3. [Building Your First Graph](#3-building-your-first-graph)
4. [Conditional Routing](#4-conditional-routing)
5. [Cycles — Loops and Retry](#5-cycles--loops-and-retry)
6. [Checkpointing — Persistent State](#6-checkpointing--persistent-state)
7. [Human-in-the-Loop (HITL)](#7-human-in-the-loop-hitl)
8. [Streaming](#8-streaming)
9. [Subgraphs — Composable Agents](#9-subgraphs--composable-agents)
10. [Advanced — Send API and Map-Reduce](#10-advanced--send-api-and-map-reduce)
11. [Advanced — Multi-Agent Systems](#11-advanced--multi-agent-systems)
12. [Advanced — Long-Running Agents](#12-advanced--long-running-agents)
13. [ReAct Agent from Scratch with LangGraph](#13-react-agent-from-scratch-with-langgraph)
14. [Real-World Use Cases](#14-real-world-use-cases)
15. [LangGraph vs LangChain Agents](#15-langgraph-vs-langchain-agents)
16. [Interview Cheat Sheet](#16-interview-cheat-sheet)
17. [Quick Revision Cards](#17-quick-revision-cards)

---

## 1. What is LangGraph — Theory

### 🧑‍🎓 Explain Like I'm in BTech 2nd Year

You know how a normal program runs? Line 1 → Line 2 → Line 3 → done. Predictable, fixed path.

Now think about how a human expert solves a problem:

```
Problem: "My code is throwing a weird error"

Expert's process (NOT a fixed sequence):
  Step 1: Read the error → "Hmm, database connection issue"
  Step 2: Check DB connection → Works fine
  Step 3: "Wait, maybe it's the query itself"
  Step 4: Look at the query → Syntax error found!
  Step 5: Fix and test → Different error now
  Step 6: Google the new error → Found Stack Overflow answer
  Step 7: Apply fix → Tests pass!
  Step 8: Done

The expert decided at each step what to do NEXT.
Some steps repeated. Some branches. All based on what they found.
```

This is what LangGraph enables — building agents that think, decide, loop, and branch, just like a human expert would.

**LangGraph models your agent as a directed graph:**
```
Each node = one step (a function that does something)
Each edge = a connection between steps
           can be conditional ("if output is X, go to Y, else go to Z")
State = shared memory that flows through the entire graph
        every node reads from state and writes updates to state
```

### The Official Definition

> LangGraph is a library for building stateful, multi-actor applications with LLMs, built on top of (and used with) LangChain. It extends the LangChain Expression Language with the ability to coordinate multiple chains (or actors) across multiple steps of computation in a cyclic manner.

**Key word: CYCLIC.** Unlike normal chains (A→B→C, one direction), graphs can have cycles (A→B→A→B→C). This is what enables retry loops, reflection, and multi-step reasoning.

**Released:** January 2024 by the LangChain team
**Why built:** LangChain's agent abstractions didn't support complex stateful flows with cycles, HITL, and checkpointing. LangGraph was built specifically for this.

### LangGraph vs LangChain Agents

```
LangChain Agents (old):
  AgentExecutor is a black box
  Hard to customise the loop
  No native state persistence
  No HITL support
  Hard to debug ("what exactly happened in that loop?")

LangGraph:
  YOU define the graph — every node, every edge
  Full control over the loop
  State persists via checkpointers (Redis, Postgres)
  Native HITL with interrupt_before / interrupt_after
  Every step is visible and debuggable
  Each node is just a Python function — easy to test
```

### Mental Model: State Machine

```
LangGraph is essentially a state machine.

State machine: a system that:
  1. Has a defined STATE (current situation)
  2. Has TRANSITIONS (rules for moving between states)
  3. Makes decisions based on current state

Traffic light example:
  States: RED | YELLOW | GREEN
  Transitions: RED → GREEN (after timer), GREEN → YELLOW, YELLOW → RED
  Current state determines what happens next

LangGraph AI agent:
  State: {question, plan, search_results, draft, feedback, final_answer}
  Transitions: plan → execute → evaluate → (if good: finish) | (if bad: revise)
  Current state determines next node to execute
```

---

## 2. Core Concepts — State, Nodes, Edges

### 2.1 State — The Shared Memory

State is a TypedDict that every node in your graph can read from and write to.

```python
from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage
import operator

# Simple state
class SimpleState(TypedDict):
    question:    str           # the user's question
    sql_query:   str | None    # generated SQL
    result:      list | None   # query results
    error:       str | None    # any error
    step_count:  int           # how many steps taken
    final_answer:str | None    # final response to user

# State with message history (for chat agents)
class ChatAgentState(TypedDict):
    messages: Annotated[list[BaseMessage], operator.add]
    # Annotated[..., operator.add] means:
    # When a node returns {"messages": [new_msg]}, it APPENDS to existing list
    # (instead of replacing it — which would lose history)
    # This is the "reducer" — how state updates are merged
    user_id:  str
    org_id:   str

# Rich state for SQL agent (used in SynapseIQ)
class SQLAgentState(TypedDict):
    question:     str
    source_ids:   list[str]
    org_id:       str
    user_id:      str
    llm_provider: str
    llm_model:    str
    messages:     Annotated[list[BaseMessage], operator.add]
    final_sql:    str | None
    final_answer: str | None
    chart_type:   str | None
    error:        str | None
    step_count:   int
```

**Key insight about state:**
```python
# Nodes return PARTIAL state updates — only what changed
# LangGraph merges the update into the full state

def some_node(state: SQLAgentState) -> dict:
    # Do some work...
    return {
        "final_sql": "SELECT * FROM orders LIMIT 10",
        "step_count": state["step_count"] + 1,
        # Don't need to return question, org_id, etc.
        # LangGraph keeps them unchanged
    }

# With the Annotated + operator.add reducer:
def agent_node(state: ChatAgentState) -> dict:
    response = llm.invoke(state["messages"])
    return {
        "messages": [response],  # APPENDS to existing messages list
        # NOT: "messages": state["messages"] + [response]
    }
```

---

### 2.2 Nodes — The Workers

A node is just a Python function. It takes the current state and returns a partial update.

```python
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
import instructor

llm = ChatGroq(model="llama-3.3-70b-versatile")

# Node 1: Understand the question, plan the approach
def plan_node(state: SQLAgentState) -> dict:
    """Node that plans how to answer the question"""
    messages = [
        SystemMessage(content="""You are a SQL expert. Given a question about data,
plan your approach. What information do you need? What tables/columns are relevant?"""),
        HumanMessage(content=state["question"]),
    ]
    response = llm.invoke(messages)
    return {
        "messages": [response],
        "step_count": state["step_count"] + 1,
    }

# Node 2: Generate SQL query
def generate_sql_node(state: SQLAgentState) -> dict:
    """Node that generates the SQL query"""
    from pydantic import BaseModel

    class SQLPlan(BaseModel):
        sql:         str
        chart_type:  str   # bar | line | pie | table
        explanation: str

    structured_llm = instructor.from_groq(llm)
    plan: SQLPlan = structured_llm.chat.completions.create(
        model=state["llm_model"],
        response_model=SQLPlan,
        messages=list(state["messages"]) + [
            HumanMessage(content="Now generate the SQL query.")
        ],
    )
    return {
        "final_sql":  plan.sql,
        "chart_type": plan.chart_type,
        "messages":   [AIMessage(content=f"Generated SQL: {plan.sql}")],
    }

# Node 3: Validate SQL safety
def validate_sql_node(state: SQLAgentState) -> dict:
    """Node that checks SQL for safety"""
    sql = state["final_sql"] or ""
    blocked = ["DROP","DELETE","INSERT","UPDATE","TRUNCATE","ALTER"]

    for keyword in blocked:
        if keyword in sql.upper():
            return {
                "error": f"Unsafe keyword '{keyword}' detected. Only SELECT allowed.",
                "final_sql": None,
            }
    return {"error": None}  # validation passed

# Node 4: Execute the SQL
def execute_sql_node(state: SQLAgentState) -> dict:
    """Node that runs the SQL via DuckDB"""
    import duckdb

    try:
        conn   = duckdb.connect()
        result = conn.execute(state["final_sql"]).df().to_dict("records")
        return {
            "messages": [AIMessage(content=f"Query returned {len(result)} rows")],
            "error":    None,
        }
    except Exception as e:
        return {"error": str(e), "step_count": state["step_count"] + 1}

# Node 5: Generate final answer
def answer_node(state: SQLAgentState) -> dict:
    """Node that writes the natural language answer"""
    response = llm.invoke([
        SystemMessage(content="Explain these SQL results in plain English."),
        HumanMessage(content=f"Question: {state['question']}\nResults: {state.get('result', [])}"),
    ])
    return {"final_answer": response.content}

# Node 6: Handle errors — retry or give up
def error_node(state: SQLAgentState) -> dict:
    """Node that handles errors from SQL execution"""
    if state["step_count"] < 8:
        # Ask LLM to fix the SQL
        fix_response = llm.invoke([
            SystemMessage(content="The SQL query failed. Fix it."),
            HumanMessage(content=f"Failed SQL: {state['final_sql']}\nError: {state['error']}"),
        ])
        return {
            "final_sql": extract_sql(fix_response.content),
            "error":     None,
            "messages":  [fix_response],
        }
    else:
        return {"final_answer": "Could not generate a valid query. Please rephrase your question."}
```

---

### 2.3 Edges — The Connections

```python
from langgraph.graph import StateGraph, END

# Unconditional edge — always goes from A to B
builder.add_edge("node_a", "node_b")

# Conditional edge — decides next node based on state
def route_after_validation(state: SQLAgentState) -> str:
    """Router function — returns the name of the next node"""
    if state.get("error"):
        return "error_handler"    # go to error node
    return "execute_sql"         # go to execution node

builder.add_conditional_edges(
    "validate_sql",               # source node
    route_after_validation,       # function that returns next node name
    {
        "error_handler": "error_node",   # mapping: return value → node name
        "execute_sql":   "execute_sql_node",
    }
)

# END — terminates the graph
builder.add_edge("answer_node", END)
```

---

## 3. Building Your First Graph

### Simple Linear Graph

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict

class SimpleState(TypedDict):
    input:  str
    step1:  str
    step2:  str
    output: str

# Define nodes
def node_a(state: SimpleState) -> dict:
    return {"step1": f"Processed: {state['input']}"}

def node_b(state: SimpleState) -> dict:
    return {"step2": f"Enhanced: {state['step1']}"}

def node_c(state: SimpleState) -> dict:
    return {"output": f"Final: {state['step2']}"}

# Build graph
builder = StateGraph(SimpleState)

# Add nodes
builder.add_node("node_a", node_a)
builder.add_node("node_b", node_b)
builder.add_node("node_c", node_c)

# Add edges
builder.set_entry_point("node_a")    # start here
builder.add_edge("node_a", "node_b")
builder.add_edge("node_b", "node_c")
builder.add_edge("node_c", END)      # finish here

# Compile — validates the graph structure
graph = builder.compile()

# Run it
result = graph.invoke({"input": "Hello", "step1": "", "step2": "", "output": ""})
print(result["output"])  # "Final: Enhanced: Processed: Hello"

# Visualise the graph (in Jupyter)
from IPython.display import Image
Image(graph.get_graph().draw_mermaid_png())
```

### Complete SQL Agent Graph (SynapseIQ Pattern)

```python
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

builder = StateGraph(SQLAgentState)

# Add all nodes
builder.add_node("plan",         plan_node)
builder.add_node("generate_sql", generate_sql_node)
builder.add_node("validate_sql", validate_sql_node)
builder.add_node("execute_sql",  execute_sql_node)
builder.add_node("error_node",   error_node)
builder.add_node("answer",       answer_node)

# Entry point
builder.set_entry_point("plan")

# Linear edges
builder.add_edge("plan", "generate_sql")
builder.add_edge("generate_sql", "validate_sql")

# Conditional: after validation
builder.add_conditional_edges(
    "validate_sql",
    lambda state: "error" if state.get("error") else "execute",
    {"error": "error_node", "execute": "execute_sql"}
)

# Conditional: after execution
builder.add_conditional_edges(
    "execute_sql",
    lambda state: "error" if state.get("error") else "answer",
    {"error": "error_node", "answer": "answer"}
)

# Conditional: after error handling
builder.add_conditional_edges(
    "error_node",
    lambda state: "done" if state.get("final_answer") or state["step_count"] >= 8 else "retry",
    {"done": END, "retry": "generate_sql"}   # ← CYCLE: retry by going back to generate
)

# Final edge
builder.add_edge("answer", END)

# Compile with Postgres checkpointer
checkpointer = AsyncPostgresSaver.from_conn_string(settings.database_url)
sql_agent = builder.compile(
    checkpointer=checkpointer,
    interrupt_before=["execute_sql"],  # pause before execution for HITL
)
```

---

## 4. Conditional Routing

### 🧑‍🎓 Explain Like I'm in BTech 2nd Year

Conditional routing is like an if-else statement, but for your agent's path.

```
Normal code:
  if condition:
      do_thing_a()
  else:
      do_thing_b()

LangGraph:
  def route(state) -> str:
      if condition_from_state:
          return "node_a"
      else:
          return "node_b"
  
  builder.add_conditional_edges("current_node", route, {"node_a": "node_a", "node_b": "node_b"})
```

### Routing Patterns

```python
# Pattern 1: Binary routing (two options)
def route_intent(state: PlannerState) -> str:
    if state["intent"] == "sql":
        return "sql_agent"
    return "rag_pipeline"

builder.add_conditional_edges(
    "classify_intent",
    route_intent,
    {"sql_agent": "sql_agent", "rag_pipeline": "rag_pipeline"},
)

# Pattern 2: Multi-way routing (many options)
def route_by_query_type(state: AgentState) -> str:
    query_type = state["query_type"]
    if query_type == "factual":
        return "rag_node"
    elif query_type == "analytical":
        return "sql_node"
    elif query_type == "creative":
        return "generation_node"
    elif query_type == "conversational":
        return "chat_node"
    else:
        return "fallback_node"

builder.add_conditional_edges(
    "classify",
    route_by_query_type,
    {
        "rag_node":        "rag_node",
        "sql_node":        "sql_node",
        "generation_node": "generation_node",
        "chat_node":       "chat_node",
        "fallback_node":   "fallback_node",
    },
)

# Pattern 3: Dynamic routing — LLM decides
from pydantic import BaseModel
from typing import Literal

class RoutingDecision(BaseModel):
    destination: Literal["sql", "rag", "hybrid", "chat"]
    reasoning:   str

def llm_router(state: AgentState) -> str:
    """Let the LLM decide where to route"""
    decision: RoutingDecision = structured_llm.chat.completions.create(
        model="llama-3.1-8b-instant",   # cheap model for routing
        response_model=RoutingDecision,
        messages=[{
            "role":    "user",
            "content": f"Route this query: '{state['question']}'\n"
                       "sql=structured data, rag=documents, hybrid=both, chat=conversational",
        }],
    )
    return decision.destination
```

---

## 5. Cycles — Loops and Retry

### 🧑‍🎓 Explain Like I'm in BTech 2nd Year

In normal code, you loop with `while` or `for`. In LangGraph, you loop by pointing an edge BACK to an earlier node. This creates a cycle in the graph.

```
Normal while loop:
  while not done:
      result = try_something()
      if result is good: done = True

LangGraph cycle:
  try_node → evaluate_node → (if good: END) | (if bad: try_node again)
                                                         ↑__________|
  This is a cycle! evaluate_node can point BACK to try_node
```

### Retry Pattern

```python
class RetryState(TypedDict):
    question:    str
    answer:      str | None
    quality:     float | None
    attempts:    int
    final:       str | None

def generate_answer(state: RetryState) -> dict:
    """Generate an answer"""
    response = llm.invoke(f"Answer: {state['question']}")
    return {
        "answer":   response.content,
        "attempts": state["attempts"] + 1,
    }

def evaluate_quality(state: RetryState) -> dict:
    """Evaluate answer quality with LLM-as-judge"""
    from pydantic import BaseModel

    class QualityScore(BaseModel):
        score:    float  # 0-10
        feedback: str

    judge = structured_llm.chat.completions.create(
        model="gpt-4o-mini",
        response_model=QualityScore,
        messages=[{
            "role":    "user",
            "content": f"Rate this answer quality (0-10): Q={state['question']} A={state['answer']}",
        }],
    )
    return {"quality": judge.score}

def should_retry(state: RetryState) -> str:
    """Decide: retry or accept?"""
    if state["quality"] >= 7.0:
        return "accept"    # good enough
    if state["attempts"] >= 3:
        return "accept"    # too many retries, accept what we have
    return "retry"         # quality too low, retry

def finalise(state: RetryState) -> dict:
    return {"final": state["answer"]}

# Build graph with CYCLE
builder = StateGraph(RetryState)
builder.add_node("generate",  generate_answer)
builder.add_node("evaluate",  evaluate_quality)
builder.add_node("finalise",  finalise)

builder.set_entry_point("generate")
builder.add_edge("generate", "evaluate")
builder.add_conditional_edges(
    "evaluate",
    should_retry,
    {
        "retry":  "generate",   # ← CYCLE: goes back to generate
        "accept": "finalise",
    },
)
builder.add_edge("finalise", END)

reflection_graph = builder.compile()
result = reflection_graph.invoke({
    "question": "Explain quantum computing",
    "answer": None, "quality": None, "attempts": 0, "final": None,
})
```

### Reflection Pattern

```python
# Agent critiques its own output and improves it
class ReflectionState(TypedDict):
    task:      str
    draft:     str | None
    critique:  str | None
    revised:   str | None
    iteration: int

def draft_node(state: ReflectionState) -> dict:
    """Write initial draft"""
    response = llm.invoke(f"Write a {state['task']}")
    return {"draft": response.content, "iteration": state["iteration"] + 1}

def critique_node(state: ReflectionState) -> dict:
    """Critique the draft"""
    response = llm.invoke(
        f"Critique this draft. Be specific about what's missing or weak:\n\n{state['draft']}"
    )
    return {"critique": response.content}

def revise_node(state: ReflectionState) -> dict:
    """Revise based on critique"""
    response = llm.invoke(
        f"Original: {state['draft']}\n\nCritique: {state['critique']}\n\nRevise the draft:"
    )
    return {"draft": response.content, "revised": response.content}

def should_continue_reflection(state: ReflectionState) -> str:
    if state["iteration"] >= 3:  # max 3 iterations
        return "done"
    # Check if critique mentions major issues
    critique = state.get("critique", "")
    if "excellent" in critique.lower() or "no issues" in critique.lower():
        return "done"
    return "continue"
```

---

## 6. Checkpointing — Persistent State

### 🧑‍🎓 Explain Like I'm in BTech 2nd Year

Without checkpointing, your agent is like a phone call. If the call drops halfway through, you have to start over from the beginning.

With checkpointing, it's like Google Docs autosave. Every step saves progress. If anything interrupts — server restart, network issue, user closes browser — the agent picks up exactly where it stopped.

### How Checkpointing Works

```python
from langgraph.checkpoint.memory import MemorySaver         # dev/testing
from langgraph.checkpoint.sqlite import SqliteSaver         # simple persistence
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver  # production

# Thread ID: identifies a specific conversation/run
# Same thread_id = same conversation history loaded from checkpoint
thread_id = "user_abc_conversation_123"

# ── Memory checkpointer (dev only — lost on restart) ──
memory_checkpointer = MemorySaver()
graph = builder.compile(checkpointer=memory_checkpointer)

# ── SQLite checkpointer (single-server persistence) ──
sqlite_checkpointer = SqliteSaver.from_conn_string("./checkpoints.db")
graph = builder.compile(checkpointer=sqlite_checkpointer)

# ── Postgres checkpointer (production — multi-server safe) ──
postgres_checkpointer = AsyncPostgresSaver.from_conn_string(
    "postgresql://user:pass@host:5432/db"
)
graph = builder.compile(checkpointer=postgres_checkpointer)
```

```python
# Running with checkpointing
config = {"configurable": {"thread_id": thread_id}}

# First run — starts fresh
result = await graph.ainvoke(
    {"question": "Show revenue by region", "step_count": 0, ...},
    config=config,
)

# Second run — RESUMES from checkpoint
# If graph was interrupted (HITL, server restart), this continues
result = await graph.ainvoke(
    None,    # None means "resume from checkpoint, don't start over"
    config=config,
)

# Get full conversation state at any point
state = await graph.aget_state(config)
print(state.values)           # current state dict
print(state.next)             # which nodes will execute next
print(state.metadata)         # step count, etc.

# Get history of all checkpoints
async for checkpoint in graph.aget_state_history(config):
    print(checkpoint.metadata["step"])   # step number
    print(checkpoint.values["question"]) # state at that step
```

### State Snapshots and Time Travel

```python
# Get all past states
checkpoints = [c async for c in graph.aget_state_history(config)]

# Go back to a specific checkpoint (time travel!)
past_checkpoint = checkpoints[3]   # 4th step back
await graph.ainvoke(
    None,
    config={
        "configurable": {
            "thread_id":   thread_id,
            "checkpoint_id": past_checkpoint.config["configurable"]["checkpoint_id"],
        }
    }
)
# Agent resumes from that past state — useful for debugging
# "Let's re-run from step 3 with different input"
```

---

## 7. Human-in-the-Loop (HITL)

### 🧑‍🎓 Explain Like I'm in BTech 2nd Year

HITL is a pause button for your agent. Instead of running everything automatically, the agent pauses at critical moments and asks "Should I proceed?" — like a surgeon stopping before making an incision to confirm with the patient.

```
Without HITL:
  Agent: "I'll delete all orders from 2023 to clean up the database"
  [executes immediately — data is gone]
  
With HITL:
  Agent: "I'm about to run: DELETE FROM orders WHERE year=2023"
  Human: "Wait, that's wrong! I said ARCHIVE, not DELETE"
  Agent: [waits for approval]
  Human: [provides correction]
  Agent: "OK, I'll run: INSERT INTO orders_archive SELECT * FROM orders WHERE year=2023"
```

### interrupt_before — Pause Before a Node

```python
# Pause BEFORE the execute_sql node
# Human reviews the SQL before it runs

graph = builder.compile(
    checkpointer=checkpointer,
    interrupt_before=["execute_sql"],  # pause before this node
)

config = {"configurable": {"thread_id": "conv_123"}}

# Step 1: Run until the interrupt
result = await graph.ainvoke(
    {"question": "Delete all test orders", "step_count": 0, ...},
    config=config,
)
# Graph runs: plan → generate_sql → validate_sql → [PAUSED before execute_sql]

# Check what the agent is about to do
state = await graph.aget_state(config)
print(state.next)               # → ("execute_sql",) — what will execute next
print(state.values["final_sql"]) # → the SQL it wants to run

# Step 2: Human reviews and either:
# A) Approves — resume
approved = await graph.ainvoke(None, config=config)   # None = resume

# B) Rejects with new input — update state then resume
await graph.aupdate_state(
    config,
    {"final_sql": "SELECT * FROM orders WHERE status='test' LIMIT 10"},  # corrected SQL
    as_node="generate_sql",  # as if this came from generate_sql node
)
result = await graph.ainvoke(None, config=config)   # resume with corrected state

# C) Cancel entirely
# Just don't call ainvoke again — the checkpoint persists but the run ends
```

### interrupt_after — Pause After a Node

```python
# Pause AFTER generate_sql — review the output before validation
graph = builder.compile(
    checkpointer=checkpointer,
    interrupt_after=["generate_sql"],   # pause after this node runs
)

# Use case: review the generated SQL before proceeding to validation
# More common than interrupt_before when you want to SEE the output first
```

### Dynamic Interrupt — Agent Decides to Pause

```python
from langgraph.errors import NodeInterrupt

def execute_sql_node(state: SQLAgentState) -> dict:
    """Execute SQL — but pause if the query might be expensive"""
    sql = state["final_sql"]

    # Agent evaluates risk itself
    if any(keyword in sql.upper() for keyword in ["DELETE", "UPDATE", "DROP"]):
        raise NodeInterrupt(
            f"Potentially dangerous operation detected. "
            f"SQL: {sql}\n\nPlease review before proceeding."
        )

    # Safe to execute
    result = duckdb.connect().execute(sql).df().to_dict("records")
    return {"result": result, "error": None}

# The agent itself decided to pause — not the developer
# Useful for: cost-based pausing, confidence-based pausing, dangerous operations
```

### Full HITL API Pattern (FastAPI)

```python
# backend/api/routes/chat.py

@router.post("/chat/sql")
async def start_sql_chat(
    body:         SQLChatRequest,
    current_user: AuthUser = Depends(get_current_user),
):
    """Start a SQL agent run — may pause for HITL"""
    config = {
        "configurable": {
            "thread_id": f"{current_user.user_id}:{body.conversation_id}",
        }
    }

    result = await sql_agent.ainvoke(
        {"question": body.question, "step_count": 0, ...},
        config=config,
    )

    state = await sql_agent.aget_state(config)

    if state.next:  # graph is paused (HITL interrupt)
        return {
            "status":      "awaiting_approval",
            "pending_sql": state.values.get("final_sql"),
            "message":     "Please review and approve the SQL query",
            "thread_id":   body.conversation_id,
        }

    return {
        "status": "complete",
        "answer": result.get("final_answer"),
    }

@router.post("/chat/sql/{conversation_id}/approve")
async def approve_sql(
    conversation_id: str,
    body:            ApprovalRequest,
    current_user:    AuthUser = Depends(get_current_user),
):
    """Approve, reject, or modify the pending SQL"""
    config = {
        "configurable": {
            "thread_id": f"{current_user.user_id}:{conversation_id}",
        }
    }

    if body.action == "reject":
        return {"status": "cancelled"}

    if body.modified_sql:
        # Update state with user's modified SQL
        await sql_agent.aupdate_state(
            config,
            {"final_sql": body.modified_sql},
            as_node="generate_sql",
        )

    # Resume the agent
    result = await sql_agent.ainvoke(None, config=config)

    return {
        "status": "complete",
        "answer": result.get("final_answer"),
    }
```

---

## 8. Streaming

### Streaming Modes

```python
# LangGraph supports 4 streaming modes:
# 1. "values"   → stream full state after each node
# 2. "updates"  → stream only what changed after each node
# 3. "messages" → stream LLM tokens as they're generated
# 4. "debug"    → detailed internal information

# Mode 1: values — see full state after each step
async for state in graph.astream(
    {"question": "Show revenue by region", ...},
    config=config,
    stream_mode="values",
):
    print(f"After node: current SQL = {state.get('final_sql')}")

# Mode 2: updates — see only what changed (more efficient)
async for node_name, update in graph.astream(
    {"question": "Show revenue by region", ...},
    config=config,
    stream_mode="updates",
):
    print(f"Node '{node_name}' updated: {list(update.keys())}")

# Mode 3: messages — stream individual tokens from LLM nodes
async for event in graph.astream_events(
    {"question": "Show revenue by region", ...},
    config=config,
    version="v2",
):
    if event["event"] == "on_chat_model_stream":
        # This fires for each token from any LLM call in any node
        token = event["data"]["chunk"].content
        if token:
            print(token, end="", flush=True)

    elif event["event"] == "on_chain_end":
        # This fires when a node completes
        print(f"\n✅ Node completed: {event['name']}")
```

### Streaming to FastAPI SSE

```python
# Stream agent steps + LLM tokens to frontend via SSE
@router.post("/chat/rag/stream")
async def rag_stream(body: RAGChatRequest, current_user: AuthUser = Depends(get_current_user)):

    async def generate():
        config = {"configurable": {"thread_id": f"{current_user.user_id}:{body.conv_id}"}}

        async for event in rag_agent.astream_events(
            {"question": body.question, ...},
            config=config,
            version="v2",
        ):
            # LLM token
            if event["event"] == "on_chat_model_stream":
                token = event["data"]["chunk"].content
                if token:
                    yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"

            # Node started
            elif event["event"] == "on_chain_start" and event["name"] in VISIBLE_NODES:
                yield f"data: {json.dumps({'type': 'step_start', 'step': event['name']})}\n\n"

            # Node completed
            elif event["event"] == "on_chain_end" and event["name"] in VISIBLE_NODES:
                duration = event.get("run_id", "")
                yield f"data: {json.dumps({'type': 'step_done', 'step': event['name']})}\n\n"

            # Graph finished
            elif event["event"] == "on_chain_end" and event["name"] == "LangGraph":
                final_state = event["data"]["output"]
                yield f"data: {json.dumps({'type': 'done', 'citations': final_state.get('citations', [])})}\n\n"

        yield "data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache"})
```

---

## 9. Subgraphs — Composable Agents

### 🧑‍🎓 Explain Like I'm in BTech 2nd Year

A subgraph is a graph inside a graph — like a function inside a program. Instead of writing one giant 20-node graph, you write small focused graphs (SQL agent, RAG pipeline) and compose them together in a parent graph (planner).

```
Parent Graph (Planner):
  classify_intent_node
       ↓
  [use SQL subgraph] OR [use RAG subgraph] OR [run both in parallel]
       ↓
  synthesise_node

SQL Subgraph (reusable, standalone):
  plan_node → generate_sql → validate → execute → answer

RAG Subgraph (reusable, standalone):
  embed → retrieve → rerank → generate
```

### Building Subgraphs

```python
# ── SQL Agent Subgraph ──
sql_builder = StateGraph(SQLAgentState)
sql_builder.add_node("plan",         plan_node)
sql_builder.add_node("generate_sql", generate_sql_node)
sql_builder.add_node("execute_sql",  execute_sql_node)
sql_builder.add_node("answer",       sql_answer_node)
sql_builder.set_entry_point("plan")
sql_builder.add_edge("plan", "generate_sql")
sql_builder.add_edge("generate_sql", "execute_sql")
sql_builder.add_edge("execute_sql", "answer")
sql_builder.add_edge("answer", END)

sql_subgraph = sql_builder.compile(checkpointer=checkpointer, interrupt_before=["execute_sql"])

# ── RAG Pipeline Subgraph ──
rag_builder = StateGraph(RAGState)
rag_builder.add_node("retrieve",  retrieve_node)
rag_builder.add_node("rerank",    rerank_node)
rag_builder.add_node("generate",  generate_node)
rag_builder.set_entry_point("retrieve")
rag_builder.add_edge("retrieve", "rerank")
rag_builder.add_edge("rerank",   "generate")
rag_builder.add_edge("generate", END)

rag_subgraph = rag_builder.compile()

# ── Planner (Parent Graph) ──
class PlannerState(TypedDict):
    question:     str
    intent:       str | None
    sql_result:   dict | None
    rag_result:   dict | None
    final_answer: str | None
    org_id:       str
    user_id:      str

def classify_node(state: PlannerState) -> dict:
    """Classify query intent"""
    intent = classify_intent(state["question"])
    return {"intent": intent}

def route_intent(state: PlannerState) -> str:
    return state["intent"]   # "sql" | "rag" | "hybrid"

def synthesise_node(state: PlannerState) -> dict:
    """Merge SQL + RAG results"""
    answer = merge_results(state["sql_result"], state["rag_result"])
    return {"final_answer": answer}

planner_builder = StateGraph(PlannerState)
planner_builder.add_node("classify",   classify_node)
planner_builder.add_node("sql_agent",  sql_subgraph)   # ← subgraph as node!
planner_builder.add_node("rag_agent",  rag_subgraph)   # ← subgraph as node!
planner_builder.add_node("synthesise", synthesise_node)

planner_builder.set_entry_point("classify")
planner_builder.add_conditional_edges(
    "classify",
    route_intent,
    {"sql": "sql_agent", "rag": "rag_agent", "hybrid": "synthesise"},
)
planner_builder.add_edge("sql_agent",  "synthesise")
planner_builder.add_edge("rag_agent",  "synthesise")
planner_builder.add_edge("synthesise", END)

planner = planner_builder.compile(checkpointer=checkpointer)
```

---

## 10. Advanced — Send API and Map-Reduce

### Map-Reduce Pattern

```python
from langgraph.types import Send

# Use case: summarise 10 documents in parallel, then combine summaries
# Map: send each document to a summarise node (parallel)
# Reduce: combine all summaries into one

class MapReduceState(TypedDict):
    documents:  list[str]
    summaries:  Annotated[list[str], operator.add]   # accumulate summaries
    final:      str | None

def split_documents(state: MapReduceState) -> list[Send]:
    """
    Send API: dynamically create parallel branches.
    Each document gets its own copy of the state for the summarise node.
    All run in parallel.
    """
    return [
        Send(
            "summarise_node",              # which node to send to
            {"document": doc, "index": i}, # what to send (added to current state)
        )
        for i, doc in enumerate(state["documents"])
    ]

def summarise_node(state: dict) -> dict:
    """Summarise a single document — runs N times in parallel"""
    summary = llm.invoke(f"Summarise in 2 sentences:\n\n{state['document']}").content
    return {"summaries": [summary]}   # appended to summaries list via operator.add

def combine_summaries(state: MapReduceState) -> dict:
    """Reduce: combine all N summaries"""
    combined = "\n\n".join(f"Doc {i+1}: {s}" for i, s in enumerate(state["summaries"]))
    final = llm.invoke(f"Create a unified summary from these:\n\n{combined}").content
    return {"final": final}

builder = StateGraph(MapReduceState)
builder.add_node("split",    split_documents)
builder.add_node("summarise_node", summarise_node)
builder.add_node("combine",  combine_summaries)

builder.set_entry_point("split")
builder.add_conditional_edges("split", lambda x: x, ["summarise_node"])  # Send API
builder.add_edge("summarise_node", "combine")
builder.add_edge("combine", END)

map_reduce_graph = builder.compile()

result = map_reduce_graph.invoke({
    "documents": ["Doc 1 content...", "Doc 2 content...", "Doc 3 content..."],
    "summaries": [],
    "final":     None,
})
print(result["final"])   # unified summary of all 3 docs
```

---

## 11. Advanced — Multi-Agent Systems

### Supervisor Pattern

```python
from langchain_core.messages import HumanMessage, AIMessage
from pydantic import BaseModel
from typing import Literal

# Define available worker agents
WORKERS = ["sql_agent", "rag_agent", "data_health_agent", "report_agent"]

class RoutingDecision(BaseModel):
    next_agent: Literal["sql_agent", "rag_agent", "data_health_agent", "report_agent", "FINISH"]
    reasoning:  str

class SupervisorState(TypedDict):
    messages:     Annotated[list[BaseMessage], operator.add]
    next:         str
    org_id:       str
    user_id:      str
    final_answer: str | None

def supervisor_node(state: SupervisorState) -> dict:
    """The supervisor LLM routes to the right specialist"""
    decision: RoutingDecision = structured_llm.chat.completions.create(
        model="llama-3.3-70b-versatile",
        response_model=RoutingDecision,
        messages=[
            SystemMessage(content=f"""You are a supervisor routing requests to specialist agents.
Available agents: {WORKERS}
- sql_agent:         handles data queries, analytics, charts
- rag_agent:         handles document questions, policy lookups
- data_health_agent: checks data quality, missing values
- report_agent:      generates PDF reports, schedules delivery
- FINISH:            task is complete, ready to respond

Current conversation: {[m.content for m in state['messages'][-3:]]}"""),
            HumanMessage(content="Who should handle the next step?"),
        ],
    )
    return {"next": decision.next_agent}

def route_supervisor(state: SupervisorState) -> str:
    return state["next"]   # returns agent name or "FINISH"

# Build multi-agent graph
builder = StateGraph(SupervisorState)
builder.add_node("supervisor",        supervisor_node)
builder.add_node("sql_agent",         sql_subgraph)
builder.add_node("rag_agent",         rag_subgraph)
builder.add_node("data_health_agent", health_subgraph)
builder.add_node("report_agent",      report_subgraph)

builder.set_entry_point("supervisor")
builder.add_conditional_edges(
    "supervisor",
    route_supervisor,
    {
        "sql_agent":         "sql_agent",
        "rag_agent":         "rag_agent",
        "data_health_agent": "data_health_agent",
        "report_agent":      "report_agent",
        "FINISH":            END,
    },
)
# All agents report back to supervisor
for agent in WORKERS:
    builder.add_edge(agent, "supervisor")

supervisor = builder.compile(checkpointer=checkpointer)
```

### Parallel Execution with Send

```python
# Hybrid query: run SQL and RAG simultaneously, merge results
def run_both_parallel(state: PlannerState) -> list[Send]:
    """For hybrid queries: run SQL and RAG in parallel"""
    return [
        Send("sql_agent", {"question": state["question"], "org_id": state["org_id"], ...}),
        Send("rag_agent", {"question": state["question"], "org_id": state["org_id"], ...}),
    ]
```

---

## 12. Advanced — Long-Running Agents

### Agents That Survive Days

```python
# Scenario: agent starts processing, user comes back tomorrow to check

# Day 1 — start the agent
config = {"configurable": {"thread_id": "weekly-report-gen-001"}}
await report_agent.ainvoke(
    {"task": "Generate weekly revenue report", "step_count": 0},
    config=config,
)
# If this hits an HITL interrupt (e.g., needs human to choose charts)
# It pauses here. Persisted in Postgres checkpoint.

# Day 2 — user comes back
# Load the current state
state = await report_agent.aget_state(config)
print(state.values["pending_decision"])   # "Please choose: bar or line chart for revenue?"
print(state.next)                          # which node will run next

# User provides input
await report_agent.aupdate_state(
    config,
    {"chart_type": "bar", "pending_decision": None},
)

# Resume — picks up from exactly where it stopped
result = await report_agent.ainvoke(None, config=config)

# ── Best practices for long-running agents ──
# 1. Always use Postgres checkpointer (not memory or SQLite)
# 2. Include timestamps in state for expiry logic
# 3. Set max_iterations to prevent infinite loops
# 4. Send SSE or webhook to notify user when done
# 5. Store thread_id in your DB so users can find their run later
```

### State Versioning

```python
# Each checkpoint has a unique ID
# You can restore to any previous checkpoint

history = [c async for c in agent.aget_state_history(config)]

# Print the history
for checkpoint in reversed(history):
    step = checkpoint.metadata.get("step", 0)
    nodes = checkpoint.next
    print(f"Step {step}: next={nodes}")

# Go back 3 steps (time travel)
past_checkpoint = history[2]
await agent.ainvoke(
    None,
    config={**config, "configurable": {
        **config["configurable"],
        "checkpoint_id": past_checkpoint.config["configurable"]["checkpoint_id"],
    }},
)
```

---

## 13. ReAct Agent from Scratch with LangGraph

```python
"""
Build the full ReAct loop from scratch.
This is what LangChain's AgentExecutor does internally,
but now you control every step.
"""
from langchain_core.messages import ToolMessage
from langgraph.prebuilt import ToolNode

# Define tools
@tool
def get_schema(source_id: str, org_id: str) -> str:
    """Get the schema (column names and types) for a data source."""
    return SourceRepository.get_schema_sync(source_id, org_id)

@tool
def execute_sql(sql: str, source_ids: list[str]) -> str:
    """Execute a SQL query against the specified data sources."""
    result = duckdb_executor.run_sync(sql, source_ids)
    return str(result[:10])  # first 10 rows

@tool
def get_sample_rows(source_id: str, n: int = 5) -> str:
    """Get N sample rows from a data source to understand the data."""
    return SourceRepository.get_samples_sync(source_id, n)

tools = [get_schema, execute_sql, get_sample_rows]

# LLM with tools bound
llm_with_tools = llm.bind_tools(tools)

# The agent node — decides what to do
def agent_node(state: SQLAgentState) -> dict:
    """LLM decides: answer directly or call a tool?"""
    system = SystemMessage(content=f"""You are an expert SQL analyst.
You have access to tools to query data sources.
Max steps: {8 - state['step_count']}. Be efficient.""")

    response = llm_with_tools.invoke([system] + list(state["messages"]))
    return {
        "messages":   [response],
        "step_count": state["step_count"] + 1,
    }

# Tool execution node (prebuilt)
tool_node = ToolNode(tools)

# Router: did LLM call a tool, or give final answer?
def should_use_tool(state: SQLAgentState) -> str:
    last_message = state["messages"][-1]

    if state["step_count"] >= 8:
        return "done"   # hit max steps

    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "use_tool"    # LLM wants to call a tool

    return "done"   # LLM gave a direct answer

# Build the ReAct graph
builder = StateGraph(SQLAgentState)
builder.add_node("agent",     agent_node)
builder.add_node("use_tool",  tool_node)

builder.set_entry_point("agent")
builder.add_conditional_edges(
    "agent",
    should_use_tool,
    {"use_tool": "use_tool", "done": END},
)
builder.add_edge("use_tool", "agent")   # ← CYCLE: after tool, back to agent

react_agent = builder.compile(
    checkpointer=checkpointer,
    interrupt_before=["use_tool"],   # HITL: pause before executing tools
)

# Run
result = await react_agent.ainvoke(
    {
        "question":    "What are the top 5 products by revenue this month?",
        "source_ids":  ["src-abc-123"],
        "messages":    [HumanMessage(content="What are the top 5 products by revenue this month?")],
        "step_count":  0,
        "org_id":      "org_xyz",
        "user_id":     "user_123",
        "llm_model":   "llama-3.3-70b-versatile",
        "final_sql":   None,
        "final_answer":None,
        "chart_type":  None,
        "error":       None,
    },
    config={"configurable": {"thread_id": "run_001"}},
)
```

---

## 14. Real-World Use Cases

### Use Case 1 — Support Agent with Escalation (SynapseIQ)

```python
"""
Real scenario: Customer support agent
- Answers product questions (RAG)
- Looks up orders (SQL)
- Escalates to human for refunds > ₹5000 (HITL)
- Logs all interactions (observer)
"""

class SupportState(TypedDict):
    messages:       Annotated[list[BaseMessage], operator.add]
    intent:         str | None       # question | order_lookup | refund
    order_data:     dict | None
    refund_amount:  float | None
    escalated:      bool
    resolution:     str | None

def classify_intent(state: SupportState) -> dict:
    ...

def lookup_order(state: SupportState) -> dict:
    ...

def answer_question(state: SupportState) -> dict:
    ...

def process_refund(state: SupportState) -> dict:
    """Process refund — but raise NodeInterrupt if > ₹5000"""
    if state["refund_amount"] > 5000:
        raise NodeInterrupt(
            f"Refund of ₹{state['refund_amount']:.0f} requires human approval. "
            f"Customer claims: order was damaged."
        )
    # Small refund: process automatically
    return {"resolution": f"Refund of ₹{state['refund_amount']:.0f} processed"}
```

### Use Case 2 — Research Agent with Reflection

```python
"""
Real scenario: Research agent that:
1. Searches the web for information
2. Reads and extracts from top results
3. Writes a draft report
4. Critiques its own draft
5. Revises until quality > 8/10
6. HITL: user reviews outline before full generation
"""
```

### Use Case 3 — PR Review Agent

```python
"""
Real scenario: Automated PR code review
1. Read PR diff from GitHub MCP
2. Analyse: bugs, security, performance, style
3. Generate inline comments
4. HITL: flag uncertain comments for senior dev
5. Post approved comments to GitHub
"""
```

---

## 15.5 — Pros, Cons & Product Decision Framework

### Full Pros and Cons

```
PROS of LangGraph:
┌─────────────────────────────────────────────────────────────────┐
│ ✅ Full control over agent loop                                   │
│    You define every node, every edge, every routing decision.   │
│    Nothing is hidden. No black box. Debug at any step.          │
│                                                                  │
│ ✅ State persistence (checkpointing)                              │
│    Postgres checkpointer: state survives server restarts.       │
│    Multi-server safe: any ECS instance can resume any thread.   │
│    Time travel: inspect or restore any past state.              │
│                                                                  │
│ ✅ Native HITL (first-class citizen)                              │
│    interrupt_before/after built into the compiler.              │
│    Resume from checkpointed state — works across HTTP requests. │
│    NodeInterrupt: agent decides dynamically when to pause.      │
│    No other framework has this as cleanly.                      │
│                                                                  │
│ ✅ Streaming built in                                             │
│    astream_events: token-level streaming from any LLM node.     │
│    4 streaming modes. No custom SSE plumbing needed.            │
│                                                                  │
│ ✅ Subgraph composition                                           │
│    SQL agent is a graph. RAG pipeline is a graph.               │
│    Planner orchestrates both as black-box nodes.                │
│    Clean separation. Each testable independently.               │
│                                                                  │
│ ✅ Provider agnostic                                               │
│    Works with Groq, OpenAI, Anthropic, Ollama — any LLM.        │
│    No LangChain dependency (only langchain-core for messages).  │
│                                                                  │
│ ✅ Production battle-tested                                        │
│    Used by LinkedIn, Uber, Klarna in production.                │
│    LangGraph Cloud: managed deployment, scheduling, webhooks.   │
│                                                                  │
│ ✅ Each node is a plain Python function                            │
│    Easy to unit test in isolation.                              │
│    No framework magic — just functions taking state + returning. │
└─────────────────────────────────────────────────────────────────┘

CONS of LangGraph:
┌─────────────────────────────────────────────────────────────────┐
│ ❌ Steep learning curve                                           │
│    State, nodes, edges, reducers, checkpointers, subgraphs...  │
│    Takes 1-2 weeks to feel comfortable. Not beginner-friendly.  │
│    "I spent 3 days understanding why my state wasn't updating." │
│                                                                  │
│ ❌ Verbose for simple tasks                                        │
│    A 3-step pipeline: define state, add 3 nodes, add edges,     │
│    compile, invoke. More code than needed for simple cases.     │
│    Direct SDK calls are cleaner for single LLM calls.           │
│                                                                  │
│ ❌ Debugging is still non-trivial                                  │
│    State machine logic can be confusing to reason about.        │
│    Graph visualisation helps but isn't always available.        │
│    Complex routing conditions can be hard to trace.             │
│                                                                  │
│ ❌ Pulls in langchain-core                                         │
│    LangGraph depends on langchain-core (message types etc.)     │
│    Not truly LangChain-free — but much lighter than LangChain.  │
│    langchain-core is stable and lightweight — acceptable.       │
│                                                                  │
│ ❌ Overkill for simple tool use                                    │
│    "LLM + 1 tool + no state" = don't need LangGraph.           │
│    AgentExecutor or even a while loop would be simpler.         │
│                                                                  │
│ ❌ Async complexity                                                │
│    Full async throughout — good but adds Python async overhead. │
│    Team needs to be comfortable with asyncio patterns.          │
└─────────────────────────────────────────────────────────────────┘
```

---

### The Product Manager Perspective — When to Pick LangGraph

```
The core question to ask:
"Does this agent need to pause, remember, retry, or run in parallel?"

If YES to any → LangGraph
If NO to all → simpler option (while loop, direct SDK call, LangChain AgentExecutor)
```

**The 4 Triggers That Demand LangGraph**

```
Trigger 1: HUMAN-IN-THE-LOOP
  "A human needs to review/approve before the agent continues"
  → interrupt_before / interrupt_after / NodeInterrupt
  
  Examples:
  - Manager approves SQL query before execution (SynapseIQ)
  - Legal team reviews AI-drafted contract before sending
  - Customer service: escalate refund > ₹5000 to senior agent
  - Finance: any transaction > ₹1 lakh needs human sign-off
  
  Can you do this without LangGraph?
  Yes, but you'd build your own state machine + persistence.
  LangGraph gives you this for free.

Trigger 2: STATE PERSISTENCE
  "The agent needs to survive server restarts or resume from where it stopped"
  → Postgres checkpointer + thread_id
  
  Examples:
  - Report generation: takes 5 minutes, server restarts at minute 3
  - Long-running research: user starts at 9 AM, comes back at 5 PM
  - Background job: queued today, runs tonight, resumes if interrupted
  - Mobile app: user closes app mid-conversation, resumes tomorrow
  
  Can you do this without LangGraph?
  Yes, but you'd build your own checkpoint system.
  LangGraph's checkpointer handles edge cases you'd miss.

Trigger 3: COMPLEX BRANCHING + CYCLES
  "The agent path is non-linear — loops back, branches conditionally"
  → Conditional edges, cycles, retry patterns
  
  Examples:
  - Write code → test → fix → test again (until tests pass)
  - Draft answer → grade quality → if bad: revise → grade again
  - Research → if incomplete: search more → if complete: write report
  - SQL fails → analyse error → fix SQL → retry (up to 8 times)
  
  Can you do this without LangGraph?
  Yes, with a while loop. But you lose: checkpointing, streaming,
  HITL integration, subgraph composition.

Trigger 4: MULTI-AGENT ORCHESTRATION
  "Multiple AI agents collaborate, hand off to each other, or run in parallel"
  → Subgraphs, supervisor pattern, Send API
  
  Examples:
  - Planner → (SQL Agent + RAG Agent in parallel) → Synthesiser
  - Supervisor → routes to Billing Agent | Support Agent | Tech Agent
  - Research team: Researcher → Editor → Fact Checker → Publisher
  - DevOps: Code Agent → Test Agent → Security Agent → Deploy Agent
  
  Can you do this without LangGraph?
  Possible with asyncio, but no shared state, no checkpointing,
  no streaming, no HITL. You'd rebuild half of LangGraph.
```

---

### Feature-to-Framework Mapping (Production Decision Table)

```
Feature/Requirement             LangChain   LangGraph   Direct SDK   Winner
─────────────────────────────────────────────────────────────────────────────
Simple RAG chatbot               ✅           ⚠️ overkill  ✅          LangChain
                                                                      or Direct SDK

Document loading (PDFs, CSV)     ✅           N/A          ✅          LangChain
                                                                      (saves time)

Multi-provider fallback          ✅           N/A          ✅ (litellm) Direct SDK
                                                                      + litellm

Structured output (JSON/Pydantic)✅           N/A          ✅ (instr.) Direct SDK
                                                                      + instructor

Conversation memory              ✅           ✅           DIY          LangGraph
                                                                      (better state)

HITL (human approval)            ❌           ✅           DIY          LangGraph

State persistence                ❌           ✅           DIY          LangGraph

Retry loops / reflection         ❌           ✅           DIY          LangGraph

Multi-agent supervisor           ⚠️ limited   ✅           DIY          LangGraph

Parallel agent execution         ❌           ✅ (Send)    asyncio.gather LangGraph

Streaming tokens to frontend     ⚠️ possible  ✅ native    DIY          LangGraph

Cold start < 1s (Lambda)         ❌ (~3s)     ⚠️ (~1.5s)   ✅ (< 0.5s)  Direct SDK

Full audit trail / compliance    ⚠️ partial   ✅           ✅           LangGraph
                                                                      or Direct SDK

Time travel / state replay       ❌           ✅           ❌           LangGraph

Prototype in < 1 day             ✅           ⚠️           ✅           LangChain
```

---

### Real-World Team Scenarios

**Scenario 1 — Customer-Facing AI Copilot (B2B SaaS)**
```
Context:
  - SaaS product, enterprise clients
  - AI copilot alongside existing app
  - Clients need audit trails (compliance)
  - Support agents need to approve AI suggestions before sending
  - Conversation must resume if user refreshes page

Pick: LangGraph ✅

Why:
  HITL: support agents approve AI responses → interrupt_before
  Persistence: page refresh = conversation resumes → Postgres checkpointer
  Audit: every state change checkpointed → full history for compliance
  Multi-turn: conversation memory handled by checkpointer

Architecture:
  classify_intent → rag_subgraph OR sql_subgraph → synthesise
  interrupt_after("synthesise") → human reviews → send
```

**Scenario 2 — Internal Data Q&A Tool**
```
Context:
  - Internal team: data analysts asking questions about company data
  - Mix of structured (DB) and unstructured (PDFs) data
  - No compliance requirements
  - Want it shipped in 2 weeks

Pick: LangChain for RAG + LangGraph for SQL agent

Why:
  RAG: LangChain RetrievalQA for document questions (fast to build)
  SQL: LangGraph for DB queries (need retry loop for bad SQL)
  No HITL: internal users, mistakes are cheap
  No persistence needed: each session is self-contained

Architecture:
  /chat/rag → LangChain RetrievalQA chain
  /chat/sql → LangGraph SQL agent (with max_steps=8 and retry cycle)
```

**Scenario 3 — Automated Background Research Agent**
```
Context:
  - Agent runs overnight, researches topics, generates reports
  - Runs for 30-60 minutes per task
  - Must survive server restarts
  - No real-time user interaction (async)
  - Runs 100 tasks simultaneously

Pick: LangGraph ✅

Why:
  Long-running: must survive restarts → Postgres checkpointer critical
  Background: no streaming to user, but need state for debugging
  100 parallel: each gets its own thread_id → isolated state
  Reliability: if 1 of 100 fails → only that thread affected
  Debugging: check state of any thread at any point

Architecture:
  plan → [parallel research tasks via Send] → synthesise → write_report
  Celery kicks off the graph. Lambda resumes if interrupted.
```

**Scenario 4 — Simple AI Feature in Existing App**
```
Context:
  - Adding "AI summarise this email" to existing app
  - Single LLM call, no tools
  - 1 week to ship
  - Team has no LLM experience

Pick: Direct SDK only ✅

Why:
  No loops, no state, no HITL, no tools
  LangGraph would be massive overkill
  LangChain would be moderate overkill
  
  Just:
  from groq import AsyncGroq
  response = await client.chat.completions.create(
      model="llama-3.3-70b-versatile",
      messages=[{"role": "user", "content": f"Summarise: {email}"}],
  )
  return response.choices[0].message.content
  
  Literally 5 lines. Done.
```

**Scenario 5 — Multi-Agent Financial Analysis Platform**
```
Context:
  - Enterprise product, financial services
  - Agents: Data Fetcher, Risk Analyser, Compliance Checker, Report Writer
  - Every step needs audit trail
  - Risk analysis needs human approval before report generation
  - Runs for 10-20 minutes per analysis
  - Regulators may request replay of any past analysis

Pick: LangGraph ✅ (most complex case, most value from LangGraph)

Why:
  Multi-agent: 4 specialists + supervisor = subgraph composition
  HITL: compliance checker output needs human sign-off → interrupt_after
  Audit: regulators need exact replay → time travel from checkpoints
  Long-running: 20 minutes → server restart risk → Postgres checkpointer
  Compliance: state at every step = full paper trail
  
Architecture:
  Supervisor → Data Fetcher (parallel: 3 sources) → Risk Analyser
  → [HITL: compliance review] → Report Writer → Distribute
```

---

### The Decision Flowchart

```
START: "Should I use LangGraph, LangChain, or Direct SDKs?"

Q1: Is it a single LLM call with no tools?
    YES → Direct SDK (5 lines, done)
    NO  → Q2

Q2: Is it a simple pipeline? (load → embed → retrieve → generate)
    YES, no loops needed → LangChain OR Direct SDK
         Fast to build? → LangChain
         Full control?  → Direct SDK
    NO  → Q3

Q3: Does it need ANY of these?
    - HITL (human approval mid-execution)
    - State persistence (survive restarts)
    - Retry loops (attempt → evaluate → retry)
    - Multi-agent coordination
    - Long-running (> 1 minute)
    - Parallel agent execution
    
    YES to any → LangGraph ✅
    NO to all  → LangChain or Direct SDK

Q4 (if LangGraph): Does it also need many integrations quickly?
    (document loaders, vector stores, etc.)
    YES → LangGraph + LangChain document loaders
    NO  → LangGraph + Direct SDKs only
```

---

### The 3-Line Decision Rule (for interviews)

```
"How do you decide between LangChain, LangGraph, and direct SDKs?"

Answer:
  Direct SDK:  single LLM call or simple pipeline, no state needed
  LangChain:   rapid prototyping, many integrations, team is exploring
  LangGraph:   any agent needing HITL, state persistence, cycles, or
               multi-agent coordination — which is most production AI

In SynapseIQ: we use LangGraph for all agents, litellm + instructor for
LLM calls directly, and no LangChain — because we're in production and
need full control, HITL, and state persistence.
```

---

## 15. LangGraph vs LangChain Agents

```
LANGGRAPH wins when:
  ✅ Complex multi-step flows with branching
  ✅ Need state persistence (checkpointing)
  ✅ Human-in-the-loop required
  ✅ Multiple agents collaborating
  ✅ Long-running tasks (minutes to days)
  ✅ Need fine-grained control over the loop
  ✅ Debugging matters (each node is traceable)
  ✅ Production systems where reliability matters

LANGCHAIN AgentExecutor wins when:
  ✅ Simple one-shot tool use
  ✅ Quick prototype in < 50 lines
  ✅ Don't need state persistence
  ✅ Team is new to agents

CHOOSE BASED ON:
  "Do I need loops, cycles, persistence, or HITL?" → LangGraph
  "Do I just need LLM + a few tools, quick and simple?" → LangChain AgentExecutor
```

---

## 16. Interview Cheat Sheet

**Q: What is LangGraph and why does it exist?**
```
LangGraph is a state machine framework for building stateful AI agents.
Built by LangChain team because AgentExecutor lacked:
1. State persistence across sessions
2. Human-in-the-loop with proper interrupt/resume
3. Complex branching and cycles
4. Multi-agent orchestration
5. Debugging visibility

Core concept: model your agent as a directed graph (nodes + edges).
State flows through the graph. Nodes update the state. Edges define routing.
```

**Q: Explain State, Nodes, Edges in LangGraph**
```
State:  TypedDict shared by all nodes. Flows through entire graph.
        Nodes read state, return partial updates.
        Annotated[list, operator.add] for append-only fields (message history).

Nodes:  Python functions. Take state → return partial state update.
        Pure functions — easier to test in isolation.

Edges:  Connections between nodes.
        Unconditional: always go A → B
        Conditional: function returns next node name based on state
        Can point backwards → creates cycles (retry loops)

Entry point: set_entry_point("first_node")
END: from langgraph.graph import END — terminates the graph
```

**Q: What is a checkpointer and why is it needed?**
```
Checkpointer: saves the graph state after every node execution.
Backends: MemorySaver (dev), SqliteSaver (local), AsyncPostgresSaver (prod)

Why needed:
1. HITL: graph pauses, user acts, graph resumes — state must survive
2. Long-running: task takes hours/days — state must persist across restarts
3. Debugging: inspect state at any past step ("time travel")
4. Multi-server: Postgres checkpointer works across multiple ECS instances

Usage:
  config = {"configurable": {"thread_id": "unique-conversation-id"}}
  await graph.ainvoke(None, config)  # None = resume from checkpoint
```

**Q: How does HITL work in LangGraph?**
```
Three mechanisms:
1. interrupt_before=["node_name"]
   Graph runs until just BEFORE that node, then pauses.
   Use when: want to review INPUT to the node.

2. interrupt_after=["node_name"]
   Graph runs that node, then pauses.
   Use when: want to review OUTPUT of the node.

3. NodeInterrupt (dynamic)
   Node itself raises NodeInterrupt exception.
   Use when: agent decides it needs human input mid-execution.

After interrupt:
  - State is checkpointed
  - graph.ainvoke(None, config) resumes
  - graph.aupdate_state(config, {...}) to inject human correction before resuming
```

**Q: What is the Send API?**
```
Send API enables dynamic fan-out — creating parallel branches at runtime.

Normal: you know the edges at compile time (builder.add_edge)
Send:   you create branches dynamically based on runtime data

Use case: map-reduce
  node that processes N documents returns [Send("worker", {"doc": doc}) for doc in docs]
  → N parallel branches created dynamically
  → All run simultaneously
  → Results accumulated via operator.add reducer

from langgraph.types import Send
def split_node(state) -> list[Send]:
    return [Send("worker_node", {"item": item}) for item in state["items"]]
```

**Q: Subgraphs — what are they and why use them?**
```
Subgraph: a compiled graph used as a node inside another graph.

Why:
  1. Reuse: SQL agent subgraph used in both "standalone SQL chat" and "planner graph"
  2. Isolation: each subgraph has its own state, easier to test
  3. Composition: planner orchestrates SQL + RAG as black boxes
  4. Clarity: 5-node planner is cleaner than 20-node monolith

How:
  sql_graph = sql_builder.compile(...)
  planner_builder.add_node("sql_agent", sql_graph)  # subgraph as node
  # Planner calls it like any other node
  # State transformation: planner state → subgraph state → planner state
```

---

## 17. Quick Revision Cards

```
CARD 1: LangGraph Core Primitives
  State:  TypedDict — shared memory, flows through graph
  Node:   Python function — reads state, returns partial update
  Edge:   Connection — unconditional or conditional
  END:    Terminates the graph
  Graph:  StateGraph(State) → add_node → add_edge → compile()

CARD 2: State Reducers
  Default: last write wins (new value replaces old)
  Annotated[list[X], operator.add]: new list APPENDED to existing
  Use for: message history (don't replace, append)
  Custom: define any function as reducer

CARD 3: Checkpointers
  MemorySaver:         dev only — lost on restart
  SqliteSaver:         single-server — local file
  AsyncPostgresSaver:  production — multi-server safe
  thread_id:           identifies a conversation/run
  None as input:       resume from checkpoint (don't restart)

CARD 4: HITL Patterns
  interrupt_before: pause before node → review INPUT
  interrupt_after:  pause after node → review OUTPUT
  NodeInterrupt:    agent decides to pause (dynamic)
  aupdate_state:    inject human correction into state
  ainvoke(None):    resume from paused state

CARD 5: Routing
  Unconditional: builder.add_edge("a", "b")
  Conditional:   builder.add_conditional_edges("a", route_fn, {val: node})
  route_fn returns a string → maps to next node name
  Can return END to terminate

CARD 6: Cycles
  Edge pointing to earlier node creates a cycle
  Use for: retry, reflection, ReAct tool-calling loop
  Always have an exit condition:
    step_count >= max_steps → END
    quality >= threshold → END
    final_answer is set → END

CARD 7: Streaming Modes
  "values":   full state after each node
  "updates":  only changed fields after each node
  "messages": LLM tokens as they generate
  astream_events: detailed event stream (on_chat_model_stream etc.)
  Use "messages" for frontend token streaming

CARD 8: Subgraphs
  Compile inner graph → use as node in outer graph
  Inner graph has its own state type
  Need state translation between inner/outer if different shapes
  Benefits: reuse, isolation, composability, clarity

CARD 9: Send API (Map-Reduce)
  from langgraph.types import Send
  Node returns [Send("target_node", {extra: data}), ...]
  → Dynamic parallel branches created at runtime
  Use for: processing N items in parallel

CARD 10: ReAct Agent Pattern
  agent_node: LLM decides action (direct answer or tool call)
  tool_node:  ToolNode(tools) executes the chosen tool
  router: if tool_calls in last message → "use_tool" else "done"
  Cycle: agent → tool → agent → tool → ... → END
  Max iterations guard: step_count >= N → "done"
```

---

## ✅ LangGraph Completion Checklist

```
THEORY
[ ] Explain State, Nodes, Edges in your own words
[ ] Understand the difference between LangGraph and LangChain AgentExecutor
[ ] Know when to use conditional vs unconditional edges
[ ] Understand state reducers (Annotated + operator.add)
[ ] Know 3 checkpointing backends and when to use each
[ ] Explain HITL: interrupt_before, interrupt_after, NodeInterrupt
[ ] Understand subgraphs and why to use them
[ ] Know 4 streaming modes (values, updates, messages, debug)
[ ] Explain Send API and map-reduce pattern

CODE
[ ] Built simple 3-node linear graph with StateGraph
[ ] Built conditional routing with add_conditional_edges
[ ] Built cycle (retry loop) with edge pointing backwards
[ ] Implemented MemorySaver checkpointer (dev)
[ ] Implemented AsyncPostgresSaver (production)
[ ] Implemented interrupt_before HITL with FastAPI approve endpoint
[ ] Implemented NodeInterrupt (dynamic interrupt from node)
[ ] Implemented aupdate_state to inject human correction
[ ] Built streaming endpoint with astream_events + SSE
[ ] Built subgraph and used it as a node in parent graph
[ ] Built ReAct agent from scratch (agent_node + ToolNode + cycle)
[ ] Built supervisor multi-agent pattern

PRODUCTION
[ ] SQL Agent with LangGraph (SynapseIQ pattern)
[ ] Planner Agent with SQL + RAG subgraphs in parallel
[ ] Long-running agent with Postgres checkpointer
[ ] HITL FastAPI: start → pause → review → approve → resume
[ ] State history inspection (time travel debugging)
```

---

*LangGraph Study Notes | GenAI + LLMOps Engineering Roadmap 2026 — Phase 4*
*Next: Phase 4 Remaining Topics → phase-4/PHASE_4_REMAINING.md*
