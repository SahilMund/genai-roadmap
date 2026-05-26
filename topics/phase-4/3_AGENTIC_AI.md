# 🤖 Phase 4 — Remaining Topics
> **Core Agent Patterns · HITL · Multi-Agent Systems · Orchestration · Fine-Tuning**
> Part of: GenAI + LLMOps Engineering Roadmap 2026 — Phase 4
> Estimated time: 35–40 hrs
> Note: MCP topics are in topics/phase-4/mcp.md | LangGraph in phase-4/LANGGRAPH.md

---

## 📑 Table of Contents

1. [4.1 — Core Agent Patterns](#41--core-agent-patterns)
2. [4.3 — Human-in-the-Loop (HITL) Deep Dive](#43--human-in-the-loop-hitl-deep-dive)
3. [4.4 — Multi-Agent Systems](#44--multi-agent-systems)
4. [4.5 — Agent Orchestration Patterns](#45--agent-orchestration-patterns)
5. [4.7 — Fine-Tuning Foundations](#47--fine-tuning-foundations)
6. [Phase 4 Projects](#-phase-4-projects)
7. [Interview Cheat Sheet](#-interview-cheat-sheet)
8. [Quick Revision Cards](#-quick-revision-cards)

---

## 4.1 — Core Agent Patterns

### 🧑‍🎓 Explain Like I'm in BTech 2nd Year

Think of agent patterns like cooking techniques. "Sautéing", "braising", "baking" — each technique has a specific use. Once you know the technique, you can apply it to any ingredient.

Agent patterns are the same. Once you know ReAct, you can apply it to any problem. Once you know Reflection, it improves any agent that makes mistakes. These patterns are the vocabulary of AI engineering.

---

### Pattern 1 — ReAct (Reasoning + Acting)

**Theory:**

ReAct is the foundational agent pattern — every other pattern builds on it. The agent alternates between:
- **Reasoning** (THOUGHT): "I need to find the customer's order history"
- **Acting** (ACTION): calling the `get_order_history(customer_id)` tool
- **Observing** (OBSERVATION): reading the tool's output
- Repeat until done

```
ReAct loop:
  THOUGHT → ACTION → OBSERVATION → THOUGHT → ACTION → OBSERVATION → ... → ANSWER

Example: "What was the revenue in South India last quarter?"
  THOUGHT: "I need to query the database for South India revenue for Q3"
  ACTION:  execute_sql("SELECT SUM(revenue) FROM orders WHERE region='South India' AND quarter='Q3'")
  OBSERVATION: "Result: ₹45.2 crore"
  THOUGHT: "I have the answer"
  ANSWER: "Revenue in South India last quarter was ₹45.2 crore"
```

**Why it works:**
```
Before ReAct (2022): agents would often skip directly to ACTION without reasoning
  → hallucinated tool arguments
  → called wrong tool
  → gave up too early

ReAct forces the agent to THINK first → THEN act
  → better tool selection
  → better argument construction
  → more reliable overall

Paper: "ReAct: Synergizing Reasoning and Acting in Language Models" (Yao et al., 2022)
Tested on: HotpotQA, Fever, ALFWorld — significant improvements over chain-of-thought alone
```

**When to use ReAct:**
```
✅ Multi-step problems requiring tool use
✅ Problems where the right approach isn't obvious upfront
✅ When the agent needs to adapt based on tool results

❌ Simple single-step problems (just call LLM directly)
❌ Well-defined workflows with fixed steps (use a chain)
```

```python
# ReAct prompt template
REACT_PROMPT = """You are a helpful AI assistant with access to tools.

Available tools:
{tools}

Use this format EXACTLY:
Thought: [your reasoning about what to do next]
Action: [tool_name]
Action Input: [tool arguments as JSON]
Observation: [tool result - this will be filled in for you]
... (repeat Thought/Action/Observation as needed)
Thought: I now have enough information to answer
Final Answer: [your answer to the original question]

Question: {input}
{agent_scratchpad}
"""
# Note: In LangGraph, you don't need this template — the loop is in the graph structure
# But understanding it helps you understand WHY the pattern works
```

---

### Pattern 2 — Plan-and-Execute

**Theory:**

ReAct is reactive — it decides one step at a time. Plan-and-Execute is strategic — it creates a complete plan FIRST, then executes each step.

```
ReAct:
  "What should I do right now?" → acts → "What should I do now?" → acts → ...
  Good for: exploratory problems where next step depends on current findings

Plan-and-Execute:
  Step 1: Create full plan: [step1, step2, step3, step4]
  Step 2: Execute step1 → result1
  Step 3: Execute step2 with result1 → result2
  Step 4: Execute step3 with result2 → result3
  Step 5: Synthesise
  Good for: well-defined multi-step tasks
```

**Why Plan-and-Execute is better for complex tasks:**
```
Problem: "Research the top 5 Indian AI startups, find their funding, founders, and products, then write a comparative report"

ReAct approach:
  Searches randomly, gets distracted, forgets to cover all 5 companies
  Takes 15-20 steps, often loses track

Plan-and-Execute approach:
  Plan: [
    "1. Find top 5 Indian AI startups",
    "2. Research Sarvam AI: founders + funding + products",
    "3. Research Krutrim: founders + funding + products",
    "4. Research Haptik: founders + funding + products",
    "5. Research Yellow.ai: founders + funding + products",
    "6. Research Vernacular.ai: founders + funding + products",
    "7. Write comparative report",
  ]
  Execute each step → structured, complete, nothing missed
```

```python
from pydantic import BaseModel

class ExecutionPlan(BaseModel):
    steps: list[str]   # ordered list of steps to execute
    goal:  str         # what we're trying to achieve

class PlanState(TypedDict):
    goal:          str
    plan:          list[str] | None
    current_step:  int
    step_results:  Annotated[list[str], operator.add]
    final_answer:  str | None

def planner_node(state: PlanState) -> dict:
    """Create a step-by-step plan for achieving the goal"""
    plan: ExecutionPlan = structured_llm.chat.completions.create(
        model="llama-3.3-70b-versatile",
        response_model=ExecutionPlan,
        messages=[{
            "role":    "user",
            "content": f"""Create a detailed step-by-step plan to achieve this goal.
Each step should be a specific, actionable task.
Goal: {state['goal']}""",
        }],
    )
    return {"plan": plan.steps, "current_step": 0}

def executor_node(state: PlanState) -> dict:
    """Execute the current step"""
    current_step = state["plan"][state["current_step"]]
    context = "\n".join(
        f"Step {i+1} result: {r}"
        for i, r in enumerate(state["step_results"])
    )
    result = llm.invoke(
        f"Previous results:\n{context}\n\nNow execute this step: {current_step}"
    )
    return {
        "step_results": [result.content],
        "current_step": state["current_step"] + 1,
    }

def synthesiser_node(state: PlanState) -> dict:
    """Combine all step results into final answer"""
    all_results = "\n\n".join(
        f"Step {i+1}: {r}" for i, r in enumerate(state["step_results"])
    )
    final = llm.invoke(f"Synthesise all results into a final answer:\n\n{all_results}")
    return {"final_answer": final.content}

def route_plan(state: PlanState) -> str:
    if state["current_step"] >= len(state["plan"] or []):
        return "synthesise"
    return "execute"

# Build plan-and-execute graph
builder = StateGraph(PlanState)
builder.add_node("plan",       planner_node)
builder.add_node("execute",    executor_node)
builder.add_node("synthesise", synthesiser_node)
builder.set_entry_point("plan")
builder.add_edge("plan", "execute")
builder.add_conditional_edges("execute", route_plan,
    {"execute": "execute", "synthesise": "synthesise"})
builder.add_edge("synthesise", END)
```

---

### Pattern 3 — Reflection

**Theory:**

Reflection is the agent equivalent of "check your work." After generating output, the agent (or a separate critic agent) evaluates quality and provides feedback. The original agent uses that feedback to improve.

```
Without reflection:
  Agent writes code → submits → may be buggy, incomplete, or wrong style

With reflection:
  Agent writes code → Critic reviews it → "Missing error handling, add try/except"
  Agent revises → Critic reviews → "Better, but the SQL injection risk on line 42"
  Agent revises → Critic reviews → "Quality threshold met"
  Agent submits → better code

This mirrors how professionals work:
  Software: code review
  Writing:  editor reviews draft
  Medicine: second opinion
  Law:      peer review
```

**Two flavours of Reflection:**

```python
# Flavour 1: Self-reflection (agent critiques itself)
class SelfReflectionState(TypedDict):
    task:       str
    draft:      str | None
    critique:   str | None
    iteration:  int
    final:      str | None

def generate_node(state: SelfReflectionState) -> dict:
    if state["draft"] and state["critique"]:
        # Revise based on critique
        prompt = f"""Original draft:\n{state['draft']}\n\nCritique:\n{state['critique']}
        \nRevise the draft addressing all critique points:"""
    else:
        # First draft
        prompt = f"Complete this task to the best of your ability: {state['task']}"
    response = llm.invoke(prompt)
    return {"draft": response.content, "iteration": state["iteration"] + 1}

def critique_node(state: SelfReflectionState) -> dict:
    critique = llm.invoke(
        f"""Critique this output strictly. Focus on:
- Accuracy (any errors or omissions?)
- Completeness (anything missing?)
- Quality (could this be significantly better?)
- Specific suggestions for improvement

Output to critique:
{state['draft']}

If quality is excellent, say "ACCEPT: no significant improvements needed"
Otherwise, provide specific critique."""
    )
    return {"critique": critique.content}

def should_accept(state: SelfReflectionState) -> str:
    if state["iteration"] >= 3:
        return "accept"
    if "ACCEPT" in (state.get("critique") or ""):
        return "accept"
    return "revise"

# Flavour 2: Multi-agent reflection (separate critic agent)
# More objective — the critic doesn't have the author's bias
# Generator agent: specialised in writing/coding/analysis
# Critic agent:    specialised in finding flaws and giving feedback
# Both are LangGraph subgraphs in a parent graph
```

---

### Pattern 4 — Parallelisation

**Theory:**

Many AI tasks have independent sub-tasks that don't need to wait for each other. Parallelisation runs them simultaneously, reducing total time.

```
Sequential (slow):
  Task A: search web (3s)
  Task B: read document (2s)
  Task C: query database (1s)
  Total: 6 seconds

Parallel (fast):
  Task A + B + C all start simultaneously
  Total: 3 seconds (time of slowest task)
```

**When to parallelise:**
```
✅ Tasks with no dependencies on each other
✅ Multiple data sources to query simultaneously
✅ Multiple analyses of the same document
✅ Multi-perspective analysis (conservative + liberal + neutral)

❌ Task B needs Task A's output as input (sequential required)
❌ Tasks share the same limited resource (DB connection pool)
❌ Order matters for the final output
```

```python
from langgraph.types import Send

# Parallel document analysis: 3 different analyses run simultaneously
def fan_out(state: AnalysisState) -> list[Send]:
    return [
        Send("analyse_sentiment",  {"document": state["document"], "analysis_type": "sentiment"}),
        Send("extract_entities",   {"document": state["document"], "analysis_type": "entities"}),
        Send("summarise",          {"document": state["document"], "analysis_type": "summary"}),
        Send("extract_key_stats",  {"document": state["document"], "analysis_type": "stats"}),
    ]
# All 4 run simultaneously → total time = slowest analysis

# Parallel multi-source retrieval
def parallel_retrieve(state: RAGState) -> list[Send]:
    return [
        Send("retrieve_from_source", {"source_id": sid, "query": state["query"]})
        for sid in state["source_ids"]
    ]
# Retrieves from all sources simultaneously
```

---

### Pattern 5 — Tool Use Design (Critical for Agent Reliability)

**Theory:**

The single biggest factor in agent reliability is tool design quality. Bad tools → bad agents. This is often ignored by beginners who focus on the agent architecture and forget about the tools.

```
Research finding (2024): 
  Agent performance improved 40% when tool descriptions were rewritten
  without changing the agent architecture, prompt, or model.
  
  The description IS the interface between agent and tool.
```

```python
# ❌ BAD tool — vague, no context, no constraints
@tool
def query(q: str) -> str:
    """Query the database"""
    return db.execute(q)
# Agent doesn't know: what database? what format? what's safe to query?

# ✅ GOOD tool — specific, contextual, constrained
@tool
def get_sales_data(
    region:     str,
    start_date: str,
    end_date:   str,
    metric:     str = "revenue",
) -> str:
    """
    Retrieve sales data from the company analytics database.

    Use this tool when you need:
    - Revenue, orders, or customer count data by region and time period
    - Historical sales trends and comparisons

    Do NOT use this for:
    - Real-time inventory (use get_inventory instead)
    - Customer contact info (use get_customer instead)
    - Financial forecasts (use get_forecast instead)

    Args:
        region:     Geographic region: "North India" | "South India" | "East India" |
                    "West India" | "All India"
        start_date: Start date in YYYY-MM-DD format (e.g., "2025-01-01")
        end_date:   End date in YYYY-MM-DD format (e.g., "2025-03-31")
        metric:     What to measure: "revenue" | "orders" | "customers"
                    Default: "revenue"

    Returns:
        JSON string with data and metadata. Example:
        {"data": [{"month": "Jan", "value": 45230000}], "total": 135690000, "unit": "INR"}

    Note:
        Dates must be within the last 3 years.
        For quarterly data, use first/last day of quarter as start/end dates.
    """
    # implementation
```

**Tool design checklist:**
```
✅ Description explains WHEN to use this tool (and when NOT to)
✅ All args have types AND descriptions with valid values listed
✅ Return format documented with an example
✅ Edge cases and limitations noted
✅ Related tools mentioned for disambiguation
✅ Tool name is a verb + noun: get_*, create_*, delete_*, search_*
```

---

### Agent vs Workflow — The Key Distinction

```
WORKFLOW (fixed path):
  You define: Step 1 → Step 2 → Step 3
  The path is predetermined. No decisions.
  Control flow is in your code.
  
  Use when: process is well-defined and predictable
  Example: "Extract text → Chunk → Embed → Store"
  
AGENT (dynamic path):
  LLM decides: what to do next, which tool to call, when to stop
  The path emerges at runtime.
  Control flow is in the LLM's reasoning.
  
  Use when: process depends on findings, multiple paths possible
  Example: "Answer this question using available tools" (LLM decides which tools)
  
RULE: Start with workflow. Add agent only when workflow isn't flexible enough.
Agents are more powerful but harder to predict, debug, and cost-control.
```

---

## 4.3 — Human-in-the-Loop (HITL) Deep Dive

### 🧑‍🎓 Explain Like I'm in BTech 2nd Year

We give AI agents a lot of power — they can query databases, send emails, process refunds, post on social media. But AI makes mistakes. Some mistakes are small (wrong answer). Some are catastrophic (accidentally deleted production data).

HITL is how you let AI be powerful while keeping humans in control of the dangerous parts.

```
Without HITL:
  Agent: "I'll execute this query: DELETE FROM users WHERE last_login < '2023-01-01'"
  [executes — 50,000 users deleted]
  "Oops, I misunderstood the task"

With HITL:
  Agent: "I want to execute: DELETE FROM users WHERE last_login < '2023-01-01'"
  Human: "WAIT. That's not what I asked for. Show me how many users this affects first."
  Agent: "SELECT COUNT(*) FROM users WHERE last_login < '2023-01-01' → 50,247 users"
  Human: "Yes, that's what I want to archive — but ARCHIVE, not DELETE."
  Agent: "Understood. Running: INSERT INTO archived_users SELECT * FROM users..."
```

### Why HITL Is Needed

```
4 reasons AI agents need human oversight:

1. Irreversibility: some actions can't be undone
   DELETE data, send email, process payment, deploy code
   → Pause and verify before executing

2. Confidence threshold: agent isn't sure
   "I'm 70% confident this is the right answer"
   → Escalate to human when confidence is low

3. Threshold-based: action exceeds a limit
   "Refund amount > ₹5000" or "Query affects > 10,000 rows"
   → Human approval required for large actions

4. Legal/compliance: decision requires human authority
   "This contract change requires a manager's sign-off"
   → Agent can prepare, but human must decide
```

### 4 HITL Interrupt Patterns

**Pattern 1 — Pre-execution Review (interrupt_before)**

```python
# Pause BEFORE executing — human reviews what's ABOUT to happen
graph = builder.compile(
    checkpointer=checkpointer,
    interrupt_before=["execute_sql", "send_email", "process_refund"],
)

# Flow: generate SQL → [PAUSE] → human reviews → approve → execute
```

**Pattern 2 — Post-execution Review (interrupt_after)**

```python
# Pause AFTER executing — human reviews what HAPPENED
graph = builder.compile(
    checkpointer=checkpointer,
    interrupt_after=["draft_response"],
)
# Flow: agent drafts → [PAUSE] → human reviews draft → approve → send
# Use case: email drafting, report generation, contract creation
```

**Pattern 3 — Dynamic Interrupt (NodeInterrupt)**

```python
from langgraph.errors import NodeInterrupt

def process_refund_node(state: SupportState) -> dict:
    refund_amount = state["refund_amount"]

    # Agent decides dynamically whether to pause
    if refund_amount > 5000:
        raise NodeInterrupt(
            f"Refund request for ₹{refund_amount:,.0f} requires manager approval.\n"
            f"Customer claim: {state['refund_reason']}\n"
            f"Order ID: {state['order_id']}"
        )

    # Small refunds: process automatically
    process_small_refund(state["order_id"], refund_amount)
    return {"resolution": f"Refund of ₹{refund_amount:,.0f} processed"}
```

**Pattern 4 — Escalation Pattern**

```python
# Agent escalates when confidence is below threshold
def answer_question_node(state: SupportState) -> dict:
    response = structured_llm.chat.completions.create(
        response_model=AnswerWithConfidence,
        messages=[...],
    )

    if response.confidence < 0.7:
        # Low confidence → escalate to human
        raise NodeInterrupt(
            f"Low confidence ({response.confidence:.0%}) on this question.\n"
            f"Question: {state['question']}\n"
            f"My best answer: {response.answer}\n"
            f"Please verify or provide the correct answer."
        )

    return {"answer": response.answer}
```

### Approval Workflow Patterns

```python
# Full approval workflow: propose → human decides → execute OR cancel

class ApprovalState(TypedDict):
    proposed_action: str
    proposed_data:   dict
    human_decision:  str | None   # "approved" | "rejected" | "modified"
    modified_data:   dict | None
    result:          str | None

def propose_node(state: ApprovalState) -> dict:
    """Agent proposes an action"""
    # ... agent generates proposal ...
    raise NodeInterrupt(
        f"Proposed action: {state['proposed_action']}\n"
        f"Data: {json.dumps(state['proposed_data'], indent=2)}\n\n"
        "Please approve, reject, or modify this action."
    )

def execute_node(state: ApprovalState) -> dict:
    """Execute the approved (possibly modified) action"""
    data = state["modified_data"] or state["proposed_data"]
    result = execute_action(state["proposed_action"], data)
    return {"result": result}

def route_after_proposal(state: ApprovalState) -> str:
    if state["human_decision"] == "approved":
        return "execute"
    elif state["human_decision"] == "modified":
        return "execute"  # execute with modified data
    else:
        return "cancel"
```

### HITL UI Patterns

```typescript
// Frontend patterns for HITL

// 1. SQL Approval Modal
function SQLApprovalModal({ pendingSql, onApprove, onReject, onModify }: Props) {
  const [editedSql, setEditedSql] = useState(pendingSql)
  return (
    <Modal>
      <h2>⚠️ Review SQL Before Execution</h2>
      <p className="text-amber-400">This query will modify data. Please review carefully.</p>
      <CodeEditor
        value={editedSql}
        onChange={setEditedSql}
        language="sql"
      />
      <div className="flex gap-3 mt-4">
        <Button onClick={() => onApprove(editedSql)} variant="success">
          ✅ Run Query
        </Button>
        <Button onClick={onReject} variant="danger">
          ❌ Cancel
        </Button>
      </div>
    </Modal>
  )
}

// 2. Review Queue (async HITL)
// When agent pauses, item added to queue
// Human reviews queue later, makes decision
// Agent resumes when decision is made

// 3. Confidence Badge
// Show confidence level on every AI response
// Low confidence → highlight in amber → "Review required"
// High confidence → show in green → "Auto-verified"
```

### Async HITL — Agent Suspends, Human Reviews Later

```python
"""
Scenario: Report generation agent needs human to choose which charts to include.
But it's 11 PM. Human will review in the morning.

Without async HITL:
  Agent starts → gets to HITL → waits for hours → HTTP timeout → lost
  
With async HITL + checkpointing:
  Agent starts (11 PM) → reaches HITL → state checkpointed → agent "sleeps"
  Human opens app (9 AM) → sees pending review → makes decision
  Agent resumes → generates report with chosen charts
  
Key: thread_id persists in Postgres → state safe for days/weeks
"""

# Backend stores HITL events
class HITLEvent(BaseModel):
    thread_id:    str
    event_type:   str       # "approval_needed" | "review_needed"
    message:      str       # what the agent needs
    proposed_data:dict | None
    created_at:   datetime
    org_id:       str
    user_id:      str

# Notify user (email/Slack) when HITL event occurs
async def on_hitl_interrupt(thread_id: str, message: str, org_id: str):
    await HITLEventRepository.create(...)
    await ses.send_email(
        to=get_user_email(org_id),
        subject="Action Required: AI Agent Needs Your Input",
        body=f"Your AI agent is waiting for your review.\n\n{message}",
    )
```

---

## 4.4 — Multi-Agent Systems

### 🧑‍🎓 Explain Like I'm in BTech 2nd Year

One agent is like one person. Powerful, but limited by what one person can know and do.

Multi-agent systems are like teams. A football team has:
- Goalkeeper (specialised in defending)
- Striker (specialised in scoring)
- Midfielder (connects defence and attack)
- Coach (decides strategy, coordinates everyone)

Multi-agent AI works the same way:
- SQL specialist agent (handles all data queries)
- RAG specialist agent (handles document questions)
- Report agent (generates PDFs)
- Planner/Supervisor (coordinates all of them)

### Why Multi-Agent?

```
Single agent limitations:
  ❌ Context overload: one agent trying to do everything → too many tools → confused
  ❌ Specialisation: one agent can't be an expert at both SQL and document analysis
  ❌ Parallelism: one agent does one thing at a time
  ❌ Reliability: one agent failure = system failure

Multi-agent benefits:
  ✅ Each agent has focused context + only relevant tools → better decisions
  ✅ Specialists: SQL agent knows SQL deeply; RAG agent knows retrieval deeply
  ✅ Parallelism: run SQL and RAG simultaneously for hybrid queries
  ✅ Isolation: if RAG agent fails, SQL agent keeps working
  ✅ Scalability: add more specialist agents as you add features
```

### Architecture 1 — Supervisor Pattern

```python
"""
One orchestrator LLM routes tasks to specialist sub-agents.
The supervisor doesn't do the work — it decides WHO should do the work.
"""

from langchain_core.messages import HumanMessage, AIMessage
from pydantic import BaseModel
from typing import Literal

# Available specialists
WORKERS = ["sql_analyst", "doc_researcher", "data_quality", "report_writer"]

class SupervisorDecision(BaseModel):
    next: Literal["sql_analyst", "doc_researcher", "data_quality", "report_writer", "FINISH"]
    reason: str

class SupervisorState(TypedDict):
    messages:     Annotated[list[BaseMessage], operator.add]
    next_agent:   str | None
    final_answer: str | None
    org_id:       str
    user_id:      str

def supervisor_node(state: SupervisorState) -> dict:
    """Supervisor decides which specialist to use next"""

    system_prompt = f"""You are a supervisor managing a team of AI specialists.
Your job: route tasks to the right specialist.

Team members and their roles:
- sql_analyst:    handles data queries, analytics, SQL, charts, statistics
- doc_researcher: handles document questions, policy lookup, information from files
- data_quality:   checks data quality, finds missing values, inconsistencies
- report_writer:  generates PDF reports, summaries, scheduled deliveries
- FINISH:         the task is complete, you have enough info to give the final answer

Current task: {state['messages'][0].content}
Work done so far: {[m.content[:200] for m in state['messages'][1:]]}

Who should handle the next step?"""

    decision: SupervisorDecision = structured_llm.chat.completions.create(
        model="llama-3.3-70b-versatile",
        response_model=SupervisorDecision,
        messages=[{"role": "user", "content": system_prompt}],
    )
    return {"next_agent": decision.next}

def route_supervisor(state: SupervisorState) -> str:
    return state["next_agent"]

# Worker agents (these are full LangGraph subgraphs in production)
def sql_analyst_node(state: SupervisorState) -> dict:
    # Full SQL agent implementation
    response = llm.invoke(f"As SQL analyst, handle: {state['messages'][-1].content}")
    return {"messages": [AIMessage(content=f"SQL Analyst: {response.content}")]}

def doc_researcher_node(state: SupervisorState) -> dict:
    response = llm.invoke(f"As document researcher, handle: {state['messages'][-1].content}")
    return {"messages": [AIMessage(content=f"Doc Researcher: {response.content}")]}

# Build supervisor graph
builder = StateGraph(SupervisorState)
builder.add_node("supervisor",    supervisor_node)
builder.add_node("sql_analyst",   sql_analyst_node)
builder.add_node("doc_researcher",doc_researcher_node)
builder.add_node("data_quality",  data_quality_node)
builder.add_node("report_writer", report_writer_node)

builder.set_entry_point("supervisor")
builder.add_conditional_edges(
    "supervisor",
    route_supervisor,
    {
        "sql_analyst":   "sql_analyst",
        "doc_researcher":"doc_researcher",
        "data_quality":  "data_quality",
        "report_writer": "report_writer",
        "FINISH":        END,
    },
)
# All workers report back to supervisor after completing
for worker in WORKERS:
    builder.add_edge(worker, "supervisor")

supervisor_system = builder.compile(checkpointer=checkpointer)
```

### Architecture 2 — Hierarchical

```
CEO Agent (strategic decisions)
  ├── Research Director (manages research team)
  │     ├── Web Researcher Agent
  │     ├── Database Analyst Agent
  │     └── Document Reader Agent
  ├── Writing Director (manages writing team)
  │     ├── Draft Writer Agent
  │     └── Editor Agent
  └── Validation Director
        ├── Fact Checker Agent
        └── Quality Assurance Agent

Use when: very complex tasks with many specialists
         nested specialisation required
         large teams of agents
```

### Architecture 3 — Swarm (Peer-to-Peer Handoff)

```python
"""
Agents decide FOR THEMSELVES when to hand off to another agent.
No central supervisor. Peer-to-peer communication.

Example: customer support
  User question → FAQ Agent
  FAQ Agent: "This needs order data I can't access" → hands off to Order Agent
  Order Agent: "This needs a refund decision" → hands off to Billing Agent
  Billing Agent: "This exceeds my authority" → hands off to Escalation Agent
"""

class SwarmState(TypedDict):
    messages:     Annotated[list[BaseMessage], operator.add]
    current_agent:str
    resolved:     bool

def faq_agent_node(state: SwarmState) -> dict:
    """FAQ agent — handles general questions, hands off when needed"""
    response = llm_with_tools.invoke([
        SystemMessage(content="""You are the FAQ specialist.
Handle general product questions using the knowledge base.
If you need order data → transfer_to_order_agent
If you need refund decisions → transfer_to_billing_agent
If you can't help → transfer_to_human"""),
        *state["messages"],
    ])
    # Check if agent wants to transfer
    if hasattr(response, "tool_calls") and response.tool_calls:
        for tc in response.tool_calls:
            if tc["name"].startswith("transfer_to_"):
                target = tc["name"].replace("transfer_to_", "")
                return {"current_agent": target, "messages": [response]}
    return {"messages": [response], "resolved": True}

def route_swarm(state: SwarmState) -> str:
    if state["resolved"]:
        return END
    return state["current_agent"]   # go to whichever agent was chosen

builder.add_conditional_edges("faq_agent", route_swarm, {
    "order_agent":   "order_agent",
    "billing_agent": "billing_agent",
    "human":         "human_handoff",
    END:             END,
})
```

### CrewAI — Role-Based Agents

```python
"""
CrewAI: simpler API for multi-agent with role definitions.
Good for: quick multi-agent prototypes, role-based workflows.
Not as flexible as LangGraph for complex state management.
"""
from crewai import Agent, Task, Crew, Process
from crewai_tools import SerperDevTool, WebsiteSearchTool

# Define agents with roles (like hiring a team)
researcher = Agent(
    role="Senior Research Analyst",
    goal="Find comprehensive information about {topic}",
    backstory="""You are an expert researcher with 10 years of experience
    in gathering and analysing information from multiple sources.""",
    tools=[SerperDevTool(), WebsiteSearchTool()],
    llm=llm,
    verbose=True,
    max_iter=5,
)

writer = Agent(
    role="Technical Writer",
    goal="Write clear, engaging content based on research",
    backstory="""You are a skilled writer who can explain complex topics
    in an accessible way.""",
    llm=llm,
    verbose=True,
)

# Define tasks
research_task = Task(
    description="Research {topic} thoroughly. Find recent developments, key players, and statistics.",
    expected_output="Comprehensive research notes with sources",
    agent=researcher,
)

writing_task = Task(
    description="Write a 500-word blog post based on the research provided.",
    expected_output="A complete, engaging blog post with introduction, body, and conclusion",
    agent=writer,
    context=[research_task],   # writing task depends on research task's output
)

# Assemble the crew
crew = Crew(
    agents=[researcher, writer],
    tasks=[research_task, writing_task],
    process=Process.sequential,  # or Process.hierarchical for manager-based
    verbose=True,
)

result = crew.kickoff(inputs={"topic": "India's AI startup ecosystem in 2026"})
print(result)
```

### When to Use Which Framework

```
LangGraph (most flexible, most control):
  ✅ Complex state management needed
  ✅ HITL required
  ✅ State persistence (checkpointing)
  ✅ Custom routing logic
  ✅ Production systems
  ✅ Need to debug exactly what happened

CrewAI (quick prototyping):
  ✅ Role-based agents (researcher, writer, editor)
  ✅ Sequential or hierarchical workflow
  ✅ Prototype in < 1 hour
  ❌ Limited state control
  ❌ Limited HITL support
  ❌ Harder to customise the loop

AutoGen/AG2 (conversational agents):
  ✅ Agents that talk to each other in natural language
  ✅ Code generation + execution
  ✅ Microsoft/Azure ecosystem
  ❌ Different paradigm from LangGraph
  ❌ Less suitable for production AI backends
```

### Failure Isolation

```python
# One sub-agent failure should not crash the supervisor

def safe_agent_call(agent, state: SupervisorState) -> dict:
    """Run an agent with error isolation"""
    try:
        return agent.invoke(state)
    except Exception as e:
        logger.error(f"Agent {agent.name} failed: {e}")
        return {
            "messages": [
                AIMessage(content=f"[{agent.name} failed: {str(e)[:200]}. Supervisor please reassign.]")
            ]
        }

# In LangGraph node:
def sql_analyst_node_safe(state: SupervisorState) -> dict:
    return safe_agent_call(sql_analyst_subgraph, state)
```

---

## 4.5 — Agent Orchestration Patterns

### 🧑‍🎓 Explain Like I'm in BTech 2nd Year

Orchestration patterns are about HOW agents work together to accomplish a task. It's the same question as: should this be a meeting (everyone at once), an email chain (sequential), or a division of labour (parallel)?

### Orchestrator Types

**Static Orchestrator — Fixed Workflow**
```
What it is:  You define every step at code time. No LLM decides routing.
Best for:    Predictable, well-understood processes
Example:     Document ingestion: always Extract → Chunk → Embed → Store

extract_node → chunk_node → embed_node → store_node → END
(the path never changes — you defined it)
```

**Dynamic Orchestrator — LLM Decides**
```
What it is:  LLM decides the next step at each node
Best for:    Open-ended problems where approach depends on findings
Example:     Research agent — decides which tool to use based on what it finds

Each node: "Given what I know so far, what should I do next?"
LLM answers: "search web" | "read document" | "query DB" | "answer"
(the path emerges — you didn't predefine it)
```

**Hybrid Orchestrator — Fixed Skeleton + LLM Decision Points**
```
What it is:  Fixed high-level structure with LLM making decisions at branches
Best for:    Most production systems — predictable with flexibility where needed
Example:     
  Always: classify → (branch) → specialise → synthesise
  Branch: LLM decides which specialist (not hardcoded)

classify_node → [LLM routes here] → sql_node OR rag_node → synthesise_node → END
(skeleton fixed; routing flexible)
```

```python
# Hybrid orchestrator example
def build_hybrid_orchestrator():
    builder = StateGraph(OrchestratorState)

    # Fixed nodes (always run)
    builder.add_node("validate_input",    validate_node)     # always first
    builder.add_node("classify_intent",   classify_node)     # always second
    builder.add_node("synthesise_output", synthesise_node)   # always before END
    builder.add_node("audit_log",         audit_node)        # always last

    # Specialist nodes (conditionally run)
    builder.add_node("sql_agent",     sql_subgraph)
    builder.add_node("rag_agent",     rag_subgraph)
    builder.add_node("hybrid_agent",  hybrid_subgraph)

    # Fixed edges
    builder.set_entry_point("validate_input")
    builder.add_edge("validate_input",    "classify_intent")
    builder.add_edge("synthesise_output", "audit_log")
    builder.add_edge("audit_log",         END)

    # Dynamic routing (LLM decides)
    builder.add_conditional_edges(
        "classify_intent",
        llm_intent_router,   # LLM decides here
        {"sql": "sql_agent", "rag": "rag_agent", "hybrid": "hybrid_agent"},
    )

    # All specialists → synthesise (fixed)
    for specialist in ["sql_agent", "rag_agent", "hybrid_agent"]:
        builder.add_edge(specialist, "synthesise_output")

    return builder.compile(checkpointer=checkpointer)
```

### Workflow Patterns

**Sequential — A → B → C**
```python
# Use for: ordered processes where each step depends on the previous
# Example: generate report → format → send email → log

builder.set_entry_point("generate")
builder.add_edge("generate",  "format")
builder.add_edge("format",    "send_email")
builder.add_edge("send_email","log")
builder.add_edge("log",       END)
```

**Parallel — (A + B) → C**
```python
# Use for: independent tasks that can run simultaneously
# Example: retrieve from multiple sources → merge

def fan_out(state) -> list[Send]:
    return [Send("retrieve", {"source": s}) for s in state["sources"]]

builder.add_conditional_edges("start", lambda x: x, ["retrieve"])  # fan-out
builder.add_edge("retrieve", "merge")  # all converge here
```

**Map-Reduce — split → (process×N in parallel) → aggregate**
```python
# Use for: same operation applied to N items
# Example: summarise N documents → combine summaries

def split(state) -> list[Send]:
    return [Send("process_item", {"item": item}) for item in state["items"]]

def process_item(state) -> dict:
    return {"results": [process(state["item"])]}  # appended via reducer

def aggregate(state) -> dict:
    final = combine(state["results"])
    return {"final": final}
```

**Retry-with-Reflection — attempt → evaluate → (accept | revise → attempt)**
```python
# Use for: quality-sensitive tasks needing iteration
# Example: write code → run tests → fix failures → re-test

def should_retry(state) -> str:
    if state["test_passed"] or state["attempts"] >= 3:
        return "accept"
    return "retry"

builder.add_conditional_edges("evaluate", should_retry,
    {"accept": END, "retry": "attempt"})  # cycle back
```

**Waterfall with HITL Gates**
```python
# Use for: multi-stage process where each stage needs human sign-off
# Example: requirements → design → implementation → review → deploy

# Each stage: generate output → HITL interrupt → human approves → next stage
graph = builder.compile(
    checkpointer=checkpointer,
    interrupt_after=["generate_requirements", "generate_design", "generate_code"],
)
```

### State Machines for Agents

```python
"""
State machines make agent behaviour explicit and predictable.
Instead of the agent deciding "what to do next" freely,
you define all valid states and transitions.

Traffic light: RED | GREEN | YELLOW
  Valid transitions: RED→GREEN, GREEN→YELLOW, YELLOW→RED
  Invalid (impossible): RED→YELLOW directly

Support ticket state machine:
  OPEN | INVESTIGATING | AWAITING_CUSTOMER | RESOLVED | ESCALATED
  Valid: OPEN→INVESTIGATING, INVESTIGATING→AWAITING_CUSTOMER
  Invalid: RESOLVED→ESCALATED (resolved tickets can't be escalated)
"""

from enum import Enum

class TicketStatus(str, Enum):
    OPEN             = "open"
    INVESTIGATING    = "investigating"
    AWAITING_CUSTOMER= "awaiting_customer"
    RESOLVED         = "resolved"
    ESCALATED        = "escalated"

VALID_TRANSITIONS = {
    TicketStatus.OPEN:              [TicketStatus.INVESTIGATING, TicketStatus.ESCALATED],
    TicketStatus.INVESTIGATING:     [TicketStatus.AWAITING_CUSTOMER, TicketStatus.RESOLVED, TicketStatus.ESCALATED],
    TicketStatus.AWAITING_CUSTOMER: [TicketStatus.INVESTIGATING, TicketStatus.RESOLVED],
    TicketStatus.RESOLVED:          [],   # terminal state
    TicketStatus.ESCALATED:         [TicketStatus.INVESTIGATING, TicketStatus.RESOLVED],
}

def validate_transition(current: TicketStatus, proposed: TicketStatus) -> bool:
    return proposed in VALID_TRANSITIONS.get(current, [])

def transition_status_node(state: TicketState) -> dict:
    proposed = determine_new_status(state)
    if not validate_transition(state["status"], proposed):
        logger.warning(f"Invalid transition: {state['status']} → {proposed}")
        return {}  # no change
    return {"status": proposed}
```

### Dead Letter Handling

```python
# What happens when an agent gets stuck?

class DeadLetterHandler:
    """Handle agents that fail, loop, or timeout"""

    def __init__(self, max_retries: int = 3, timeout_seconds: int = 300):
        self.max_retries = max_retries
        self.timeout = timeout_seconds

    def handle_stuck_agent(self, thread_id: str, state: dict) -> dict:
        """Called when agent exceeds max_iterations or timeout"""
        logger.error(f"Agent stuck: thread_id={thread_id}, step={state.get('step_count')}")

        # 1. Log for debugging
        DeadLetterRepository.create({
            "thread_id": thread_id,
            "state":     state,
            "reason":    "max_iterations_exceeded",
        })

        # 2. Notify user
        notify_user(state["user_id"], "Your request could not be completed. Our team has been notified.")

        # 3. Alert operations team
        slack_alert(f"Agent stuck: {thread_id}")

        # 4. Return graceful failure
        return {"final_answer": "I was unable to complete this request. Please try rephrasing your question."}
```

### Agent Timeout Budgets

```python
# Always set total time limits — never let an agent run forever

import asyncio

async def run_agent_with_timeout(
    agent,
    input: dict,
    config: dict,
    timeout_seconds: int = 120,  # 2 minute hard limit
) -> dict:
    """Run agent with a total wall-clock time limit"""
    try:
        result = await asyncio.wait_for(
            agent.ainvoke(input, config=config),
            timeout=timeout_seconds,
        )
        return result
    except asyncio.TimeoutError:
        logger.error(f"Agent timed out after {timeout_seconds}s")
        # Clean up the checkpoint? Or leave for debugging?
        return {
            "final_answer": f"Request timed out after {timeout_seconds} seconds. "
                           "Please try a simpler query.",
            "error": "timeout",
        }
    except Exception as e:
        logger.error(f"Agent error: {e}")
        return {"final_answer": "An error occurred. Please try again.", "error": str(e)}
```

---

## 4.7 — Fine-Tuning Foundations

### 🧑‍🎓 Explain Like I'm in BTech 2nd Year

A pre-trained LLM (like Llama or GPT-4) is like a brilliant student who graduated from university. They're smart, know a lot, but don't know your specific company's processes, terminology, or style.

Fine-tuning is like giving them a 3-month internship at your company:
- They read your internal documents
- They practice your specific format
- They learn your domain's vocabulary
- At the end: they respond like an expert in YOUR domain

```
Before fine-tuning:
  User: "What's the CLTV for Premium segment in Q3?"
  LLM:  "CLTV stands for Customer Lifetime Value. It is calculated as..."
        (general textbook answer)

After fine-tuning on your company's data:
  User: "What's the CLTV for Premium segment in Q3?"
  LLM:  "Based on our Q3 data, Premium segment CLTV is ₹45,230 (up 12% YoY),
        driven primarily by higher retention in the South India market."
        (knows your business context)
```

### When to Fine-Tune vs RAG

```
Decision Framework:

                         Knowledge Available?
                         ┌──────────────────────────────────────┐
                         │ YES (have the docs/data)             │
                         │                                       │
           Need it       │  → Use RAG                           │
           dynamic?      │  (knowledge updates without retraining)│
           (changes      ├──────────────────────────────────────┤
           frequently?)  │ NO (knowledge is in model's weights   │
                         │     or doesn't exist in any document) │
                         │  → Consider Fine-tuning              │
                         └──────────────────────────────────────┘

USE RAG WHEN:
  ✅ Information changes frequently (prices, policies, news)
  ✅ You need source citations
  ✅ Information is in documents/databases
  ✅ You want to add knowledge without retraining

USE FINE-TUNING WHEN:
  ✅ Style/format consistency matters more than content
     (e.g., "always respond in this exact JSON format")
  ✅ Domain vocabulary/jargon not in base model
     (e.g., your company's internal acronyms)
  ✅ Need faster/cheaper inference (smaller fine-tuned model vs large base)
  ✅ Privacy: can't send proprietary data to API
  ✅ Consistent tone/persona (always respond like Brand Voice Guide says)

COMBINE BOTH WHEN:
  ✅ Fine-tune for style + RAG for facts
  → Best of both worlds
  → Fine-tuned model knows how to respond; RAG provides what to say
```

### LoRA — Low-Rank Adaptation

**Theory:**

When you fine-tune a model, you're adjusting its weights. But a 7B parameter model has 7 BILLION numbers to adjust. That's expensive.

LoRA's insight: **weight updates during fine-tuning are naturally low-rank**. You don't need to update all 7B numbers. You can approximate the update with two much smaller matrices.

```
Full fine-tuning:
  Weight matrix W (e.g., 4096 × 4096 = 16.7M params) → update all of them
  
LoRA:
  W stays frozen (never changes)
  Add: ΔW = A × B
    where A: 4096 × 8 matrix (tiny!)
          B: 8 × 4096 matrix (tiny!)
    rank r = 8 (instead of full 4096)
  
  During inference: W' = W + ΔW
  Training: only update A and B (two tiny matrices)
  
  Parameter count: 4096×8 + 8×4096 = 65,536 trainable parameters
  vs full fine-tuning: 16,777,216 parameters (256x fewer!)
```

```python
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load base model
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")

# Configure LoRA
lora_config = LoraConfig(
    r=16,                          # rank — higher = more capacity, more VRAM
                                   # typical: 8, 16, 32, 64
    lora_alpha=32,                 # scaling = alpha/r = 2x rank (typical)
    target_modules=[               # which layers to adapt
        "q_proj",                  # query projection (attention)
        "k_proj",                  # key projection
        "v_proj",                  # value projection
        "o_proj",                  # output projection
        "gate_proj",               # MLP gate
        "up_proj",                 # MLP up
        "down_proj",               # MLP down
    ],
    lora_dropout=0.05,             # regularisation
    bias="none",                   # don't adapt bias terms
    task_type=TaskType.CAUSAL_LM,  # language model task
)

# Apply LoRA to model
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# trainable params: 41,943,040 || all params: 8,072,806,400 || trainable%: 0.52%
# 0.52% of parameters are trainable! Everything else is frozen.
```

**LoRA hyperparameters explained:**
```
r (rank):
  Controls how much the model can adapt.
  r=4:  minimal adaptation, very fast, less VRAM
  r=16: good balance (most common)
  r=64: high adaptation, slower, more VRAM
  Rule: start with r=16, tune from there

lora_alpha:
  Scaling factor for LoRA updates.
  Effective scale = alpha / r
  alpha=32, r=16 → scale=2 (recommended starting point)
  Rule: alpha = 2 × r

target_modules:
  Which weight matrices to add LoRA to.
  More modules = more capacity + more VRAM.
  Start with attention layers (q_proj, v_proj) for minimal LoRA.
  Add MLP layers (gate_proj, up_proj, down_proj) for more capacity.

lora_dropout:
  Regularisation — prevents overfitting on small datasets.
  0.05: standard for datasets < 10K examples
  0: for large datasets (>100K examples)
```

### QLoRA — Fine-Tuning on Consumer Hardware

```
Problem: 7B model in full precision (FP16) takes ~14GB VRAM.
         Training requires ~28-56GB VRAM. Needs A100.

QLoRA solution:
  Step 1: Quantise base model to 4-bit NF4 (Normal Float 4)
          7B model: 28GB → 4-5GB VRAM! Fits on RTX 3090!
  Step 2: Add LoRA adapters on top of quantised model
          Only adapters train — base model is frozen AND quantised
  Step 3: Compute in BF16 precision despite 4-bit storage
          (dequantise for computation, requantise for storage)
```

```python
import torch
from transformers import BitsAndBytesConfig, AutoModelForCausalLM

# QLoRA quantisation config
quantisation_config = BitsAndBytesConfig(
    load_in_4bit=True,                         # 4-bit quantisation
    bnb_4bit_quant_type="nf4",                # NormalFloat4 (better than int4)
    bnb_4bit_compute_dtype=torch.bfloat16,    # compute in BF16
    bnb_4bit_use_double_quant=True,           # double quantisation (extra savings)
)

# Load quantised model (fits in much less VRAM)
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct",
    quantization_config=quantisation_config,
    device_map="auto",            # automatically place on GPU/CPU
    torch_dtype=torch.bfloat16,
)

# Add LoRA on top (same as before)
model = get_peft_model(model, lora_config)
```

**Hardware requirements:**

| Model | Full Fine-tune | LoRA | QLoRA |
|-------|---------------|------|-------|
| 7B   | 80GB A100     | 40GB A100  | **24GB RTX 3090** ✅ |
| 13B  | 2×A100 (160GB)| 80GB A100  | **40GB A100** |
| 70B  | 8×A100 (640GB)| 4×A100     | **2×A100 (160GB)** |

**Cloud options for QLoRA training:**
```
Google Colab Pro:  A100 40GB free/cheap — great for 7B QLoRA
Vast.ai:           RTX 3090 at $0.15/hr — cheapest option
RunPod:            A100 80GB — for 13B and larger
AWS p3.2xlarge:    V100 16GB — small models only
Lambda Labs:       A100 at competitive pricing
```

### Training Data Format

```python
# JSONL instruction format — what your training file looks like
# Each line is one training example

# Example: training data for SQL agent
training_data = [
    {
        "messages": [
            {
                "role":    "system",
                "content": "You are an expert SQL analyst. Generate accurate SQL queries.",
            },
            {
                "role":    "user",
                "content": "Show me the top 10 customers by total revenue for Q3 2025.",
            },
            {
                "role":    "assistant",
                "content": """I'll query the orders table to find top customers by revenue.

```sql
SELECT
    c.customer_name,
    c.customer_id,
    SUM(o.revenue) as total_revenue,
    COUNT(o.order_id) as order_count
FROM customers c
JOIN orders o ON c.customer_id = o.customer_id
WHERE o.order_date BETWEEN '2025-07-01' AND '2025-09-30'
GROUP BY c.customer_id, c.customer_name
ORDER BY total_revenue DESC
LIMIT 10;
```

This query joins customers with their orders, filters to Q3 2025,
aggregates revenue per customer, and returns the top 10.""",
            },
        ]
    },
    # ... more examples
]

# Save as JSONL
import json
with open("training_data.jsonl", "w") as f:
    for example in training_data:
        f.write(json.dumps(example) + "\n")
```

**Data quality principles:**
```
Quality > Quantity:
  500 expertly curated examples > 50,000 noisy ones
  Bad data teaches bad behaviour — garbage in, garbage out

What makes good training data:
  ✅ Covers all the scenarios your model will face
  ✅ Correct, complete answers (verified by domain expert)
  ✅ Consistent format/style throughout
  ✅ Diverse (different phrasings of same question)
  ✅ Balanced (not all easy questions)

Data sources:
  1. Human-written (best quality, expensive)
  2. Synthetic — GPT-4/Claude generates training data
     (scale cheap, verify sample of outputs)
  3. Distillation — larger model teachers smaller model
```

### Training with Unsloth (Fastest, Recommended)

```python
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments

# Load model with Unsloth (2x faster training, 70% less VRAM vs HuggingFace)
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Llama-3.1-8B-Instruct",
    max_seq_length=2048,
    dtype=None,          # auto-detect
    load_in_4bit=True,   # QLoRA
)

# Apply LoRA with Unsloth
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    lora_alpha=32,
    lora_dropout=0,      # 0 for Unsloth (their optimisation)
    bias="none",
    use_gradient_checkpointing="unsloth",  # saves VRAM
    random_state=42,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
)

# Training
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,       # your JSONL dataset
    dataset_text_field="text",
    max_seq_length=2048,
    args=TrainingArguments(
        output_dir="./outputs",
        num_train_epochs=3,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,   # effective batch = 2×4 = 8
        learning_rate=2e-4,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=10,
        save_steps=100,
        warmup_steps=50,
        lr_scheduler_type="cosine",
        report_to="wandb",    # W&B logging
    ),
)

trainer.train()

# Save LoRA adapters
model.save_pretrained("lora_adapters")
tokenizer.save_pretrained("lora_adapters")
# Adapter files are ~50-200MB (vs 14GB base model)
```

### Merge and Deploy

```python
# Option 1: Keep adapter separate (flexible — swap adapters at runtime)
from peft import PeftModel

base_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
model = PeftModel.from_pretrained(base_model, "lora_adapters")
# Use model for inference — slightly slower (adapter applied at runtime)

# Option 2: Merge adapter into base (faster inference — no adapter overhead)
merged_model = model.merge_and_unload()
merged_model.save_pretrained("merged_model")
# merged_model is just like a regular model — no PEFT overhead
# Deploy this to production

# Serve with vLLM (production serving)
# vllm serve merged_model --tensor-parallel-size 2
```

### Evaluation — Before/After Comparison

```python
# How to measure if fine-tuning actually helped

# 1. ROUGE-L (automated, fast, approximate)
from rouge_score import rouge_scorer

scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

def evaluate_rouge(predictions: list[str], references: list[str]) -> float:
    scores = [
        scorer.score(ref, pred)["rougeL"].fmeasure
        for pred, ref in zip(predictions, references)
    ]
    return sum(scores) / len(scores)

# 2. LLM-as-Judge (more accurate, automated)
def llm_judge(question: str, base_answer: str, ft_answer: str) -> str:
    response = llm.invoke(f"""Compare these two answers to the question.
Which is better? Reply ONLY with "BASE" or "FINE_TUNED" and a brief reason.

Question: {question}
Base model answer: {base_answer}
Fine-tuned answer: {ft_answer}""")
    return response.content

# Run on 50 test examples
results = [
    llm_judge(q, base_answers[i], ft_answers[i])
    for i, q in enumerate(test_questions)
]
ft_wins = sum(1 for r in results if "FINE_TUNED" in r)
print(f"Fine-tuned wins: {ft_wins}/50 ({ft_wins*2}%)")

# 3. Task-specific accuracy (most meaningful)
# For SQL generation: execute both queries, compare results
# For classification: measure exact match
# For extraction: measure F1 on extracted fields
```

---

## 🏗️ Phase 4 Projects

### Project 4A — Research + Report Agent (LangGraph)

**What you build:**
```
Input: research topic
Output: structured report with citations + confidence per claim

Flow:
  1. User inputs topic
  2. Agent creates research plan (Plan-and-Execute)
  3. Searches web for each plan item (Tavily tool)
  4. Reads top 3 results per search (web reader tool)
  5. Extracts key points and credibility
  6. HITL: shows outline, user adds/removes sections
  7. Writes full report section by section
  8. Self-reflection: evaluates each section, revises if needed
  9. Final report with citations + confidence scores

Stack: LangGraph + Tavily + Groq + React frontend
Time: 12 hrs
```

### Project 4B — Multi-Agent Customer Support (SynapseIQ style)

**What you build:**
```
Supervisor routes to:
  FAQ Agent (RAG over docs)
  Order Agent (SQL over orders DB)
  Billing Agent (processes refunds)
  Escalation Agent (creates tickets)

HITL: refunds > ₹5000 → pause → manager approves
Full LangGraph traces for every conversation
React frontend with real-time agent step display

Stack: LangGraph multi-agent + Qdrant + DuckDB + FastAPI + React
Time: 15 hrs
```

### Project 4C — SQL Intelligence Agent

**What you build:**
```
Natural language → multi-step SQL reasoning

Complex queries handled:
  "Compare revenue YoY for top 5 products, broken down by region"
  Agent: get schema → understand tables → write SQL for each product →
         join results → compute YoY → format with chart

HITL: show generated SQL before execution
Recharts visualisation of results

Stack: LangGraph + DuckDB + FastAPI + React
Time: 10 hrs
```

### Project 4D — PR Reviewer Agent

**What you build:**
```
GitHub webhook → PR created → agent triggered
Reads diff → analyses code → posts inline comments

Reviews for:
  Bugs and logic errors
  Security vulnerabilities
  Performance issues
  Test coverage gaps
  Style consistency

HITL: low-confidence suggestions flagged for human review
Agent posts directly to GitHub PR via GitHub API

Stack: LangGraph + GitHub API + FastAPI + GitHub Actions
Time: 12 hrs
```

---

## 📋 Interview Cheat Sheet

**Q: What is the ReAct pattern?**
```
ReAct = Reasoning + Acting.
The agent alternates:
  THOUGHT: "I need to find X"
  ACTION:  calls tool to get X
  OBSERVATION: reads tool result
  THOUGHT: "Now I need Y because X showed..."
  ...until final answer

Why it works: forces explicit reasoning before acting.
Without reasoning step: agent calls wrong tool, passes wrong args.
Paper: Yao et al. 2022. Significantly outperforms chain-of-thought alone.
```

**Q: When would you use Plan-and-Execute instead of ReAct?**
```
ReAct: reactive, one step at a time, good for exploratory problems
Plan-and-Execute: strategic, full plan first, good for complex well-defined tasks

Use Plan-and-Execute when:
  Task has many steps that are predictable upfront
  Completeness matters (don't want to forget any step)
  Want to show user the plan before executing (HITL opportunity)

Use ReAct when:
  Next step depends on what you find (truly exploratory)
  Simple tool-using tasks
  Unknown territory
```

**Q: What is Reflection in AI agents?**
```
Agent critiques its own output (or has a separate critic agent do it).
Iterates until quality threshold met or max iterations reached.

Two types:
1. Self-reflection: same agent generates and critiques
2. Multi-agent: specialist generator + specialist critic

When to use: quality-sensitive tasks where first attempt is often good but not great.
Examples: code generation, report writing, complex reasoning.
Cost: 2-3x more LLM calls. Worth it for quality-critical outputs.
```

**Q: Supervisor vs Swarm multi-agent architecture?**
```
Supervisor: central orchestrator routes tasks to specialists
  - One LLM decides who handles what
  - Specialists are focused (better at their job)
  - Easier to debug (who handled what is clear)
  - Good for: most production systems

Swarm: agents hand off to each other peer-to-peer
  - Each agent decides when to pass the baton
  - More autonomous
  - Harder to debug (who decided what?)
  - Good for: dynamic workflows where handoff logic is complex
```

**Q: When do you fine-tune vs use RAG?**
```
RAG: information is in documents, changes frequently, need citations
     → Add knowledge without retraining

Fine-tune: style consistency, domain vocabulary, smaller+faster model,
           privacy (can't send data to API), proprietary behaviour
           → Teach the model HOW to behave, not what to know

Best practice: combine both
  Fine-tune for consistent style and domain expertise
  RAG for up-to-date, citable factual information
```

**Q: What is LoRA and why is it used?**
```
LoRA = Low-Rank Adaptation.
Fine-tuning technique that adds small trainable matrices (A×B) 
to frozen base model weights.

Why: full fine-tuning of 7B model needs 40+GB VRAM.
     LoRA only trains 0.5-1% of parameters.
     7B with LoRA: 8-16GB VRAM.

Key params:
  r (rank): 8-64, controls capacity
  lora_alpha: scaling = alpha/r, typically 2×r
  target_modules: which layers to adapt (q_proj, v_proj etc.)

QLoRA: same as LoRA but base model is 4-bit quantised.
  7B model: 28GB → 4-5GB VRAM. Runs on RTX 3090!
```

---

## 🃏 Quick Revision Cards

```
CARD 1: Core Agent Patterns
  ReAct:              Thought → Action → Observe → repeat
  Plan-and-Execute:   Plan all steps → execute each
  Reflection:         Generate → Critique → Revise → repeat
  Parallelisation:    Run independent tasks simultaneously
  Key insight:        Tool description quality = agent reliability

CARD 2: Agent vs Workflow
  Workflow: fixed path, you define control flow in code
  Agent:    dynamic path, LLM defines control flow at runtime
  Rule:     start with workflow, add agent when flexibility needed
  Cost:     agents are more expensive (more LLM calls) and less predictable

CARD 3: HITL Patterns
  interrupt_before:  pause before node (review INPUT)
  interrupt_after:   pause after node (review OUTPUT)
  NodeInterrupt:     agent decides to pause itself
  aupdate_state:     human corrects state before resuming
  When needed:       irreversible actions, low confidence, thresholds, compliance

CARD 4: Multi-Agent Architectures
  Supervisor:     one orchestrator routes to specialists (most common)
  Hierarchical:   supervisor → sub-supervisors → workers
  Swarm:          peer-to-peer handoff, agents decide themselves
  CrewAI:         role-based, quick prototype (less control)
  LangGraph:      most flexible, production-grade, full state control

CARD 5: Orchestrator Types
  Static:   fixed path, no LLM routing, predictable, fast
  Dynamic:  LLM decides next step, flexible, harder to debug
  Hybrid:   fixed skeleton + LLM at branch points (best for production)

CARD 6: Workflow Patterns
  Sequential:     A → B → C (ordered dependencies)
  Parallel:       A + B → C (independent tasks)
  Map-reduce:     split → N×process → aggregate
  Retry-reflect:  attempt → evaluate → (accept | retry)
  HITL waterfall: stage → human approval → next stage

CARD 7: LoRA vs QLoRA
  LoRA:   freeze base, add trainable A×B matrices, 40GB VRAM for 7B
  QLoRA:  4-bit quantise base + LoRA on top, 24GB for 7B (RTX 3090!)
  r=16, alpha=32: standard starting point
  target: q_proj + v_proj (minimal) or all 7 attention+MLP layers (full)
  Result: 0.5-1% of params trainable, ~same quality as full fine-tune

CARD 8: Fine-Tune vs RAG Decision
  RAG:        knowledge in docs, changes frequently, need citations
  Fine-tune:  style consistency, domain vocab, privacy, smaller model
  Both:       fine-tune for style, RAG for facts (best combination)

CARD 9: Training Data
  Format: JSONL with system/user/assistant messages
  Quality > Quantity: 500 curated > 50K noisy
  Diversity: different phrasings of same question
  Verification: sample check on synthetic data
  Frameworks: Unsloth (fastest), HF PEFT (standard), LLaMA Factory (UI)

CARD 10: Agent Failure Handling
  Max iterations guard: step_count >= N → END (never infinite loop)
  Timeout budget: asyncio.wait_for(agent.run(), timeout=120)
  Dead letter: log, notify user, alert ops team
  Graceful degradation: return partial answer with disclaimer
  State machine: validate all transitions — no invalid state jumps
```

---

## ✅ Phase 4 Completion Checklist

```
CORE AGENT PATTERNS
[ ] Implemented ReAct agent from scratch (without using AgentExecutor)
[ ] Implemented Plan-and-Execute agent
[ ] Implemented Reflection loop (generate → critique → revise)
[ ] Implemented parallelisation with asyncio.gather / Send API
[ ] Can explain agent vs workflow and when to use each
[ ] Know why tool descriptions are critical for reliability

HITL
[ ] Implemented interrupt_before with FastAPI approve/reject endpoint
[ ] Implemented interrupt_after pattern
[ ] Implemented NodeInterrupt (dynamic interrupt from node)
[ ] Built async HITL (agent pauses, human reviews later, resumes)
[ ] Built HITL UI components (review modal, approve/reject buttons)
[ ] Implemented aupdate_state for human corrections

MULTI-AGENT
[ ] Built supervisor pattern with 3+ specialist agents
[ ] Built with LangGraph (subgraphs composing in parent graph)
[ ] Experimented with CrewAI (know the trade-offs)
[ ] Implemented failure isolation (one agent fails, others continue)
[ ] Implemented agent specialisation (each agent has focused tools)

ORCHESTRATION
[ ] Built sequential workflow
[ ] Built parallel workflow (fan-out)
[ ] Built map-reduce workflow (Send API)
[ ] Built retry-with-reflection cycle
[ ] Implemented state machine with valid transition validation
[ ] Implemented dead letter handling
[ ] Implemented agent timeout budget

FINE-TUNING
[ ] Understand LoRA math (W' = W + A×B, rank r << d)
[ ] Understand QLoRA (4-bit quantisation + LoRA)
[ ] Know key hyperparameters (r, lora_alpha, target_modules)
[ ] Prepared a training dataset in JSONL instruction format
[ ] Fine-tuned a 7B model with QLoRA on Google Colab
[ ] Evaluated before/after using LLM-as-judge
[ ] Merged LoRA adapter into base model for deployment
[ ] Can explain fine-tune vs RAG decision framework

PROJECTS
[ ] Project 4A: Research + Report Agent with reflection + HITL
[ ] Project 4B: Multi-agent customer support with supervisor pattern
[ ] Project 4C: SQL Intelligence Agent with LangGraph
[ ] Project 4D: PR Reviewer Agent with GitHub integration
```

---

*Phase 4 Remaining Topics | GenAI + LLMOps Engineering Roadmap 2026*
*MCP topics: topics/phase-4/mcp.md | LangGraph: phase-4/LANGGRAPH.md | LangChain: phase-4/LANGCHAIN.md*
*Next: Phase 5 — Design Patterns for Python GenAI*
