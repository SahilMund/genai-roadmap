# 🔍 CodeSense — AI PR Reviewer & Code Intelligence Platform
> **Type:** Capstone Project C · Production-Grade AI Developer Tool
> **Stack:** FastAPI · LangGraph · GitHub MCP · Claude Sonnet 4 · Qdrant · PostgreSQL · Redis · React 18 · Clerk · Langfuse · Docker · GitHub Actions
> **Timeline:** 3–4 weeks (50–65 hrs)
> **Demonstrates:** MCP + Agents + HITL + LLMOps + CI/CD + RAG + Code Analysis — all in one real product

---

## 📑 Table of Contents

1. [The Origin Story](#1-the-origin-story)
2. [What CodeSense Does](#2-what-codesense-does)
3. [Why This Is The Right Capstone](#3-why-this-is-the-right-capstone)
4. [System Architecture](#4-system-architecture)
5. [GitHub Webhook → Agent Pipeline](#5-github-webhook--agent-pipeline)
6. [LangGraph Review Agent](#6-langgraph-review-agent)
7. [Code Intelligence — Codebase RAG](#7-code-intelligence--codebase-rag)
8. [HITL — Senior Dev Review Queue](#8-hitl--senior-dev-review-queue)
9. [Custom YAML Configuration per Repo](#9-custom-yaml-configuration-per-repo)
10. [Weekly Codebase Health Report](#10-weekly-codebase-health-report)
11. [LLMOps Layer](#11-llmops-layer)
12. [Database Schema](#12-database-schema)
13. [API Reference](#13-api-reference)
14. [Folder Structure](#14-folder-structure)
15. [Phase-by-Phase Build Plan](#15-phase-by-phase-build-plan)
16. [Resume Deliverables + Interview Story](#16-resume-deliverables--interview-story)

---

## 1. The Origin Story

### The Problem

Engineering teams are bottlenecked on code review. A senior engineer gets 10-15 PR review requests a week. Each takes 20-40 minutes to do thoroughly. Most engineers review for correctness but miss security issues, edge cases, and documentation gaps under time pressure.

Meanwhile, junior engineers wait 1-3 days for review. They make the same mistakes repeatedly — not because they're bad engineers, but because feedback loops are slow and inconsistent.

The result:
- Slow cycle time (PR open → merged: avg 2.3 days at most companies)
- Senior engineers burnt out on review
- Security issues slip through
- Codebase quality degrades quietly

### The Solution

CodeSense is a GitHub-native AI code review agent. It installs as a GitHub App, hooks into every PR, and provides:

1. **Automated first-pass review** within 60 seconds of PR open (catches 80% of common issues)
2. **Inline comments** on specific diff lines with suggested code fixes
3. **HITL queue** for uncertain suggestions — routes to senior dev, not the author
4. **Codebase context** — uses RAG over the repo's own history to catch patterns and inconsistencies
5. **Weekly health report** — trend tracking, most common issue types, improvement over time

This is not a linter. Linters catch syntax. CodeSense catches logic bugs, security vulnerabilities, missing error handling, and architectural inconsistencies — using the actual diff in context with the full codebase.

---

## 2. What CodeSense Does

```
PR OPENED (GitHub event) ─────────────────────────────────────────────
  GitHub sends webhook to CodeSense
  CodeSense fetches: PR diff, changed files, PR description, author history
  LangGraph agent runs:
    [classify_pr]  → size, risk level, affected systems
    [fetch_context] → RAG: similar past PRs, related code, team standards
    [review_bugs]   → logic errors, null dereferences, off-by-one
    [review_security] → SQL injection, hardcoded secrets, auth bypass
    [review_tests]  → missing test cases, edge cases not covered
    [review_style]  → naming, docs, comment quality
    [review_arch]   → consistency with existing patterns
    [route_hitl]    → uncertain findings → senior dev queue
    [post_comments] → GitHub inline comments + summary comment

INLINE COMMENTS (on specific diff lines)
  Line 47: "This SQL query is vulnerable to injection. Suggested fix:"
           ```python
           cursor.execute("SELECT * FROM users WHERE id = %s", (user_id,))
           ```
  Line 83: "This function doesn't handle the case where `data` is None.
            Add a guard clause or default value."
  Line 102: "Missing error handling for network timeout in this API call."

PR SUMMARY COMMENT (posted by CodeSense bot)
  ## CodeSense Review Summary
  **Risk Level:** 🟡 Medium
  **Files Changed:** 8 | **Lines Added:** 247 | **Lines Removed:** 89

  ### Issues Found (6)
  | Severity | Category    | Count |
  |----------|-------------|-------|
  | 🔴 High  | Security    | 1     |
  | 🟡 Med   | Error Handling | 2  |
  | 🟢 Low   | Style       | 3     |

  ### Suggested Reviewers
  @sarah (owns the auth module) @rahul (most familiar with this service)

  ### HITL Queue
  2 findings escalated to senior dev review (low confidence)

WEEKLY CODEBASE HEALTH REPORT (every Monday 9 AM)
  Security issues introduced: 3 (down from 7 last week ✅)
  Test coverage delta: +2.3%
  Most common issue: missing error handling in API calls (12 occurrences)
  PRs reviewed: 34 | Avg review time: 48 seconds
  Comment acceptance rate: 67%
  Top contributor to debt: /services/payment/ (14 issues this week)
```

---

## 3. Why This Is The Right Capstone

```
INTERVIEWER ASKS:   "What have you built?"
MOST CANDIDATES:    "A RAG chatbot that answers questions about PDFs."
YOU:                "An AI code review platform that hooks into GitHub via
                     MCP, runs a LangGraph agent to analyse PRs, posts inline
                     comments, and routes uncertain findings to a HITL queue.
                     It uses RAG over the codebase's own history to detect
                     pattern violations. It has full LLMOps: Langfuse traces,
                     RAGAS eval on review quality, CI/CD with eval gate."

WHAT THIS SHOWS:
  ✅ GitHub MCP integration (real-world tool use)
  ✅ LangGraph multi-node agent (not just a single LLM call)
  ✅ HITL pattern (production-safe AI)
  ✅ RAG over code (not just documents — differentiated)
  ✅ LLMOps: Langfuse, eval, prompt versioning
  ✅ CI/CD: GitHub Actions, webhook security
  ✅ Product thinking: configurable, per-repo rules, health reports
  ✅ Real users can install it (GitHub App)

COMPARED TO SYNAPSEIQ (your main project):
  CodeSense is scoped, shippable in 3 weeks, and immediately relatable to
  every engineer interviewer — they all feel the pain of slow PR reviews.
  It's a demo you can literally run on your interviewer's public repo.
```

---

## 4. System Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         CodeSense Platform                              │
│                                                                         │
│  GitHub ──── webhook ────▶  FastAPI  ───────────────────────────────┐  │
│  (PR events)               /webhooks/github                         │  │
│                                      │                              │  │
│                            ┌─────────▼──────────────────────────┐  │  │
│                            │    Celery Task Queue               │  │  │
│                            │    (async PR processing)           │  │  │
│                            └─────────┬──────────────────────────┘  │  │
│                                      │                              │  │
│                            ┌─────────▼──────────────────────────┐  │  │
│                            │    LangGraph Review Agent          │  │  │
│                            │                                    │  │  │
│                            │  classify_pr                       │  │  │
│                            │      ↓                             │  │  │
│                            │  fetch_context (RAG)               │  │  │
│                            │      ↓                             │  │  │
│                            │  ┌─────────────────────────┐      │  │  │
│                            │  │  Parallel Review Nodes  │      │  │  │
│                            │  │  bugs | security | tests│      │  │  │
│                            │  │  style | arch | docs    │      │  │  │
│                            │  └───────────┬─────────────┘      │  │  │
│                            │      ↓       ↓                    │  │  │
│                            │  aggregate_findings               │  │  │
│                            │      ↓                             │  │  │
│                            │  route_hitl (uncertain → queue)   │  │  │
│                            │      ↓                             │  │  │
│                            │  post_to_github (MCP)             │  │  │
│                            └────────────────────────────────────┘  │  │
│                                                                     │  │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐           │  │
│  │  Qdrant  │  │ Postgres │  │  Redis   │  │ Langfuse │           │  │
│  │(code RAG)│  │(reviews) │  │(sessions)│  │(traces)  │           │  │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘           │  │
│                                                                     │  │
│  Dashboard (React + Clerk)                                          │  │
│  HITL review queue │ Repo health │ Config editor │ Analytics        │  │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 5. GitHub Webhook → Agent Pipeline

### GitHub App Setup + Webhook Security

```python
# app/api/webhooks.py

import hashlib, hmac, json
from fastapi import FastAPI, Request, HTTPException, BackgroundTasks
from app.tasks.review import run_pr_review_task

GITHUB_WEBHOOK_SECRET = os.getenv("GITHUB_WEBHOOK_SECRET")

@app.post("/webhooks/github")
async def github_webhook(
    request:    Request,
    background: BackgroundTasks,
):
    # ── Step 1: Verify webhook signature (security-critical) ─────────
    signature = request.headers.get("X-Hub-Signature-256", "")
    body      = await request.body()

    expected  = "sha256=" + hmac.new(
        GITHUB_WEBHOOK_SECRET.encode(),
        body,
        hashlib.sha256,
    ).hexdigest()

    if not hmac.compare_digest(signature, expected):
        raise HTTPException(401, "Invalid webhook signature")

    # ── Step 2: Parse event ────────────────────────────────────────────
    event   = request.headers.get("X-GitHub-Event")
    payload = json.loads(body)
    action  = payload.get("action")

    # ── Step 3: Route events ──────────────────────────────────────────
    if event == "pull_request" and action in ("opened", "synchronize", "reopened"):
        pr_data = {
            "repo_full_name":   payload["repository"]["full_name"],
            "pr_number":        payload["pull_request"]["number"],
            "pr_title":         payload["pull_request"]["title"],
            "pr_body":          payload["pull_request"]["body"] or "",
            "author":           payload["pull_request"]["user"]["login"],
            "base_branch":      payload["pull_request"]["base"]["ref"],
            "head_sha":         payload["pull_request"]["head"]["sha"],
            "installation_id":  payload["installation"]["id"],
        }
        # Async — respond to GitHub immediately (< 10s requirement)
        background.add_task(run_pr_review_task, pr_data)
        return {"status": "queued", "pr": pr_data["pr_number"]}

    # Ignore other events
    return {"status": "ignored", "event": event}

# Celery task — runs in background worker
@celery_app.task(name="run_pr_review", max_retries=2)
def run_pr_review_task(pr_data: dict):
    import asyncio
    asyncio.run(_run_review(pr_data))

async def _run_review(pr_data: dict):
    config = {"configurable": {"thread_id": f"pr_{pr_data['repo_full_name']}_{pr_data['pr_number']}"}}
    await review_graph.ainvoke({"pr_data": pr_data, "findings": []}, config=config)
```

### GitHub MCP Tool Integration

```python
# app/agent/tools/github_mcp.py
"""
GitHub MCP gives the agent tools to:
  - Fetch PR diff (line-by-line changes)
  - Read file contents from the repo
  - Post inline review comments
  - Post PR summary comments
  - Request reviewers
  - Get commit history + blame
"""

from langchain_mcp_adapters.client import MultiServerMCPClient

mcp_client = MultiServerMCPClient({
    "github": {
        "command": "npx",
        "args":    ["-y", "@modelcontextprotocol/server-github"],
        "env":     {"GITHUB_PERSONAL_ACCESS_TOKEN": os.getenv("GITHUB_TOKEN")},
        "transport":"stdio",
    }
})

# Tools the agent gets:
# get_pull_request       → PR metadata + description
# get_pull_request_files → list of changed files with patch/diff
# get_file_contents      → read any file at any commit SHA
# create_review          → post review with inline comments
# create_issue_comment   → post PR-level summary comment
# list_commits           → recent commit history
# get_blame              → who last changed each line

async def get_pr_tools():
    tools = await mcp_client.get_tools()
    return {t.name: t for t in tools if t.name in [
        "get_pull_request",
        "get_pull_request_files",
        "get_file_contents",
        "create_review",
        "create_issue_comment",
        "list_commits",
    ]}
```

---

## 6. LangGraph Review Agent

### State

```python
# app/agent/state.py

from typing import TypedDict, Annotated
from operator import add

class ReviewFinding(TypedDict):
    file:        str
    line:        int
    end_line:    int | None
    category:    str    # bug | security | test | style | arch | docs
    severity:    str    # critical | high | medium | low | info
    title:       str
    body:        str    # markdown — shown in GitHub comment
    suggestion:  str | None   # code suggestion block
    confidence:  float  # 0.0–1.0 (< 0.70 → HITL queue)
    hitl_needed: bool

class ReviewState(TypedDict):
    # ── PR metadata ───────────────────────────────────────────────────
    pr_data:        dict        # from GitHub webhook payload
    repo_config:    dict        # loaded from .codesense.yml

    # ── Fetched content ───────────────────────────────────────────────
    pr_diff:        str         # raw unified diff
    changed_files:  list[dict]  # [{filename, patch, additions, deletions}]
    codebase_context: str       # RAG: similar patterns from history

    # ── Analysis ──────────────────────────────────────────────────────
    pr_classification: dict     # {size, risk_level, affected_systems}
    findings:        Annotated[list[ReviewFinding], add]

    # ── Routing ───────────────────────────────────────────────────────
    high_confidence_findings:   list[ReviewFinding]
    hitl_findings:              list[ReviewFinding]

    # ── Output ────────────────────────────────────────────────────────
    github_review_posted: bool
    review_id:            str | None
```

### Graph

```python
# app/agent/graph.py

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.redis import RedisSaver
import asyncio

def build_review_graph(redis_client) -> "CompiledGraph":
    builder = StateGraph(ReviewState)

    # Nodes
    builder.add_node("load_config",       load_repo_config_node)
    builder.add_node("fetch_pr_content",  fetch_pr_content_node)
    builder.add_node("classify_pr",       classify_pr_node)
    builder.add_node("fetch_context",     fetch_codebase_context_node)

    # Parallel review nodes (run simultaneously via Send API)
    builder.add_node("review_bugs",       review_bugs_node)
    builder.add_node("review_security",   review_security_node)
    builder.add_node("review_tests",      review_tests_node)
    builder.add_node("review_style",      review_style_node)
    builder.add_node("review_arch",       review_arch_node)

    builder.add_node("aggregate",         aggregate_findings_node)
    builder.add_node("route_hitl",        route_hitl_node)
    builder.add_node("post_github",       post_to_github_node)
    builder.add_node("create_hitl_tasks", create_hitl_tasks_node)

    # Edges
    builder.set_entry_point("load_config")
    builder.add_edge("load_config",       "fetch_pr_content")
    builder.add_edge("fetch_pr_content",  "classify_pr")
    builder.add_edge("classify_pr",       "fetch_context")

    # Fan-out to parallel review nodes
    builder.add_conditional_edges(
        "fetch_context",
        fan_out_reviews,   # returns Send() objects for each enabled check
        ["review_bugs", "review_security", "review_tests",
         "review_style", "review_arch"],
    )

    # All parallel nodes converge into aggregate
    for node in ["review_bugs", "review_security", "review_tests",
                 "review_style", "review_arch"]:
        builder.add_edge(node, "aggregate")

    builder.add_edge("aggregate",     "route_hitl")
    builder.add_edge("route_hitl",    "post_github")
    builder.add_edge("post_github",   "create_hitl_tasks")
    builder.add_edge("create_hitl_tasks", END)

    return builder.compile(checkpointer=RedisSaver(redis_client))

from langgraph.types import Send

def fan_out_reviews(state: ReviewState) -> list:
    """Fan out to parallel review nodes based on repo config"""
    config   = state["repo_config"]
    checks   = config.get("checks", {})
    pr_diff  = state["pr_diff"]
    context  = state["codebase_context"]
    base     = {"pr_diff": pr_diff, "context": context,
                "repo_config": state["repo_config"]}

    sends = []
    if checks.get("bugs",     True): sends.append(Send("review_bugs",     base))
    if checks.get("security", True): sends.append(Send("review_security", base))
    if checks.get("tests",    True): sends.append(Send("review_tests",    base))
    if checks.get("style",    True): sends.append(Send("review_style",    base))
    if checks.get("arch",     True): sends.append(Send("review_arch",     base))
    return sends
```

### Review Nodes

```python
# app/agent/nodes/review_nodes.py

from pydantic import BaseModel
from langfuse.decorators import observe
import instructor
from anthropic import AsyncAnthropic

# Claude Sonnet 4 — best for code understanding
anthropic = AsyncAnthropic()
client    = instructor.from_anthropic(anthropic)

class ReviewFindings(BaseModel):
    findings: list[ReviewFinding]

REVIEW_PROMPTS = {
    "bugs": """You are a senior software engineer reviewing a pull request for bugs.

Analyse the diff below for:
- Logic errors and incorrect conditionals
- Off-by-one errors and boundary conditions
- Null/None pointer dereferences
- Race conditions in concurrent code
- Unhandled return values or errors
- Incorrect data type assumptions
- Missing edge cases (empty list, zero, negative values)

For each finding, provide:
- Exact file and line number from the diff
- Clear title (< 10 words)
- Explanation of the bug and why it's wrong
- Suggested fix as a code block
- Confidence (0.0–1.0): how certain you are this is actually a bug

Only report actual bugs — not style preferences.
confidence < 0.70 means you're not sure — flag for human review.""",

    "security": """You are a security engineer reviewing a pull request.

Check for:
- SQL injection (string formatting in queries)
- Command injection (subprocess with user input)
- Hardcoded secrets, API keys, passwords
- Insecure direct object references
- Missing authentication/authorisation checks
- Unsafe deserialization
- Path traversal vulnerabilities
- XSS vulnerabilities in template rendering
- SSRF (Server-Side Request Forgery)
- Insecure cryptography (MD5, SHA1 for passwords)

Be precise — security false positives destroy trust.
confidence < 0.80 for security → always flag for human review.""",

    "tests": """You are a test engineer reviewing a pull request.

Identify:
- New code paths not covered by new tests
- Edge cases the tests don't cover (empty input, None, max values)
- Missing error case tests (what happens when the API returns 500?)
- Tests that don't actually assert anything meaningful
- Test data that doesn't represent real scenarios

Do NOT comment on test style or naming.
Only flag missing coverage for logic that matters.""",

    "style": """You are reviewing code style and documentation.

Check only for:
- Public functions/classes missing docstrings
- Misleading variable/function names
- Magic numbers without constants
- Dead code (unused variables, imports, functions)
- Functions > 50 lines that should be split

Do NOT flag:
- Minor formatting (use a linter for that)
- Personal style preferences
- Things already covered by Black/ESLint/Prettier

confidence < 0.60 → skip, not worth the noise.""",

    "arch": """You are a senior architect reviewing code consistency.

The codebase context below shows existing patterns.
Identify where this PR:
- Uses a different pattern from what already exists (e.g., direct DB call where a repository is used elsewhere)
- Duplicates logic that already exists in another module
- Introduces a dependency that breaks layering (e.g., service calling another service's DB directly)
- Misses an existing utility that should be reused

This requires codebase context — only flag if the context shows a clear pattern violation.""",
}

@observe(name="review_bugs")
async def review_bugs_node(state: dict) -> dict:
    return await _run_review_node(state, "bugs")

@observe(name="review_security")
async def review_security_node(state: dict) -> dict:
    return await _run_review_node(state, "security")

@observe(name="review_tests")
async def review_tests_node(state: dict) -> dict:
    return await _run_review_node(state, "tests")

@observe(name="review_style")
async def review_style_node(state: dict) -> dict:
    return await _run_review_node(state, "style")

@observe(name="review_arch")
async def review_arch_node(state: dict) -> dict:
    return await _run_review_node(state, "arch")

async def _run_review_node(state: dict, category: str) -> dict:
    """Generic review node — sends diff + category-specific prompt to Claude"""
    system_prompt = REVIEW_PROMPTS[category]
    diff          = state["pr_diff"]
    context       = state.get("context", "")

    result: ReviewFindings = await client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=2000,
        response_model=ReviewFindings,
        system=system_prompt,
        messages=[{
            "role":    "human",
            "content": (
                f"Codebase context (existing patterns):\n{context[:2000]}\n\n"
                f"PR diff to review:\n```diff\n{diff[:8000]}\n```"
            ),
        }],
    )

    # Tag each finding with its category
    for finding in result.findings:
        finding["category"]    = category
        finding["hitl_needed"] = finding["confidence"] < 0.70

    langfuse_context.update_current_observation(
        metadata={"category": category, "findings_count": len(result.findings)}
    )

    return {"findings": result.findings}
```

### Post to GitHub

```python
# app/agent/nodes/post_github.py

@observe(name="post_to_github")
async def post_to_github_node(state: ReviewState) -> dict:
    """Post inline comments + summary via GitHub MCP"""
    tools    = await get_pr_tools()
    pr       = state["pr_data"]
    findings = state["high_confidence_findings"]

    if not findings:
        return {"github_review_posted": True}

    # Build inline comments (GitHub review format)
    comments = []
    for f in findings:
        body = f"**[{f['severity'].upper()}] {f['title']}**\n\n{f['body']}"
        if f.get("suggestion"):
            body += f"\n\n```suggestion\n{f['suggestion']}\n```"

        comments.append({
            "path":       f["file"],
            "line":       f["line"],
            "body":       body,
            "side":       "RIGHT",   # comment on the new code
        })

    # Post all inline comments as a single review
    severity_emoji = {"critical": "🔴", "high": "🔴", "medium": "🟡",
                       "low": "🟢", "info": "ℹ️"}

    # Group by severity for summary
    by_severity = {}
    for f in findings:
        by_severity.setdefault(f["severity"], []).append(f)

    summary_table = "| Severity | Category | Count |\n|---|---|---|\n"
    for sev in ["critical", "high", "medium", "low"]:
        group = by_severity.get(sev, [])
        if group:
            cats = ", ".join(set(f["category"] for f in group))
            summary_table += f"| {severity_emoji[sev]} {sev.title()} | {cats} | {len(group)} |\n"

    hitl_count = len(state["hitl_findings"])
    body = f"""## 🔍 CodeSense Review

**Risk Level:** {_risk_emoji(state['pr_classification']['risk_level'])} {state['pr_classification']['risk_level'].title()}

### Issues Found ({len(findings)})
{summary_table}

{"### ⏳ HITL Queue" if hitl_count else ""}
{"f'{hitl_count} finding(s) with low confidence escalated to senior dev review.' if hitl_count else ''}

---
*Reviewed by CodeSense · [View Dashboard](https://codesense.dev) · [Configure](.codesense.yml)*"""

    # Create GitHub review via MCP
    review_result = await tools["create_review"].ainvoke({
        "owner":      pr["repo_full_name"].split("/")[0],
        "repo":       pr["repo_full_name"].split("/")[1],
        "pull_number":pr["pr_number"],
        "commit_id":  pr["head_sha"],
        "body":       body,
        "event":      "COMMENT",   # COMMENT (not APPROVE/REQUEST_CHANGES — AI shouldn't block)
        "comments":   comments,
    })

    return {
        "github_review_posted": True,
        "review_id":            str(review_result.get("id")),
    }

def _risk_emoji(risk: str) -> str:
    return {"critical": "🔴", "high": "🟠", "medium": "🟡", "low": "🟢"}.get(risk, "⚪")
```

---

## 7. Code Intelligence — Codebase RAG

```python
# app/services/code_rag.py
"""
RAG over the codebase itself — not just documentation.
Index: every Python/JS/TS file in the repo → chunked → embedded → Qdrant.
At review time: retrieve similar code patterns, related files, past PR fixes.

Why this matters:
  Without codebase context: "You should use a repository pattern."
  With codebase context:    "You're calling DB directly here (line 47), but
                             the rest of the codebase uses UserRepository.
                             See: /services/user_service.py line 23."

That second comment is actionable. The first is just advice.
"""

from pathlib import Path
from tree_sitter import Language, Parser   # parse code into AST chunks
import tree_sitter_python as tspython
import tree_sitter_javascript as tsjs

SUPPORTED_EXTENSIONS = {".py", ".js", ".ts", ".tsx", ".go", ".java", ".rs"}

class CodebaseIndexer:
    """Index an entire codebase into Qdrant for context-aware code review"""

    def __init__(self, qdrant_client, embedder):
        self.qdrant  = qdrant_client
        self.embedder= embedder
        self.COLLECTION = "codebase"

    async def index_repo(self, repo_path: str, repo_name: str):
        """Index all code files in a repo"""
        chunks = []
        for path in Path(repo_path).rglob("*"):
            if path.suffix not in SUPPORTED_EXTENSIONS:
                continue
            if any(p in str(path) for p in ["node_modules", ".git", "__pycache__",
                                              "venv", ".venv", "dist", "build"]):
                continue
            try:
                code    = path.read_text(encoding="utf-8", errors="ignore")
                file_chunks = self._chunk_code(code, str(path), repo_name)
                chunks.extend(file_chunks)
            except Exception:
                continue

        await self._embed_and_store(chunks, repo_name)
        print(f"Indexed {len(chunks)} code chunks from {repo_name}")

    def _chunk_code(self, code: str, file_path: str, repo_name: str) -> list[dict]:
        """
        Chunk code by function/class boundaries (not fixed token windows).
        A function is a natural unit — it's self-contained.
        Splitting inside a function loses context.
        """
        chunks = []
        lines  = code.split("\n")

        # Simple heuristic: split on function/class definitions
        BLOCK_STARTERS = ("def ", "class ", "function ", "const ", "async def ",
                           "export default", "export function", "export class")

        current_chunk   = []
        current_start   = 0

        for i, line in enumerate(lines):
            stripped = line.strip()
            # New block starting — save current chunk
            if any(stripped.startswith(s) for s in BLOCK_STARTERS) and current_chunk:
                chunks.append({
                    "content":   "\n".join(current_chunk),
                    "file_path": file_path,
                    "start_line":current_start,
                    "end_line":  i - 1,
                    "repo":      repo_name,
                    "language":  Path(file_path).suffix.lstrip("."),
                })
                current_chunk = [line]
                current_start = i
            else:
                current_chunk.append(line)
                # Hard limit: max 80 lines per chunk
                if len(current_chunk) >= 80:
                    chunks.append({
                        "content":   "\n".join(current_chunk),
                        "file_path": file_path,
                        "start_line":current_start,
                        "end_line":  i,
                        "repo":      repo_name,
                        "language":  Path(file_path).suffix.lstrip("."),
                    })
                    current_chunk = []
                    current_start = i + 1

        # Last chunk
        if current_chunk:
            chunks.append({
                "content":   "\n".join(current_chunk),
                "file_path": file_path,
                "start_line":current_start,
                "end_line":  len(lines) - 1,
                "repo":      repo_name,
                "language":  Path(file_path).suffix.lstrip("."),
            })

        return [c for c in chunks if c["content"].strip()]

    async def _embed_and_store(self, chunks: list[dict], repo_name: str):
        """Embed code chunks and store in Qdrant"""
        from qdrant_client.models import PointStruct, VectorParams, Distance
        import hashlib, uuid

        # Ensure collection exists
        try:
            self.qdrant.get_collection(self.COLLECTION)
        except Exception:
            self.qdrant.create_collection(
                self.COLLECTION,
                vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
            )

        BATCH = 50
        for i in range(0, len(chunks), BATCH):
            batch     = chunks[i:i+BATCH]
            texts     = [c["content"] for c in batch]
            embeddings= await self.embedder.embed(texts)

            points = [
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=emb,
                    payload={k: v for k, v in chunk.items() if k != "content"}
                    | {"content": chunk["content"][:500]},  # truncate for payload
                )
                for chunk, emb in zip(batch, embeddings)
            ]
            self.qdrant.upsert(self.COLLECTION, points=points)

    async def get_context_for_pr(self, pr_diff: str, changed_files: list[str],
                                  repo_name: str, k: int = 6) -> str:
        """
        Given a PR diff, retrieve the most relevant codebase patterns.
        Uses: (a) semantic search on diff, (b) file-adjacent search on changed files.
        """
        # (a) Semantic: what patterns in the diff match existing code?
        diff_embedding = await self.embedder.embed([pr_diff[:2000]])[0]
        semantic_results = self.qdrant.search(
            self.COLLECTION,
            query_vector=diff_embedding,
            query_filter={"must": [{"key": "repo", "match": {"value": repo_name}}]},
            limit=k // 2,
        )

        # (b) Adjacent: retrieve code from the same modules as changed files
        adjacent_results = []
        for filepath in changed_files[:3]:   # top 3 changed files
            module    = str(Path(filepath).parent)
            results   = self.qdrant.scroll(
                self.COLLECTION,
                scroll_filter={"must": [
                    {"key": "repo",      "match": {"value": repo_name}},
                    {"key": "file_path", "match": {"text":  module}},
                ]},
                limit=2,
            )
            adjacent_results.extend(results[0])

        all_chunks = (
            [r.payload for r in semantic_results] +
            [r.payload for r in adjacent_results]
        )

        # Format as context
        context_parts = []
        seen = set()
        for chunk in all_chunks:
            key = chunk["file_path"] + str(chunk.get("start_line", 0))
            if key in seen:
                continue
            seen.add(key)
            context_parts.append(
                f"// {chunk['file_path']} (lines {chunk.get('start_line')}-{chunk.get('end_line')})\n"
                f"{chunk['content']}"
            )

        return "\n\n---\n\n".join(context_parts[:k])
```

---

## 8. HITL — Senior Dev Review Queue

```python
# app/agent/nodes/route_hitl.py

async def route_hitl_node(state: ReviewState) -> dict:
    """Split findings into high-confidence (auto-post) and HITL (human review)"""
    all_findings  = state["findings"]
    repo_config   = state["repo_config"]

    # Threshold from repo config (default: 0.70)
    hitl_threshold = repo_config.get("hitl_threshold", 0.70)

    # Security findings always go to HITL regardless of confidence
    security_hitl_threshold = repo_config.get("security_hitl_threshold", 0.80)

    auto_post = []
    hitl      = []

    for finding in all_findings:
        threshold = (security_hitl_threshold if finding["category"] == "security"
                     else hitl_threshold)
        if finding["confidence"] < threshold or finding.get("hitl_needed"):
            hitl.append(finding)
        else:
            auto_post.append(finding)

    # Store HITL findings in Postgres for dashboard
    for finding in hitl:
        await HITLRepository.create({
            "pr_data":     state["pr_data"],
            "finding":     finding,
            "status":      "pending",
            "priority":    "high" if finding["category"] == "security" else "normal",
        })

    return {
        "high_confidence_findings": auto_post,
        "hitl_findings":            hitl,
    }

# app/api/hitl.py — human reviewer endpoints
@router.get("/hitl/queue")
async def get_hitl_queue(current_user = Depends(get_current_user)):
    return await HITLRepository.get_pending(org_id=current_user.org_id)

@router.post("/hitl/{finding_id}/post")
async def post_hitl_finding(finding_id: str, body: PostFindingRequest,
                              current_user = Depends(get_current_user)):
    """Human confirms finding → post to GitHub"""
    finding = await HITLRepository.get(finding_id)
    # Post to GitHub via MCP with human-edited body
    tools   = await get_pr_tools()
    await tools["create_review"].ainvoke({
        "owner":   finding["pr_data"]["repo_full_name"].split("/")[0],
        "repo":    finding["pr_data"]["repo_full_name"].split("/")[1],
        "pull_number": finding["pr_data"]["pr_number"],
        "commit_id":   finding["pr_data"]["head_sha"],
        "body":    body.edited_body or finding["finding"]["body"],
        "event":   "COMMENT",
        "comments":[{"path": finding["finding"]["file"],
                      "line": finding["finding"]["line"],
                      "body": body.edited_body or finding["finding"]["body"]}],
    })
    await HITLRepository.update(finding_id, {"status": "posted", "posted_by": current_user.user_id})
    return {"status": "posted"}

@router.post("/hitl/{finding_id}/dismiss")
async def dismiss_hitl_finding(finding_id: str, body: DismissRequest,
                                 current_user = Depends(get_current_user)):
    """Human dismisses finding — false positive"""
    await HITLRepository.update(finding_id, {
        "status":     "dismissed",
        "dismissed_by": current_user.user_id,
        "dismiss_reason": body.reason,
    })
    # This feedback trains the model — dismissed findings = false positives
    await FeedbackStore.record_false_positive(finding_id)
    return {"status": "dismissed"}
```

---

## 9. Custom YAML Configuration per Repo

```yaml
# .codesense.yml  (checked into the repo root)

version: "1.0"

# Which checks to run
checks:
  bugs:     true
  security: true
  tests:    true
  style:    false    # team uses ESLint for style, don't duplicate
  arch:     true
  docs:     false    # not a priority for this team

# Confidence thresholds
hitl_threshold:          0.70   # findings below this → human review
security_hitl_threshold: 0.80   # stricter for security

# Severity rules
severity_rules:
  fail_on:               []          # don't block PRs — COMMENT only
  skip_below:            "info"      # don't post info-level findings

# File exclusions
exclude_paths:
  - "tests/**"
  - "migrations/**"
  - "*.min.js"
  - "vendor/**"

# Custom rules (appended to standard prompts)
custom_rules:
  bugs: |
    This repo uses async SQLAlchemy. Check for:
    - Missing await on async DB calls
    - Session not closed in finally block
    - N+1 queries in loops
  security: |
    This is a financial application. Extra scrutiny on:
    - Any code handling payment data
    - JWT validation and expiry checks
    - Admin endpoint authorization

# Auto-assign reviewers based on changed files
reviewer_rules:
  - pattern: "services/payment/**"
    reviewers: ["@payment-team"]
  - pattern: "auth/**"
    reviewers: ["@security-team"]
  - pattern: "**"
    reviewers: []   # CodeSense will suggest based on git blame

# Weekly report
report:
  enabled:   true
  schedule:  "0 9 * * 1"   # Monday 9 AM
  recipients: ["engineering@company.com"]
  slack_channel: "#engineering"
```

```python
# app/services/config_loader.py

import yaml
from pathlib import Path

DEFAULT_CONFIG = {
    "version": "1.0",
    "checks": {"bugs": True, "security": True, "tests": True,
                "style": True, "arch": True, "docs": False},
    "hitl_threshold":          0.70,
    "security_hitl_threshold": 0.80,
    "severity_rules":          {"fail_on": [], "skip_below": "info"},
    "exclude_paths":           ["tests/**", "*.min.js"],
    "custom_rules":            {},
    "reviewer_rules":          [],
    "report":                  {"enabled": True, "schedule": "0 9 * * 1"},
}

async def load_repo_config_node(state: ReviewState) -> dict:
    """Load .codesense.yml from the repo root using GitHub MCP"""
    pr  = state["pr_data"]
    tools = await get_pr_tools()

    try:
        result = await tools["get_file_contents"].ainvoke({
            "owner": pr["repo_full_name"].split("/")[0],
            "repo":  pr["repo_full_name"].split("/")[1],
            "path":  ".codesense.yml",
            "ref":   pr["head_sha"],
        })
        import base64
        content = base64.b64decode(result["content"]).decode("utf-8")
        config  = yaml.safe_load(content)
        # Merge with defaults (user config overrides defaults)
        merged  = {**DEFAULT_CONFIG, **config}
        merged["checks"] = {**DEFAULT_CONFIG["checks"], **config.get("checks", {})}
    except Exception:
        # .codesense.yml not found → use defaults
        merged = DEFAULT_CONFIG

    return {"repo_config": merged}
```

---

## 10. Weekly Codebase Health Report

```python
# app/tasks/weekly_report.py
# Celery beat: runs every Monday at 9 AM

from celery.schedules import crontab

@celery_app.task(name="generate_weekly_report")
def generate_weekly_report_task():
    import asyncio
    asyncio.run(_generate_report())

async def _generate_report():
    """Aggregate last week's findings → LLM summary → send to Slack/email"""
    from_date = datetime.utcnow() - timedelta(days=7)

    # Pull all findings from last 7 days
    stats = await ReviewRepository.get_weekly_stats(from_date)
    # stats = {total_prs, total_findings, by_severity, by_category,
    #           acceptance_rate, false_positive_rate, top_files, trend}

    # LLM generates the narrative section
    narrative = await groq.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{
            "role": "user",
            "content": f"""Write a concise engineering weekly report based on these code review stats.
Tone: data-driven, constructive, actionable.

Stats: {json.dumps(stats, indent=2)}

Include:
1. One-sentence summary of codebase health (better/worse/stable vs last week)
2. Top 2 recurring issues with specific recommendation
3. One positive trend to celebrate
4. Single action item for next week

Max 200 words.""",
        }],
        max_tokens=300,
    )

    report = {
        "week":             from_date.strftime("%Y-W%U"),
        "prs_reviewed":     stats["total_prs"],
        "findings":         stats["total_findings"],
        "by_severity":      stats["by_severity"],
        "by_category":      stats["by_category"],
        "acceptance_rate":  stats["acceptance_rate"],
        "top_files":        stats["top_files"][:5],
        "narrative":        narrative.choices[0].message.content,
        "trend":            stats["trend"],   # up/down vs last week
    }

    # Post to Slack
    await post_slack_report(report)
    # Store in Postgres for dashboard
    await ReportRepository.create(report)
```

---

## 11. LLMOps Layer

### Eval Suite

```python
# evals/eval_review_quality.py
"""
Golden dataset: 30 real PRs with known bugs/issues (manually identified).
Eval: does CodeSense find the known issues? Does it avoid false positives?

Metrics:
  recall:      % of real issues found by CodeSense (target: > 70%)
  precision:   % of CodeSense findings that are real issues (target: > 60%)
  false positive rate: dismissed findings / total findings (target: < 20%)
"""

from ragas import evaluate
from ragas.metrics import AnswerRelevancyMetric
from deepeval.metrics import GEval, LLMTestCaseParams
from deepeval.test_case import LLMTestCase

# Custom eval: is the code review comment actionable?
actionability_metric = GEval(
    name="ReviewActionability",
    criteria="The code review comment is specific, actionable, and includes a suggested fix.",
    evaluation_steps=[
        "Does the comment identify a specific line or code block?",
        "Does it explain WHY this is a problem (not just that it is)?",
        "Does it suggest a concrete fix or improvement?",
        "Is it free from vague advice like 'consider improving this'?",
    ],
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
    threshold=0.70,
    model="gpt-4o-mini",
)

GOLDEN_DATASET = [
    {
        "pr_diff": "...",           # real diff with known SQL injection on line 47
        "expected_findings": [
            {"category": "security", "line": 47, "issue": "sql_injection"}
        ],
    },
    # ... 29 more examples
]

async def run_eval():
    results = {"recall": [], "precision": [], "actionability": []}
    for case in GOLDEN_DATASET:
        found = await graphrag_query(case["pr_diff"])  # run the agent
        # Compare found vs expected
        recall    = len(found & expected) / len(expected)
        precision = len(found & expected) / max(len(found), 1)
        results["recall"].append(recall)
        results["precision"].append(precision)

    avg = {k: sum(v)/len(v) for k, v in results.items()}
    print(f"Recall: {avg['recall']:.2f} | Precision: {avg['precision']:.2f}")

    # CI/CD gate
    if avg["recall"] < 0.70 or avg["precision"] < 0.60:
        print("❌ Eval gate FAILED")
        exit(1)
    print("✅ Eval gate PASSED")
```

### Prompt Versioning + A/B

```
prompts/
  review_bugs/
    v1.txt        ← current production prompt
    v2.txt        ← testing: stricter about null checks
    meta.json     ← {current: v1, ab_test: {v1: 0.80, v2: 0.20}}
  review_security/
    v1.txt
    meta.json
```

### Langfuse Integration

Every LLM call tagged with: `repo`, `pr_number`, `review_category`, `confidence`, `posted_to_github`. Dashboard shows: cost per PR review, which category finds the most issues, false positive rate by category, prompt version performance.

---

## 12. Database Schema

```sql
CREATE TABLE repos (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    full_name       TEXT UNIQUE NOT NULL,   -- "org/repo"
    installation_id TEXT NOT NULL,          -- GitHub App installation
    config          JSONB,                  -- cached .codesense.yml
    indexed_at      TIMESTAMPTZ,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE pr_reviews (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    repo_id         UUID REFERENCES repos(id),
    pr_number       INTEGER NOT NULL,
    pr_title        TEXT,
    author          TEXT,
    head_sha        TEXT,
    status          TEXT DEFAULT 'processing',  -- processing|completed|failed
    risk_level      TEXT,
    total_findings  INTEGER DEFAULT 0,
    auto_posted     INTEGER DEFAULT 0,
    hitl_count      INTEGER DEFAULT 0,
    github_review_id TEXT,
    processing_ms   INTEGER,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX ON pr_reviews (repo_id, created_at DESC);

CREATE TABLE findings (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    review_id   UUID REFERENCES pr_reviews(id),
    file_path   TEXT NOT NULL,
    line_number INTEGER,
    category    TEXT NOT NULL,
    severity    TEXT NOT NULL,
    title       TEXT NOT NULL,
    body        TEXT NOT NULL,
    suggestion  TEXT,
    confidence  FLOAT NOT NULL,
    auto_posted BOOLEAN DEFAULT FALSE,
    created_at  TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX ON findings (review_id);
CREATE INDEX ON findings (category, severity);

CREATE TABLE hitl_queue (
    id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    finding_id   UUID REFERENCES findings(id),
    review_id    UUID REFERENCES pr_reviews(id),
    repo_id      UUID REFERENCES repos(id),
    status       TEXT DEFAULT 'pending',   -- pending|posted|dismissed
    priority     TEXT DEFAULT 'normal',    -- high|normal
    posted_by    TEXT,                     -- Clerk user_id
    dismissed_by TEXT,
    dismiss_reason TEXT,
    reviewed_at  TIMESTAMPTZ,
    created_at   TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX ON hitl_queue (status, priority, created_at);

CREATE TABLE weekly_reports (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    repo_id         UUID REFERENCES repos(id),
    week            TEXT NOT NULL,          -- "2026-W10"
    prs_reviewed    INTEGER,
    total_findings  JSONB,                  -- {critical:2, high:5, medium:12, low:8}
    acceptance_rate FLOAT,
    false_pos_rate  FLOAT,
    top_files       JSONB,
    narrative       TEXT,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE feedback (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    finding_id  UUID REFERENCES findings(id),
    signal      TEXT NOT NULL,   -- accepted | dismissed | edited
    edited_body TEXT,
    created_at  TIMESTAMPTZ DEFAULT NOW()
);
-- feedback table feeds continuous improvement of review quality
```

---

## 13. API Reference

```
── WEBHOOK ────────────────────────────────────────────────────────────
POST  /webhooks/github            GitHub App webhook (all events)

── REPOS ──────────────────────────────────────────────────────────────
GET   /repos                      List connected repos
GET   /repos/{id}                 Repo details + config
POST  /repos/{id}/index           Trigger codebase indexing (RAG)
GET   /repos/{id}/config          Current .codesense.yml config
PATCH /repos/{id}/config          Override config via dashboard

── REVIEWS ────────────────────────────────────────────────────────────
GET   /reviews                    List all PR reviews (paginated)
GET   /reviews/{id}               Review details + findings
POST  /reviews/manual             Manually trigger review for any PR
GET   /reviews/{id}/findings      All findings for a review

── HITL QUEUE ─────────────────────────────────────────────────────────
GET   /hitl/queue                 Pending findings needing human review
GET   /hitl/{finding_id}          Finding detail (code, context, suggestion)
POST  /hitl/{finding_id}/post     Confirm and post to GitHub
POST  /hitl/{finding_id}/dismiss  Dismiss as false positive
POST  /hitl/{finding_id}/edit     Edit body then post

── ANALYTICS ──────────────────────────────────────────────────────────
GET   /analytics/overview         Total PRs, findings, acceptance rate
GET   /analytics/trends           Weekly trend: findings by severity
GET   /analytics/categories       Finding breakdown by category
GET   /analytics/false-positives  False positive rate by category
GET   /analytics/top-files        Most flagged files in the codebase

── REPORTS ────────────────────────────────────────────────────────────
GET   /reports                    All weekly reports
GET   /reports/latest             Most recent report

── FEEDBACK ───────────────────────────────────────────────────────────
POST  /feedback/{finding_id}      Accept/dismiss inline (from GitHub comment)
```

---

## 14. Folder Structure

```
codesense/
├── app/
│   ├── main.py
│   ├── agent/
│   │   ├── graph.py               ← LangGraph review graph
│   │   ├── state.py               ← ReviewState TypedDict
│   │   ├── nodes/
│   │   │   ├── load_config.py     ← .codesense.yml loader
│   │   │   ├── fetch_pr.py        ← GitHub MCP: get diff + files
│   │   │   ├── classify_pr.py     ← size, risk, affected systems
│   │   │   ├── fetch_context.py   ← Qdrant codebase RAG
│   │   │   ├── review_bugs.py
│   │   │   ├── review_security.py
│   │   │   ├── review_tests.py
│   │   │   ├── review_style.py
│   │   │   ├── review_arch.py
│   │   │   ├── aggregate.py       ← merge all parallel findings
│   │   │   ├── route_hitl.py      ← confidence routing
│   │   │   └── post_github.py     ← GitHub MCP: post review
│   │   └── tools/
│   │       └── github_mcp.py      ← MCP client + tool wrappers
│   ├── api/
│   │   ├── webhooks.py
│   │   ├── repos.py
│   │   ├── reviews.py
│   │   ├── hitl.py
│   │   ├── analytics.py
│   │   └── feedback.py
│   ├── services/
│   │   ├── code_rag.py            ← codebase indexer + retriever
│   │   └── config_loader.py
│   ├── tasks/
│   │   ├── celery_app.py
│   │   ├── review_task.py         ← async PR review runner
│   │   └── weekly_report.py       ← scheduled Monday report
│   └── repositories/
│       ├── review_repo.py
│       ├── hitl_repo.py
│       └── feedback_repo.py
│
├── prompts/
│   ├── review_bugs/
│   │   ├── v1.txt
│   │   └── meta.json
│   ├── review_security/
│   │   ├── v1.txt
│   │   └── meta.json
│   ├── review_tests/v1.txt
│   ├── review_style/v1.txt
│   └── review_arch/v1.txt
│
├── evals/
│   ├── golden_prs/                ← 30 PRs with known issues (JSON)
│   ├── eval_review_quality.py     ← recall, precision, actionability
│   └── golden_dataset.json
│
├── frontend/                      ← React 18 + Vite + Tailwind + Clerk
│   └── src/pages/
│       ├── Dashboard.tsx          ← overview + metrics
│       ├── Reviews.tsx            ← PR review history
│       ├── HITLQueue.tsx          ← pending human review
│       ├── RepoConfig.tsx         ← .codesense.yml editor UI
│       └── Reports.tsx            ← weekly health reports
│
├── tests/
│   ├── test_webhook_security.py   ← signature verification
│   ├── test_review_graph.py       ← graph runs end to end
│   ├── test_hitl_routing.py       ← confidence threshold routing
│   ├── test_config_loader.py      ← YAML parse + defaults merge
│   └── test_code_rag.py           ← index + retrieve round trip
│
├── .github/
│   └── workflows/
│       ├── ci.yml                 ← test + lint + eval gate
│       └── deploy.yml             ← ECS or Railway deploy
│
├── docker-compose.yml             ← api + celery + postgres + redis + qdrant
├── Dockerfile
└── README.md
```

---

## 15. Phase-by-Phase Build Plan

### Phase 0 — Foundation (3 days)
```
[ ] GitHub App created (github.com/settings/apps/new)
    Permissions: pull_requests (read+write), contents (read), metadata (read)
    Webhook events: pull_request
[ ] FastAPI + webhook endpoint + signature verification
[ ] Celery + Redis broker
[ ] Postgres schema migrated
[ ] Docker Compose: api + celery + postgres + redis + qdrant
[ ] Clerk auth on dashboard routes

Showable: Open a PR → GitHub sends webhook → FastAPI receives it
          CloudWatch (or console): "Webhook received: PR #42 opened"
```

### Phase 1 — GitHub MCP + Basic Review (5 days)
```
[ ] GitHub MCP server wired into LangGraph
[ ] fetch_pr_content_node: get diff + changed files via MCP
[ ] Single review node (review_bugs only) producing findings
[ ] post_to_github_node: post inline comments via MCP create_review
[ ] load_repo_config_node: read .codesense.yml or use defaults
[ ] classify_pr_node: size + risk level

Showable: Open a PR with a deliberate bug →
          CodeSense posts an inline comment on the bug line within 60 seconds
```

### Phase 2 — All Review Nodes + Parallel (4 days)
```
[ ] All 5 review nodes: bugs, security, tests, style, arch
[ ] fan_out_reviews: Send API for parallel execution
[ ] aggregate_findings_node: merge + deduplicate
[ ] Custom rules from .codesense.yml injected into prompts
[ ] PR summary comment with severity table
[ ] Suggested reviewers based on git blame

Showable: PR with 3 different issue types → all caught, summary comment posted
          .codesense.yml disables style check → no style comments appear
```

### Phase 3 — Codebase RAG (4 days)
```
[ ] CodebaseIndexer: chunk by function boundary, embed, store in Qdrant
[ ] POST /repos/{id}/index triggers full repo indexing
[ ] fetch_context_node: semantic + adjacent file search
[ ] Architecture review node uses codebase context
[ ] Context injected into each review node's prompt

Showable: "You're calling DB directly here, but /services/ uses UserRepository"
          Diff without context: generic advice
          Diff with context: specific, codebase-aware comment
```

### Phase 4 — HITL Queue + Dashboard (3 days)
```
[ ] route_hitl_node: confidence < threshold → HITLRepository
[ ] GET /hitl/queue returns pending findings
[ ] HITLQueue.tsx: finding detail, approve/edit/dismiss
[ ] POST /hitl/{id}/post resumes GitHub posting
[ ] false positive feedback stored in feedback table
[ ] Slack notification on new HITL finding

Showable: Low-confidence finding → appears in HITL queue → human posts it
          Dismiss a false positive → logged as negative feedback
```

### Phase 5 — LLMOps + Weekly Report (4 days)
```
[ ] Prompt versioning: all prompts in prompts/ directory
[ ] Langfuse traces: every LLM call tagged (repo, pr, category, confidence)
[ ] Eval suite: 30 golden PRs, recall + precision computed
[ ] CI/CD: GitHub Actions → lint → test → eval gate → deploy
[ ] Weekly report Celery beat task (Monday 9 AM)
[ ] Analytics dashboard page: trends, category breakdown, top files

Showable: Full CI/CD run — push to main → eval gate passes → auto-deploy
          Weekly report in Slack: "34 PRs reviewed, 67% acceptance rate"
          Langfuse dashboard: cost per PR, best/worst performing category
```

---

## 16. Resume Deliverables + Interview Story

### Deliverables

```
GitHub repo:     codesense — architecture diagram, demo GIF of review being posted
GitHub App:      installable by others (github.com/apps/codesense-review)
Live dashboard:  Railway or Fly.io
Key metrics:
  Avg review time: 48 seconds after PR opened
  Recall on golden dataset: 72%
  Precision: 65%
  Comment acceptance rate: 67%
  False positive rate (dismissed): 18%
Langfuse dashboard: cost per PR review, per-category performance
Eval report: RAGAS-style table showing precision/recall by category
```

### The 2-Minute Interview Story

> *"My capstone is CodeSense — an AI code review platform that installs as a GitHub App and reviews every PR automatically.*
>
> *The agent is built with LangGraph and runs five specialised review nodes in parallel — bugs, security, missing tests, style, and architectural consistency. Each node uses Claude Sonnet 4 with a category-specific system prompt. The parallel execution via LangGraph's Send API means a full PR review completes in about 48 seconds.*
>
> *The codebase context is what makes the architectural review actually useful. I index the entire repo by function boundaries into Qdrant. When a PR comes in, I retrieve semantically similar code patterns and adjacent module code. So instead of generic advice like 'use a repository pattern,' the agent says 'you're calling the database directly on line 47, but this codebase uses UserRepository — see /services/user_service.py line 23.' That's actionable.*
>
> *For HITL: any finding below 0.70 confidence goes to a human review queue instead of being posted directly. Security findings have a stricter 0.80 threshold. Reviewers see the finding, the context, and the suggested fix, and can post, edit, or dismiss it. Dismissed findings feed back as negative training signal.*
>
> *The LLMOps layer: prompts are versioned in Git, every LLM call is traced in Langfuse with cost and latency, and there's a CI/CD eval gate that runs against 30 real PRs with known issues. The gate fails if recall drops below 70%. On the golden dataset we're hitting 72% recall with 65% precision and 18% false positive rate."*

---

*CodeSense — AI PR Reviewer & Code Intelligence Platform · Capstone Project C*
*LangGraph · GitHub MCP · Claude Sonnet 4 · Qdrant Code RAG · HITL Queue · LLMOps · GitHub Actions*
