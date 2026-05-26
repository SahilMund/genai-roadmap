# 🔌 MCP — Model Context Protocol
> **Complete Deep-Dive Notes | Architecture | JSON-RPC 2.0 | Coding Guide | FastMCP**  
> Spec version: **2025-11-25** | Released by Anthropic: November 2024  
> Part of: GenAI + LLMOps Engineering Roadmap 2026 — Phase 1.5

---

## Watch these videos for MCP projects

[![Watch the video](https://img.youtube.com/vi/3_TN1i3MTEU/maxresdefault.jpg)](https://youtu.be/3_TN1i3MTEU)
[![Watch the video](https://img.youtube.com/vi/j5f2EQf5hkw/maxresdefault.jpg)](https://www.youtube.com/watch?v=j5f2EQf5hkw)


## 📑 Table of Contents

1. [Before MCP — The World Without It](#-before-mcp--the-world-without-it)
2. [Why MCP Was Needed](#-why-mcp-was-needed)
3. [Function Calling vs MCP](#-function-calling-vs-mcp)
4. [What is MCP — In-Depth](#-what-is-mcp--in-depth)
5. [MCP Architecture](#-mcp-architecture)
6. [MCP Primitives](#-mcp-primitives)
7. [Data Layer — JSON-RPC 2.0](#-data-layer--json-rpc-20)
8. [Standard Methods — Full Reference](#-standard-methods--full-reference)
9. [Why JSON-RPC, Not REST](#-why-json-rpc-not-rest)
10. [Transport Layer](#-transport-layer)
11. [Local (stdio) vs Remote (HTTP+SSE)](#-local-stdio-vs-remote-httpsse)
12. [Full Architecture Diagram](#-full-architecture-diagram)
13. [MCP Lifecycle](#-mcp-lifecycle)
14. [Error Handling, Cancellation, Timeouts](#-error-handling-cancellation-timeouts)
15. [Connection Types — Config Files vs Connectors](#-connection-types)
16. [Coding Guide](#-coding-guide)
17. [FastMCP Deep-Dive](#-fastmcp-deep-dive)
18. [Building an Expense Tracker MCP Server](#-building-an-expense-tracker-mcp-server)
19. [FastAPI + MCP Integration](#-fastapi--mcp-integration)
20. [Deploying MCP Servers](#-deploying-mcp-servers)
21. [Is MCP Dead? Alternatives](#-is-mcp-dead-alternatives)
22. [Resources & Links](#-resources--links)

---

## 🌎 Before MCP — The World Without It

### 🧑‍🎓 Explain Like I'm in BTech 2nd Year

Imagine every mobile app needed a **custom cable** to charge — iPhone had one connector, Samsung had another, laptops had their own. That was the nightmare before USB-C.

The AI tooling world before MCP was exactly that nightmare:

```
Before MCP (The M × N Problem):

  Claude ──── custom code ──── GitHub
  Claude ──── custom code ──── Postgres
  Claude ──── custom code ──── Slack
  Claude ──── custom code ──── Notion

  GPT-4  ──── custom code ──── GitHub    ← different custom code again!
  GPT-4  ──── custom code ──── Postgres  ← different custom code again!
  GPT-4  ──── custom code ──── Slack     ← different custom code again!

  N AI apps × M tools = N × M custom integrations 😱
  5 AI apps × 20 tools = 100 integrations to maintain
```

Every AI company had to write their own tool-calling code, their own authentication patterns, their own serialisation logic — for every single tool, for every single AI. When one API changed, everything broke.

### The Real Pain Points

- **No standard** — every integration was bespoke, built specifically for one LLM and one tool
- **No reuse** — a GitHub integration built for Claude couldn't be reused with Cursor or GPT-4
- **No discovery** — AI couldn't dynamically discover what tools were available
- **No stateful sessions** — every call was a fresh REST request with no session context
- **Maintenance nightmare** — M tools × N AIs = M×N integrations to keep up to date

---

## 🎯 Why MCP Was Needed

### The USB-C Moment for AI

> **MCP = USB-C for AI tools.** One standard port. Any device. Any cable. Just works.

```
After MCP (The M + N Solution):

  [Claude Desktop]──┐
  [Cursor]          ├──── MCP Protocol ──── [GitHub MCP Server]
  [Windsurf]        │                       [Postgres MCP Server]
  [GPT-4 client]   ──┘                       [Slack MCP Server]
  [Your custom app]                          [Your custom MCP Server]

  N AI apps + M MCP servers = N + M things to maintain ✅
  5 AI apps + 20 servers = 25 things to maintain (vs 100 before!)
```

**MCP was announced by Anthropic on November 25, 2024.** By March 2025, ChatGPT adopted it. Microsoft Copilot, Google Gemini's agentic surface, Cursor, and Windsurf followed. It is now the de facto open standard for AI ↔ tool integration.

### What MCP Actually Solves

| Problem Before MCP | How MCP Fixes It |
|---|---|
| Custom code per integration | One standard protocol, write once |
| No dynamic tool discovery | Servers declare capabilities at handshake |
| No stateful sessions | Persistent JSON-RPC session per connection |
| No reuse across AI clients | Any MCP client talks to any MCP server |
| Authentication inconsistency | OAuth 2.1 + PKCE standardised in spec |
| No bidirectional comms | JSON-RPC supports server→client messages |

---

## ⚔️ Function Calling vs MCP

This is one of the most common interview questions. They are **NOT competing** — they solve different problems at different layers.

### Function Calling — What It Is

Function calling (tool use) is a feature of LLM APIs where the model outputs a structured JSON object requesting a specific function to be called. It is **ad-hoc and per-provider**.

```python
# Function calling — defined INSIDE every API call
# Works only with OpenAI's specific format

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_github_issues",
            "description": "List open issues on a GitHub repo",
            "parameters": {
                "type": "object",
                "properties": {
                    "repo": {"type": "string"},
                    "state": {"type": "string", "enum": ["open", "closed"]}
                },
                "required": ["repo"]
            }
        }
    }
]

# Must redefine this schema for EVERY API call
# Must rewrite completely for Anthropic (different format)
# Must rewrite again for Gemini (different format again)
```

### MCP — What It Is

MCP is a **protocol standard** — a persistent, stateful session where the AI client discovers tool schemas *dynamically* from the server rather than having them hardcoded into every API call.

```
Function Calling:
  Defined in every API call → per-provider → can't be reused → bespoke

MCP:
  Defined once in MCP server → any client discovers it → standard protocol → reusable
```

### Side-by-Side Comparison

| Dimension | Function Calling | MCP |
|-----------|-----------------|-----|
| **Standard** | Provider-specific (OpenAI ≠ Anthropic ≠ Gemini) | Universal open standard |
| **Schema location** | Hardcoded in every API call | Declared once in MCP server |
| **Discovery** | Manual — you define tools per call | Automatic — client queries `tools/list` |
| **Session** | Stateless — no session concept | Stateful — persistent JSON-RPC session |
| **Bidirectional** | No (LLM → tool only) | Yes (client ↔ server) |
| **Reusability** | Zero — rewrite per AI provider | Full — one server, all MCP clients |
| **Auth standard** | None (roll your own) | OAuth 2.1 + PKCE in spec |
| **Notifications** | Not supported | Yes — server pushes updates to client |
| **Use when** | Quick one-off tool in a single app | Building reusable tool ecosystem |

### They Work Together

```
In practice: MCP uses function-calling concepts INSIDE the protocol

1. MCP client connects to MCP server (persistent session)
2. Client calls tools/list → discovers all available tools WITH their schemas
3. When AI decides to use a tool, client sends tools/call request via MCP
4. MCP server executes and returns result

MCP is the infrastructure layer.
Function calling is the mechanism inside that infrastructure.
```

---

## 🔍 What is MCP — In-Depth

### Formal Definition

> MCP (Model Context Protocol) is an open, stateful JSON-RPC 2.0 based standard that lets any AI application discover tools, reusable prompts, resources, and other context from remote MCP servers, then invoke them through a persistent session — without writing any provider-specific integration code.

Published by Anthropic on **November 25, 2024**, the protocol is now maintained as an open-source spec on GitHub and supported by a growing ecosystem of clients, servers, and tools.

### The Core Insight

MCP took direct inspiration from the **Language Server Protocol (LSP)** — the standard that let every text editor (VS Code, Neovim, Emacs) work with every programming language (Python, TypeScript, Rust) without M×N custom extensions.

```
Language Server Protocol (LSP):
  VS Code  ─── LSP ─── Python language server
  Neovim   ─── LSP ─── TypeScript language server
  Any editor ─ LSP ─── Any language server

Model Context Protocol (MCP):
  Claude Desktop ─── MCP ─── GitHub server
  Cursor         ─── MCP ─── Postgres server
  Any AI client  ─── MCP ─── Any MCP server
```

### What MCP Enables

- **Tools** — AI can call external functions (search web, query DB, send email)
- **Resources** — AI can read external data (file contents, API responses, DB records)
- **Prompts** — Reusable prompt templates the AI can surface as workflows
- **Sampling** — Server can request the AI to generate text (server → LLM)
- **Dynamic discovery** — AI learns what's available at runtime, not compile time

---

## 🏗️ MCP Architecture

### Three-Layer Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         MCP HOST                                    │
│  (Claude Desktop, Cursor, Windsurf, your custom app)               │
│                                                                     │
│  The HOST is the top-level application that:                       │
│  - Contains the LLM                                                │
│  - Creates and manages MCP Client instances                        │
│  - Enforces security and user consent                              │
│  - Decides WHICH tools the LLM can see                            │
│                                                                     │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐               │
│  │ MCP Client 1│  │ MCP Client 2│  │ MCP Client 3│               │
│  │ (isolated)  │  │ (isolated)  │  │ (isolated)  │               │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘               │
└─────────┼────────────────┼────────────────┼─────────────────────┘
          │ stdio          │ stdio          │ HTTP+SSE
          ▼                ▼                ▼
  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
  │ MCP Server   │  │ MCP Server   │  │ MCP Server   │
  │  (Local)     │  │  (Local)     │  │  (Remote)    │
  │              │  │              │  │              │
  │ Tools:       │  │ Tools:       │  │ Tools:       │
  │ - read_file  │  │ - run_query  │  │ - get_tweet  │
  │ - write_file │  │ - list_tables│  │ - post_tweet │
  │              │  │              │  │              │
  │ Resources:   │  │ Resources:   │  │ Resources:   │
  │ - file://    │  │ - schema://  │  │ - timeline:/ │
  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘
         │                 │                  │
         ▼                 ▼                  ▼
    File System        PostgreSQL          Twitter API
```

### The Three Roles

```
HOST      = The application containing the LLM and orchestrating everything
            Think: Claude Desktop, Cursor, your LangGraph agent

CLIENT    = A connection manager created BY the host for EACH server
            Maintains the JSON-RPC session, handles negotiation
            One client per server, isolated from other clients

SERVER    = Exposes tools/resources/prompts to connected clients
            Knows nothing about the host or other servers
            Lightweight, focused, single-purpose
```

### Key Design Principles

1. **Servers are isolated** — each server only sees its own client, not the host or other servers
2. **Clients are isolated** — each client manages exactly one server connection
3. **Host controls security** — the host decides which tools the LLM can access and enforces user consent
4. **Servers can be dumb** — a server doesn't need to know what LLM is on the other end

---

## 🧩 MCP Primitives

MCP servers expose exactly **three primitives** to clients (server-side), and clients expose **three primitives** to servers (client-side).

### Server-Side Primitives

#### 1. 🔧 Tools — "Functions the AI Can Call"

The most used primitive. Tools are executable actions — the AI requests them, the server runs them, results come back.

```
Analogy: Tools are like API endpoints. The AI calls them like a developer calls an API.

Examples:
  - create_github_issue(title, body, labels)
  - run_sql_query(query)
  - send_slack_message(channel, text)
  - search_web(query)
  - read_file(path)
  - get_weather(city)
```

```python
from fastmcp import FastMCP

mcp = FastMCP("github-server")

@mcp.tool()
def create_issue(
    repo: str,
    title: str,
    body: str,
    labels: list[str] = []
) -> dict:
    """
    Create a new GitHub issue.
    
    Args:
        repo: Repository in format 'owner/repo' e.g. 'microsoft/vscode'
        title: Issue title (required)
        body: Issue description in markdown
        labels: List of label names to apply
    """
    # Implementation calls GitHub API
    response = github_client.create_issue(repo, title, body, labels)
    return {"issue_number": response.number, "url": response.html_url}
```

**Tool characteristics:**
- Have a name, description, and JSON Schema input spec
- Can return text, JSON, images, or error content
- Descriptions are critical — the LLM uses them to decide WHEN to call the tool
- Bad descriptions = LLM uses wrong tools or wrong arguments

#### 2. 📄 Resources — "Data the AI Can Read"

Resources are read-only data sources. Like GET-only endpoints. The AI reads them for context — they don't execute actions.

```
Analogy: Resources are like files or database views. You read them; you don't modify through them.

Examples:
  - file:///project/README.md      — file contents
  - db://schema/users              — database schema definition
  - github://repos/microsoft/vscode/issues — list of open issues
  - config://valid-expense-categories     — JSON list of valid categories
```

```python
@mcp.resource("config://expense-categories")
def get_expense_categories() -> str:
    """
    Returns the list of valid expense categories.
    Clients should use these to ensure consistent categorisation.
    """
    categories = {
        "Food": ["Groceries", "Restaurants", "Coffee"],
        "Transport": ["Fuel", "Uber", "Metro", "Flight"],
        "Tech": ["Software", "Hardware", "Cloud"],
        "Education": ["Courses", "Books", "Certifications"],
    }
    return json.dumps(categories, indent=2)

# URI templates — parameterised resources
@mcp.resource("github://repos/{owner}/{repo}/readme")
def get_repo_readme(owner: str, repo: str) -> str:
    """Fetch the README for any GitHub repository"""
    return github_client.get_readme(owner, repo)
```

#### 3. 💬 Prompts — "Reusable Workflow Templates"

Pre-defined prompt templates that the AI can surface as slash-commands or structured workflows. Less commonly used but critical for consistency.

```
Analogy: Prompts are like form templates. Instead of free-form text, they guide structured input.

Without prompt:
  User: "Create an issue for the login bug"  ← too vague

With MCP prompt template:
  User: /create-bug-report
  Server provides template with: title, steps to reproduce, expected vs actual, severity
  → consistent, structured, complete bug reports every time
```

```python
@mcp.prompt()
def bug_report_template(
    component: str,
    severity: str = "medium"
) -> str:
    """
    Generate a structured bug report prompt template.
    
    Args:
        component: Affected system component (e.g., "auth", "payment")
        severity: Bug severity - low/medium/high/critical
    """
    return f"""
    Create a GitHub issue for a {severity} severity bug in the {component} component.
    
    Include these sections:
    ## Bug Summary
    [One-line description]
    
    ## Steps to Reproduce
    1. [Step 1]
    2. [Step 2]
    
    ## Expected Behaviour
    [What should happen]
    
    ## Actual Behaviour
    [What actually happens]
    
    ## Environment
    - Browser/OS: 
    - Version:
    """
```

### Client-Side Primitives

Clients also expose primitives **back to servers** (less commonly discussed):

| Primitive | Direction | Purpose |
|-----------|-----------|---------|
| **Roots** | Client → Server | Tell server which file paths/URLs the client can access |
| **Sampling** | Server → Client | Server requests the LLM to generate text (enables recursive agent workflows) |
| **Elicitation** | Server → Client | Server asks user for additional input mid-operation |

```python
# Sampling — server requests LLM generation
# Server sends this to client when it needs the LLM's help mid-execution
sampling_request = {
    "jsonrpc": "2.0",
    "id": 99,
    "method": "sampling/createMessage",
    "params": {
        "messages": [
            {"role": "user", "content": {"type": "text", "text": "Summarise this log: ..."}}
        ],
        "maxTokens": 500,
    }
}
# Client (host) passes this to the LLM, returns completion to server
```

---

## 📡 Data Layer — JSON-RPC 2.0

### 🧑‍🎓 Explain Like I'm in BTech 2nd Year

Think of JSON-RPC 2.0 as the **language** that clients and servers use to talk. It's like defining grammar rules for a conversation:

- Every request has a unique ID (so you can match responses)
- Every response references that same ID
- Some messages are "fire-and-forget" — notifications — no ID, no response expected
- Errors have standardised codes so both sides know exactly what went wrong

```
Without JSON-RPC (custom protocol):
  client → server: "hey, do this thing: {action: 'call_tool', data: {...}, reqID: 'abc'}"
  server → client: "done, here's result for abc: {...}"
  Problem: Every team invents their own format. No tools understand it. No SDKs support it.

With JSON-RPC 2.0:
  client → server: {"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {...}}
  server → client: {"jsonrpc": "2.0", "id": 1, "result": {...}}
  Every SDK, every debugger, every proxy understands this format.
```

### The Three JSON-RPC Message Types

```json
// TYPE 1: REQUEST (client or server initiates, expects response)
// Must have: jsonrpc, id, method
// May have: params
{
  "jsonrpc": "2.0",
  "id": 42,
  "method": "tools/call",
  "params": {
    "name": "create_issue",
    "arguments": {"repo": "myorg/myapp", "title": "Login bug"}
  }
}

// TYPE 2: RESPONSE (success)
// Must have: jsonrpc, id matching the request, result
{
  "jsonrpc": "2.0",
  "id": 42,
  "result": {
    "content": [{"type": "text", "text": "Issue #123 created: https://github.com/..."}]
  }
}

// TYPE 2b: RESPONSE (error)
// Must have: jsonrpc, id matching the request, error object
{
  "jsonrpc": "2.0",
  "id": 42,
  "error": {
    "code": -32602,
    "message": "Invalid params",
    "data": {"detail": "Required field 'title' is missing"}
  }
}

// TYPE 3: NOTIFICATION (one-way, no response expected)
// Must have: jsonrpc, method
// NO id field — this is the key distinguisher from a request
{
  "jsonrpc": "2.0",
  "method": "notifications/tools/list_changed"
}
```

---

## 📋 Standard Methods — Full Reference

### Lifecycle Methods

#### `initialize` — Session Handshake

```json
// CLIENT → SERVER: Start the session
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "initialize",
  "params": {
    "protocolVersion": "2025-03-26",
    "capabilities": {
      "roots": {"listChanged": true},
      "sampling": {}
    },
    "clientInfo": {
      "name": "claude-desktop",
      "version": "1.0.0"
    }
  }
}

// SERVER → CLIENT: Capability negotiation response
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {
    "protocolVersion": "2025-03-26",
    "capabilities": {
      "tools": {"listChanged": true},
      "resources": {"subscribe": true},
      "prompts": {}
    },
    "serverInfo": {
      "name": "github-mcp-server",
      "version": "2.1.0"
    },
    "instructions": "Use this server to interact with GitHub repos. Always check rate limits."
  }
}

// CLIENT → SERVER: Acknowledge initialization (notification, no response)
{
  "jsonrpc": "2.0",
  "method": "initialized"
}
```

---

### Tools Methods

#### `tools/list` — Discover Available Tools

```json
// REQUEST
{
  "jsonrpc": "2.0",
  "id": 2,
  "method": "tools/list",
  "params": null
}

// RESPONSE — GitHub MCP Server example
{
  "jsonrpc": "2.0",
  "id": 2,
  "result": {
    "tools": [
      {
        "name": "create_issue",
        "description": "Create a new GitHub issue in a repository",
        "inputSchema": {
          "type": "object",
          "properties": {
            "repo": {
              "type": "string",
              "description": "Repository in format 'owner/repo'"
            },
            "title": {
              "type": "string",
              "description": "Issue title"
            },
            "body": {
              "type": "string",
              "description": "Issue body in Markdown"
            },
            "labels": {
              "type": "array",
              "items": {"type": "string"},
              "description": "Labels to apply"
            }
          },
          "required": ["repo", "title"]
        }
      },
      {
        "name": "list_issues",
        "description": "List open issues for a repository",
        "inputSchema": {
          "type": "object",
          "properties": {
            "repo": {"type": "string"},
            "state": {
              "type": "string",
              "enum": ["open", "closed", "all"],
              "default": "open"
            },
            "limit": {"type": "integer", "default": 20}
          },
          "required": ["repo"]
        }
      }
    ]
  }
}
```

#### `tools/call` — Execute a Tool

```json
// REQUEST — GitHub: Create issue
{
  "jsonrpc": "2.0",
  "id": 3,
  "method": "tools/call",
  "params": {
    "name": "create_issue",
    "arguments": {
      "repo": "myorg/myapp",
      "title": "Login button broken on mobile Safari",
      "body": "## Bug\nThe login button doesn't respond to tap on iOS 17 Safari.\n\n## Steps\n1. Open app on iPhone\n2. Tap Login",
      "labels": ["bug", "mobile", "high-priority"]
    }
  }
}

// SUCCESS RESPONSE
{
  "jsonrpc": "2.0",
  "id": 3,
  "result": {
    "content": [
      {
        "type": "text",
        "text": "Issue #247 created successfully.\nURL: https://github.com/myorg/myapp/issues/247\nLabels applied: bug, mobile, high-priority"
      }
    ],
    "isError": false
  }
}

// ERROR RESPONSE — e.g. repo not found
{
  "jsonrpc": "2.0",
  "id": 3,
  "result": {
    "content": [
      {
        "type": "text",
        "text": "Error: Repository 'myorg/myapp' not found or you don't have access."
      }
    ],
    "isError": true
  }
}
```

#### `tools/call` — Twitter (X) Example

```json
// REQUEST — Post a tweet
{
  "jsonrpc": "2.0",
  "id": 10,
  "method": "tools/call",
  "params": {
    "name": "post_tweet",
    "arguments": {
      "text": "Just shipped MCP support for our AI assistant! 🚀 #AI #MCP #BuildingInPublic",
      "reply_to_id": null
    }
  }
}

// SUCCESS RESPONSE
{
  "jsonrpc": "2.0",
  "id": 10,
  "result": {
    "content": [
      {
        "type": "text",
        "text": "Tweet posted successfully.\nID: 1849283746291\nURL: https://x.com/user/status/1849283746291\nCharacters used: 78/280"
      }
    ],
    "isError": false
  }
}

// REQUEST — Search tweets
{
  "jsonrpc": "2.0",
  "id": 11,
  "method": "tools/call",
  "params": {
    "name": "search_tweets",
    "arguments": {
      "query": "#MCP OR #ModelContextProtocol lang:en",
      "limit": 10,
      "sort": "recency"
    }
  }
}
```

---

### Resources Methods

#### `resources/list` — Discover Resources

```json
// REQUEST
{
  "jsonrpc": "2.0",
  "id": 4,
  "method": "resources/list",
  "params": null
}

// RESPONSE — GitHub MCP Server
{
  "jsonrpc": "2.0",
  "id": 4,
  "result": {
    "resources": [
      {
        "uri": "github://repos/{owner}/{repo}/readme",
        "name": "Repository README",
        "description": "Fetch the README.md of any GitHub repository",
        "mimeType": "text/markdown"
      },
      {
        "uri": "github://user/repos",
        "name": "Your repositories",
        "description": "List all repositories accessible to the authenticated user",
        "mimeType": "application/json"
      }
    ]
  }
}
```

#### `resources/read` — Read a Resource

```json
// REQUEST — X (Twitter) MCP: Read user timeline
{
  "jsonrpc": "2.0",
  "id": 5,
  "method": "resources/read",
  "params": {
    "uri": "twitter://users/@elonmusk/timeline?limit=5"
  }
}

// RESPONSE
{
  "jsonrpc": "2.0",
  "id": 5,
  "result": {
    "contents": [
      {
        "uri": "twitter://users/@elonmusk/timeline",
        "mimeType": "application/json",
        "text": "[{\"id\": \"184920...\", \"text\": \"...\", \"created_at\": \"...\"}]"
      }
    ]
  }
}
```

---

### Notifications (Server → Client)

```json
// Server tells client: tool list has changed (new tools added)
{
  "jsonrpc": "2.0",
  "method": "notifications/tools/list_changed"
}

// Server sends progress update for long-running tool
{
  "jsonrpc": "2.0",
  "method": "notifications/progress",
  "params": {
    "progressToken": "token-abc-123",
    "progress": 65,
    "total": 100,
    "message": "Indexing repository files: 650/1000"
  }
}

// Server sends log message to client
{
  "jsonrpc": "2.0",
  "method": "notifications/message",
  "params": {
    "level": "warning",
    "logger": "github-server",
    "data": "GitHub API rate limit: 42 remaining (resets in 3200s)"
  }
}

// Server notifies: subscribed resource has changed
{
  "jsonrpc": "2.0",
  "method": "notifications/resources/updated",
  "params": {
    "uri": "file:///project/config.yaml"
  }
}

// Client cancels a running request
{
  "jsonrpc": "2.0",
  "method": "notifications/cancelled",
  "params": {
    "requestId": 42,
    "reason": "User cancelled the operation"
  }
}
```

---

### Batch Requests

JSON-RPC 2.0 supports batching — send multiple requests in one message:

```json
// CLIENT sends batch of 3 requests at once
[
  {"jsonrpc": "2.0", "id": 10, "method": "tools/list", "params": null},
  {"jsonrpc": "2.0", "id": 11, "method": "resources/list", "params": null},
  {"jsonrpc": "2.0", "id": 12, "method": "prompts/list", "params": null}
]

// SERVER responds with batch
[
  {"jsonrpc": "2.0", "id": 10, "result": {"tools": [...]}},
  {"jsonrpc": "2.0", "id": 11, "result": {"resources": [...]}},
  {"jsonrpc": "2.0", "id": 12, "result": {"prompts": [...]}}
]
// Server can process in parallel internally — order not guaranteed
```

---

## 🤔 Why JSON-RPC, Not JSON + REST API?

This is an interview question worth knowing deeply.

```
Why not just build a REST API for MCP?
GET /tools       → list tools
POST /tools/call → call a tool
GET /resources   → list resources
```

### 5 Reasons JSON-RPC Beats REST for MCP

**1. Bidirectional Communication**
```
REST:     Client → Server (one direction only)
JSON-RPC: Client ↔ Server (both directions, same connection)

MCP needs server → client: sampling, progress notifications, resource updates
REST cannot do this without WebSockets or polling hacks.
```

**2. Notifications (Fire-and-Forget)**
```
REST:     Every message needs an HTTP response (200, 404, etc.)
JSON-RPC: Notifications have no ID → no response expected → true async

MCP needs notifications for: progress updates, list changes, log messages
REST would require polling every second or a separate WebSocket — complex.
```

**3. Batching**
```
REST:     3 separate HTTP requests to list tools + resources + prompts
JSON-RPC: One batch message → parallel processing → one batch response

At scale this matters: initialization needs all three lists at once.
```

**4. Lightweight and Simple**
```
REST overhead: HTTP method + URL + headers + body + status codes
JSON-RPC: just {"jsonrpc":"2.0","id":1,"method":"tools/list"}

MCP runs over stdio (no network, just pipes) → HTTP overhead would be wasteful.
JSON-RPC works over any transport: stdio, WebSocket, HTTP — same message format.
```

**5. Transport Agnosticism (the big one)**
```
REST is fundamentally tied to HTTP (URLs, methods, status codes).
JSON-RPC is just messages — works over:
  - stdio (standard input/output) — for local processes
  - WebSocket — for persistent connections
  - HTTP+SSE — for streaming remote servers
  - Any custom transport

This is why MCP can run locally AS A SUBPROCESS with zero network overhead,
while the same server code can also be deployed remotely over HTTP.
Same protocol, different transport.
```

---

## 🚌 Transport Layer

### What is the Transport Layer?

The **data layer** (JSON-RPC 2.0) defines the *language* — the format and grammar of messages.  
The **transport layer** defines the *medium* — HOW those messages physically travel.

```
JSON-RPC message (same format regardless of transport):
  {"jsonrpc":"2.0","id":1,"method":"tools/call","params":{...}}

Travels over:
  Transport A (stdio): written to process stdin, read from stdout
  Transport B (HTTP):  sent as HTTP POST body, response in HTTP response body
  Transport C (SSE):   HTTP POST + Server-Sent Events stream

The message is identical. Only the delivery mechanism changes.
```

### How JSON-RPC is Transport-Agnostic

```python
# Same server code, different transports

from fastmcp import FastMCP

mcp = FastMCP("weather-server")

@mcp.tool()
def get_weather(city: str) -> str:
    return f"Weather in {city}: 28°C, partly cloudy"

# Run over stdio (local)
if __name__ == "__main__":
    mcp.run()                       # default: stdio

# Run over HTTP (remote)
if __name__ == "__main__":
    mcp.run(transport="streamable-http", host="0.0.0.0", port=8000)

# Same tool definition. Same JSON-RPC messages. Different delivery pipe.
```

---

## 📡 Local (stdio) vs Remote (HTTP+SSE)

### Local Transport — stdio

```
How stdio works:
  1. Host spawns MCP server as a child process
  2. Host writes JSON-RPC messages to server's STDIN
  3. Server processes message, writes JSON-RPC response to STDOUT
  4. Host reads from server's STDOUT

Host Process
    │
    ├── STDIN  ──────▶ MCP Server Process
    │                      │
    └── STDOUT ◀────────── │
                           └── Does the actual work
```

```bash
# How Claude Desktop launches a stdio MCP server:
# The host runs: python my_mcp_server.py
# Then communicates via pipe (stdin/stdout)

# You can test this manually:
echo '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-03-26","capabilities":{},"clientInfo":{"name":"test","version":"1.0"}}}' | python my_server.py
```

**Why stdio is used for local servers:**
- Zero network overhead (no TCP stack, no port, no firewall)
- Process-level security (server can only be reached by parent process)
- No authentication needed (OS-level process isolation)
- Perfect for developer tools: Claude Desktop, Cursor, IDE integrations
- Simplest to set up — just a Python script

### Remote Transport — Streamable HTTP (HTTP+SSE)

```
How Streamable HTTP works:
  Client sends requests via HTTP POST
  Server streams responses via Server-Sent Events (SSE)
  Same JSON-RPC messages, different delivery

Client                          Server (deployed on cloud)
  │                                 │
  │── HTTP POST /mcp ──────────────▶│
  │   {jsonrpc, method, params}     │
  │                                 │── processes...
  │◀── HTTP 200 + SSE stream ───────│
  │   data: {jsonrpc, id, result}   │
  │   data: {notifications...}      │
  │                                 │
  │── HTTP POST /mcp ──────────────▶│  (next request, same session)
```

**Why HTTP+SSE is used for remote servers:**
- Accessible over the internet (multiple clients, different machines)
- Supports OAuth 2.1 authentication for enterprise security
- Works with load balancers and cloud hosting (Docker, Railway, AWS)
- Supports multiple concurrent clients (vs stdio: one client per process)
- SSE gives server ability to push notifications to client

### Comparison Table

| Feature | stdio (Local) | Streamable HTTP (Remote) |
|---------|--------------|--------------------------|
| **Network** | None (process pipe) | Internet/LAN |
| **Multi-client** | No (1 client per process) | Yes (many clients) |
| **Authentication** | OS process isolation | OAuth 2.1 + PKCE |
| **Latency** | ~0ms | ~10-200ms |
| **Deploy to cloud** | No | Yes |
| **Setup complexity** | Config file only | Docker + hosting required |
| **Security** | Very high (no network exposure) | Requires auth implementation |
| **Use for** | Developer tools, Claude Desktop | Production APIs, shared access |

---

## 🗺️ Full Architecture Diagram

```
╔══════════════════════════════════════════════════════════════════════════════╗
║                              MCP HOST                                        ║
║                    (Claude Desktop / Your Custom App)                       ║
║                                                                              ║
║  ┌─────────────────────────────────────────────────────────────────────┐   ║
║  │                          LLM ENGINE                                  │   ║
║  │              (Claude Sonnet 4 / GPT-4o / Local LLM)                │   ║
║  │                                                                      │   ║
║  │  When LLM wants to use a tool:                                      │   ║
║  │  "I need to create a GitHub issue" → routes to MCP Client 1        │   ║
║  └─────────────────────────────────────────────────────────────────────┘   ║
║                │                    │                    │                   ║
║                ▼                    ▼                    ▼                   ║
║  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────────┐     ║
║  │   MCP CLIENT 1   │  │   MCP CLIENT 2   │  │    MCP CLIENT 3      │     ║
║  │   [Isolated]     │  │   [Isolated]     │  │    [Isolated]        │     ║
║  │                  │  │                  │  │                      │     ║
║  │ Manages session  │  │ Manages session  │  │ Manages session      │     ║
║  │ Handles protocol │  │ Handles protocol │  │ Handles protocol     │     ║
║  └────────┬─────────┘  └────────┬─────────┘  └──────────┬───────────┘     ║
╚═══════════╪════════════════════╪══════════════════════════╪════════════════╝
            │                    │                          │
            │ stdio              │ stdio                    │ HTTPS + SSE
            │ (local process)    │ (local process)          │ (remote server)
            ▼                    ▼                          ▼
┌─────────────────────┐ ┌─────────────────────┐ ┌─────────────────────────┐
│   LOCAL MCP SERVER  │ │   LOCAL MCP SERVER  │ │    REMOTE MCP SERVER    │
│   📁 File System    │ │   🐘 PostgreSQL      │ │    🐦 X (Twitter)        │
│                     │ │                     │ │                         │
│   TOOLS:            │ │   TOOLS:            │ │   TOOLS:                │
│   ✦ read_file       │ │   ✦ run_select_query│ │   ✦ post_tweet          │
│   ✦ write_file      │ │   ✦ list_tables     │ │   ✦ search_tweets       │
│   ✦ list_directory  │ │   ✦ get_schema      │ │   ✦ get_user_profile    │
│                     │ │                     │ │   ✦ get_timeline        │
│   RESOURCES:        │ │   RESOURCES:        │ │                         │
│   ✦ file://path     │ │   ✦ db://schema     │ │   RESOURCES:            │
│   ✦ dir://path      │ │   ✦ db://tables     │ │   ✦ twitter://timeline  │
│                     │ │                     │ │   ✦ twitter://user      │
│   PROMPTS:          │ │   PROMPTS:          │ │                         │
│   ✦ summarise_file  │ │   ✦ explain_query   │ │   PROMPTS:              │
└──────────┬──────────┘ └──────────┬──────────┘ │   ✦ thread_composer    │
           │                       │             └────────────┬────────────┘
           ▼                       ▼                          ▼
      File System             PostgreSQL DB           Twitter/X API
```

---

## 🔄 MCP Lifecycle

### Phase 1: Initialisation

```
CLIENT                                    SERVER
  │                                          │
  │── initialize (protocolVersion, caps) ──▶│
  │                                          │  Negotiate version
  │                                          │  Declare capabilities
  │◀── result (version, caps, instructions)──│
  │                                          │
  │── notifications/initialized (no response)▶│
  │                                          │
  │    ✅ Session established               │
  │    ✅ Both sides know each other's caps │
```

```json
// Rules during initialisation:
// - ONLY ping and server logging notifications are allowed before initialized
// - All other requests (tools/call, resources/read) are FORBIDDEN until initialized
// - Version mismatch → server closes connection immediately
```

### Phase 2: Operation

```
CLIENT                                    SERVER
  │                                          │
  │── tools/list ──────────────────────────▶│
  │◀── result (tools array) ────────────────│
  │                                          │
  │── resources/list ──────────────────────▶│
  │◀── result (resources array) ────────────│
  │                                          │
  │  [LLM decides to create GitHub issue]   │
  │── tools/call (create_issue, args) ─────▶│
  │                                    work │
  │◀── notifications/progress (50%) ────────│  (server pushes progress)
  │◀── notifications/progress (100%) ───────│
  │◀── result (issue URL) ──────────────────│
  │                                          │
  │  [Server adds a new tool dynamically]   │
  │◀── notifications/tools/list_changed ────│  (server pushes update)
  │── tools/list (refresh) ────────────────▶│
  │◀── result (updated tools) ──────────────│
```

### Phase 3: Shutdown

```
CLIENT                                    SERVER
  │                                          │
  │  [User closes Claude Desktop]           │
  │── transport disconnect ────────────────▶│
  │                                          │
  │  For stdio: Host kills child process    │
  │  For HTTP:  Connection closed           │
  │             Session state cleaned up   │
```

---

## ⚠️ Error Handling, Cancellation, Timeouts

### Standard JSON-RPC Error Codes

```json
// JSON-RPC 2.0 standard codes (mandatory)
-32700   // Parse error — invalid JSON received
-32600   // Invalid Request — not a valid JSON-RPC Request object
-32601   // Method not found — method doesn't exist on server
-32602   // Invalid params — wrong arguments for the method
-32603   // Internal error — server-side error

// MCP-specific codes (optional, -32000 to -32099 range)
-32001   // Tool not found — requested tool name doesn't exist
-32002   // Tool execution failed — tool ran but encountered an error
-32003   // Resource not found — URI doesn't match any resource
-32004   // Resource unavailable — resource exists but can't be read now
-32005   // Resource access denied — authorisation failure
```

```json
// Error response example — GitHub server, repo not found
{
  "jsonrpc": "2.0",
  "id": 7,
  "error": {
    "code": -32003,
    "message": "Resource not found",
    "data": {
      "uri": "github://repos/nonexistent/repo/readme",
      "detail": "Repository 'nonexistent/repo' does not exist or is private"
    }
  }
}
```

### Cancellation

```json
// CLIENT cancels a long-running request
// Must be sent BEFORE the response arrives
{
  "jsonrpc": "2.0",
  "method": "notifications/cancelled",
  "params": {
    "requestId": 42,
    "reason": "User pressed Escape"
  }
}
// Server should abort processing for requestId=42
// Server may still send the result, client should ignore it
```

### Timeouts

```python
# FastMCP server — per-tool timeout
from fastmcp import FastMCP
import asyncio

mcp = FastMCP("my-server")

@mcp.tool()
async def slow_operation(query: str) -> str:
    """Long-running operation with timeout"""
    try:
        result = await asyncio.wait_for(
            do_heavy_computation(query),
            timeout=30.0  # 30 second hard timeout
        )
        return result
    except asyncio.TimeoutError:
        return "Operation timed out after 30 seconds. Try a simpler query."

# Client-side timeout (FastMCP client)
from fastmcp import Client

async with Client("server.py", timeout=30) as client:
    result = await client.call_tool("slow_operation", {"query": "..."})
```

---

## 🔌 Connection Types

### Type 1 — Config File (Most Common for Local)

Used for connecting Claude Desktop (or any MCP client) to local MCP servers.

**MacOS/Linux config file:** `~/.config/claude/claude_desktop_config.json`  
**Windows:** `%APPDATA%\Claude\claude_desktop_config.json`

```json
{
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/Users/sahil/Documents"],
      "env": {}
    },
    "github": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-github"],
      "env": {
        "GITHUB_PERSONAL_ACCESS_TOKEN": "ghp_xxxxxxxxxxxxx"
      }
    },
    "postgres": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-postgres", "postgresql://localhost/mydb"],
      "env": {}
    },
    "my-expense-tracker": {
      "command": "python",
      "args": ["/Users/sahil/projects/expense-mcp/server.py"],
      "env": {
        "DB_PATH": "/Users/sahil/projects/expense-mcp/expenses.db"
      }
    },
    "twitter-remote": {
      "url": "https://twitter-mcp.mycompany.com/mcp",
      "headers": {
        "Authorization": "Bearer my-api-key-here"
      }
    }
  }
}
```

**When to use:** Development, local tools, Claude Desktop integrations, private data that shouldn't be on a server.

### Type 2 — Remote URL / Connectors

For remote HTTP MCP servers. Used in cloud deployments and team-shared servers.

```python
# Connecting to a remote MCP server via FastMCP Client
from fastmcp import Client

# Remote HTTP server
async with Client("https://api.mycompany.com/mcp") as client:
    tools = await client.list_tools()
    result = await client.call_tool("search_crm", {"query": "Rahul"})

# With auth headers
from fastmcp.client.auth import BearerAuth
async with Client(
    "https://api.mycompany.com/mcp",
    auth=BearerAuth("my-oauth-token")
) as client:
    result = await client.call_tool("create_ticket", {...})
```

**When to use:** Shared team servers, production APIs, servers with authentication, multi-user access.

### Config File vs Remote — Decision Guide

| | Config File (stdio) | Remote URL (HTTP) |
|--|--|--|
| **Data stays local** | Yes | No (leaves machine) |
| **Multi-user** | No | Yes |
| **Authentication** | OS-level | OAuth 2.1 |
| **Cloud deploy** | No | Yes |
| **Team sharing** | No | Yes |
| **Setup effort** | Low (edit JSON) | Medium (deploy server) |
| **Use for** | Personal tools, local files, dev | Shared company tools, APIs |

---

## 💻 Coding Guide

### Part 1 — Connect Claude Desktop to Existing MCP Servers

```bash
# Install popular existing MCP servers

# 1. Filesystem server (read/write local files)
npm install -g @modelcontextprotocol/server-filesystem

# 2. GitHub server
npm install -g @modelcontextprotocol/server-github

# 3. Postgres server
npm install -g @modelcontextprotocol/server-postgres

# 4. Weather server (community)
pip install mcp-server-weather

# 5. Brave Search
npm install -g @modelcontextprotocol/server-brave-search
```

```json
// claude_desktop_config.json — connect all at once
{
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem",
               "/Users/sahil/Desktop", "/Users/sahil/Documents"],
      "env": {}
    },
    "github": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-github"],
      "env": {"GITHUB_PERSONAL_ACCESS_TOKEN": "ghp_your_token_here"}
    },
    "postgres": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-postgres",
               "postgresql://localhost:5432/mydb"],
      "env": {}
    }
  }
}
```

Now in Claude Desktop you can say:
- *"List all PDF files in my Documents folder"*
- *"Create a GitHub issue in myorg/myrepo about the login bug"*
- *"Show me the top 10 users from the database"*

---

### Part 2 — Custom MCP Server + Claude Desktop

```bash
# Setup
pip install fastmcp
# Or with uv (recommended):
uv init my-mcp-server
cd my-mcp-server
uv add fastmcp
```

```python
# dice_server.py — simple custom server to start with
from fastmcp import FastMCP
import random

mcp = FastMCP(
    "dice-server",
    instructions="A dice rolling server. Use roll_dice to roll any dice."
)

@mcp.tool()
def roll_dice(sides: int = 6, count: int = 1) -> dict:
    """
    Roll one or more dice.
    
    Args:
        sides: Number of sides on each die (4, 6, 8, 10, 12, 20, 100)
        count: How many dice to roll (1-10)
    
    Returns a dict with individual rolls and total.
    """
    if count < 1 or count > 10:
        raise ValueError("count must be between 1 and 10")
    if sides not in [4, 6, 8, 10, 12, 20, 100]:
        raise ValueError(f"Invalid dice: d{sides}")
    
    rolls = [random.randint(1, sides) for _ in range(count)]
    return {
        "dice": f"{count}d{sides}",
        "rolls": rolls,
        "total": sum(rolls),
        "min_possible": count,
        "max_possible": count * sides,
    }

@mcp.tool()
def add_numbers(a: float, b: float) -> float:
    """Add two numbers together."""
    return a + b

if __name__ == "__main__":
    mcp.run()  # stdio by default
```

```bash
# Test with MCP Inspector (like Postman for MCP)
fastmcp dev dice_server.py

# Opens browser UI at http://localhost:5173
# You can call tools and see JSON-RPC messages in real time
```

```json
// Add to claude_desktop_config.json
{
  "mcpServers": {
    "dice": {
      "command": "python",
      "args": ["/absolute/path/to/dice_server.py"]
    }
  }
}
```

```bash
# Or use fastmcp install (manages virtualenv for you)
fastmcp install dice_server.py --name "Dice Roller"
# --name sets the display name in Claude Desktop
```

---

### Part 3 — Custom Client + Custom Server

```python
# server.py — the custom server
from fastmcp import FastMCP

mcp = FastMCP("calculator-server")

@mcp.tool()
def calculate(expression: str) -> float:
    """
    Evaluate a mathematical expression.
    Supports: +, -, *, /, **, sqrt, sin, cos
    Example: '2 + 3 * 4' or 'sqrt(16)'
    """
    import math
    allowed = {
        "sqrt": math.sqrt, "sin": math.sin,
        "cos": math.cos, "pi": math.pi, "e": math.e
    }
    return eval(expression, {"__builtins__": {}}, allowed)

@mcp.resource("math://constants")
def get_constants() -> str:
    """Common mathematical constants"""
    import json, math
    return json.dumps({"pi": math.pi, "e": math.e, "phi": 1.618033988749})

if __name__ == "__main__":
    mcp.run()
```

```python
# client.py — your custom client
import asyncio
from fastmcp import Client

async def main():
    # Connect to the server (stdio — it runs as a subprocess)
    async with Client("server.py") as client:
        
        # Discover what's available
        tools     = await client.list_tools()
        resources = await client.list_resources()
        
        print("Available tools:")
        for tool in tools:
            print(f"  - {tool.name}: {tool.description}")
        
        # Call a tool
        result = await client.call_tool(
            "calculate",
            {"expression": "sqrt(144) + 3 * 4"}
        )
        print(f"\nResult: {result[0].text}")  # → 24.0
        
        # Read a resource
        content = await client.read_resource("math://constants")
        print(f"\nConstants: {content[0].text}")

asyncio.run(main())
```

```bash
# Run the client (it spawns the server automatically)
python client.py
```

---

## 🚀 FastMCP Deep-Dive

### The History

```
Timeline:
  Nov 2024 → Anthropic releases MCP + official Python SDK
              SDK is powerful but verbose (lots of boilerplate)
  
  Dec 2024 → Jeremiah Lowin (Prefect founder) creates FastMCP 1.0
              "FastAPI but for MCP" — decorator-based, minimal boilerplate
              Becomes immediately popular
  
  Early 2025 → Anthropic ADOPTS FastMCP 1.0 into the official SDK
               You can now import: from mcp.server.fastmcp import FastMCP
               
  Mid 2025 → FastMCP 2.0 released by Prefect team
             Goes far beyond SDK: proxying, server composition, FastAPI generation,
             enterprise auth (Google, GitHub, Azure, Auth0), deployment tools
             
  2026 → FastMCP 2.0 powers 70% of all MCP servers across all languages
         Downloaded 1 million times per day
```

### FastMCP vs MCP SDK

```python
# Building the same tool:

# === MCP SDK (official, verbose) ===
from mcp.server import Server
from mcp.types import Tool, TextContent, CallToolResult

server = Server("my-server")

@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="get_weather",
            description="Get current weather for a city",
            inputSchema={
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name"}
                },
                "required": ["city"]
            }
        )
    ]

@server.call_tool()
async def call_tool(name: str, arguments: dict) -> CallToolResult:
    if name == "get_weather":
        city = arguments["city"]
        return CallToolResult(
            content=[TextContent(type="text", text=f"Weather in {city}: 28°C")]
        )
    raise ValueError(f"Unknown tool: {name}")

# Start server...
import asyncio
from mcp.server.stdio import stdio_server
asyncio.run(stdio_server(server))

# === FastMCP 2.0 (Pythonic, concise) ===
from fastmcp import FastMCP

mcp = FastMCP("my-server")

@mcp.tool()
def get_weather(city: str) -> str:
    """Get current weather for a city"""
    return f"Weather in {city}: 28°C"

mcp.run()  # handles everything automatically
```

**FastMCP extracts schema from Python type hints and docstrings automatically.** No manual JSON schema writing.

### FastMCP Relationship with FastAPI

```python
# FastMCP can generate an MCP server FROM an existing FastAPI app
from fastapi import FastAPI
from fastmcp import FastMCP

# Your existing FastAPI app (already in production)
app = FastAPI()

@app.get("/weather/{city}")
async def get_weather(city: str) -> dict:
    return {"city": city, "temp": 28, "condition": "sunny"}

@app.post("/issues")
async def create_issue(repo: str, title: str, body: str) -> dict:
    return {"id": 123, "url": f"https://github.com/{repo}/issues/123"}

# One line to generate MCP server from FastAPI app
mcp = FastMCP.from_fastapi(app)

# Now your existing REST API is ALSO available as an MCP server
# AI clients can call your API's endpoints as MCP tools
# No duplicate code — FastMCP reads your OpenAPI spec automatically
```

### FastMCP CLI Commands

```bash
# Install FastMCP
pip install fastmcp
# Or with uv (strongly recommended):
uv add fastmcp

# Development — test with MCP Inspector UI
fastmcp dev server.py
fastmcp dev server.py --with pandas       # add temp dependencies
fastmcp dev server.py --with-editable .   # with local package

# Install into Claude Desktop (creates isolated venv automatically)
fastmcp install server.py
fastmcp install server.py --name "My Tool"
fastmcp install server.py --with requests -v API_KEY=mykey -f .env

# Run directly (for custom deployments)
python server.py          # stdio mode
python server.py --transport streamable-http --port 8000  # HTTP mode

# If your FastMCP instance isn't named 'mcp', 'server', or 'app':
fastmcp dev my_module.py:my_instance_name
```

### FastMCP Context — Access MCP Capabilities Inside Tools

```python
from fastmcp import FastMCP, Context

mcp = FastMCP("context-demo")

@mcp.tool()
async def process_large_file(file_path: str, ctx: Context) -> str:
    """Process a large file with progress reporting"""
    
    # Log messages (sent to client as notifications)
    await ctx.info(f"Starting to process: {file_path}")
    
    lines = open(file_path).readlines()
    results = []
    
    for i, line in enumerate(lines):
        # Report progress
        await ctx.report_progress(i, len(lines))
        results.append(process_line(line))
        
        # Log warnings
        if i % 100 == 0:
            await ctx.warning(f"Processed {i}/{len(lines)} lines")
    
    # Request LLM sampling from within the tool
    summary = await ctx.sample(
        f"Summarise these {len(results)} processed results in 2 sentences: {results[:10]}"
    )
    
    await ctx.info("Processing complete!")
    return summary.text
```

---

## 🏦 Building an Expense Tracker MCP Server

### Project Overview

```
Goal: Natural language expense tracking
  "Add milk expense of ₹20"      → addExpense(amount=20, category="Groceries", note="milk")
  "Show all expenses from May"   → listExpenses(start="2025-05-01", end="2025-05-31")
  "How much did I spend on food?" → summarizeExpenses(category="Food")

Stack:
  FastMCP (server framework)
  SQLite (local database — simple, no server needed)
  uv (dependency management)
```

### Step 1 — Project Setup

```bash
uv init expense-tracker-mcp
cd expense-tracker-mcp
uv add fastmcp
```

### Step 2 — Database Schema

```python
# db.py
import sqlite3
from pathlib import Path

DB_PATH = Path.home() / ".expense-tracker" / "expenses.db"
DB_PATH.parent.mkdir(exist_ok=True)

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    with get_db() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS expenses (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                date        TEXT NOT NULL DEFAULT (date('now')),
                amount      REAL NOT NULL,
                category    TEXT NOT NULL,
                subcategory TEXT,
                note        TEXT,
                created_at  TEXT DEFAULT (datetime('now'))
            )
        """)
        conn.commit()
    print(f"Database initialised at: {DB_PATH}")
```

### Step 3 — MCP Server with Tools + Resources

```python
# server.py
import json
from datetime import date, timedelta
from fastmcp import FastMCP
from db import get_db, init_db

# Initialise database
init_db()

mcp = FastMCP(
    "expense-tracker",
    instructions="""
    Expense tracking server. Use tools to add, list, and summarise expenses.
    Always check the valid-categories resource first to use consistent category names.
    Dates default to today unless specified. Accept natural language dates like 'yesterday', 'last week'.
    """
)

# ── RESOURCE: valid categories ─────────────────────────────────────────────
@mcp.resource("config://valid-categories")
def get_valid_categories() -> str:
    """
    JSON list of valid expense categories and subcategories.
    Always use these exact names for consistency. Read this before adding expenses.
    """
    categories = {
        "Food": ["Groceries", "Restaurants", "Coffee", "Delivery", "Snacks"],
        "Transport": ["Fuel", "Cab/Uber", "Metro/Bus", "Auto", "Flight", "Train"],
        "Tech": ["Software/SaaS", "Hardware", "Cloud/Hosting", "Courses"],
        "Health": ["Medicine", "Doctor", "Gym", "Supplements"],
        "Shopping": ["Clothes", "Electronics", "Home", "Personal Care"],
        "Entertainment": ["Movies", "Games", "Subscriptions", "Events"],
        "Bills": ["Electricity", "Internet", "Mobile", "Rent", "Insurance"],
        "Other": ["Misc"],
    }
    return json.dumps(categories, indent=2, ensure_ascii=False)

# ── TOOL 1: Add expense ────────────────────────────────────────────────────
@mcp.tool()
def add_expense(
    amount: float,
    category: str,
    subcategory: str = "",
    note: str = "",
    expense_date: str = "",
) -> dict:
    """
    Add a new expense entry.
    
    Args:
        amount: Amount in rupees (positive number)
        category: Main category — must match valid-categories resource
        subcategory: Sub-category (optional but recommended)
        note: Free-text note e.g. "lunch with team", "monthly subscription"
        expense_date: Date as YYYY-MM-DD. Defaults to today if empty.
    
    Returns: The created expense record with its ID.
    """
    if amount <= 0:
        raise ValueError("Amount must be positive")
    
    if not expense_date:
        expense_date = date.today().isoformat()
    
    with get_db() as conn:
        cursor = conn.execute(
            """INSERT INTO expenses (date, amount, category, subcategory, note)
               VALUES (?, ?, ?, ?, ?)""",
            (expense_date, amount, category, subcategory, note)
        )
        conn.commit()
        expense_id = cursor.lastrowid
    
    return {
        "id": expense_id,
        "date": expense_date,
        "amount": amount,
        "category": category,
        "subcategory": subcategory,
        "note": note,
        "message": f"✅ Added ₹{amount:.0f} for {category}" + (f" ({note})" if note else ""),
    }

# ── TOOL 2: List expenses ──────────────────────────────────────────────────
@mcp.tool()
def list_expenses(
    start_date: str = "",
    end_date: str = "",
    category: str = "",
    limit: int = 50,
) -> dict:
    """
    List expenses, optionally filtered by date range and category.
    
    Args:
        start_date: Start date YYYY-MM-DD (defaults to 30 days ago)
        end_date: End date YYYY-MM-DD (defaults to today)
        category: Filter by category (optional, empty = all categories)
        limit: Max entries to return (default 50, max 200)
    
    Returns: List of expenses with total count and sum.
    """
    if not start_date:
        start_date = (date.today() - timedelta(days=30)).isoformat()
    if not end_date:
        end_date = date.today().isoformat()
    if limit > 200:
        limit = 200
    
    query = "SELECT * FROM expenses WHERE date BETWEEN ? AND ?"
    params = [start_date, end_date]
    
    if category:
        query += " AND category = ?"
        params.append(category)
    
    query += " ORDER BY date DESC, id DESC LIMIT ?"
    params.append(limit)
    
    with get_db() as conn:
        rows = conn.execute(query, params).fetchall()
    
    expenses = [dict(r) for r in rows]
    total = sum(e["amount"] for e in expenses)
    
    return {
        "expenses":   expenses,
        "count":      len(expenses),
        "total":      round(total, 2),
        "period":     f"{start_date} to {end_date}",
        "message":    f"Found {len(expenses)} expenses totalling ₹{total:.0f}",
    }

# ── TOOL 3: Summarise expenses ─────────────────────────────────────────────
@mcp.tool()
def summarise_expenses(
    start_date: str = "",
    end_date: str = "",
) -> dict:
    """
    Summarise expenses by category for a date range.
    Shows totals per category and overall grand total.
    
    Args:
        start_date: Start date YYYY-MM-DD (defaults to this month's start)
        end_date: End date YYYY-MM-DD (defaults to today)
    """
    if not start_date:
        today = date.today()
        start_date = today.replace(day=1).isoformat()
    if not end_date:
        end_date = date.today().isoformat()
    
    with get_db() as conn:
        rows = conn.execute("""
            SELECT
                category,
                subcategory,
                COUNT(*)       AS count,
                SUM(amount)    AS total,
                AVG(amount)    AS avg,
                MAX(amount)    AS max_amount
            FROM expenses
            WHERE date BETWEEN ? AND ?
            GROUP BY category, subcategory
            ORDER BY total DESC
        """, [start_date, end_date]).fetchall()
        
        grand_total = conn.execute(
            "SELECT SUM(amount) FROM expenses WHERE date BETWEEN ? AND ?",
            [start_date, end_date]
        ).fetchone()[0] or 0
    
    summary = [dict(r) for r in rows]
    for s in summary:
        s["total"] = round(s["total"], 2)
        s["avg"]   = round(s["avg"], 2)
    
    return {
        "summary":    summary,
        "grand_total": round(grand_total, 2),
        "period":     f"{start_date} to {end_date}",
        "message":    f"Total spend: ₹{grand_total:.0f} across {len(summary)} categories",
    }

# ── Run ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    mcp.run()
```

### Step 4 — Test with MCP Inspector

```bash
fastmcp dev server.py

# Opens at http://localhost:5173
# You can:
#   - See all tools and their schemas
#   - Call tools with test arguments
#   - Inspect raw JSON-RPC messages
#   - Check resources
```

### Step 5 — Install in Claude Desktop

```bash
fastmcp install server.py --name "Expense Tracker" -v DB_PATH=/Users/sahil/.expense-tracker/expenses.db
```

Now in Claude Desktop:
- *"Add ₹150 for a restaurant lunch with the team today"*
- *"Show me all tech expenses from this month"*
- *"How much did I spend total in May 2025?"*

---

## ⚡ FastAPI + MCP Integration

```python
# existing_api.py — Your existing FastAPI backend
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Company API")

class Issue(BaseModel):
    repo: str
    title: str
    body: str

@app.post("/issues", summary="Create a GitHub issue")
async def create_issue(issue: Issue) -> dict:
    # ... existing implementation
    return {"id": 123, "url": "https://github.com/..."}

@app.get("/issues/{repo}", summary="List issues for repo")
async def list_issues(repo: str, state: str = "open") -> list:
    # ... existing implementation
    return [{"id": 1, "title": "Bug in login"}]

# ─── Add MCP support to your existing FastAPI app ─────────────────────────

from fastmcp import FastMCP

# Generate MCP server from your existing FastAPI app
# FastMCP reads your OpenAPI spec and creates tools from your endpoints
mcp = FastMCP.from_fastapi(
    app,
    name="company-api-mcp",
    # Optional: exclude endpoints you don't want as MCP tools
    exclude_paths=["/docs", "/redoc", "/openapi.json", "/health"]
)

# Now run BOTH FastAPI and MCP simultaneously:
if __name__ == "__main__":
    import uvicorn, threading
    
    # Start FastAPI in a thread
    api_thread = threading.Thread(
        target=uvicorn.run,
        args=(app,),
        kwargs={"host": "0.0.0.0", "port": 8000},
        daemon=True
    )
    api_thread.start()
    
    # Run MCP server on stdio (for Claude Desktop)
    mcp.run()
```

**Use case:** Companies that already have FastAPI backends for their web/mobile apps can instantly expose that backend as an MCP server — no duplicate code, no separate maintenance.

---

## ☁️ Deploying MCP Servers

### Option 1 — FastMCP Cloud (Easiest)

```python
# server.py
from fastmcp import FastMCP

mcp = FastMCP("my-server")

@mcp.tool()
def my_tool(input: str) -> str:
    """My cloud-deployed tool"""
    return f"Processed: {input}"
```

```bash
# Deploy to FastMCP Cloud (managed hosting)
fastmcp deploy server.py

# → Returns a public URL: https://my-server.fastmcp.run/mcp
# → No infrastructure to manage
# → Automatic HTTPS, scaling, monitoring
```

### Option 2 — Docker + Any Cloud

```python
# server.py — HTTP transport for remote deployment
from fastmcp import FastMCP

mcp = FastMCP("my-server")

# ... your tools here ...

if __name__ == "__main__":
    mcp.run(
        transport="streamable-http",
        host="0.0.0.0",
        port=8000,
    )
```

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install fastmcp

COPY server.py .

EXPOSE 8000
CMD ["python", "server.py"]
```

```bash
# Build and deploy
docker build -t my-mcp-server .
docker run -p 8000:8000 -e API_KEY=xxx my-mcp-server

# Deploy to Railway
railway up

# Deploy to Fly.io
fly deploy

# Deploy to AWS ECS / Lambda (via Mangum adapter)
```

### Option 3 — Integrate into Existing FastAPI

```python
from fastapi import FastAPI
from fastmcp import FastMCP

app = FastAPI()
mcp = FastMCP("embedded-mcp")

@mcp.tool()
def my_tool(x: str) -> str:
    return f"Result: {x}"

# Mount MCP under FastAPI at /mcp endpoint
mcp_app = mcp.get_asgi_app()
app.mount("/mcp", mcp_app)

# Now your FastAPI serves:
# - REST API at /api/...
# - MCP at /mcp
# Single deployment, single container
```

---

## 💀 Is MCP Dead? Alternatives

### The "Is MCP Dead?" Debate (2026)

**Short answer: No. MCP is winning. But it has competition.**

**What people mean when they ask this:**
- Some companies (especially enterprise) are moving to **A2A** for agent-to-agent communication
- MCP has had stateful session challenges with load balancers at scale
- The November 2025 spec added stateless server operation to address this

### MCP vs A2A — They Are Complementary, Not Competing

```
MCP = Agent ↔ Tool integration (VERTICAL)
  One agent, many tools
  "AI brain" connects to its "hands"
  Standard: What tools are available? How to call them?

A2A = Agent ↔ Agent collaboration (HORIZONTAL)
  Multiple agents delegating to each other
  "Team of AI specialists working together"
  Standard: How does one agent give work to another agent?

Real-world combined:
  Customer support agent ──MCP──▶ CRM database (tool)
  Customer support agent ──MCP──▶ Knowledge base (tool)
  Customer support agent ──A2A──▶ Technical specialist agent (peer agent)
  Technical specialist   ──MCP──▶ Internal bug tracker (tool)
```

### Other Alternatives

| Alternative | What It Is | Why MCP Wins |
|-------------|-----------|-------------|
| **OpenAPI + REST** | Traditional API integration | No discovery, no session, no notifications, M×N problem remains |
| **LangChain Tools** | Framework-specific tool layer | Tied to LangChain, not universal, not cross-client |
| **A2A** | Agent-to-agent protocol (Google) | Complementary, not competing — solves different problem |
| **Custom JSON-RPC** | Roll your own | Missing ecosystem, no SDK support, reinventing the wheel |
| **LlamaIndex Tools** | Framework-specific | Same as LangChain tools — not universal |

### MCP Adoption in 2026

- Claude Desktop, Claude Code: ✅ Native
- ChatGPT: ✅ Adopted March 2025
- Microsoft Copilot Studio: ✅ Supported
- Google Gemini agentic surface: ✅ Supported
- Cursor, Windsurf, Zed: ✅ Native
- VS Code (GitHub Copilot): ✅ MCP support added

**MCP has won the standardisation battle. It is the USB-C of AI tools.**

---
**production-grade authenticated MCP server** with OAuth/JWT security. [look here](https://github.com/techwithtim/AdvancedMCPServerWithAuth)
---

## 📁 File 1 — `database.py`

### What it sets up

```
SQLite database
    └── "notes" table
            ├── id        (auto-increment primary key)
            ├── user_id   (which user owns this note)
            └── content   (the note text)
```

### Line by line

```python
engine = create_engine('sqlite:///database.db')
```
Creates a SQLite database file called `database.db` in the current directory. SQLAlchemy is the ORM — you write Python, it writes SQL.

```python
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
```
A **session factory** — every time you call `SessionLocal()` you get a fresh DB connection. `autocommit=False` means you must explicitly call `.commit()` to save changes.

```python
class Note(Base):
    __tablename__ = "notes"
    id      = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, nullable=False, index=True)
    content = Column(Text, nullable=False)
```
Python class → SQL table mapping. The `index=True` on `user_id` means lookups by user are fast (important — you'll query by user_id constantly).

```python
Base.metadata.create_all(bind=engine)
```
Creates the table in the DB **if it doesn't exist yet**. Runs on import — safe to call multiple times.

```python
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
```
A **FastAPI-style dependency injector** — yields a session, guarantees `.close()` even if an exception happens. Not actually used in this codebase (the Repository pattern is used instead), but kept for potential FastAPI integration.

```python
class NoteRepository:
    @staticmethod
    def get_notes_by_user(user_id: str) -> List[Note]:
        db = SessionLocal()
        try:
            return db.query(Note).filter(Note.user_id == user_id).all()
        finally:
            db.close()
```
**Repository pattern** — all DB logic is centralised here. Creates its own session, queries, closes. The SQL equivalent is:
```sql
SELECT * FROM notes WHERE user_id = ?
```

```python
    @staticmethod
    def create_note(user_id: str, content: str) -> Note:
        db = SessionLocal()
        try:
            note = Note(user_id=user_id, content=content)
            db.add(note)
            db.commit()
            db.refresh(note)  # ← fetches the auto-generated id back from DB
            return note
        finally:
            db.close()
```
Creates a note row. `db.refresh(note)` is important — after commit, the `id` field is populated from the DB, so you refresh the Python object to get it.

---

## 📁 File 2 — `server.py`

### The Big Picture

```
Claude Desktop / any MCP client
        │
        │  Bearer token (JWT from Stytch)
        ▼
  FastMCP Server (HTTP transport, port 8000)
        │
        ├── validates JWT signature against Stytch JWKS
        ├── extracts user_id from JWT claims
        │
        ├── tool: get_my_notes  ──▶  NoteRepository.get_notes_by_user(user_id)
        └── tool: add_note      ──▶  NoteRepository.create_note(user_id, content)
```

This is a **multi-user, authenticated MCP server**. Each user's notes are isolated — user A can never see user B's notes.

---

### Authentication Setup

```python
auth = BearerAuthProvider(
    jwks_uri=f"{os.getenv('STYTCH_DOMAIN')}/.well-known/jwks.json",
    issuer=os.getenv("STYTCH_DOMAIN"),
    algorithm="RS256",
    audience=os.getenv("STYTCH_PROJECT_ID")
)
```

**Stytch** is an auth-as-a-service provider (like Auth0). When a user logs in via Stytch, they get a **JWT (JSON Web Token)**.

This `BearerAuthProvider` tells FastMCP:
- **Where to get public keys** (`jwks_uri`) — Stytch publishes public keys at this URL so anyone can verify its tokens
- **Who issued the token** (`issuer`) — must match the `iss` claim in the JWT
- **What algorithm was used** (`RS256`) — RSA-based asymmetric signing (Stytch signs with private key, server verifies with public key)
- **Who the token is for** (`audience`) — must match `aud` claim in JWT

FastMCP automatically rejects any request without a valid JWT. Your tools never even run if auth fails.

```
Client sends request:
  Authorization: Bearer eyJhbGci...  ← JWT token

FastMCP intercepts:
  1. Decodes JWT header → gets key ID
  2. Fetches matching public key from JWKS endpoint
  3. Verifies signature (was this really signed by Stytch?)
  4. Checks issuer + audience + expiry
  5. If valid → request proceeds to your tool
  6. If invalid → 401 Unauthorized, tool never runs
```

---

### The MCP Server

```python
mcp = FastMCP(name="Notes App", auth=auth)
```

Creates the MCP server with auth middleware attached. Every tool call is now authenticated.

---

### Tool 1 — `get_my_notes`

```python
@mcp.tool()
def get_my_notes() -> str:
    """Get all notes for a user"""
    access_token: AccessToken = get_access_token()
    user_id = jwt.get_unverified_claims(access_token.token)["sub"]
    notes = NoteRepository.get_notes_by_user(user_id)
    ...
```

**Step by step:**

```
get_access_token()
    └── FastMCP dependency — retrieves the validated JWT from the request context
        (FastMCP already verified it's valid — you just need the claims now)

jwt.get_unverified_claims(access_token.token)["sub"]
    └── Decodes the JWT payload WITHOUT re-verifying
        Safe here because FastMCP already verified it above
        "sub" = subject = the user's unique ID in Stytch
        e.g. "user-live-abc123def456"

NoteRepository.get_notes_by_user(user_id)
    └── SQL: SELECT * FROM notes WHERE user_id = 'user-live-abc123def456'
        Only this user's notes. Never another user's.
```

Note the tool takes **zero parameters** — the user identity comes from the JWT, not from user input. This is secure by design. A malicious user can't pass someone else's `user_id`.

---

### Tool 2 — `add_note`

```python
@mcp.tool()
def add_note(content: str) -> str:
    """Add a note for a user"""
    access_token: AccessToken = get_access_token()
    user_id = jwt.get_unverified_claims(access_token.token)["sub"]
    note = NoteRepository.create_note(user_id, content)
    return f"added note: {note.content}"
```

Same pattern — `user_id` comes from JWT, not from the user input. The user only provides `content`. They cannot forge notes for another user.

---

### OAuth Metadata Endpoint

```python
@mcp.custom_route("/.well-known/oauth-protected-resource", methods=["GET", "OPTIONS"])
def oauth_metadata(request: StarletteRequest) -> JSONResponse:
    base_url = str(request.base_url).rstrip("/")
    return JSONResponse({
        "resource": base_url,
        "authorization_servers": [os.getenv("STYTCH_DOMAIN")],
        "scopes_supported": ["read", "write"],
        "bearer_methods_supported": ["header", "body"]
    })
```

This is a **standard OAuth 2.0 Protected Resource Metadata** endpoint (RFC 9396). MCP clients that support OAuth discovery call this URL to learn:

```
"Who should I get a token from?"  →  authorization_servers: [Stytch URL]
"How do I send the token?"        →  bearer_methods_supported: ["header"]
"What permissions exist?"         →  scopes_supported: ["read", "write"]
```

Without this, MCP clients would have no automatic way to know which auth server to use. With it, a client can auto-discover the entire auth flow.

---

### Server Startup

```python
mcp.run(
    transport="http",          # remote HTTP server (not stdio)
    host="127.0.0.1",          # localhost only (put nginx/tunnel in front for prod)
    port=8000,
    middleware=[
        Middleware(
            CORSMiddleware,
            allow_origins=["*"],      # allow any frontend origin
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    ]
)
```

- **`transport="http"`** — runs as an HTTP server, not a subprocess (stdio). This means it can serve multiple concurrent clients over the network
- **CORS middleware** — allows browser-based MCP clients to connect (without CORS, browsers block cross-origin requests)
- **`host="127.0.0.1"`** — only accessible locally for now. In production you'd put Nginx or a cloud load balancer in front

---

## 🗺️ Full Flow — End to End

```
1. User logs in via Stytch in their browser
        └── Gets JWT: eyJhbGciOiJSUzI1NiJ9...

2. MCP client (Claude Desktop / custom client) sends request:
   POST /mcp
   Authorization: Bearer eyJhbGciOiJSUzI1NiJ9...
   Body: {"jsonrpc":"2.0","method":"tools/call","params":{"name":"get_my_notes"}}

3. FastMCP intercepts → BearerAuthProvider validates JWT
        ├── Fetches Stytch public key from JWKS URI
        ├── Verifies RS256 signature
        ├── Checks issuer + audience + expiry
        └── ✅ Valid → proceeds

4. get_my_notes() runs
        ├── get_access_token() → retrieves validated JWT from context
        ├── jwt.get_unverified_claims()["sub"] → "user-live-abc123"
        └── NoteRepository.get_notes_by_user("user-live-abc123")
                └── SELECT * FROM notes WHERE user_id = 'user-live-abc123'

5. Response sent back:
   {"jsonrpc":"2.0","id":1,"result":{"content":[{"type":"text","text":"Your notes:\n1: Buy milk\n2: Call dentist"}]}}
```

---

## 🔑 Key Patterns Being Used

| Pattern | Where | Why |
|---|---|---|
| **Bearer JWT auth** | `BearerAuthProvider` | Industry standard, stateless, verifiable |
| **JWKS validation** | `jwks_uri` | Never trust tokens without verifying the signature |
| **Identity from JWT** | `jwt.get_unverified_claims()["sub"]` | User can't forge their own identity |
| **Repository pattern** | `NoteRepository` | DB logic isolated, testable independently |
| **OAuth metadata** | `/.well-known/oauth-protected-resource` | Auto-discovery for MCP OAuth clients |
| **CORS middleware** | `CORSMiddleware` | Allow browser clients to connect |
| **HTTP transport** | `transport="http"` | Multi-user remote server (vs stdio single-user local) |

This is a solid pattern for any **multi-user production MCP server** — the same approach works for any per-user data (expenses, tasks, calendar, CRM records).

---

MCP is becoming the universal interface between AI agents and the outside world.
But "build an MCP server" can mean seven very different things.

## Here are the patterns I keep coming back to.

- 𝟬𝟭 𝗧𝗼𝗼𝗹 𝗪𝗿𝗮𝗽𝗽𝗲𝗿
→ One server, one service. Each API endpoint becomes one MCP tool.
→ Use when you just need to give the agent a clean handle on a specific service.
→ Think GitHub MCP, Slack MCP, Stripe MCP. This is where 90% of teams should start. Simple, debuggable, easy to version.

- 𝟬𝟮 𝗥𝗲𝘀𝗼𝘂𝗿𝗰𝗲 𝗣𝗿𝗼𝘃𝗶𝗱𝗲𝗿
→ Exposes data as MCP resources, not tools. The agent reads context, doesn't take action.
→ Use when you want the model to pull files, docs, or database rows into its context window on demand.
→ Tools mutate. Resources inform. Confusing the two is the most common MCP design mistake.

- 𝟬𝟯 𝗔𝗴𝗴𝗿𝗲𝗴𝗮𝘁𝗼𝗿 𝗚𝗮𝘁𝗲𝘄𝗮𝘆
→ One MCP server fronting many downstream services. Consolidates auth, rate limits, observability.
→ Use when you have a dozen internal services and don't want a dozen separate MCP connections per client.
→ The hidden value: one place to enforce policy, log every tool call, and rotate credentials.

- 𝟬𝟰 𝗦𝘁𝗮𝘁𝗲𝗳𝘂𝗹 𝗦𝗲𝘀𝘀𝗶𝗼𝗻
→ The server holds live state across tool calls. Browser sessions, DB transactions, file handles.
→ Use when the agent needs continuity, like driving a browser or staying inside a SQL transaction.
→ Think Playwright MCP, Postgres MCP. Powerful and expensive. Session expiry and cleanup are not optional.

- 𝟬𝟱 𝗦𝗮𝗻𝗱𝗯𝗼𝘅 𝗘𝘅𝗲𝗰𝘂𝘁𝗼𝗿
→ Isolated execution environment for code, shell commands, or filesystem operations.
→ Use when the agent needs to run untrusted operations safely. Think E2B-style code interpreters and ephemeral containers.
→ The MCP server is the trust boundary. Get that boundary wrong and everything inside it leaks out.

- 𝟬𝟲 𝗪𝗼𝗿𝗸𝗳𝗹𝗼𝘄 𝗢𝗿𝗰𝗵𝗲𝘀𝘁𝗿𝗮𝘁𝗼𝗿
→ Each tool wraps a multi-step internal workflow. The server orchestrates. The agent just calls.
→ Use when the steps are deterministic and you don't want to burn tokens on the model reasoning through them.
→ Anything you'd encode as a runbook belongs here, not in the agent's context window.

- 𝟬𝟳 𝗦𝘂𝗯𝗮𝗴𝗲𝗻𝘁
→ The MCP server is itself an agent. Claude calls another Claude, or a specialized model, for a delegated task.
→ Use when a sub-task needs its own reasoning loop, its own tools, and isolation from the parent's context.
→ Powerful for deep research, code review, and any task with a clean input-output contract.

The pattern you pick is a design decision, not a default.

Wrong pattern means wasted tokens, broken state, or a brittle agent that fills its context window with 40 tools it never needs and times out under load.

Most teams default to Tool Wrapper for everything, then wonder why their agent stops picking the right tool once the count crosses 20.

<img src="https://media.licdn.com/dms/image/v2/D4D22AQGc1oooiczTlQ/feedshare-shrink_800/B4DZ43Ave6I8Ac-/0/1779039402229?e=1781136000&v=beta&t=oBu0H6tkx7dXKObxh1IfIHNLJZaYvnP1dcnNJawvTyU" alt="top-7-mcp-server-patterns">

---

## 📚 Resources & Links

### Official Docs & Spec

| Resource | URL |
|----------|-----|
| MCP Official Docs | https://modelcontextprotocol.io |
| MCP Specification (latest) | https://modelcontextprotocol.io/specification/2025-11-25 |
| MCP Python SDK | https://github.com/modelcontextprotocol/python-sdk |
| MCP GitHub Org | https://github.com/modelcontextprotocol |
| FastMCP 2.0 Docs | https://gofastmcp.com |
| FastMCP GitHub | https://github.com/PrefectHQ/fastmcp |
| MCP Inspector | https://github.com/modelcontextprotocol/inspector |

### Pre-Built MCP Servers

| Server | What It Does | URL |
|--------|-------------|-----|
| Filesystem | Read/write local files | `@modelcontextprotocol/server-filesystem` |
| GitHub | Issues, PRs, repos | `@modelcontextprotocol/server-github` |
| PostgreSQL | Database queries | `@modelcontextprotocol/server-postgres` |
| Brave Search | Web search | `@modelcontextprotocol/server-brave-search` |
| Google Drive | Drive files | `@modelcontextprotocol/server-gdrive` |
| Slack | Channels, messages | `@modelcontextprotocol/server-slack` |
| Notion | Pages, databases | Community maintained |
| MCP Server registry | Browse all servers | https://github.com/modelcontextprotocol/servers |

### Learning Resources

| Resource | Type | Link |
|----------|------|-------|
| MCP Introduction | Official guide | https://modelcontextprotocol.io/introduction |
| Building MCP with FastMCP | Tutorial | https://gofastmcp.com/getting-started/quickstart |
| Complete MCP Guide 2026 | Article | https://dev.to/x4nent/complete-guide-to-mcp-model-context-protocol-in-2026 |
| MCP Developer Guide (GitHub) | Deep-dive | https://github.com/cyanheads/model-context-protocol-resources |
| JSON-RPC in MCP | Reference | https://mcpcat.io/guides/understanding-json-rpc-protocol-mcp |
| MCP Cheat Sheet | Quick reference | https://www.webfuse.com/mcp-cheat-sheet |
| MCP Message Types | JSON-RPC ref | https://portkey.ai/blog/mcp-message-types-complete-json-rpc-reference-guide |
| DataCamp FastMCP Tutorial | Hands-on | https://www.datacamp.com/tutorial/building-mcp-server-client-fastmcp |

---

## ✅ MCP Mastery Checklist

```
CONCEPTS
[ ] Can explain MCP vs function calling clearly with the N×M analogy
[ ] Know the three server-side primitives: tools, resources, prompts
[ ] Know the three client-side primitives: roots, sampling, elicitation
[ ] Understand why JSON-RPC was chosen over REST (5 reasons)
[ ] Know stdio vs HTTP+SSE — when to use each
[ ] Understand the 3-phase lifecycle: init → operation → shutdown
[ ] Know all standard error codes (-32700 to -32603, -32001 to -32005)
[ ] Understand how capabilities are negotiated at handshake

HANDS-ON
[ ] Connected Claude Desktop to filesystem + GitHub + Postgres MCP servers
[ ] Built a custom MCP server with FastMCP (dice roller / calculator)
[ ] Tested server with MCP Inspector (viewed raw JSON-RPC messages)
[ ] Installed custom server into Claude Desktop with fastmcp install
[ ] Built a custom client that connects to your custom server
[ ] Built the Expense Tracker MCP server (all 3 tools + 1 resource)
[ ] Deployed a remote MCP server with HTTP transport
[ ] Integrated MCP into an existing FastAPI app

ADVANCED
[ ] Know the difference between FastMCP 1.0, MCP SDK, and FastMCP 2.0
[ ] Used Context in a FastMCP tool for progress reporting and sampling
[ ] Generated MCP server from FastAPI using FastMCP.from_fastapi()
[ ] Can explain MCP vs A2A and when to use each
```

---

*MCP Deep-Dive Notes | GenAI + LLMOps Engineering Roadmap 2026*  
*Spec version: 2025-11-25 | FastMCP version: 2.x | Last updated: May 2026*