# ☁️ Phase 7 — Cloud AI Platforms
> **Complete Study Notes | Theory + Code + Use Cases | Interview Prep**
> Part of: GenAI + LLMOps Engineering Roadmap 2026
> Estimated time: 70–90 hrs over 5–6 weeks
> Prerequisites: Phase 1–6 (LLM APIs, RAG, FastAPI, LangGraph, Design Patterns, Cost Optimization)

---

## 📑 Table of Contents

1. [Why Cloud Matters for AI Engineers](#1-why-cloud-matters-for-ai-engineers)
2. [7.1 — AWS Bedrock](#71--aws-bedrock)
3. [7.2 — AWS Lambda for AI Workloads](#72--aws-lambda-for-ai-workloads)
4. [7.3 — AWS SageMaker](#73--aws-sagemaker)
5. [7.4 — GCP Vertex AI + Agent Development Kit (ADK)](#74--gcp-vertex-ai--agent-development-kit-adk)
6. [7.5 — AWS Core Infrastructure for AI Apps](#75--aws-core-infrastructure-for-ai-apps)
7. [7.6 — GCP Cloud Run for AI](#76--gcp-cloud-run-for-ai)
8. [7.7 — Kubernetes for AI Engineers](#77--kubernetes-for-ai-engineers)
9. [Platform Decision Framework](#8-platform-decision-framework)
10. [Phase 7 Projects](#-phase-7-projects)
11. [Interview Cheat Sheet](#-interview-cheat-sheet)
12. [Quick Revision Cards](#-quick-revision-cards)

---

## 1. Why Cloud Matters for AI Engineers

### 🧑‍🎓 Explain Like I'm in BTech 2nd Year

When you build a personal project on your laptop, you control everything — start/stop the server, manage the database, add memory if needed. But a production AI system that serves 10,000 users simultaneously is a different beast.

Think of cloud platforms as a city's infrastructure. You could build your own power plant, water system, and roads — or you could plug into the city's existing infrastructure and focus on building your building. Cloud platforms are that existing infrastructure for AI.

```
What you get from cloud platforms for AI:

Compute:
  GPU instances (for model training/fine-tuning)
  Serverless (for variable load — pay only when running)
  Containers (for consistent deployment)

Managed AI Services:
  Pre-built RAG pipelines (Bedrock Knowledge Bases)
  Pre-built agent frameworks (Bedrock Agents, ADK)
  Pre-built safety layers (Bedrock Guardrails)
  Fine-tuning APIs (SageMaker, Vertex AI)
  → No managing your own vector DB, embedding pipeline, safety layer

Storage:
  Object storage (S3, GCS) for documents/models
  Managed databases (RDS, Cloud SQL) for state
  Managed caches (ElastiCache, Memorystore) for performance

Networking:
  Load balancers (auto-scale AI endpoints)
  CDN (serve model outputs globally, fast)
  VPCs (isolate AI services for compliance)
```

### Two Clouds, One Decision

```
AWS (Amazon Web Services):
  Market share: ~32% of cloud
  AI services: Bedrock (managed LLM access) + SageMaker (ML platform)
  Dominant in: financial services, healthcare, enterprise
  Your advantage: Bedrock + Lambda = cheapest scalable AI API

GCP (Google Cloud Platform):
  Market share: ~11% of cloud
  AI services: Vertex AI + ADK (Gemini-native agents)
  Dominant in: startups, media, gaming, data-intensive companies
  Your advantage: ADK + Cloud Run = fastest GenAI agent deployment

You don't need to choose — senior AI engineers know both.
Every enterprise uses a mix. Knowing both is a hiring signal.
```

---

## 7.1 — AWS Bedrock

### 🧑‍🎓 Explain Like I'm in BTech 2nd Year

If you want to use GPT-4o in your app, you call OpenAI's API. If you want to use Claude in your app within AWS (for data residency, compliance, or enterprise contracts), you call AWS Bedrock's API.

Bedrock is essentially a managed marketplace for foundation models, hosted on AWS infrastructure. It adds enterprise features on top: managed RAG, managed agents, built-in safety layers, compliance (HIPAA, SOC 2, GDPR), and billing through your AWS account.

```
Without Bedrock (direct API):
  Your App → OpenAI/Anthropic API → response
  Issues:
    Data leaves AWS (compliance concern)
    Separate billing + contract per provider
    No managed RAG — build your own
    No managed agents — build your own
    No managed guardrails — build your own

With Bedrock:
  Your App → Bedrock → routes to Claude/Llama/Mistral/Titan
  Benefits:
    Data stays in AWS (HIPAA, GDPR, FedRAMP compliant)
    One AWS bill for all models
    Bedrock Knowledge Bases = managed RAG (no Qdrant/LangChain)
    Bedrock Agents = managed agent framework (no LangGraph for simple cases)
    Bedrock Guardrails = managed safety (no custom guardrail code)
```

### Available Models on Bedrock (2026)

```
ANTHROPIC:
  anthropic.claude-opus-4-v1          ← most capable
  anthropic.claude-sonnet-4-v1        ← best price/performance
  anthropic.claude-haiku-4-v1         ← fastest, cheapest

META:
  meta.llama3-3-70b-instruct-v1       ← open-source, customisable
  meta.llama3-1-8b-instruct-v1        ← small, fast, cheap
  meta.llama4-maverick-v1             ← multimodal

AMAZON:
  amazon.titan-text-express-v1        ← AWS native, optimised
  amazon.nova-lite-v1                 ← cheap for classification
  amazon.nova-pro-v1                  ← production quality

MISTRAL:
  mistral.mistral-large-2402-v1
  mistral.mixtral-8x7b-instruct-v0
```

### 7.1.1 Basic Bedrock API

```python
# Install: pip install boto3

import boto3
import json

# boto3 automatically uses IAM credentials from environment
# Set via: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION
# Or better: EC2 instance role / Lambda execution role (no keys needed)

bedrock = boto3.client(
    service_name="bedrock-runtime",
    region_name="ap-south-1",  # Mumbai — data stays in India
)

# ── InvokeModel — complete response (non-streaming) ──────────────────
def invoke_bedrock(model_id: str, prompt: str) -> str:
    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 1000,
        "messages": [{"role": "user", "content": prompt}],
    })
    response = bedrock.invoke_model(
        modelId=model_id,
        contentType="application/json",
        accept="application/json",
        body=body,
    )
    result = json.loads(response["body"].read())
    return result["content"][0]["text"]

answer = invoke_bedrock(
    "anthropic.claude-haiku-4-20250514-v1:0",
    "What is RAG in AI systems?"
)
print(answer)

# ── InvokeModelWithResponseStream — streaming ─────────────────────────
def stream_bedrock(model_id: str, prompt: str):
    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 1000,
        "messages": [{"role": "user", "content": prompt}],
    })
    response = bedrock.invoke_model_with_response_stream(
        modelId=model_id,
        contentType="application/json",
        accept="application/json",
        body=body,
    )
    for event in response["body"]:
        chunk = json.loads(event["chunk"]["bytes"])
        if chunk["type"] == "content_block_delta":
            yield chunk["delta"].get("text", "")

# Stream to console
for token in stream_bedrock("anthropic.claude-haiku-4-20250514-v1:0", "Explain RAG"):
    print(token, end="", flush=True)
```

### 7.1.2 Bedrock + FastAPI — Streaming to Browser

```python
# backend/api/routes/bedrock_chat.py

import boto3
import json
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

app = FastAPI()
bedrock = boto3.client("bedrock-runtime", region_name="ap-south-1")

class ChatRequest(BaseModel):
    message: str
    model:   str = "anthropic.claude-haiku-4-20250514-v1:0"

@app.post("/chat/stream")
async def chat_stream(body: ChatRequest):
    """Stream Bedrock tokens via SSE to browser"""

    async def generate():
        request_body = json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens":        1000,
            "messages":          [{"role": "user", "content": body.message}],
        })

        try:
            response = bedrock.invoke_model_with_response_stream(
                modelId=body.model,
                contentType="application/json",
                accept="application/json",
                body=request_body,
            )
            for event in response["body"]:
                chunk = json.loads(event["chunk"]["bytes"])

                if chunk["type"] == "content_block_delta":
                    token = chunk["delta"].get("text", "")
                    if token:
                        yield f"data: {json.dumps({'token': token})}\n\n"

                elif chunk["type"] == "message_stop":
                    usage = chunk.get("amazon-bedrock-invocationMetrics", {})
                    yield f"data: {json.dumps({'done': True, 'usage': usage})}\n\n"
                    break

        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no"})
```

### 7.1.3 Bedrock Knowledge Bases — Managed RAG

```python
"""
Bedrock Knowledge Bases: managed RAG pipeline
  You bring: documents in S3
  AWS provides: chunking, embedding (Titan), vector store (managed)
  You get: a managed retrieval API — no Qdrant, no LangChain needed

When to use Bedrock KB vs your own RAG:
  Bedrock KB:   enterprise, compliance, minimal infra, AWS-only stack
  Custom RAG:   full control, custom chunking, multiple vector stores,
               hybrid retrieval, cost sensitive (Bedrock KB is expensive)
"""

import boto3

# Pre-requisite: create Knowledge Base in AWS Console
# → Add data source: S3 bucket with your PDFs
# → Choose embedding model: amazon.titan-embed-text-v2
# → Knowledge Base ID: "XXXXXXXX"

bedrock_agent_runtime = boto3.client(
    "bedrock-agent-runtime",
    region_name="ap-south-1",
)

def query_knowledge_base(
    query:              str,
    knowledge_base_id:  str,
    n_results:          int = 5,
) -> dict:
    """Query Bedrock Knowledge Base — managed RAG retrieval"""

    response = bedrock_agent_runtime.retrieve(
        knowledgeBaseId=knowledge_base_id,
        retrievalQuery={"text": query},
        retrievalConfiguration={
            "vectorSearchConfiguration": {
                "numberOfResults": n_results,
                "overrideSearchType": "HYBRID",  # BM25 + vector (like our Phase 5A)
            }
        },
    )

    results = []
    for r in response["retrievalResults"]:
        results.append({
            "content":    r["content"]["text"],
            "score":      r["score"],
            "source":     r["location"]["s3Location"]["uri"],
        })
    return results

def retrieve_and_generate(
    query:             str,
    knowledge_base_id: str,
    model_id:          str = "anthropic.claude-haiku-4-20250514-v1:0",
) -> str:
    """
    RetrieveAndGenerate: one API call does retrieval + generation
    Bedrock automatically injects retrieved context into the prompt.
    """
    response = bedrock_agent_runtime.retrieve_and_generate(
        input={"text": query},
        retrieveAndGenerateConfiguration={
            "type": "KNOWLEDGE_BASE",
            "knowledgeBaseConfiguration": {
                "knowledgeBaseId":  knowledge_base_id,
                "modelArn":         f"arn:aws:bedrock:ap-south-1::foundation-model/{model_id}",
                "generationConfiguration": {
                    "promptTemplate": {
                        "textPromptTemplate":
                            "Answer using ONLY the context below. "
                            "If not in context: 'I don't have that information.'\n\n"
                            "Context:\n$search_results$\n\nQuestion: $query$"
                    }
                },
            }
        },
    )
    return response["output"]["text"]
```

### 7.1.4 Bedrock Agents — Managed Agent Builder

```python
"""
Bedrock Agents = managed agent framework
  You define: action groups (your Lambda functions the agent can call)
  AWS provides: the reasoning loop, memory, knowledge base integration
  
  Architecture:
    Bedrock Agent
      ├── Knowledge Base (RAG over your documents)
      ├── Action Group 1: get_order_status → Lambda function
      ├── Action Group 2: generate_report → Lambda function
      └── Memory: per-session conversation history

When to use Bedrock Agents vs LangGraph:
  Bedrock Agents:   AWS-native, less code, managed infra, compliance-friendly
  LangGraph:        full control, HITL, complex routing, non-AWS stack,
                    custom state management, open-source
  
  "For a compliance-sensitive enterprise app where everything is already
   in AWS → Bedrock Agents. For a complex custom agent with HITL and
   intricate state → LangGraph."
"""

import boto3

bedrock_agent_runtime = boto3.client(
    "bedrock-agent-runtime",
    region_name="ap-south-1",
)

def invoke_bedrock_agent(
    agent_id:        str,
    agent_alias_id:  str,
    session_id:      str,     # unique per conversation (memory key)
    user_message:    str,
) -> str:
    """
    Invoke a Bedrock Agent with memory.
    The agent handles: reasoning, tool selection, knowledge base retrieval.
    """
    response = bedrock_agent_runtime.invoke_agent(
        agentId=agent_id,
        agentAliasId=agent_alias_id,
        sessionId=session_id,          # agent remembers this session
        inputText=user_message,
        enableTrace=True,              # see agent reasoning steps (for debugging)
    )

    # Stream response chunks
    full_response = ""
    for event in response["completion"]:
        if "chunk" in event:
            chunk = event["chunk"]["bytes"].decode("utf-8")
            full_response += chunk

        if "trace" in event:
            # Agent's reasoning trace — useful for debugging + Langfuse integration
            trace = event["trace"]["trace"]
            if "orchestrationTrace" in trace:
                step = trace["orchestrationTrace"]
                if "modelInvocationInput" in step:
                    print(f"  [Agent thinking] {step['modelInvocationInput']['text'][:100]}")

    return full_response

# Lambda function for action group: get_order_status
# This runs in AWS Lambda — no server to manage
# ─────────────────────────────────────────────────
# def lambda_handler(event, context):
#     """Bedrock calls this Lambda when agent decides to use this action"""
#     action = event["actionGroup"]
#     function = event["function"]
#     params = {p["name"]: p["value"] for p in event.get("parameters", [])}
#
#     if function == "get_order_status":
#         order_id = params["order_id"]
#         # query DB, return result
#         return {"response": {"actionGroup": action, "function": function,
#                              "functionResponse": {"responseBody":
#                                  {"TEXT": {"body": f"Order {order_id}: shipped, ETA 2 days"}}}}}
```

### 7.1.5 Bedrock Guardrails — Managed Safety

```python
"""
Bedrock Guardrails = managed content safety layer.
Apply to any model call on Bedrock.
Features:
  - Content filtering: violence, hate speech, sexual content (configurable severity)
  - PII detection + redaction: names, emails, phone numbers, Aadhaar, PAN
  - Topic blocking: block specific topics ("don't discuss competitor X")
  - Grounding check: flag LLM responses not grounded in retrieved context
  - Word blocking: custom bad word list
"""

def invoke_with_guardrails(
    model_id:        str,
    prompt:          str,
    guardrail_id:    str,
    guardrail_version: str = "DRAFT",
) -> dict:
    """Call Bedrock model with guardrails applied to both input and output"""
    request_body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 1000,
        "messages": [{"role": "user", "content": prompt}],
        "system": "You are a helpful customer support assistant.",
    })

    response = bedrock.invoke_model(
        modelId=model_id,
        contentType="application/json",
        accept="application/json",
        body=request_body,
        guardrailIdentifier=guardrail_id,
        guardrailVersion=guardrail_version,
        trace="ENABLED",
    )

    result = json.loads(response["body"].read())

    # Check if guardrail intervened
    if result.get("stop_reason") == "guardrail_intervened":
        actions = result.get("amazon-bedrock-guardrailAction", [])
        return {
            "blocked":    True,
            "reason":     actions,
            "response":   "I cannot respond to that request.",
        }

    return {
        "blocked":  False,
        "response": result["content"][0]["text"],
    }

# Create guardrail config (done in AWS Console or CDK):
# {
#   "contentPolicyConfig": {
#     "filtersConfig": [
#       {"type": "HATE",     "inputStrength": "HIGH",   "outputStrength": "HIGH"},
#       {"type": "VIOLENCE", "inputStrength": "MEDIUM", "outputStrength": "HIGH"},
#     ]
#   },
#   "piiEntityTypes": ["EMAIL", "PHONE", "NAME", "AADHAAR", "PAN"],
#   "piiAction": "ANONYMIZE",   # or "BLOCK"
#   "deniedTopics": [
#     {"name": "CompetitorMention", "definition": "Any mention of competitor products"}
#   ]
# }
```

### 7.1.6 IAM Roles for Bedrock — Least Privilege

```python
"""
CRITICAL: Never use root credentials or user access keys for Bedrock in production.
Always use IAM roles attached to compute (Lambda, ECS task, EC2).

Principle of Least Privilege:
  Only grant exactly the Bedrock actions your service needs.
  Never: "bedrockruntime:*" (all actions on all models)
  Yes:   "bedrock:InvokeModel" on specific model ARNs only
"""

# IAM policy (define in CDK, Terraform, or AWS Console)
BEDROCK_POLICY = {
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "bedrock:InvokeModel",
                "bedrock:InvokeModelWithResponseStream",
            ],
            "Resource": [
                # Only allow specific models — not wildcard
                "arn:aws:bedrock:ap-south-1::foundation-model/anthropic.claude-haiku-4-*",
                "arn:aws:bedrock:ap-south-1::foundation-model/anthropic.claude-sonnet-4-*",
            ],
        },
        {
            "Effect": "Allow",
            "Action": [
                "bedrock:Retrieve",
                "bedrock:RetrieveAndGenerate",
            ],
            "Resource": [
                # Only the specific Knowledge Base this service uses
                f"arn:aws:bedrock:ap-south-1:123456789:knowledge-base/XXXXXXXXXX",
            ],
        },
    ]
}

# In code: assume role via EC2 instance profile (no credentials in code!)
# boto3 automatically discovers credentials in this order:
# 1. Environment variables (dev local)
# 2. ~/.aws/credentials file (dev local)
# 3. IAM role attached to EC2/Lambda/ECS (production — PREFERRED)

bedrock_client = boto3.client("bedrock-runtime")
# That's it — no keys in code. Role discovered automatically.
```

---

## 7.2 — AWS Lambda for AI Workloads

### 🧑‍🎓 Explain Like I'm in BTech 2nd Year

Lambda is serverless computing. You upload a function. AWS runs it when triggered. You pay only for the time it runs — down to milliseconds. When nobody calls it: zero cost, zero servers running.

```
Traditional server (ECS/EC2):
  Server runs 24/7 even if no requests
  You pay: $50/month even if 0 requests on Tuesday
  You manage: deployment, scaling, patches

Lambda:
  Function exists only when called
  You pay: per invocation × duration
  Zero requests = $0
  AWS manages: scaling, infra, patches
  
For AI workloads specifically:
  Perfect for: intermittent LLM tasks (batch scoring, async processing)
  Not ideal for:  always-on services with high concurrency (use ECS instead)
  
Example costs:
  1,000,000 Lambda invocations × 1 second each = $2/month
  ECS Fargate for same load (if always on) = $50/month
  AWS gives first 1M requests/month FREE
```

### 7.2.1 Lambda Basics for AI

```python
# Basic Lambda handler for LLM inference
# This file = lambda_function.py → zip it → upload to Lambda

import json
import boto3

bedrock = boto3.client("bedrock-runtime", region_name="ap-south-1")

def lambda_handler(event, context):
    """
    Lambda entry point.
    event: request payload (from API Gateway, SQS, EventBridge, etc.)
    context: Lambda runtime metadata (function_name, remaining_time_in_ms, etc.)
    
    This is triggered by:
    - API Gateway HTTP request
    - SQS message (batch job)
    - EventBridge scheduled rule
    - S3 event (new document uploaded)
    """

    # Extract from API Gateway proxy integration
    body    = json.loads(event.get("body", "{}"))
    query   = body.get("query", "")
    user_id = body.get("user_id", "anonymous")

    if not query:
        return {
            "statusCode": 400,
            "body": json.dumps({"error": "query is required"}),
        }

    # Call Bedrock (IAM role handles auth — no keys in code)
    response = bedrock.invoke_model(
        modelId="anthropic.claude-haiku-4-20250514-v1:0",
        contentType="application/json",
        accept="application/json",
        body=json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens":        500,
            "messages":          [{"role": "user", "content": query}],
        }),
    )
    result  = json.loads(response["body"].read())
    answer  = result["content"][0]["text"]
    tokens  = result["usage"]

    return {
        "statusCode": 200,
        "headers": {
            "Content-Type":                "application/json",
            "Access-Control-Allow-Origin": "*",   # CORS
        },
        "body": json.dumps({
            "answer":         answer,
            "input_tokens":   tokens["input_tokens"],
            "output_tokens":  tokens["output_tokens"],
        }),
    }

# Lambda timeout: default 3s, max 15 minutes
# Set to 60-120s for LLM calls (they can be slow)
# Memory: 128MB-10GB; more memory = more CPU = faster cold starts
# Recommended for AI: 512MB-1GB
```

### 7.2.2 Lambda Response Streaming

```python
"""
Lambda Streaming: stream tokens as they're generated
Without streaming: client waits 3-10s → gets full response at once
With streaming: client gets first token in 400ms → reads as it generates

Requirements:
  - Lambda Python 3.12+ runtime
  - RESPONSE_STREAM invoke mode (in Lambda config)
  - API Gateway HTTP API (not REST API) for streaming to browser
"""

import awslambdaric.bootstrap as bootstrap
from awslambdaric.runtime_client import RuntimeClient

def handler(event, context):
    """Streaming Lambda handler"""

    body  = json.loads(event.get("body", "{}"))
    query = body.get("query", "")

    # Must use @streaming decorator with awslambdaric
    # Lambda streaming handler MUST be defined differently:
    pass

# ── Correct streaming handler ──────────────────────────────────────────
# requirements.txt: awslambdaric>=1.0.0

import boto3, json
from awslambdaric.streaming import streaming_response

bedrock = boto3.client("bedrock-runtime")

@streaming_response
def lambda_handler(event, context, response_stream):
    body  = json.loads(event.get("body", "{}"))
    query = body.get("query", "")

    # Write SSE headers
    response_stream.set_headers({
        "Content-Type":    "text/event-stream",
        "Cache-Control":   "no-cache",
        "X-Accel-Buffering":"no",
    })

    bedrock_response = bedrock.invoke_model_with_response_stream(
        modelId="anthropic.claude-haiku-4-20250514-v1:0",
        contentType="application/json",
        accept="application/json",
        body=json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 500,
            "messages": [{"role": "user", "content": query}],
        }),
    )

    for event_chunk in bedrock_response["body"]:
        chunk = json.loads(event_chunk["chunk"]["bytes"])
        if chunk["type"] == "content_block_delta":
            token = chunk["delta"].get("text", "")
            if token:
                # Write token to stream — browser receives immediately
                response_stream.write(f"data: {json.dumps({'token': token})}\n\n".encode())

    response_stream.write(b"data: [DONE]\n\n")
```

### 7.2.3 Mangum — Run FastAPI on Lambda

```python
"""
Mangum converts your FastAPI app into a Lambda handler.
Zero code changes to your FastAPI app needed.

Architecture:
  HTTP request
       ↓
  API Gateway (routes to Lambda)
       ↓
  Lambda (your FastAPI app wrapped in Mangum)
       ↓
  FastAPI processes request normally
       ↓
  Response back through same chain

Benefits:
  - Run your existing FastAPI AI app on Lambda (serverless, scale-to-zero)
  - Pay only when requests come in
  - No ECS/EC2 to manage
  
Limitations:
  - Cold starts: first request after idle period takes 1-3 extra seconds
  - No WebSockets (use API Gateway WebSocket API separately)
  - Execution timeout: max 15 minutes (fine for LLM calls; not for long agents)
"""

# requirements.txt: mangum>=0.17.0, fastapi, boto3

from fastapi import FastAPI
from mangum import Mangum
from pydantic import BaseModel
import boto3, json

app = FastAPI(title="AI API on Lambda")

bedrock = boto3.client("bedrock-runtime", region_name="ap-south-1")

class QueryRequest(BaseModel):
    query: str

@app.post("/api/chat")
async def chat(body: QueryRequest) -> dict:
    """Standard FastAPI endpoint — works locally AND on Lambda"""
    response = bedrock.invoke_model(
        modelId="anthropic.claude-haiku-4-20250514-v1:0",
        contentType="application/json",
        accept="application/json",
        body=json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 500,
            "messages": [{"role": "user", "content": body.query}],
        }),
    )
    result = json.loads(response["body"].read())
    return {"answer": result["content"][0]["text"]}

@app.get("/health")
async def health():
    return {"status": "ok"}

# This line makes the entire FastAPI app work as a Lambda handler
handler = Mangum(app, lifespan="off")

# Deploy:
# 1. pip install -r requirements.txt -t ./package
# 2. cp lambda_function.py package/
# 3. cd package && zip -r ../deployment.zip .
# 4. aws lambda update-function-code --function-name my-ai-api \
#    --zip-file fileb://deployment.zip
#
# Or better: use SAM / CDK for automated deploy
```

### 7.2.4 Cold Start Mitigation

```python
"""
Cold start: Lambda initialises a new container on first request after idle.
For AI: cold start = loading boto3, fastapi, langchain → can be 3-5 seconds.
That's unacceptable for a user-facing API.

Solutions:
"""

# Solution 1: Provisioned Concurrency
# Keep N Lambda instances always warm (pre-started, ready to respond)
# Cost: you pay for reserved concurrency even when idle
# Use for: latency-sensitive endpoints (user-facing chat)

# Configure in Lambda console or CDK:
# provisioned_concurrency_config = {"ProvisionedConcurrentExecutions": 2}
# Keeps 2 warm instances → first request is instant

# Solution 2: Lambda Layer for heavy dependencies
# Move heavy imports to a Layer so they're shared across invocations
# ──────────────────────────────────────────────────────────────
# layers/python/python/
#   ├── boto3/
#   ├── fastapi/
#   └── pydantic/
# → Zip and upload as Lambda Layer
# → Layer is cached separately → not re-uploaded with each deploy

# Solution 3: Minimise imports at module level
# ── SLOW (cold start): ──
# import langchain      # 500ms to import
# import pandas         # 200ms to import
# import torch          # 2000ms to import

# ── FAST: lazy imports ──
def heavy_function():
    import langchain   # imported only when function is called
    # ...

# Solution 4: Use smaller runtime
# Python 3.12 with minimal deps = faster cold start than 3.9 with many deps
# Keep Lambda deployment zip < 50MB for faster cold start

# Solution 5: Ping Lambda on a schedule (not recommended)
# EventBridge rule: run Lambda every 5 minutes with warm-up payload
# {is_warmup: true} → handler returns immediately without processing
```

### 7.2.5 EventBridge + Lambda — Scheduled AI Jobs

```python
"""
EventBridge = AWS's event bus + cron scheduler.
Use it to trigger Lambda on a schedule for:
  - Daily RAG re-ingestion (new documents appeared in S3)
  - Nightly batch LLM scoring (process accumulated jobs)
  - Weekly model drift checks
  - Hourly cost summaries
"""

# EventBridge rule (define in CDK/Terraform/Console):
# rate(1 day) → Lambda: daily_rag_reingestion
# cron(0 2 * * ? *) → Lambda: nightly_batch_scoring (2 AM UTC daily)

# Lambda handler for scheduled job
def lambda_handler(event, context):
    """
    EventBridge cron event — runs daily at 2 AM.
    Re-indexes any new documents added to S3 in the last 24 hours.
    """
    import datetime

    # Check if this is a warmup ping (from scheduled warmup)
    if event.get("is_warmup"):
        return {"status": "warm"}

    source     = event.get("source")
    detail_type= event.get("detail-type")

    if detail_type == "Scheduled Event":
        # Triggered by cron
        return run_daily_reingestion()

    elif source == "s3-upload-notifier":
        # Triggered by S3 event (new document)
        s3_key = event["detail"]["object"]["key"]
        return process_single_document(s3_key)

def run_daily_reingestion():
    """Find and index new S3 documents from last 24 hours"""
    s3  = boto3.client("s3")
    bedrock_agent = boto3.client("bedrock-agent")
    since = (datetime.datetime.utcnow() - datetime.timedelta(days=1)).isoformat()

    # List objects modified in last 24 hours
    paginator = s3.get_paginator("list_objects_v2")
    new_docs  = []
    for page in paginator.paginate(Bucket="my-docs-bucket"):
        for obj in page.get("Contents", []):
            if obj["LastModified"].isoformat() > since:
                new_docs.append(obj["Key"])

    if new_docs:
        # Trigger Bedrock Knowledge Base sync
        bedrock_agent.start_ingestion_job(
            knowledgeBaseId="XXXXXXXXXX",
            dataSourceId="YYYYYYYYYY",
        )
        print(f"Started re-ingestion for {len(new_docs)} new documents")

    return {"status": "done", "new_docs": len(new_docs)}
```

### 7.2.6 Lambda + SQS — Async Batch Processing

```python
"""
SQS + Lambda: the standard async AI job pattern.

Flow:
  API request → put job in SQS → return immediately to caller (202 Accepted)
  SQS → triggers Lambda → Lambda processes job (could take 60 seconds)
  Lambda writes result to DB → (optional) webhook or WebSocket notifies client

Use cases:
  Resume parsing (400 PDFs → 400 SQS messages → Lambda processes each)
  Batch LLM scoring (1000 items → process 10 at a time with concurrency)
  Report generation (slow LLM job → don't make user wait)
"""

# FastAPI: accept job, queue it, return immediately
@app.post("/api/batch-score")
async def batch_score(body: BatchScoreRequest) -> dict:
    sqs  = boto3.client("sqs")
    jobs = []

    for resume_id in body.resume_ids:
        # Send each resume as its own SQS message
        resp = sqs.send_message(
            QueueUrl=SQS_QUEUE_URL,
            MessageBody=json.dumps({
                "job_type":   "score_resume",
                "resume_id":  resume_id,
                "job_id":     body.job_id,
                "request_id": str(uuid.uuid4()),
            }),
            # Delay: process resumes 5 seconds after upload (let S3 settle)
            DelaySeconds=5,
        )
        jobs.append(resp["MessageId"])

    return {
        "status":    "queued",
        "job_count": len(jobs),
        "message":   "Processing in background. Check /api/jobs/{job_id}/status",
    }

# Lambda: process SQS messages
def lambda_handler(event, context):
    """
    SQS trigger: Lambda gets batches of up to 10 messages at once.
    Batch size + concurrency controlled in Lambda → SQS trigger settings.
    """
    processed = []
    failed    = []

    for record in event["Records"]:
        body    = json.loads(record["body"])
        job_type= body["job_type"]

        try:
            if job_type == "score_resume":
                result = score_resume_with_bedrock(body["resume_id"], body["job_id"])
                write_score_to_db(body["resume_id"], result)
                processed.append(body["resume_id"])

        except Exception as e:
            # Return failed messages to SQS for retry
            # Don't add to batchItemFailures → message requeued automatically
            failed.append({
                "itemIdentifier": record["messageId"],
                "error":          str(e),
            })

    print(f"Processed: {len(processed)}, Failed: {len(failed)}")
    return {"batchItemFailures": [{"itemIdentifier": f["itemIdentifier"]} for f in failed]}
```

---

## 7.3 — AWS SageMaker

### 🧑‍🎓 Explain Like I'm in BTech 2nd Year

If Bedrock is "use pre-trained models as-is", SageMaker is "train, fine-tune, and host your own models". Think of Bedrock as buying a chef from a restaurant, and SageMaker as building your own kitchen and training your own chef.

```
Bedrock:   use existing foundation models (Claude, Llama, etc.)
SageMaker: train/fine-tune/host any model (your fine-tuned Llama, custom classifier)

When you need SageMaker:
  Fine-tuned model: you've done QLoRA on Llama for your domain
  Custom ML model: your resume screener, churn predictor, fraud detector
  Private model:   can't send data to external APIs (healthcare, finance)
  Cost at scale:   hosting Llama locally on SageMaker cheaper than per-token pricing
```

### 7.3.1 SageMaker JumpStart — Deploy in Minutes

```python
"""
SageMaker JumpStart: one-click deploy of popular models.
No training needed — just deploy a pre-trained model as an endpoint.
Models: Llama 3, Mistral, Falcon, Stable Diffusion, BERT, etc.
"""

import boto3

sagemaker_client = boto3.client("sagemaker", region_name="ap-south-1")
sm_runtime = boto3.client("sagemaker-runtime", region_name="ap-south-1")

# Step 1: Create endpoint from JumpStart model
# (In production: use CDK or Terraform; here shown as boto3 for understanding)
def deploy_jumpstart_model(
    model_id:      str = "meta-textgeneration-llama-3-3-70b-instruct",
    instance_type: str = "ml.g5.12xlarge",   # 4x A10G GPUs
    endpoint_name: str = "llama-3-3-70b-prod",
):
    from sagemaker.jumpstart.model import JumpStartModel

    model = JumpStartModel(model_id=model_id)
    predictor = model.deploy(
        initial_instance_count=1,
        instance_type=instance_type,
        endpoint_name=endpoint_name,
        accept_eula=True,     # required for Llama models
    )
    print(f"Endpoint deployed: {endpoint_name}")
    return predictor

# Step 2: Call the deployed endpoint
def invoke_sagemaker_endpoint(
    endpoint_name: str,
    prompt:        str,
    max_tokens:    int = 500,
) -> str:
    """Call a SageMaker endpoint — same boto3 API regardless of model"""
    payload = json.dumps({
        "inputs":     prompt,
        "parameters": {
            "max_new_tokens": max_tokens,
            "temperature":    0.1,
            "top_p":          0.9,
            "do_sample":      True,
        },
    })
    response = sm_runtime.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType="application/json",
        Body=payload,
    )
    result = json.loads(response["Body"].read())
    return result[0]["generated_text"]
```

### 7.3.2 SageMaker Pipelines — MLOps Automation

```python
"""
SageMaker Pipelines: managed MLOps pipeline.
Define steps: PreProcess → Train → Evaluate → Register → Deploy
Each step is a SageMaker job (runs in managed compute).
Re-run on new data, compare model versions, auto-deploy if metrics pass.

This is what you use for HireSense resume screener retraining in production.
(Phase 7 upgrade from the Celery-based local retraining in HireSense PRD)
"""

import boto3
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import ProcessingStep, TrainingStep, ConditionStep
from sagemaker.workflow.conditions import ConditionGreaterThan
from sagemaker.workflow.parameters import ParameterFloat, ParameterString
from sagemaker.processing import ScriptProcessor
from sagemaker.estimator import Estimator
from sagemaker import Session

sagemaker_session = Session()
ROLE = "arn:aws:iam::123456789012:role/SageMakerRole"  # IAM role for SageMaker

# ── Pipeline parameters (can be overridden at runtime) ────────────────
min_f1_threshold = ParameterFloat(name="MinF1Threshold", default_value=0.75)
input_data_uri   = ParameterString(name="InputDataUri",
                                    default_value="s3://my-bucket/training-data/")

# ── Step 1: PreProcessing ─────────────────────────────────────────────
preprocess_processor = ScriptProcessor(
    image_uri=f"763104351884.dkr.ecr.ap-south-1.amazonaws.com/sklearn:1.2-1-cpu-py3",
    command=["python3"],
    instance_type="ml.m5.xlarge",
    instance_count=1,
    base_job_name="resume-preprocess",
    role=ROLE,
    sagemaker_session=sagemaker_session,
)

preprocess_step = ProcessingStep(
    name="PreprocessData",
    processor=preprocess_processor,
    inputs=[{"InputName": "raw", "source": input_data_uri}],
    outputs=[{"OutputName": "train"}, {"OutputName": "validation"}],
    code="scripts/preprocess.py",   # your preprocessing script
)

# ── Step 2: Training ──────────────────────────────────────────────────
sklearn_estimator = Estimator(
    image_uri=f"763104351884.dkr.ecr.ap-south-1.amazonaws.com/sklearn:1.2-1-cpu-py3",
    entry_point="scripts/train.py",    # your training script
    role=ROLE,
    instance_count=1,
    instance_type="ml.m5.2xlarge",
    output_path="s3://my-bucket/models/",
    hyperparameters={
        "n_estimators":  200,
        "max_depth":     10,
        "min_f1":        min_f1_threshold,
    },
    sagemaker_session=sagemaker_session,
)

train_step = TrainingStep(
    name="TrainModel",
    estimator=sklearn_estimator,
    inputs={
        "train":      preprocess_step.properties.ProcessingOutputConfig.Outputs["train"],
        "validation": preprocess_step.properties.ProcessingOutputConfig.Outputs["validation"],
    },
)

# ── Step 3: Evaluate ──────────────────────────────────────────────────
eval_processor = ScriptProcessor(
    image_uri="...", command=["python3"], instance_type="ml.m5.xlarge",
    instance_count=1, role=ROLE, sagemaker_session=sagemaker_session,
)

eval_step = ProcessingStep(
    name="EvaluateModel",
    processor=eval_processor,
    inputs=[{"InputName": "model", "source": train_step.properties.ModelArtifacts.S3ModelArtifacts}],
    outputs=[{"OutputName": "evaluation"}],
    code="scripts/evaluate.py",
    property_files=[{
        "PropertyFileName": "evaluation_report",
        "OutputName":       "evaluation",
        "Path":             "evaluation.json",   # {"f1": 0.81, "auc": 0.88}
    }],
)

# ── Step 4: Conditional — only register if F1 is high enough ──────────
condition_step = ConditionStep(
    name="CheckModelQuality",
    conditions=[
        ConditionGreaterThan(
            left=eval_step.properties.Outputs["evaluation_report"]["f1"],
            right=min_f1_threshold,
        )
    ],
    if_steps=[register_step],    # register model if condition passes
    else_steps=[notify_step],    # notify if model is not good enough
)

# ── Assemble Pipeline ─────────────────────────────────────────────────
pipeline = Pipeline(
    name="ResumeScreenerPipeline",
    parameters=[min_f1_threshold, input_data_uri],
    steps=[preprocess_step, train_step, eval_step, condition_step],
    sagemaker_session=sagemaker_session,
)

pipeline.upsert(role_arn=ROLE)   # create/update pipeline

# Run the pipeline
pipeline.start(
    parameters={"MinF1Threshold": 0.78}  # override threshold at runtime
)
```

### 7.3.3 SageMaker Real-Time vs Async Inference

```python
"""
TWO ENDPOINT TYPES:

Real-time endpoint:
  - Synchronous: call → response in milliseconds to seconds
  - Best for: user-facing APIs, interactive applications
  - Scales automatically based on traffic
  - Cost: you pay while endpoint is running (even if idle)
  
Async inference endpoint:
  - Accepts request → returns immediately → processes async → writes to S3
  - Best for: long-running jobs (3-60 seconds), batch processing
  - Cost: only pay per invocation (no idle cost when no requests)
  - Use for: RAG report generation, resume batch scoring
"""

# Async inference configuration
async_config = {
    "AsyncInferenceConfig": {
        "OutputConfig": {
            "S3OutputPath": "s3://my-bucket/async-outputs/",
            "NotificationConfig": {
                "SuccessTopic": "arn:aws:sns:...:inference-success",
                "ErrorTopic":   "arn:aws:sns:...:inference-error",
            }
        },
        "ClientConfig": {
            "MaxConcurrentInvocationsPerInstance": 4,
        },
    }
}

# Invoke async endpoint
response = sm_runtime.invoke_endpoint_async(
    EndpointName="resume-scorer-async",
    ContentType="application/json",
    InputLocation="s3://my-bucket/inputs/batch-001.json",  # input in S3
)
output_location = response["OutputLocation"]  # poll this S3 path for results

# Check result
import time
def poll_async_result(s3_path: str, timeout: int = 120) -> dict:
    """Poll S3 for async inference result"""
    s3     = boto3.client("s3")
    bucket = s3_path.split("/")[2]
    key    = "/".join(s3_path.split("/")[3:])

    for _ in range(timeout // 2):
        try:
            response = s3.get_object(Bucket=bucket, Key=key)
            return json.loads(response["Body"].read())
        except s3.exceptions.NoSuchKey:
            time.sleep(2)   # not ready yet, wait 2s
    raise TimeoutError(f"Async inference timed out after {timeout}s")
```

---

## 7.4 — GCP Vertex AI + Agent Development Kit (ADK)

### 🧑‍🎓 Explain Like I'm in BTech 2nd Year

Vertex AI is Google's answer to AWS SageMaker + Bedrock combined. But what makes it unique in 2026 is the Agent Development Kit (ADK) — Google's open-source Python SDK for building agents that use Gemini natively. Think of it as Google's version of LangGraph, but with built-in Gemini integration, Google Search as a native tool, and Cloud Run as the deployment target.

```
AWS stack:           GCP stack:
  Bedrock            ↔  Vertex AI (Gemini API)
  Bedrock Agents     ↔  Google ADK
  SageMaker          ↔  Vertex AI Training
  Lambda             ↔  Cloud Run
  API Gateway        ↔  Cloud Endpoints
  IAM roles          ↔  Service Accounts + Workload Identity
```

### 7.4.1 Vertex AI Gemini API

```python
# pip install google-cloud-aiplatform vertexai

import vertexai
from vertexai.generative_models import GenerativeModel, GenerationConfig, Part

# Initialize (uses Application Default Credentials from gcloud auth)
vertexai.init(project="my-gcp-project", location="us-central1")

# ── Basic Gemini call ─────────────────────────────────────────────────
model = GenerativeModel("gemini-2.0-flash-001")

response = model.generate_content(
    "Explain retrieval augmented generation in 3 sentences.",
    generation_config=GenerationConfig(
        temperature=0.1,
        max_output_tokens=256,
    ),
)
print(response.text)

# ── Streaming ─────────────────────────────────────────────────────────
for chunk in model.generate_content("Write a short story", stream=True):
    print(chunk.text, end="", flush=True)

# ── Chat (multi-turn) ─────────────────────────────────────────────────
chat = model.start_chat()
r1 = chat.send_message("My name is Rahul. I work in AI.")
r2 = chat.send_message("What should I focus on next in my career?")
# Gemini remembers "Rahul works in AI" from r1

# ── Multimodal — analyse image + text ─────────────────────────────────
image_part = Part.from_uri(
    uri="gs://my-bucket/architecture-diagram.png",
    mime_type="image/png",
)
response = model.generate_content([
    image_part,
    "What are the potential bottlenecks in this system architecture?",
])
print(response.text)
```

### 7.4.2 Google ADK — Agent Development Kit

```python
"""
Google ADK (April 2025): Python SDK for building Gemini-native agents.
  - Simpler than LangGraph for straightforward agents
  - Built-in Google Search, code execution as tools
  - Native MCP support (connect any MCP server)
  - Deployment: Cloud Run or Vertex AI Agent Engine (managed)
  - Sessions API: persistent conversation state

When to use ADK vs LangGraph:
  ADK:       GCP stack, Gemini, Google Search, quick agent, simple routing
  LangGraph: complex state, HITL, non-GCP, Redis checkpointing, custom graphs
"""

# pip install google-adk

from google.adk.agents import LlmAgent, SequentialAgent
from google.adk.tools import google_search, built_in_code_execution
from google.adk.tools import FunctionTool
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from vertexai.generative_models import GenerativeModel

# ── Define custom tools ───────────────────────────────────────────────
def get_weather(city: str) -> dict:
    """
    Get current weather for a city.
    Args:
        city: City name (e.g., "Mumbai", "Bangalore")
    Returns:
        Weather data dict with temperature and condition
    """
    # In production: call a weather API
    return {"city": city, "temperature": 28, "condition": "sunny", "humidity": 65}

def search_company_data(company: str, metric: str) -> str:
    """
    Search internal company database for specific metrics.
    Args:
        company: Company name to search for
        metric:  What to look up (revenue, headcount, growth_rate)
    Returns:
        The metric value as a string
    """
    data = {
        "TechCorp": {"revenue": "₹450 crore", "headcount": 1200, "growth_rate": "32%"},
        "StartupXYZ": {"revenue": "₹80 crore", "headcount": 250, "growth_rate": "67%"},
    }
    return str(data.get(company, {}).get(metric, "Not found"))

# ── Build ADK agent ───────────────────────────────────────────────────
research_agent = LlmAgent(
    name="research_agent",
    model="gemini-2.0-flash-001",
    description="A research agent that can search the web and analyse company data",
    instruction="""You are a professional business research analyst.
When asked about companies:
  1. Search the web for recent news and financial information
  2. Check the internal company database for key metrics
  3. Synthesise and provide a comprehensive analysis

Always cite your sources. Be specific with numbers.
Keep responses under 400 words unless more detail is requested.""",
    tools=[
        google_search,                            # built-in Google Search
        FunctionTool(func=search_company_data),   # custom tool
        FunctionTool(func=get_weather),           # another custom tool
    ],
)

# ── ADK with MCP Server ───────────────────────────────────────────────
from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset, StdioServerParameters

# Connect to a local Postgres MCP server
postgres_mcp = MCPToolset(
    connection_params=StdioServerParameters(
        command="npx",
        args=["@modelcontextprotocol/server-postgres", "postgresql://localhost/mydb"],
    )
)

db_agent = LlmAgent(
    name="db_agent",
    model="gemini-2.5-pro-001",  # more capable for SQL
    description="Agent that can query the database",
    instruction="You are a SQL expert. Query the database to answer user questions. Always explain your SQL.",
    tools=[postgres_mcp],
)

# ── Run the agent ─────────────────────────────────────────────────────
async def run_adk_agent():
    session_service = InMemorySessionService()
    runner = Runner(
        agent=research_agent,
        app_name="research_app",
        session_service=session_service,
    )

    session_id = "user_001_session"

    # Turn 1
    response = await runner.run_async(
        user_id="user_001",
        session_id=session_id,
        new_message="Research TechCorp — I need their latest news, revenue, and growth rate.",
    )
    print("Agent:", response.text)

    # Turn 2 — agent remembers context from turn 1
    response = await runner.run_async(
        user_id="user_001",
        session_id=session_id,
        new_message="How does that growth compare to StartupXYZ?",
    )
    print("Agent:", response.text)

import asyncio
asyncio.run(run_adk_agent())
```

### 7.4.3 ADK Multi-Agent Orchestration

```python
"""
ADK supports multi-agent patterns:
  Sequential:   agent A → output → agent B → output → agent C
  Parallel:     all agents run simultaneously, results merged
  Hierarchical: orchestrator routes to sub-agents
"""

from google.adk.agents import LlmAgent, SequentialAgent, ParallelAgent

# Sub-agents (specialists)
web_researcher = LlmAgent(
    name="web_researcher",
    model="gemini-2.0-flash-001",
    description="Searches the web for information",
    instruction="Research the given topic thoroughly using web search. Return factual findings.",
    tools=[google_search],
)

data_analyst = LlmAgent(
    name="data_analyst",
    model="gemini-2.5-pro-001",
    description="Analyses data and runs calculations",
    instruction="Analyse the provided data. Write Python code to calculate insights. Be precise.",
    tools=[built_in_code_execution],
)

report_writer = LlmAgent(
    name="report_writer",
    model="gemini-2.0-flash-001",
    description="Writes professional reports",
    instruction="Write a clear, concise executive report based on the provided research and analysis.",
)

# Sequential pipeline: research → analyse → write
research_pipeline = SequentialAgent(
    name="research_pipeline",
    description="Full research-to-report pipeline",
    sub_agents=[web_researcher, data_analyst, report_writer],
)

# Parallel: run multiple researchers simultaneously
parallel_research = ParallelAgent(
    name="parallel_researcher",
    description="Parallel research from multiple sources",
    sub_agents=[web_researcher, data_analyst],
    # Both run simultaneously — total time = slowest agent
)

# Hierarchical: orchestrator decides which sub-agent to use
orchestrator = LlmAgent(
    name="orchestrator",
    model="gemini-2.5-pro-001",
    description="Routes tasks to the right specialist agent",
    instruction="""You are an orchestrator. Route tasks to specialists:
  - Research tasks → web_researcher
  - Data analysis tasks → data_analyst
  - Report writing → report_writer
  - Complex tasks → research_pipeline""",
    sub_agents=[web_researcher, data_analyst, report_writer, research_pipeline],
)
```

### 7.4.4 ADK Evaluation

```python
"""
ADK has built-in evaluation — trajectory and final response quality.
This is the equivalent of our RAGAS eval, but native to ADK.
"""

from google.adk.evaluation import AgentEvaluator

evaluator = AgentEvaluator()

# Define test cases
TEST_CASES = [
    {
        "query":          "What is the revenue of TechCorp?",
        "expected_tools": ["search_company_data"],
        "expected_answer": "₹450 crore",
    },
    {
        "query":          "Latest AI news from Google",
        "expected_tools": ["google_search"],
        "expected_answer": None,  # ground truth not available for real-time news
    },
]

# Evaluate agent
results = await evaluator.evaluate(
    agent=research_agent,
    test_cases=TEST_CASES,
    metrics=["tool_use_accuracy", "response_quality", "trajectory_completeness"],
)

for case, result in zip(TEST_CASES, results):
    print(f"Query: {case['query'][:50]}")
    print(f"  Tool accuracy:      {result.tool_use_accuracy:.2f}")
    print(f"  Response quality:   {result.response_quality:.2f}")
    print(f"  Trajectory:         {result.trajectory_completeness:.2f}")
```

---

## 7.5 — AWS Core Infrastructure for AI Apps

### 7.5.1 The Standard AWS AI App Stack

```
Request
  ↓
Route 53 (DNS)
  ↓
CloudFront (CDN — cache static assets globally)
  ↓
ALB (Application Load Balancer — distribute to ECS instances)
  ↓
ECS Fargate (your FastAPI AI service — containerised, auto-scaling)
  │                ↓
  │         ElastiCache Redis (session cache, semantic cache)
  │                ↓
  │         RDS Aurora Postgres (conversation history, user data)
  │                ↓
  │         S3 (documents for RAG, model artifacts)
  │                ↓
  │         Bedrock (LLM calls)
  ↓
SQS (async jobs queue)
  ↓
Lambda (process SQS jobs — parsing, scoring, embeddings)
  ↓
CloudWatch (logs, metrics, alarms)
  ↓
Sentry (error tracking)
```

### 7.5.2 ECS Fargate — Containerised AI Service

```python
"""
ECS Fargate: run Docker containers without managing servers.
"Serverless containers" — you define the container, AWS runs it.
Best for: always-on AI APIs (vs Lambda which is for intermittent)
"""

# Dockerfile for FastAPI AI service
DOCKERFILE = """
FROM python:3.12-slim

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app
COPY app/ /app/
WORKDIR /app

# Expose port
EXPOSE 8000

# Health check (ECS uses this to know if container is healthy)
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Start server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", \
     "--workers", "4", "--timeout-keep-alive", "120"]
"""

# ECS Task Definition (simplified — use CDK in production)
TASK_DEFINITION = {
    "family": "synapseiq-api",
    "networkMode": "awsvpc",
    "requiresCompatibilities": ["FARGATE"],
    "cpu": "1024",      # 1 vCPU
    "memory": "2048",   # 2 GB RAM
    "executionRoleArn": "arn:aws:iam::123456789012:role/ecsTaskExecutionRole",
    "taskRoleArn":      "arn:aws:iam::123456789012:role/ecsTaskRole",
    "containerDefinitions": [
        {
            "name":  "api",
            "image": "123456789012.dkr.ecr.ap-south-1.amazonaws.com/synapseiq:latest",
            "portMappings": [{"containerPort": 8000, "protocol": "tcp"}],
            "environment": [
                {"name": "AWS_REGION", "value": "ap-south-1"},
                {"name": "ENVIRONMENT", "value": "production"},
            ],
            "secrets": [
                # Inject secrets from AWS Secrets Manager (never in env vars)
                {"name": "OPENAI_API_KEY",  "valueFrom": "arn:aws:secretsmanager:..."},
                {"name": "DATABASE_URL",    "valueFrom": "arn:aws:secretsmanager:..."},
            ],
            "logConfiguration": {
                "logDriver": "awslogs",
                "options": {
                    "awslogs-group":  "/ecs/synapseiq-api",
                    "awslogs-region": "ap-south-1",
                    "awslogs-stream-prefix": "ecs",
                },
            },
            "healthCheck": {
                "command": ["CMD-SHELL", "curl -f http://localhost:8000/health || exit 1"],
                "interval": 30, "timeout": 5, "retries": 3,
            },
        }
    ],
}

# Auto-scaling (scale out when CPU > 70% for 2 minutes)
AUTO_SCALING_POLICY = {
    "TargetValue": 70.0,
    "ScaleInCooldown":  60,
    "ScaleOutCooldown": 30,
    "PredefinedMetricSpecification": {
        "PredefinedMetricType": "ECSServiceAverageCPUUtilization"
    },
}
```

### 7.5.3 S3 Patterns for AI

```python
"""
S3: the default storage for everything in AI systems.
  Documents for RAG ingestion
  Model artifacts (weights, checkpoints)
  Uploaded files (resumes, PDFs)
  Async job inputs/outputs
  Backups and logs
"""

import boto3
from botocore.config import Config

s3 = boto3.client(
    "s3",
    region_name="ap-south-1",
    config=Config(
        retries={"max_attempts": 3, "mode": "standard"},
        max_pool_connections=50,
    ),
)

# ── Upload file to S3 with metadata ──────────────────────────────────
async def upload_document(
    file_bytes: bytes,
    filename:   str,
    org_id:     str,
    source_id:  str,
) -> str:
    """Upload document to S3. Returns S3 key."""
    key = f"orgs/{org_id}/sources/{source_id}/{filename}"
    s3.put_object(
        Bucket="synapseiq-documents",
        Key=key,
        Body=file_bytes,
        ContentType="application/pdf",
        Metadata={
            "org_id":    org_id,
            "source_id": source_id,
            "uploaded":  datetime.utcnow().isoformat(),
        },
        # Server-side encryption (always enable for sensitive docs)
        ServerSideEncryption="aws:kms",
    )
    return key

# ── Pre-signed URL — let client upload directly to S3 (skip your server) ──
def generate_upload_url(key: str, expires_in: int = 3600) -> str:
    """
    Generate a pre-signed URL for direct-to-S3 upload from browser.
    Client uploads directly to S3 — your server never handles the file.
    Reduces bandwidth + cost.
    """
    return s3.generate_presigned_url(
        "put_object",
        Params={"Bucket": "synapseiq-documents", "Key": key,
                "ContentType": "application/pdf"},
        ExpiresIn=expires_in,
    )

# ── S3 event → Lambda trigger ─────────────────────────────────────────
# When a new document is uploaded to S3, trigger Lambda to start ingestion
# Configure in S3 bucket → Event Notifications → Lambda function
# Event: s3:ObjectCreated:*
# Filter: prefix=orgs/, suffix=.pdf
```

### 7.5.4 CloudWatch — AI-Specific Monitoring

```python
"""
CloudWatch: AWS's built-in monitoring.
For AI systems: log custom metrics (token counts, cache hit rate, LLM latency).
"""

import boto3
from datetime import datetime

cloudwatch = boto3.client("cloudwatch", region_name="ap-south-1")

# ── Push custom metrics ───────────────────────────────────────────────
def track_llm_call(
    model:         str,
    input_tokens:  int,
    output_tokens: int,
    latency_ms:    int,
    cost_usd:      float,
    cache_hit:     bool,
    org_id:        str,
):
    """Push per-LLM-call metrics to CloudWatch"""
    cloudwatch.put_metric_data(
        Namespace="SynapseIQ/LLM",
        MetricData=[
            {
                "MetricName": "InputTokens",
                "Value":      input_tokens,
                "Unit":       "Count",
                "Dimensions": [
                    {"Name": "Model",  "Value": model},
                    {"Name": "OrgId",  "Value": org_id},
                ],
                "Timestamp": datetime.utcnow(),
            },
            {
                "MetricName": "Latency",
                "Value":      latency_ms,
                "Unit":       "Milliseconds",
                "Dimensions": [{"Name": "Model", "Value": model}],
            },
            {
                "MetricName": "CacheHitRate",
                "Value":      1.0 if cache_hit else 0.0,
                "Unit":       "Count",
            },
            {
                "MetricName": "CostUSD",
                "Value":      cost_usd,
                "Unit":       "None",
                "Dimensions": [{"Name": "OrgId", "Value": org_id}],
            },
        ],
    )

# ── Create alarm: alert if P95 LLM latency > 5 seconds ───────────────
def create_latency_alarm():
    cloudwatch.put_metric_alarm(
        AlarmName="LLM-P95-Latency-High",
        MetricName="Latency",
        Namespace="SynapseIQ/LLM",
        Statistic="p95",
        Period=300,               # 5 minutes
        EvaluationPeriods=2,      # alert if high for 2 consecutive periods
        Threshold=5000,           # 5 seconds in ms
        ComparisonOperator="GreaterThanThreshold",
        AlarmActions=["arn:aws:sns:ap-south-1:...:engineering-alerts"],
    )
```

---

## 7.6 — GCP Cloud Run for AI

### 🧑‍🎓 Explain Like I'm in BTech 2nd Year

Cloud Run is GCP's version of "serverless containers". You give it a Docker image, it runs it when requests come in, and scales to zero when idle.

```
Lambda:     serverless FUNCTIONS (code, not containers; short-lived, stateless)
Cloud Run:  serverless CONTAINERS (full app; longer timeout, more memory, streaming)

Cloud Run vs Lambda for AI:
  Lambda:     cheap for infrequent jobs, cold start matters less
              max 15 min timeout, max 10GB memory
  Cloud Run:  better for long-running LLM jobs (60 min timeout)
              supports HTTP/2 + SSE natively (great for streaming)
              larger containers (up to 32GB RAM)
              → better for AI API that needs streaming + longer LLM timeouts
```

### 7.6.1 Deploy FastAPI to Cloud Run

```bash
# Deploy your existing FastAPI app to Cloud Run
# Zero code changes needed

# 1. Authenticate
gcloud auth login
gcloud config set project my-gcp-project

# 2. Build and push Docker image
gcloud builds submit --tag gcr.io/my-gcp-project/synapseiq-api:latest .
# Or: Artifact Registry (newer):
gcloud builds submit --tag us-central1-docker.pkg.dev/my-gcp-project/apps/synapseiq-api:latest .

# 3. Deploy to Cloud Run
gcloud run deploy synapseiq-api \
  --image      us-central1-docker.pkg.dev/my-gcp-project/apps/synapseiq-api:latest \
  --region     asia-south1 \
  --platform   managed \
  --port       8000 \
  --cpu        2 \
  --memory     2Gi \
  --timeout    300 \
  --min-instances 1 \        # keep 1 warm instance (avoid cold starts)
  --max-instances 20 \       # scale up to 20 on peak traffic
  --concurrency   10 \       # 10 concurrent requests per instance
  --set-env-vars  ENVIRONMENT=production \
  --set-secrets   OPENAI_API_KEY=openai-api-key:latest \
  --allow-unauthenticated    # or: --no-allow-unauthenticated for private

# 4. View URL
gcloud run services describe synapseiq-api --region asia-south1 \
  --format='value(status.url)'
```

```python
# Cloud Run + Vertex AI — call Gemini from Cloud Run (no API keys needed!)
# Service Account → Workload Identity → automatic auth

from google.cloud import aiplatform
import vertexai
from vertexai.generative_models import GenerativeModel

# When running on Cloud Run:
# The Cloud Run service account automatically gets credentials
# No GOOGLE_APPLICATION_CREDENTIALS needed
vertexai.init(project="my-gcp-project", location="asia-south1")

@app.post("/api/chat")
async def chat(body: ChatRequest) -> dict:
    model    = GenerativeModel("gemini-2.0-flash-001")
    response = model.generate_content(body.message)
    return {"answer": response.text}
    # No API key management — IAM handles auth entirely
```

### 7.6.2 Cloud Run Streaming

```python
"""
Cloud Run supports HTTP/2 and SSE natively.
FastAPI + StreamingResponse works out of the box.
No special config needed (unlike Lambda which needs streaming mode).
"""

from fastapi.responses import StreamingResponse
import vertexai
from vertexai.generative_models import GenerativeModel

vertexai.init(project="my-gcp-project", location="asia-south1")

@app.post("/api/chat/stream")
async def chat_stream(body: ChatRequest):
    """Stream Gemini tokens via SSE — works natively on Cloud Run"""
    model = GenerativeModel("gemini-2.0-flash-001")

    async def generate():
        try:
            # Gemini streaming
            for chunk in model.generate_content(body.message, stream=True):
                if chunk.text:
                    yield f"data: {json.dumps({'token': chunk.text})}\n\n"
            yield f"data: {json.dumps({'done': True})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
```

### 7.6.3 Cloud Run + Pub/Sub — Async LLM Jobs

```python
"""
Cloud Pub/Sub = GCP's equivalent of AWS SQS.
Use it to queue async LLM jobs from Cloud Run.
"""

from google.cloud import pubsub_v1
import json, base64

publisher = pubsub_v1.PublisherClient()
TOPIC_PATH = "projects/my-gcp-project/topics/llm-jobs"

# Producer: queue a job
@app.post("/api/ingest")
async def queue_ingestion_job(body: IngestRequest) -> dict:
    """Queue document for async ingestion — return immediately"""
    message = json.dumps({
        "job_type":  "ingest_document",
        "document_id": body.document_id,
        "org_id":      body.org_id,
    }).encode("utf-8")

    future = publisher.publish(TOPIC_PATH, message)
    message_id = future.result()

    return {"status": "queued", "message_id": message_id}

# Consumer: Cloud Run service receiving Pub/Sub push messages
@app.post("/pubsub/ingest-worker")
async def pubsub_worker(request: Request) -> dict:
    """Pub/Sub pushes messages here (Cloud Run worker)"""
    body = await request.json()
    # Pub/Sub wraps message in a Pub/Sub envelope
    message_data = base64.b64decode(body["message"]["data"])
    job = json.loads(message_data)

    if job["job_type"] == "ingest_document":
        await process_document_ingestion(job["document_id"], job["org_id"])

    return {"status": "ok"}  # ACK to Pub/Sub
```

### 7.6.4 Deploy ADK Agent to Cloud Run

```python
# Deploy an ADK multi-agent system as a Cloud Run service
# ADK provides a FastAPI app you can wrap and deploy

from google.adk.cli.fast_api_app import get_fast_api_app

# Import your agents
from agents.research_pipeline import orchestrator

# Get ADK's built-in FastAPI app
# This includes: /run, /sessions, /events endpoints
adk_app = get_fast_api_app(
    agents=[orchestrator],
    session_service_uri="redis://redis:6379",   # or Cloud Memorystore
)

# Add your own custom routes on top
@adk_app.get("/health")
async def health():
    return {"status": "ok"}

# Deploy: gcloud run deploy adk-agent --image ... (same as regular Cloud Run)
```

---

## 7.7 — Kubernetes for AI Engineers

### 🧑‍🎓 Explain Like I'm in BTech 2nd Year

Kubernetes (K8s) is the operating system for distributed systems. If ECS/Cloud Run are managed apartments (AWS handles the building), Kubernetes is building your own building — you manage more, but have full control.

```
You don't need to be a K8s admin.
You need to know enough to:
  1. Deploy your AI service on K8s
  2. Read and modify YAML configs
  3. Debug issues with kubectl
  4. Talk to DevOps teams about GPU scheduling
  5. Interview: explain K8s to a hiring manager who doesn't care about K8s internals
```

### 7.7.1 Core Concepts

```yaml
# ── Pod: one container (or a few tightly coupled) ────────────────────
# You almost never create Pods directly. Use Deployments.
apiVersion: v1
kind: Pod
metadata:
  name: ai-api-pod
spec:
  containers:
  - name: api
    image: gcr.io/my-project/synapseiq-api:latest
    ports:
    - containerPort: 8000
    env:
    - name: ENVIRONMENT
      value: production
    resources:
      requests:
        memory: "512Mi"  # minimum guaranteed
        cpu:    "250m"   # 0.25 vCPU
      limits:
        memory: "2Gi"    # hard cap
        cpu:    "1000m"  # 1 vCPU
```

```yaml
# ── Deployment: manage pod replicas, rolling updates ─────────────────
apiVersion: apps/v1
kind: Deployment
metadata:
  name: synapseiq-api
  namespace: production
spec:
  replicas: 3              # always keep 3 pods running
  selector:
    matchLabels:
      app: synapseiq-api
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge:       1    # create 1 new pod before killing old
      maxUnavailable: 0    # never have less than replica count (zero-downtime)
  template:
    metadata:
      labels:
        app: synapseiq-api
    spec:
      containers:
      - name: api
        image: gcr.io/my-project/synapseiq-api:v2.1.0
        ports:
        - containerPort: 8000
        envFrom:
        - secretRef:
            name: api-secrets           # mount all secrets as env vars
        - configMapRef:
            name: api-config            # mount config as env vars
        resources:
          requests: {memory: "512Mi", cpu: "250m"}
          limits:   {memory: "2Gi",   cpu: "1000m"}
        readinessProbe:                 # K8s: is this pod ready to receive traffic?
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 5
        livenessProbe:                  # K8s: is this pod still alive?
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 15
          periodSeconds: 10
```

```yaml
# ── Service: expose Deployment to other pods or the internet ─────────
apiVersion: v1
kind: Service
metadata:
  name: synapseiq-api-service
spec:
  selector:
    app: synapseiq-api
  ports:
  - port: 80
    targetPort: 8000
  type: ClusterIP          # internal only; use LoadBalancer for external

---
# ── ConfigMap: non-secret config ─────────────────────────────────────
apiVersion: v1
kind: ConfigMap
metadata:
  name: api-config
data:
  ENVIRONMENT:  production
  LOG_LEVEL:    info
  REGION:       ap-south-1

---
# ── Secret: sensitive config (base64-encoded in cluster) ─────────────
apiVersion: v1
kind: Secret
metadata:
  name: api-secrets
type: Opaque
data:
  OPENAI_API_KEY: <base64-encoded-key>    # never store raw keys
  DATABASE_URL:   <base64-encoded-url>
# In production: use External Secrets Operator to sync from AWS Secrets Manager
```

### 7.7.2 AI-Specific K8s Patterns

```yaml
# ── GPU Scheduling — run LLM inference on GPU nodes ──────────────────
apiVersion: apps/v1
kind: Deployment
metadata:
  name: vllm-inference
spec:
  replicas: 1
  template:
    spec:
      containers:
      - name: vllm
        image: vllm/vllm-openai:latest
        args:
        - --model
        - meta-llama/Llama-3.1-8B-Instruct
        - --tensor-parallel-size
        - "1"
        resources:
          requests:
            nvidia.com/gpu: "1"   # request 1 GPU
          limits:
            nvidia.com/gpu: "1"   # hard limit: can't exceed 1
            memory: "16Gi"
            cpu: "4"
      nodeSelector:
        cloud.google.com/gke-accelerator: nvidia-l4   # only schedule on GPU nodes
      tolerations:
      - key:    "nvidia.com/gpu"     # allow scheduling on GPU-tainted nodes
        operator: "Exists"
        effect: "NoSchedule"

---
# ── PersistentVolume — cache model weights so they don't re-download ──
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: model-weights-cache
spec:
  accessModes:
  - ReadWriteOnce
  resources:
    requests:
      storage: 20Gi         # Llama 3.1 8B ≈ 16GB
  storageClassName: premium-rwo

# Mount in pod:
# volumes:
# - name: model-cache
#   persistentVolumeClaim:
#     claimName: model-weights-cache
# volumeMounts:
# - name: model-cache
#   mountPath: /root/.cache/huggingface  # HuggingFace cache dir

---
# ── Init Container — download model before main container starts ──────
spec:
  initContainers:
  - name: download-model
    image: python:3.12-slim
    command:
    - python3
    - -c
    - |
      from huggingface_hub import snapshot_download
      snapshot_download("meta-llama/Llama-3.1-8B-Instruct",
                        cache_dir="/model-cache")
    volumeMounts:
    - name: model-cache
      mountPath: /model-cache

---
# ── HPA — Auto-scale based on GPU utilisation ─────────────────────────
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: api-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: synapseiq-api
  minReplicas: 2
  maxReplicas: 20
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70   # scale out when avg CPU > 70%

---
# ── CronJob — scheduled RAG re-ingestion ─────────────────────────────
apiVersion: batch/v1
kind: CronJob
metadata:
  name: rag-reingestion
spec:
  schedule: "0 2 * * *"    # 2 AM UTC daily
  jobTemplate:
    spec:
      template:
        spec:
          containers:
          - name: reingestion
            image: gcr.io/my-project/reingestion-worker:latest
            command: ["python", "scripts/daily_reingestion.py"]
          restartPolicy: OnFailure
```

### 7.7.3 kubectl Commands — Daily Use

```bash
# ── View ──────────────────────────────────────────────────────────────
kubectl get pods                          # list all pods in current namespace
kubectl get pods -n production            # specific namespace
kubectl get pods -o wide                  # show node each pod is on
kubectl describe pod synapseiq-api-7f8b   # detailed pod info (events, status)
kubectl get deployments                   # list deployments
kubectl get services                      # list services
kubectl get hpa                           # list autoscalers

# ── Debug ─────────────────────────────────────────────────────────────
kubectl logs synapseiq-api-7f8b           # last 100 lines of logs
kubectl logs synapseiq-api-7f8b -f        # follow logs (like tail -f)
kubectl logs synapseiq-api-7f8b --previous # logs from last crashed container
kubectl exec -it synapseiq-api-7f8b -- bash  # shell into running pod

# ── Port-forward — access pod directly for debugging ─────────────────
kubectl port-forward pod/synapseiq-api-7f8b 8080:8000
# Now: curl localhost:8080/health (bypasses load balancer)

# ── Deploy ────────────────────────────────────────────────────────────
kubectl apply -f deployment.yaml          # apply a manifest
kubectl set image deployment/synapseiq-api api=gcr.io/.../api:v2.2.0  # update image
kubectl rollout status deployment/synapseiq-api  # watch rollout progress
kubectl rollout undo deployment/synapseiq-api    # rollback to previous version

# ── Scale ─────────────────────────────────────────────────────────────
kubectl scale deployment synapseiq-api --replicas=5   # manual scale
```

### 7.7.4 GitHub Actions → K8s CI/CD

```yaml
# .github/workflows/deploy-k8s.yml

name: Deploy to K8s

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Authenticate to GCP
        uses: google-github-actions/auth@v2
        with:
          credentials_json: ${{ secrets.GCP_CREDENTIALS }}

      - name: Build and push image
        run: |
          gcloud builds submit \
            --tag us-central1-docker.pkg.dev/$PROJECT_ID/apps/synapseiq:$GITHUB_SHA .

      - name: Run eval gate before deploy
        run: |
          python scripts/eval_before_deploy.py
          # Fails (exit 1) if RAGAS score < threshold → blocks deploy

      - name: Update K8s deployment
        run: |
          gcloud container clusters get-credentials prod-cluster --region us-central1
          kubectl set image deployment/synapseiq-api \
            api=us-central1-docker.pkg.dev/$PROJECT_ID/apps/synapseiq:$GITHUB_SHA \
            -n production
          kubectl rollout status deployment/synapseiq-api -n production
          # Waits for rollout to complete, fails if pods crash

      - name: Verify deployment
        run: |
          kubectl get pods -n production
          # Smoke test
          ENDPOINT=$(kubectl get service synapseiq-api -n production -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
          curl -f http://$ENDPOINT/health
```

---

## 8. Platform Decision Framework

### When to Use What

```
DECISION: "Where should I deploy my AI service?"

START HERE:
  ├── Already in AWS ecosystem (RDS, S3, VPC)?
  │     → ECS Fargate (always-on API) or Lambda (intermittent)
  │
  ├── Already in GCP ecosystem (BigQuery, Cloud SQL, GKE)?
  │     → Cloud Run (always-on API) or Cloud Run Jobs (batch)
  │
  ├── Need GPU for inference (hosting your own model)?
  │     → K8s + GPU node pool (EKS/GKE) or SageMaker endpoint
  │
  ├── Intermittent workload, pay-per-request, short duration (< 15 min)?
  │     → Lambda (AWS) or Cloud Run (GCP, up to 60 min)
  │
  ├── Enterprise, compliance, everything must stay in one cloud?
  │     → AWS: Bedrock + Lambda + ECS + RDS + ElastiCache
  │     → GCP: Vertex AI + Cloud Run + Cloud SQL + Memorystore
  │
  ├── Startup, moving fast, want managed everything?
  │     → Cloud Run + Vertex AI Gemini + Cloud SQL + Firestore
  │     (GCP Cloud Run has the simplest deploy DX)
  │
  └── Complex workloads, need full control, team has K8s experience?
        → EKS or GKE (managed Kubernetes)

DECISION: "Which LLM API should I use in production?"

  Cost matters most?          → Groq (cheapest) or Bedrock Nova
  Compliance/enterprise?      → Bedrock (data stays in AWS)
  GCP stack?                  → Vertex AI Gemini
  Best quality for the buck?  → Claude Haiku 4 on Bedrock
  Open source?                → Llama on SageMaker or K8s

DECISION: "Bedrock Agents vs LangGraph?"

  AWS-native, compliance, minimal code → Bedrock Agents
  Complex routing, HITL, custom state  → LangGraph
  Multi-agent with GCP Gemini          → ADK
```

---

## 🏗️ Phase 7 Projects

### Project A — RAG Chatbot on Bedrock (AWS)
```
What:  Same RAG system from SynapseIQ, deployed on Bedrock infrastructure
Stack: Bedrock Knowledge Bases + Bedrock Agents + Bedrock Guardrails
       Lambda (Mangum) + API Gateway + S3

Build:
  1. Upload docs to S3
  2. Create Bedrock Knowledge Base (S3 → auto-chunked → embedded → stored)
  3. Create Bedrock Agent with action group (Lambda for order lookup)
  4. Add Bedrock Guardrail (PII masking, topic blocking)
  5. Wrap FastAPI with Mangum → deploy on Lambda
  6. API Gateway routes → Lambda → Bedrock Agent

Showable: chat with a document, PII masked, tool call logs visible
```

### Project B — Serverless LLM API (Lambda + Mangum)
```
What:  FastAPI AI app deployed entirely serverless on Lambda
Stack: FastAPI + Mangum + Lambda + API Gateway + Bedrock
       EventBridge for daily batch jobs

Build:
  1. Existing FastAPI app (from Phase 3) + Mangum wrapper
  2. Lambda layers for heavy dependencies
  3. API Gateway HTTP API
  4. Lambda streaming for SSE tokens
  5. EventBridge cron: daily re-index job → Lambda

Showable: invoke via curl, stream tokens, cost = $0 at low volume
```

### Project C — Research Agent with ADK (GCP)
```
What:  Multi-agent research system using Google ADK + Gemini
Stack: ADK + Gemini 2.5 Pro + Google Search + MCP Postgres
       Cloud Run + Pub/Sub

Build:
  1. ADK agent with Google Search + custom tools
  2. MCP Postgres tool (connect to local PG via MCP)
  3. Multi-agent: researcher → analyst → writer (sequential)
  4. Deploy to Cloud Run (gcloud run deploy, one command)
  5. Pub/Sub queue for async research jobs

Showable: research a topic, agent uses Google Search + DB, generates report
```

### Project D — LLM Service on K8s
```
What:  vLLM + FastAPI AI service deployed on local K8s (minikube/kind)
Stack: vLLM (Ollama as substitute for local), K8s YAML, HPA, CronJob

Build:
  1. Ollama as local inference server (simulates vLLM)
  2. FastAPI service that calls Ollama
  3. Deployment.yaml + Service.yaml + HPA.yaml
  4. CronJob.yaml for scheduled eval run
  5. GitHub Actions: build → kubectl apply

Showable: kubectl get pods, curl the service, watch HPA scale
```

---

## 📋 Interview Cheat Sheet

**Q: When would you use Bedrock over calling OpenAI/Anthropic directly?**
```
3 reasons:
1. Compliance: data stays in AWS (HIPAA, GDPR, data residency)
2. Unified billing: one AWS account for all models (Claude, Llama, Titan)
3. Managed services: Bedrock Knowledge Bases (managed RAG), Bedrock Agents
   (managed agent framework), Bedrock Guardrails (managed safety)

Direct API:
  Simpler setup, more provider choice, lower overhead for small projects
  Better when: not AWS-native, cost-sensitive (Bedrock adds ~10-15% overhead),
               need full control over RAG/agent implementation

Choose Bedrock: enterprise, compliance, AWS-native, want managed infra
Choose direct:  startups, GCP, multi-cloud, full control needed
```

**Q: Lambda vs ECS/Fargate — when do you use each?**
```
Lambda:
  - Intermittent workloads (not always receiving requests)
  - Short-lived (< 15 min)
  - Event-driven (SQS trigger, S3 event, EventBridge cron)
  - Scale to zero (no requests = no cost)
  - Cold start acceptable (< 2s for non-latency-critical)
  Example: async resume parser, daily RAG re-indexer, batch scorer

ECS/Fargate:
  - Always-on APIs (user-facing chat, real-time inference)
  - Long-running processes (WebSocket connections, streaming)
  - Need consistent low latency (no cold start tolerance)
  - Heavy containers (LangGraph + many dependencies)
  Example: production RAG API, voice agent, LangGraph service

Rule: user-facing API with consistent traffic → ECS; event-driven background job → Lambda
```

**Q: How does Bedrock Knowledge Bases compare to building your own RAG?**
```
Bedrock Knowledge Bases (managed):
  AWS handles: chunking, embedding (Titan), vector store (OpenSearch)
  You handle: S3 bucket with documents
  Pro: zero infra, one API call (RetrieveAndGenerate)
  Con: less control, more expensive, locked to AWS, limited chunking options

Custom RAG (our Phase 2-5):
  You handle: PDFMiner → text splitter → embeddings → Qdrant → hybrid search → reranking
  Pro: full control, cheaper at scale, custom chunking, hybrid search + reranking
  Con: more code, more infra, more maintenance

Choose Bedrock KB: enterprise compliance, already in AWS, simple RAG, quick to ship
Choose custom:     cost-sensitive, complex retrieval, multi-cloud, custom chunking
```

**Q: What is Google ADK and why would you use it over LangGraph?**
```
ADK = Agent Development Kit. Google's Python SDK for Gemini-native agents.
Released April 2025.

Use ADK when:
  - GCP stack (Vertex AI Gemini is the LLM)
  - Need Google Search as a native tool (built-in, no API key needed)
  - Want managed deployment (Vertex AI Agent Engine)
  - Simple agent patterns (sequential, parallel, hierarchical)
  - Quick iteration — less boilerplate than LangGraph

Use LangGraph when:
  - Need precise HITL (interrupt_before/after)
  - Complex custom state management (TypedDict with reducers)
  - Non-GCP infrastructure (AWS Bedrock, Groq, Anthropic)
  - Need Redis/Postgres checkpointing for long-running agents
  - Very complex routing logic

ADK is simpler and more opinionated.
LangGraph is more powerful and more flexible.
```

**Q: What K8s concepts matter most for AI engineers?**
```
5 concepts that actually come up:

1. GPU scheduling: nvidia.com/gpu: 1 resource request
   + node selectors to target GPU node pool
   (You'll need this for vLLM, fine-tuning workloads)

2. Resource limits: always set requests + limits for AI pods
   They're memory-hungry. Without limits → OOM kill → pod crash

3. PersistentVolumes: cache model weights
   Without PV: 7B model re-downloads on every pod restart (8 minutes)
   
4. HPA: auto-scale on CPU/custom metrics
   LLM workloads are spiky — HPA handles demand spikes
   
5. CronJob: scheduled eval runs, daily re-ingestion
   Same as EventBridge but in K8s-native way

You don't need to know: etcd internals, controller manager, scheduler algorithm
```

---

## 🃏 Quick Revision Cards

```
CARD 1: Bedrock Models
  Claude:  anthropic.claude-haiku-4-v1 / sonnet-4-v1 / opus-4-v1
  Llama:   meta.llama3-3-70b-instruct-v1 / llama3-1-8b-v1
  Amazon:  amazon.nova-lite-v1 / nova-pro-v1
  Invoke:  bedrock.invoke_model(modelId, body)
  Stream:  bedrock.invoke_model_with_response_stream(modelId, body)

CARD 2: Bedrock Managed Services
  Knowledge Bases: managed RAG (S3 → auto-embed → query)
  Agents:          managed agent (action groups → Lambda → tools)
  Guardrails:      content filter + PII redact + topic block
  Model Eval:      automated eval jobs with custom metrics
  All accessed via: boto3 bedrock-agent-runtime client

CARD 3: Lambda for AI
  Mangum:       FastAPI → Lambda (zero code change)
  Streaming:    @streaming_response decorator
  Cold start:   provisioned concurrency (keep N warm)
  Timeout:      max 15 min (set 60-120s for LLM calls)
  Trigger:      API Gateway | SQS | EventBridge | S3 event
  IAM:          never use access keys in code — use execution role

CARD 4: SageMaker
  JumpStart:   deploy Llama/Mistral/Falcon with one click
  Endpoints:   real-time (sync) vs async (S3 in/out)
  Pipelines:   preprocess → train → eval → condition → register
  Eval gate:   ConditionGreaterThan(F1, threshold) → only deploy if better
  HuggingFace: native integration (from_huggingface_hub)

CARD 5: Google ADK
  LlmAgent:      single agent with tools + instructions
  SequentialAgent: A → B → C (output flows between)
  ParallelAgent:  A + B simultaneously (fastest = total time)
  Built-in tools: google_search, built_in_code_execution
  MCP support:    MCPToolset(StdioServerParameters(...))
  Sessions:       InMemorySessionService or Cloud Redis
  Deploy:         Cloud Run (gcloud run deploy)

CARD 6: Cloud Run vs Lambda
  Cloud Run:   container, up to 60 min, SSE native, 32GB RAM, GCP
  Lambda:      function/container, up to 15 min, streaming mode, 10GB, AWS
  Both:        scale to zero, pay per request, serverless
  Cloud Run:   better for AI APIs (longer timeout, streaming DX)
  Lambda:      better for event-driven jobs (SQS, S3, EventBridge)

CARD 7: K8s AI Essentials
  GPU:     nvidia.com/gpu: 1 + nodeSelector for GPU pool
  Limits:  always set requests+limits (AI pods are memory-hungry)
  PV:      cache model weights (avoid re-download on restart)
  HPA:     auto-scale on CPU/custom metric (70% target)
  CronJob: scheduled tasks (daily eval, re-ingestion)
  kubectl: apply/get/describe/logs/exec/port-forward/rollout

CARD 8: AWS vs GCP AI Stack
  AWS: Bedrock + Lambda + ECS + RDS + ElastiCache + SQS + S3
  GCP: Vertex AI + Cloud Run + Cloud SQL + Memorystore + Pub/Sub + GCS
  K8s: EKS (AWS) or GKE (GCP) — same kubectl, different managed control plane
  IAM: AWS IAM roles → GCP Service Accounts + Workload Identity

CARD 9: IAM / Security Best Practices
  Never hardcode API keys in code
  Use IAM execution roles (Lambda/ECS/Cloud Run picks up creds automatically)
  Least privilege: only grant exact actions on exact resources
  Secrets: AWS Secrets Manager / GCP Secret Manager (not env vars)
  KMS: encrypt S3 objects with KMS key (ServerSideEncryption: aws:kms)

CARD 10: Platform Decision
  Always-on API, AWS:                ECS Fargate
  Event-driven job, AWS:             Lambda
  Always-on API, GCP:                Cloud Run
  Async job, GCP:                    Cloud Run Jobs + Pub/Sub
  Host fine-tuned model:             SageMaker endpoint / K8s GPU pod
  Managed RAG, AWS:                  Bedrock Knowledge Bases
  Managed agent, AWS:                Bedrock Agents
  Gemini agent, GCP:                 ADK + Cloud Run
  Full control, complex:             K8s (EKS or GKE)
```

---

## ✅ Phase 7 Completion Checklist

```
AWS BEDROCK
[ ] Called InvokeModel and InvokeModelWithResponseStream with boto3
[ ] Streamed Bedrock tokens to FastAPI → SSE → browser
[ ] Created Bedrock Knowledge Base (S3 → auto-embed → query)
[ ] Used RetrieveAndGenerate API (one-call RAG)
[ ] Created Bedrock Guardrail (PII masking + topic blocking)
[ ] Invoked Bedrock Agent with session memory
[ ] IAM role attached to Lambda/ECS (no keys in code)
[ ] Region selection: ap-south-1 for India data residency

AWS LAMBDA
[ ] FastAPI app running on Lambda with Mangum
[ ] Lambda streaming response working (SSE)
[ ] Cold start measured and mitigated (layers + provisioned)
[ ] EventBridge cron → Lambda (scheduled job)
[ ] SQS → Lambda batch processor (async AI job queue)
[ ] Deployment zip with Lambda layers

AWS SAGEMAKER
[ ] SageMaker JumpStart endpoint deployed (Llama or Mistral)
[ ] Endpoint invoked via boto3 sagemaker-runtime
[ ] Async inference endpoint configured
[ ] SageMaker Pipeline built (3+ steps)
[ ] Conditional step with F1 threshold gate

GCP VERTEX AI + ADK
[ ] Vertex AI Gemini API called (basic + streaming)
[ ] Chat session (multi-turn) working
[ ] ADK LlmAgent built with 2+ tools
[ ] ADK SequentialAgent built (3 sub-agents)
[ ] MCP tool connected to ADK agent
[ ] ADK agent deployed to Cloud Run
[ ] ADK evaluation run (trajectory + response quality)

GCP CLOUD RUN
[ ] FastAPI app deployed to Cloud Run (gcloud run deploy)
[ ] Streaming SSE from Cloud Run working
[ ] Pub/Sub push → Cloud Run worker
[ ] Cloud Run + Vertex AI (no API keys, service account auth)
[ ] Min-instances=1 configured (avoid cold start)

K8S
[ ] Deployment.yaml + Service.yaml written and applied
[ ] ConfigMap + Secret (external secrets) configured
[ ] HPA configured (CPU-based autoscaling)
[ ] GPU pod spec written (nvidia.com/gpu resource request)
[ ] CronJob for scheduled task
[ ] PersistentVolume for model weight caching
[ ] GitHub Actions → kubectl apply CI/CD pipeline
[ ] 5+ kubectl commands used in debugging

PROJECTS
[ ] Project A: Bedrock RAG chatbot (KB + Agent + Guardrail + Lambda)
[ ] Project B: Serverless LLM API (Mangum + Lambda + streaming)
[ ] Project C: ADK research agent (ADK + Gemini + Search + Cloud Run)
[ ] Project D: K8s LLM service (local K8s + vLLM/Ollama + HPA)
```

---

*Phase 7 — Cloud AI Platforms | GenAI + LLMOps Engineering Roadmap 2026*
*Next: Phase 8 — LLMOps (Evaluation · Observability · Reliability · Guardrails · CI/CD)*
