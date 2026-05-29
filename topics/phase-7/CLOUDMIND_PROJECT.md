# ☁️ CloudMind — Multi-Cloud AI Research Agent
> **Type:** Phase 7 Capstone Mini-Project
> **Purpose:** Touch every Phase 7 concept in one coherent, production-like system
> **Stack:** FastAPI · AWS Bedrock · Lambda · SQS · EventBridge · GCP Vertex AI · ADK · Cloud Run · K8s (local minikube) · Langfuse
> **Timeline:** 3–4 weeks (50–60 hrs)

---

## 📑 Table of Contents

1. [The Idea](#1-the-idea)
2. [What Phase 7 Concepts It Covers](#2-what-phase-7-concepts-it-covers)
3. [System Architecture](#3-system-architecture)
4. [How Each Component Fits Together](#4-how-each-component-fits-together)
5. [Phase-by-Phase Build Plan](#5-phase-by-phase-build-plan)
6. [Component Deep Dives](#6-component-deep-dives)
   - [6.1 FastAPI Core + Mangum (Lambda)](#61-fastapi-core--mangum-lambda)
   - [6.2 Bedrock — Direct LLM + Knowledge Base + Guardrails](#62-bedrock--direct-llm--knowledge-base--guardrails)
   - [6.3 SQS + Lambda — Async Research Jobs](#63-sqs--lambda--async-research-jobs)
   - [6.4 EventBridge — Scheduled Digest](#64-eventbridge--scheduled-digest)
   - [6.5 ADK Agent — GCP Side](#65-adk-agent--gcp-side)
   - [6.6 Cloud Run — GCP Deployment](#66-cloud-run--gcp-deployment)
   - [6.7 Kubernetes — Local Orchestration](#67-kubernetes--local-orchestration)
   - [6.8 Provider Router — Bedrock vs ADK](#68-provider-router--bedrock-vs-adk)
7. [Folder Structure](#7-folder-structure)
8. [Docker Compose — Local Dev](#8-docker-compose--local-dev)
9. [GitHub Actions CI/CD](#9-github-actions-cicd)
10. [Demo Script](#10-demo-script)
11. [Phase 7 Coverage Checklist](#11-phase-7-coverage-checklist)

---

## 1. The Idea

**CloudMind** is a multi-cloud AI research agent system. You give it a research topic. It:

1. Routes the request to the best available AI backend (Bedrock or Vertex AI / ADK)
2. Searches the web, queries a knowledge base, synthesises findings
3. Returns a streaming answer immediately (synchronous path)
4. Queues a deep-research job to SQS (asynchronous path) for a longer, structured report
5. Every night, an EventBridge cron triggers a "digest" Lambda that summarises all research done that day and emails it

The whole thing deploys in three ways:
- **AWS**: Lambda (Mangum) + SQS + Bedrock + EventBridge
- **GCP**: Cloud Run + ADK + Vertex AI Gemini
- **Local**: minikube K8s with Ollama as local inference

This is intentionally lean — one domain, one usecase — but every infrastructure piece is real.

```
Why "research agent" as the use case?

It naturally needs:
  ✅ Streaming (user wants tokens as they arrive)
  ✅ Async jobs (deep research takes 30-60 seconds → can't block HTTP)
  ✅ Knowledge base (Bedrock KB for indexed docs)
  ✅ Guardrails (block harmful queries)
  ✅ Scheduled job (daily digest makes sense)
  ✅ Multi-cloud routing (Bedrock for compliance, ADK for Gemini quality)
  ✅ Managed agent (Bedrock Agents OR ADK depending on cloud)
  ✅ K8s (local dev with GPU-aware manifests even if no real GPU)
```

---

## 2. What Phase 7 Concepts It Covers

```
CONCEPT                        WHERE IN CLOUDMIND
──────────────────────────────────────────────────────────────────────
Bedrock InvokeModel            FastAPI /chat → Bedrock sync call
Bedrock streaming              FastAPI /chat/stream → SSE to browser
Bedrock Knowledge Bases        /chat → KB retrieval (your indexed docs)
Bedrock Agents                 /research/deep → Bedrock Agent with tool
Bedrock Guardrails             Applied to every Bedrock call
IAM roles (no keys in code)    Lambda execution role, ECS task role
Lambda + Mangum                FastAPI wrapped in Mangum → Lambda
Lambda streaming               /chat/stream on Lambda
Lambda layers                  boto3 + pydantic in shared layer
EventBridge cron               Daily digest job at 8 AM
SQS + Lambda trigger           /research/queue → SQS → worker Lambda
Cold start mitigation          Provisioned concurrency on /chat endpoint
Vertex AI Gemini               GCP route → Gemini 2.0 Flash
Google ADK agent               ADK agent with Google Search + custom tool
ADK SequentialAgent            research → analyse → report pipeline
Cloud Run deploy               GCP version deployed to Cloud Run
Cloud Run streaming            SSE streaming on Cloud Run
K8s Deployment + Service       All services as K8s manifests
K8s ConfigMap + Secret         Config and keys injected correctly
K8s HPA                        Auto-scale API pods on CPU
K8s CronJob                    Daily digest as K8s CronJob
K8s PersistentVolume           Ollama model weights cached
Provider Router                Runtime switch: AWS vs GCP vs local
GitHub Actions CI/CD           Build → push → deploy (ECS or Cloud Run)
```

---

## 3. System Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         CloudMind System                                │
│                                                                         │
│   Browser / curl                                                        │
│       │                                                                 │
│  ┌────▼──────────────────────────────────────────────────────────────┐  │
│  │                  FastAPI App (main.py)                            │  │
│  │                                                                   │  │
│  │  POST /chat              → sync LLM call (stream)                 │  │
│  │  POST /chat/stream       → SSE stream of tokens                   │  │
│  │  POST /research/queue    → async deep research (SQS / Pub/Sub)   │  │
│  │  GET  /research/{id}     → poll job status + result               │  │
│  │  GET  /health            → health check (K8s probe)               │  │
│  │                                                                   │  │
│  │  ProviderRouter: AWS_MODE | GCP_MODE | LOCAL_MODE                 │  │
│  └────┬──────────────────────────────────────────────────────────────┘  │
│       │                                                                  │
│  ┌────▼───────────────┐    ┌──────────────────────────────────────────┐ │
│  │   AWS PATH         │    │   GCP PATH                               │ │
│  │                    │    │                                          │ │
│  │  Bedrock           │    │  Vertex AI Gemini 2.0 Flash              │ │
│  │  (Claude Haiku)    │    │  OR ADK Agent (research_pipeline)        │ │
│  │       ↓            │    │                                          │ │
│  │  Bedrock KB        │    │  ADK SequentialAgent:                    │ │
│  │  (knowledge base)  │    │    web_researcher → data_analyst         │ │
│  │       ↓            │    │    → report_writer                       │ │
│  │  Bedrock Guardrail │    │                                          │ │
│  │  (safety layer)    │    │  Deployed: Cloud Run                     │ │
│  │       ↓            │    └──────────────────────────────────────────┘ │
│  │  Bedrock Agent     │                                                  │
│  │  (deep research)   │    ┌──────────────────────────────────────────┐ │
│  └────────────────────┘    │   LOCAL PATH                             │ │
│                            │                                          │ │
│  ┌─────────────────────┐   │  Ollama (llama3.1:8b)                    │ │
│  │  ASYNC LAYER        │   │  Running in K8s pod                      │ │
│  │                     │   │  PersistentVolume for model cache         │ │
│  │  SQS (AWS)          │   └──────────────────────────────────────────┘ │
│  │  OR Pub/Sub (GCP)   │                                                  │
│  │       ↓             │   ┌──────────────────────────────────────────┐ │
│  │  Worker Lambda      │   │  SCHEDULED JOBS                          │ │
│  │  (deep research)    │   │                                          │ │
│  │       ↓             │   │  EventBridge cron (AWS) / K8s CronJob    │ │
│  │  Result → Redis     │   │  → digest_lambda (daily 8 AM)            │ │
│  │  (TTL 1 hour)       │   │  → summarise today's research            │ │
│  └─────────────────────┘   │  → write to S3 / GCS                    │ │
│                            └──────────────────────────────────────────┘ │
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │  OBSERVABILITY                                                  │    │
│  │  Langfuse: every LLM call traced (provider, tokens, cost, ms)  │    │
│  │  CloudWatch: Lambda errors + latency                           │    │
│  │  Structured logging (JSON): all FastAPI requests               │    │
│  └─────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────┘

DEPLOYMENT TARGETS:
  AWS:   Lambda (Mangum) + API Gateway
  GCP:   Cloud Run
  Local: minikube K8s (kubectl apply -f k8s/)
```

---

## 4. How Each Component Fits Together

```
REQUEST PATH (sync):
  User → POST /chat/stream?topic=quantum+computing
  ProviderRouter checks env: CLOUD_PROVIDER=aws|gcp|local
  
  AWS:   FastAPI → Bedrock KB (retrieve) → Bedrock InvokeModelWithResponseStream
         → SSE tokens back to user
  
  GCP:   FastAPI → Vertex AI Gemini 2.0 Flash (stream=True)
         → SSE tokens back to user
  
  Local: FastAPI → Ollama /api/generate (stream=True)
         → SSE tokens back to user

REQUEST PATH (async deep research):
  User → POST /research/queue {topic: "quantum computing"}
  FastAPI → SQS.send_message (AWS) or Publisher.publish (GCP)
  Returns: {job_id: "job_abc123", status: "queued"}
  
  SQS → triggers deep_research_worker Lambda
  Lambda runs: Bedrock Agent OR ADK SequentialAgent (5-30 seconds)
  Lambda writes: result to Redis (TTL 1hr) keyed by job_id
  
  User polls: GET /research/job_abc123
  FastAPI reads from Redis → returns result when ready

SCHEDULED PATH:
  AWS EventBridge: cron(0 8 * * ? *) → digest_lambda
  digest_lambda: reads all research from today → Bedrock summarise → S3
  
  K8s CronJob: same digest logic running locally on schedule
```

---

## 5. Phase-by-Phase Build Plan

### Phase 0 — Skeleton (2 days)
```
[ ] FastAPI app with all routes returning stubs
    POST /chat, POST /chat/stream, POST /research/queue, GET /research/{id}, GET /health
[ ] ProviderRouter class: reads CLOUD_PROVIDER env var, dispatches to right backend
[ ] BaseLLMBackend ABC: complete(messages) + stream(messages) interface
[ ] Redis client setup (job result store)
[ ] Langfuse client setup (traces all LLM calls)
[ ] .env.example with all required variables
[ ] docker-compose.yml: fastapi + redis + (optional) ollama

Showable: curl /health → OK. curl /chat → "stub response". All routes exist.
```

### Phase 1 — AWS Bedrock Backend (5 days)
```
[ ] BedrockBackend implementing BaseLLMBackend
    .complete()  → boto3 invoke_model
    .stream()    → boto3 invoke_model_with_response_stream
[ ] /chat/stream endpoint returning SSE (EventSourceResponse)
[ ] Bedrock Knowledge Base integration
    Create KB in AWS Console: upload 5-10 docs to S3
    .retrieve() → KB retrieval before generation
    .retrieve_and_generate() → one-call RAG
[ ] Bedrock Guardrails
    Create guardrail: PII masking + block harmful topics
    Apply guardrailIdentifier to every invoke_model call
[ ] Bedrock Agent for deep research
    Create agent in AWS Console: one action group → Lambda tool
    .research() → invoke_agent with session_id
[ ] Test all three: direct model, KB RAG, Agent
[ ] IAM: verify no hardcoded keys (use local ~/.aws/credentials for dev)

Showable:
  curl /chat?topic=LLMs → streams Claude Haiku tokens with KB context
  Guardrail blocks: curl /chat?topic="how to make a bomb" → blocked response
  Bedrock Agent: curl /research/deep → agent uses tool, returns structured report
```

### Phase 2 — SQS Async Queue + Worker Lambda (3 days)
```
[ ] /research/queue endpoint: validate request → SQS.send_message → return job_id
[ ] SQS queue created (CloudFormation or console): research-jobs-queue
[ ] deep_research_worker Lambda function:
    triggered by SQS messages
    runs Bedrock Agent (deep research, ~20 seconds)
    writes result to Redis: redis.setex(job_id, 3600, result)
[ ] GET /research/{job_id}: reads from Redis, returns result or "pending"
[ ] Lambda packaging: zip + boto3 layer
[ ] DLQ (Dead Letter Queue): failed messages after 3 retries → dead-letter queue

Showable:
  POST /research/queue {topic: "..."} → {job_id: "abc", status: "queued"}
  Wait 20 seconds
  GET /research/abc → {status: "complete", report: "..."}
  DLQ: force a failure → message appears in dead-letter queue
```

### Phase 3 — Mangum + Lambda Deployment (3 days)
```
[ ] Add Mangum to FastAPI app: handler = Mangum(app, lifespan="off")
[ ] Lambda layer: boto3 + pydantic + fastapi in shared layer
[ ] Deployment script: pip install -t package/ + zip
[ ] API Gateway HTTP API: route all paths to Lambda
[ ] Lambda streaming: enable RESPONSE_STREAM mode
[ ] Cold start test: measure first request vs warm request latency
[ ] Provisioned concurrency on /chat endpoint (keep 1 warm)

Showable:
  curl https://xxxx.execute-api.ap-south-1.amazonaws.com/chat/stream → SSE tokens
  CloudWatch: see Lambda invocation logs
  Latency: warm = 400ms, cold = 3200ms, provisioned = 420ms
```

### Phase 4 — EventBridge Scheduled Digest (2 days)
```
[ ] digest_lambda: separate Lambda function
    reads all Redis keys matching "research:*" from today
    sends all topics to Bedrock for summarisation
    writes digest to S3: digests/YYYY-MM-DD.json
[ ] EventBridge rule: cron(0 8 * * ? *) → digest_lambda
[ ] Test: invoke digest_lambda manually → see S3 file created
[ ] CloudWatch alarm: alert if digest_lambda fails

Showable:
  Manually invoke digest_lambda → see S3 file created
  EventBridge rule visible in AWS Console
  CloudWatch alarm configured (won't fire unless Lambda fails)
```

### Phase 5 — GCP: ADK Agent + Cloud Run (5 days)
```
[ ] GCPBackend implementing BaseLLMBackend
    .complete()  → Vertex AI GenerativeModel.generate_content()
    .stream()    → model.generate_content(stream=True)
[ ] ADK research agent:
    web_researcher sub-agent (google_search tool)
    data_analyst sub-agent (built_in_code_execution)
    report_writer sub-agent
    SequentialAgent: researcher → analyst → writer
[ ] Wire ADK agent into /research/queue on GCP path
    GCP: POST /research/queue → Pub/Sub → Cloud Run worker → ADK agent
[ ] ADK evaluation: test agent on 5 sample research queries
[ ] Cloud Run deployment:
    gcloud builds submit → Artifact Registry
    gcloud run deploy with streaming, min-instances=1
[ ] Verify streaming SSE works from Cloud Run

Showable:
  CLOUD_PROVIDER=gcp curl /chat/stream → Gemini tokens via Cloud Run
  ADK SequentialAgent: watch web_researcher → analyst → writer chain
  ADK eval: tool_use_accuracy and response_quality scores printed
  gcloud run services list → service URL + traffic stats
```

### Phase 6 — Kubernetes (Local minikube) (4 days)
```
[ ] minikube start (or kind create cluster)
[ ] All K8s manifests in k8s/ directory:
    deployment.yaml  (FastAPI app)
    service.yaml     (ClusterIP + LoadBalancer)
    configmap.yaml   (non-secret config)
    secret.yaml      (API keys base64-encoded)
    hpa.yaml         (CPU-based autoscaling)
    ollama-deploy.yaml  (Ollama for local inference)
    ollama-pvc.yaml  (20Gi PersistentVolumeClaim for model cache)
    ollama-init.yaml (InitContainer: pull llama3.1:8b before start)
    cronjob.yaml     (daily digest at 8 AM)
[ ] OllamaBackend: BaseLLMBackend using httpx → Ollama API
[ ] Local mode: CLOUD_PROVIDER=local → routes to Ollama pod
[ ] Apply all manifests: kubectl apply -f k8s/
[ ] Test HPA: stress-test with hey or k6 → watch replicas scale

Showable:
  kubectl get pods → all pods Running
  kubectl port-forward svc/cloudmind 8000:80 →
    curl localhost:8000/chat → Ollama responds
  kubectl get hpa → watch REPLICAS increase under load
  kubectl get pvc → model-weights-cache BOUND
  kubectl logs -l app=cloudmind → structured JSON logs
```

### Phase 7 — CI/CD + Observability (3 days)
```
[ ] GitHub Actions workflow:
    test job: pytest tests/ (unit tests for all 3 backends)
    build job: docker build + push (ECR or Artifact Registry)
    deploy-aws job: aws lambda update-function-code
    deploy-gcp job: gcloud run deploy
    deploy-k8s job: kubectl set image (on minikube via kubeconfig secret)
[ ] Langfuse: add @observe decorator to all LLM calls
    Trace: provider, model, tokens, cost, latency per call
[ ] CloudWatch dashboard:
    Lambda invocations/errors/duration
    SQS queue depth
    Custom metric: research jobs completed per hour
[ ] README with architecture diagram + demo instructions

Showable:
  Push to main → GitHub Actions runs → all 3 deploy jobs succeed
  Langfuse dashboard: see traces for AWS and GCP calls side-by-side
  CloudWatch: Lambda metrics chart
```

---

## 6. Component Deep Dives

### 6.1 FastAPI Core + Mangum (Lambda)

```python
# app/main.py

import os, json, uuid, time
from fastapi import FastAPI, Query, BackgroundTasks
from fastapi.responses import StreamingResponse
from mangum import Mangum
from pydantic import BaseModel

from app.router import ProviderRouter
from app.store import ResultStore
from app.observability import tracer

app = FastAPI(title="CloudMind Research Agent", version="1.0.0")
router_obj = ProviderRouter()
store      = ResultStore()   # Redis-backed

# ── /health ────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {
        "status":   "ok",
        "provider": os.getenv("CLOUD_PROVIDER", "aws"),
        "version":  "1.0.0",
    }

# ── /chat — sync with optional streaming ──────────────────────────────
class ChatRequest(BaseModel):
    topic:    str
    mode:     str = "concise"    # concise | detailed
    stream:   bool = False

@app.post("/chat")
async def chat(body: ChatRequest):
    """Sync chat — returns full response at once"""
    with tracer.trace("chat", provider=os.getenv("CLOUD_PROVIDER")):
        messages = [
            {"role": "system", "content": "You are a concise research assistant."},
            {"role": "user",   "content": f"Research this topic briefly: {body.topic}"},
        ]
        answer = await router_obj.complete(messages)
    return {"answer": answer, "topic": body.topic}

@app.post("/chat/stream")
async def chat_stream(body: ChatRequest):
    """SSE streaming — tokens arrive one by one"""
    messages = [
        {"role": "system", "content": "You are a research assistant."},
        {"role": "user",   "content": f"Research: {body.topic}"},
    ]

    async def generate():
        async for token in router_obj.stream(messages):
            yield f"data: {json.dumps({'token': token})}\n\n"
        yield f"data: {json.dumps({'done': True})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )

# ── /research/queue — async deep research ─────────────────────────────
class ResearchRequest(BaseModel):
    topic:    str
    depth:    str = "standard"   # standard | deep
    format:   str = "report"     # report | bullets | summary

@app.post("/research/queue")
async def queue_research(body: ResearchRequest):
    """Queue a deep research job — returns job_id immediately"""
    job_id = f"job_{uuid.uuid4().hex[:8]}"

    # Mark as queued in Redis
    await store.set_status(job_id, "queued")

    # Send to queue (SQS on AWS, Pub/Sub on GCP)
    await router_obj.enqueue_research_job({
        "job_id":  job_id,
        "topic":   body.topic,
        "depth":   body.depth,
        "format":  body.format,
    })

    return {
        "job_id":    job_id,
        "status":    "queued",
        "poll_url":  f"/research/{job_id}",
        "message":   "Deep research queued. Poll the URL every 5s for results.",
    }

@app.get("/research/{job_id}")
async def get_research_result(job_id: str):
    """Poll for async research result"""
    status = await store.get_status(job_id)
    if not status:
        return {"status": "not_found", "job_id": job_id}

    if status == "queued":
        return {"status": "queued", "job_id": job_id, "message": "Still processing..."}

    if status == "failed":
        return {"status": "failed", "job_id": job_id}

    result = await store.get_result(job_id)
    return {"status": "complete", "job_id": job_id, "report": result}

# ── Mangum — wraps entire app as Lambda handler ───────────────────────
# Lambda calls this when triggered via API Gateway or Function URL
handler = Mangum(app, lifespan="off")
```

### 6.2 Bedrock — Direct LLM + Knowledge Base + Guardrails

```python
# app/backends/bedrock_backend.py

import boto3, json, os
from app.backends.base import BaseLLMBackend
from langfuse.decorators import observe

GUARDRAIL_ID      = os.getenv("BEDROCK_GUARDRAIL_ID", "")
GUARDRAIL_VERSION = os.getenv("BEDROCK_GUARDRAIL_VERSION", "DRAFT")
KB_ID             = os.getenv("BEDROCK_KB_ID", "")
AGENT_ID          = os.getenv("BEDROCK_AGENT_ID", "")
AGENT_ALIAS_ID    = os.getenv("BEDROCK_AGENT_ALIAS_ID", "TSTALIASID")
MODEL_ID          = os.getenv("BEDROCK_MODEL_ID", "anthropic.claude-haiku-4-20250514-v1:0")

class BedrockBackend(BaseLLMBackend):
    def __init__(self):
        self._rt    = boto3.client("bedrock-runtime",       region_name="ap-south-1")
        self._agent = boto3.client("bedrock-agent-runtime", region_name="ap-south-1")

    def _build_body(self, messages: list[dict], max_tokens: int = 800) -> str:
        return json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens":        max_tokens,
            "messages":          messages,
        })

    @observe(name="bedrock_complete")
    async def complete(self, messages: list[dict]) -> str:
        """Bedrock sync call with guardrail applied"""
        kwargs = dict(
            modelId=MODEL_ID,
            contentType="application/json",
            accept="application/json",
            body=self._build_body(messages),
        )
        if GUARDRAIL_ID:
            kwargs["guardrailIdentifier"] = GUARDRAIL_ID
            kwargs["guardrailVersion"]    = GUARDRAIL_VERSION
            kwargs["trace"]               = "ENABLED"

        response = self._rt.invoke_model(**kwargs)
        result   = json.loads(response["body"].read())

        # Check if guardrail blocked the response
        if result.get("stop_reason") == "guardrail_intervened":
            return "I can't help with that request."

        return result["content"][0]["text"]

    @observe(name="bedrock_stream")
    async def stream(self, messages: list[dict]):
        """Bedrock streaming — yields tokens one by one"""
        kwargs = dict(
            modelId=MODEL_ID,
            contentType="application/json",
            accept="application/json",
            body=self._build_body(messages),
        )
        if GUARDRAIL_ID:
            kwargs["guardrailIdentifier"] = GUARDRAIL_ID
            kwargs["guardrailVersion"]    = GUARDRAIL_VERSION

        response = self._rt.invoke_model_with_response_stream(**kwargs)
        for event in response["body"]:
            chunk = json.loads(event["chunk"]["bytes"])
            if chunk["type"] == "content_block_delta":
                token = chunk["delta"].get("text", "")
                if token:
                    yield token

    @observe(name="bedrock_kb_retrieve")
    async def retrieve_and_generate(self, query: str) -> str:
        """One-call RAG using Bedrock Knowledge Base"""
        if not KB_ID:
            return await self.complete([{"role": "user", "content": query}])

        response = self._agent.retrieve_and_generate(
            input={"text": query},
            retrieveAndGenerateConfiguration={
                "type": "KNOWLEDGE_BASE",
                "knowledgeBaseConfiguration": {
                    "knowledgeBaseId": KB_ID,
                    "modelArn": f"arn:aws:bedrock:ap-south-1::foundation-model/{MODEL_ID}",
                    "retrievalConfiguration": {
                        "vectorSearchConfiguration": {
                            "numberOfResults":    5,
                            "overrideSearchType": "HYBRID",
                        }
                    },
                    "generationConfiguration": {
                        "promptTemplate": {
                            "textPromptTemplate":
                                "Answer using ONLY this context. "
                                "If not in context: say so.\n\n$search_results$\n\nQ: $query$"
                        }
                    },
                }
            },
        )
        return response["output"]["text"]

    @observe(name="bedrock_agent")
    async def research_with_agent(self, topic: str, session_id: str) -> str:
        """Deep research using Bedrock Agent (manages tool use internally)"""
        if not AGENT_ID:
            return await self.complete([
                {"role": "user", "content": f"Write a detailed research report on: {topic}"}
            ])

        response  = self._agent.invoke_agent(
            agentId=AGENT_ID,
            agentAliasId=AGENT_ALIAS_ID,
            sessionId=session_id,
            inputText=f"Research this topic thoroughly and write a structured report: {topic}",
            enableTrace=True,
        )
        full_text = ""
        for event in response["completion"]:
            if "chunk" in event:
                full_text += event["chunk"]["bytes"].decode("utf-8")
        return full_text
```

### 6.3 SQS + Lambda — Async Research Jobs

```python
# app/queue/sqs_queue.py

import boto3, json, os

SQS_QUEUE_URL = os.getenv("SQS_QUEUE_URL", "")

class SQSQueue:
    def __init__(self):
        self._sqs = boto3.client("sqs", region_name="ap-south-1")

    async def enqueue(self, job: dict) -> str:
        """Send a research job to SQS"""
        response = self._sqs.send_message(
            QueueUrl=SQS_QUEUE_URL,
            MessageBody=json.dumps(job),
            MessageAttributes={
                "job_type": {
                    "StringValue": "deep_research",
                    "DataType":    "String",
                }
            },
        )
        return response["MessageId"]

# ── Worker Lambda (separate file: workers/deep_research_worker.py) ────
# This file is packaged separately and deployed as its own Lambda.
# Triggered by SQS.

import boto3, json, os, redis

r = redis.Redis(
    host=os.getenv("REDIS_HOST"),
    port=int(os.getenv("REDIS_PORT", "6379")),
    decode_responses=True,
)

bedrock_agent = boto3.client("bedrock-agent-runtime", region_name="ap-south-1")

def lambda_handler(event, context):
    """SQS batch trigger — process research jobs"""
    processed, failed = [], []

    for record in event["Records"]:
        body   = json.loads(record["body"])
        job_id = body["job_id"]
        topic  = body["topic"]

        try:
            # Mark as in-progress
            r.setex(f"status:{job_id}", 3600, "processing")

            # Run Bedrock Agent (or fall back to direct model call)
            agent_id       = os.getenv("BEDROCK_AGENT_ID", "")
            agent_alias_id = os.getenv("BEDROCK_AGENT_ALIAS_ID", "TSTALIASID")

            if agent_id:
                response  = bedrock_agent.invoke_agent(
                    agentId=agent_id,
                    agentAliasId=agent_alias_id,
                    sessionId=job_id,
                    inputText=f"Write a structured research report on: {topic}",
                )
                result = ""
                for evt in response["completion"]:
                    if "chunk" in evt:
                        result += evt["chunk"]["bytes"].decode("utf-8")
            else:
                # Fallback: direct model call
                bedrock_rt = boto3.client("bedrock-runtime", region_name="ap-south-1")
                resp       = bedrock_rt.invoke_model(
                    modelId="anthropic.claude-haiku-4-20250514-v1:0",
                    contentType="application/json",
                    accept="application/json",
                    body=json.dumps({
                        "anthropic_version": "bedrock-2023-05-31",
                        "max_tokens": 2000,
                        "messages": [{"role": "user",
                                      "content": f"Write a structured research report on: {topic}"}],
                    }),
                )
                result = json.loads(resp["body"].read())["content"][0]["text"]

            # Write result to Redis (TTL 1 hour)
            r.setex(f"result:{job_id}", 3600, result)
            r.setex(f"status:{job_id}", 3600, "complete")
            processed.append(job_id)

        except Exception as e:
            r.setex(f"status:{job_id}", 3600, "failed")
            print(f"ERROR processing job {job_id}: {e}")
            # Return to SQS for retry (batchItemFailures)
            failed.append({"itemIdentifier": record["messageId"]})

    print(f"Processed: {len(processed)}, Failed: {len(failed)}")
    return {"batchItemFailures": failed}
```

### 6.4 EventBridge — Scheduled Digest

```python
# workers/digest_lambda.py
# Deployed as separate Lambda.
# Triggered by EventBridge rule: cron(0 8 * * ? *)

import boto3, json, os, redis
from datetime import datetime, timedelta

r    = redis.Redis(host=os.getenv("REDIS_HOST"), decode_responses=True)
s3   = boto3.client("s3")
brt  = boto3.client("bedrock-runtime", region_name="ap-south-1")

S3_BUCKET = os.getenv("S3_DIGEST_BUCKET", "cloudmind-digests")

def lambda_handler(event, context):
    """
    Runs daily at 8 AM.
    1. Collects all research results from yesterday stored in Redis
    2. Asks Bedrock to summarise them into a daily digest
    3. Writes digest to S3
    """
    # Handle warm-up ping
    if event.get("is_warmup"):
        return {"status": "warm"}

    today    = datetime.utcnow().strftime("%Y-%m-%d")
    yesterday= (datetime.utcnow() - timedelta(days=1)).strftime("%Y-%m-%d")

    # Collect all research results from Redis
    # Keys: result:{job_id} — we stored job_ids with date prefix in prod
    # For simplicity: scan all result: keys
    all_results = []
    for key in r.scan_iter("result:*"):
        result = r.get(key)
        if result:
            job_id = key.replace("result:", "")
            all_results.append({"job_id": job_id, "content": result[:500]})

    if not all_results:
        print("No research results found. Skipping digest.")
        return {"status": "skipped", "reason": "no_results"}

    # Summarise with Bedrock
    combined = "\n\n---\n\n".join(
        f"Research {i+1}: {r['content']}" for i, r in enumerate(all_results[:20])
    )
    digest_response = brt.invoke_model(
        modelId="anthropic.claude-haiku-4-20250514-v1:0",
        contentType="application/json",
        accept="application/json",
        body=json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 1000,
            "messages": [{
                "role": "user",
                "content": f"""Summarise the following research results into a 
                              concise daily digest with key themes and insights:
                              
                              {combined}""",
            }],
        }),
    )
    digest = json.loads(digest_response["body"].read())["content"][0]["text"]

    # Write to S3
    s3_key  = f"digests/{today}/research_digest.json"
    s3.put_object(
        Bucket=S3_BUCKET,
        Key=s3_key,
        Body=json.dumps({
            "date":         today,
            "job_count":    len(all_results),
            "digest":       digest,
            "generated_at": datetime.utcnow().isoformat(),
        }, indent=2),
        ContentType="application/json",
    )

    print(f"Digest written to s3://{S3_BUCKET}/{s3_key}")
    return {"status": "done", "s3_key": s3_key, "jobs_summarised": len(all_results)}
```

### 6.5 ADK Agent — GCP Side

```python
# app/backends/adk_backend.py

import os
import vertexai
from vertexai.generative_models import GenerativeModel
from google.adk.agents import LlmAgent, SequentialAgent
from google.adk.tools import google_search, built_in_code_execution, FunctionTool
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner

from app.backends.base import BaseLLMBackend
from langfuse.decorators import observe

vertexai.init(
    project=os.getenv("GCP_PROJECT", ""),
    location=os.getenv("GCP_REGION", "us-central1"),
)

# ── ADK Research Pipeline ─────────────────────────────────────────────
def get_company_data(company: str, metric: str) -> str:
    """
    Fetch metric for a company from internal data store.
    Args:
        company: Company name to look up
        metric:  Metric name (revenue, employees, founded)
    Returns:
        Value as string, or 'not found'
    """
    # In production: query your DB or API
    mock_data = {
        "Google":    {"revenue": "$307B",  "employees": "180k", "founded": "1998"},
        "Microsoft": {"revenue": "$245B",  "employees": "220k", "founded": "1975"},
        "OpenAI":    {"revenue": "$2B est","employees": "1.7k", "founded": "2015"},
    }
    return str(mock_data.get(company, {}).get(metric, "not found"))

web_researcher = LlmAgent(
    name="web_researcher",
    model="gemini-2.0-flash-001",
    description="Searches the web for recent information on a topic",
    instruction="""You are a research specialist.
Search the web to find recent, factual information on the given topic.
Extract: key facts, statistics, recent developments, notable sources.
Be specific. Use google_search tool.""",
    tools=[google_search],
)

data_analyst = LlmAgent(
    name="data_analyst",
    model="gemini-2.0-flash-001",
    description="Analyses data and extracts quantitative insights",
    instruction="""You are a data analyst.
Given research findings, extract and verify numerical data.
Use the company data tool when relevant.
Calculate any meaningful ratios or trends.
Use code execution for calculations if needed.""",
    tools=[FunctionTool(func=get_company_data), built_in_code_execution],
)

report_writer = LlmAgent(
    name="report_writer",
    model="gemini-2.0-flash-001",
    description="Writes a clear research report from findings",
    instruction="""You are a professional technical writer.
Given research findings and data analysis, write a structured report.
Format: Executive Summary → Key Findings → Data Analysis → Conclusion.
Keep under 500 words. Use clear language.""",
)

research_pipeline = SequentialAgent(
    name="research_pipeline",
    description="Full research pipeline: web search → data analysis → report writing",
    sub_agents=[web_researcher, data_analyst, report_writer],
)

_session_service = InMemorySessionService()
_runner = Runner(
    agent=research_pipeline,
    app_name="cloudmind",
    session_service=_session_service,
)

class ADKBackend(BaseLLMBackend):
    """GCP backend using Vertex AI directly or ADK agent"""

    def __init__(self):
        self._model = GenerativeModel("gemini-2.0-flash-001")

    @observe(name="adk_complete")
    async def complete(self, messages: list[dict]) -> str:
        """Direct Gemini call (for fast /chat)"""
        # Convert to Gemini format
        user_msg = messages[-1]["content"]
        response = self._model.generate_content(user_msg)
        return response.text

    @observe(name="adk_stream")
    async def stream(self, messages: list[dict]):
        """Stream Gemini tokens"""
        user_msg = messages[-1]["content"]
        for chunk in self._model.generate_content(user_msg, stream=True):
            if chunk.text:
                yield chunk.text

    @observe(name="adk_agent_research")
    async def research_with_agent(self, topic: str, session_id: str) -> str:
        """Deep research via ADK SequentialAgent pipeline"""
        response = await _runner.run_async(
            user_id="cloudmind_user",
            session_id=session_id,
            new_message=f"Research this topic and write a structured report: {topic}",
        )
        return response.text
```

### 6.6 Cloud Run — GCP Deployment

```python
# app/backends/pubsub_queue.py  (GCP async queue)

from google.cloud import pubsub_v1
import json, os

TOPIC_PATH = f"projects/{os.getenv('GCP_PROJECT')}/topics/cloudmind-research-jobs"

class PubSubQueue:
    def __init__(self):
        self._pub = pubsub_v1.PublisherClient()

    async def enqueue(self, job: dict) -> str:
        future    = self._pub.publish(TOPIC_PATH, json.dumps(job).encode("utf-8"))
        return future.result()

# Deployment commands (run from project root):
"""
# Build and push image to Artifact Registry
gcloud builds submit \
  --tag us-central1-docker.pkg.dev/$GCP_PROJECT/cloudmind/api:$GITHUB_SHA .

# Deploy API to Cloud Run
gcloud run deploy cloudmind-api \
  --image   us-central1-docker.pkg.dev/$GCP_PROJECT/cloudmind/api:$GITHUB_SHA \
  --region  asia-south1 \
  --port    8000 \
  --cpu     2 \
  --memory  2Gi \
  --timeout 300 \
  --min-instances 1 \
  --max-instances 10 \
  --concurrency   8 \
  --set-env-vars  CLOUD_PROVIDER=gcp,GCP_PROJECT=$GCP_PROJECT \
  --set-secrets   LANGFUSE_SECRET_KEY=langfuse-secret:latest \
  --allow-unauthenticated

# Deploy ADK research worker to Cloud Run (separate service)
gcloud run deploy cloudmind-worker \
  --image   us-central1-docker.pkg.dev/$GCP_PROJECT/cloudmind/worker:$GITHUB_SHA \
  --region  asia-south1 \
  --port    8080 \
  --cpu     2 \
  --memory  4Gi \
  --timeout 600 \
  --min-instances 0 \
  --max-instances 5 \
  --set-env-vars CLOUD_PROVIDER=gcp \
  --no-allow-unauthenticated  # only Pub/Sub can trigger this
"""
```

### 6.7 Kubernetes — Local Orchestration

```yaml
# k8s/namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: cloudmind

---
# k8s/configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: cloudmind-config
  namespace: cloudmind
data:
  CLOUD_PROVIDER: "local"
  REDIS_HOST:     "redis-service"
  REDIS_PORT:     "6379"
  LOG_LEVEL:      "info"

---
# k8s/secret.yaml
# In production: use External Secrets Operator to sync from AWS Secrets Manager
# For local dev: base64-encode your keys
apiVersion: v1
kind: Secret
metadata:
  name: cloudmind-secrets
  namespace: cloudmind
type: Opaque
data:
  LANGFUSE_SECRET_KEY: <base64-encoded>
  AWS_ACCESS_KEY_ID:   <base64-encoded>   # only for local dev
  AWS_SECRET_ACCESS_KEY: <base64-encoded>

---
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: cloudmind-api
  namespace: cloudmind
spec:
  replicas: 2
  selector:
    matchLabels:
      app: cloudmind-api
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 0
  template:
    metadata:
      labels:
        app: cloudmind-api
    spec:
      containers:
      - name: api
        image: cloudmind-api:latest   # built locally with: docker build -t cloudmind-api .
        imagePullPolicy: Never        # use local image in minikube
        ports:
        - containerPort: 8000
        envFrom:
        - configMapRef:
            name: cloudmind-config
        - secretRef:
            name: cloudmind-secrets
        resources:
          requests:
            memory: "256Mi"
            cpu:    "100m"
          limits:
            memory: "1Gi"
            cpu:    "500m"
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 10

---
# k8s/service.yaml
apiVersion: v1
kind: Service
metadata:
  name: cloudmind-service
  namespace: cloudmind
spec:
  selector:
    app: cloudmind-api
  ports:
  - port: 80
    targetPort: 8000
  type: LoadBalancer   # NodePort on minikube: kubectl tunnel or minikube service

---
# k8s/hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: cloudmind-hpa
  namespace: cloudmind
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: cloudmind-api
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type:               Utilization
        averageUtilization: 70

---
# k8s/ollama-pvc.yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: ollama-model-cache
  namespace: cloudmind
spec:
  accessModes:
  - ReadWriteOnce
  resources:
    requests:
      storage: 10Gi   # llama3.1:8b ≈ 4.7GB

---
# k8s/ollama-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ollama
  namespace: cloudmind
spec:
  replicas: 1
  selector:
    matchLabels:
      app: ollama
  template:
    metadata:
      labels:
        app: ollama
    spec:
      initContainers:
      - name: pull-model
        image: ollama/ollama:latest
        command: ["/bin/sh", "-c"]
        args:
        - |
          ollama serve &
          sleep 5
          ollama pull llama3.1:8b
          kill %1
        volumeMounts:
        - name: model-cache
          mountPath: /root/.ollama
      containers:
      - name: ollama
        image: ollama/ollama:latest
        ports:
        - containerPort: 11434
        resources:
          requests:
            memory: "4Gi"
            cpu:    "2"
          limits:
            memory: "8Gi"
            cpu:    "4"
          # Uncomment if GPU available:
          # nvidia.com/gpu: "1"
        volumeMounts:
        - name: model-cache
          mountPath: /root/.ollama
      volumes:
      - name: model-cache
        persistentVolumeClaim:
          claimName: ollama-model-cache

---
# k8s/ollama-service.yaml
apiVersion: v1
kind: Service
metadata:
  name: ollama-service
  namespace: cloudmind
spec:
  selector:
    app: ollama
  ports:
  - port: 11434
    targetPort: 11434
  type: ClusterIP   # internal only — API talks to it within cluster

---
# k8s/cronjob.yaml
apiVersion: batch/v1
kind: CronJob
metadata:
  name: daily-digest
  namespace: cloudmind
spec:
  schedule: "0 8 * * *"    # 8 AM UTC daily
  jobTemplate:
    spec:
      template:
        spec:
          containers:
          - name: digest
            image: cloudmind-api:latest
            imagePullPolicy: Never
            command: ["python", "-m", "workers.digest_worker"]
            envFrom:
            - configMapRef:
                name: cloudmind-config
            - secretRef:
                name: cloudmind-secrets
          restartPolicy: OnFailure

---
# k8s/redis-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: redis
  namespace: cloudmind
spec:
  replicas: 1
  selector:
    matchLabels:
      app: redis
  template:
    metadata:
      labels:
        app: redis
    spec:
      containers:
      - name: redis
        image: redis:7-alpine
        ports:
        - containerPort: 6379
        args: ["--appendonly", "yes"]   # persistence
        resources:
          requests: {memory: "64Mi",  cpu: "50m"}
          limits:   {memory: "256Mi", cpu: "200m"}
---
apiVersion: v1
kind: Service
metadata:
  name: redis-service
  namespace: cloudmind
spec:
  selector:
    app: redis
  ports:
  - port: 6379
    targetPort: 6379
  type: ClusterIP
```

### 6.8 Provider Router — Bedrock vs ADK vs Ollama

```python
# app/router.py

import os
from app.backends.base import BaseLLMBackend
from app.queue.base import BaseQueue

class ProviderRouter:
    """
    Single dispatcher for all LLM + queue operations.
    Reads CLOUD_PROVIDER env var at startup.
    Never changes at runtime (restart to switch provider).
    
    CLOUD_PROVIDER=aws   → BedrockBackend + SQSQueue
    CLOUD_PROVIDER=gcp   → ADKBackend + PubSubQueue
    CLOUD_PROVIDER=local → OllamaBackend + InMemoryQueue
    """

    def __init__(self):
        provider = os.getenv("CLOUD_PROVIDER", "local").lower()
        print(f"[ProviderRouter] Initialising with provider: {provider}")

        if provider == "aws":
            from app.backends.bedrock_backend import BedrockBackend
            from app.queue.sqs_queue import SQSQueue
            self._llm   = BedrockBackend()
            self._queue = SQSQueue()

        elif provider == "gcp":
            from app.backends.adk_backend import ADKBackend
            from app.queue.pubsub_queue import PubSubQueue
            self._llm   = ADKBackend()
            self._queue = PubSubQueue()

        elif provider == "local":
            from app.backends.ollama_backend import OllamaBackend
            from app.queue.memory_queue import InMemoryQueue
            self._llm   = OllamaBackend()
            self._queue = InMemoryQueue()

        else:
            raise ValueError(f"Unknown CLOUD_PROVIDER: {provider}. Use aws | gcp | local")

    async def complete(self, messages: list[dict]) -> str:
        return await self._llm.complete(messages)

    async def stream(self, messages: list[dict]):
        async for token in self._llm.stream(messages):
            yield token

    async def enqueue_research_job(self, job: dict) -> str:
        return await self._queue.enqueue(job)

    async def research_deep(self, topic: str, session_id: str) -> str:
        return await self._llm.research_with_agent(topic, session_id)

# app/backends/ollama_backend.py  (local fallback)
import httpx, json, os
from app.backends.base import BaseLLMBackend

OLLAMA_URL   = os.getenv("OLLAMA_URL", "http://ollama-service:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")

class OllamaBackend(BaseLLMBackend):
    async def complete(self, messages: list[dict]) -> str:
        async with httpx.AsyncClient(timeout=120.0) as client:
            r = await client.post(f"{OLLAMA_URL}/api/chat", json={
                "model":    OLLAMA_MODEL,
                "messages": messages,
                "stream":   False,
            })
            return r.json()["message"]["content"]

    async def stream(self, messages: list[dict]):
        async with httpx.AsyncClient(timeout=120.0) as client:
            async with client.stream("POST", f"{OLLAMA_URL}/api/chat", json={
                "model":    OLLAMA_MODEL,
                "messages": messages,
                "stream":   True,
            }) as response:
                async for line in response.aiter_lines():
                    if line:
                        chunk = json.loads(line)
                        token = chunk.get("message", {}).get("content", "")
                        if token:
                            yield token

    async def research_with_agent(self, topic: str, session_id: str) -> str:
        # Local: just a detailed prompt (no real agent framework)
        return await self.complete([{
            "role": "user",
            "content": f"Write a detailed research report on: {topic}",
        }])
```

---

## 7. Folder Structure

```
cloudmind/
├── app/
│   ├── main.py                  ← FastAPI app + Mangum handler
│   ├── router.py                ← ProviderRouter (aws|gcp|local)
│   ├── store.py                 ← ResultStore (Redis)
│   ├── observability.py         ← Langfuse tracer
│   │
│   ├── backends/
│   │   ├── base.py              ← BaseLLMBackend ABC
│   │   ├── bedrock_backend.py   ← AWS: invoke_model + KB + Agent + Guardrail
│   │   ├── adk_backend.py       ← GCP: Vertex AI + ADK SequentialAgent
│   │   └── ollama_backend.py    ← Local: Ollama via httpx
│   │
│   └── queue/
│       ├── base.py              ← BaseQueue ABC
│       ├── sqs_queue.py         ← AWS SQS
│       ├── pubsub_queue.py      ← GCP Pub/Sub
│       └── memory_queue.py      ← Local: in-memory + background task
│
├── workers/
│   ├── deep_research_worker.py  ← SQS Lambda handler (AWS)
│   └── digest_lambda.py         ← EventBridge Lambda handler (AWS)
│
├── k8s/
│   ├── namespace.yaml
│   ├── configmap.yaml
│   ├── secret.yaml
│   ├── deployment.yaml          ← FastAPI app
│   ├── service.yaml
│   ├── hpa.yaml
│   ├── redis-deployment.yaml
│   ├── ollama-deployment.yaml
│   ├── ollama-pvc.yaml          ← PVC for model weights cache
│   └── cronjob.yaml             ← daily digest
│
├── tests/
│   ├── test_bedrock_backend.py  ← mock boto3 calls
│   ├── test_adk_backend.py
│   ├── test_ollama_backend.py
│   ├── test_router.py
│   └── test_api.py              ← TestClient against all endpoints
│
├── scripts/
│   ├── deploy_aws.sh            ← zip + update Lambda + update ECS
│   ├── deploy_gcp.sh            ← gcloud builds + gcloud run deploy
│   └── apply_k8s.sh             ← kubectl apply -f k8s/ -n cloudmind
│
├── .github/
│   └── workflows/
│       └── deploy.yml           ← full CI/CD
│
├── Dockerfile
├── docker-compose.yml           ← local dev: fastapi + redis + ollama
├── requirements.txt
├── requirements-aws.txt         ← boto3 + mangum (Lambda layer)
├── requirements-gcp.txt         ← google-adk + google-cloud-aiplatform
├── .env.example
└── README.md
```

---

## 8. Docker Compose — Local Dev

```yaml
# docker-compose.yml

services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      CLOUD_PROVIDER: local       # change to aws or gcp as needed
      REDIS_HOST:     redis
      REDIS_PORT:     "6379"
      OLLAMA_URL:     http://ollama:11434
      LANGFUSE_HOST:  http://langfuse:3000
      LANGFUSE_PUBLIC_KEY: ${LANGFUSE_PUBLIC_KEY}
      LANGFUSE_SECRET_KEY: ${LANGFUSE_SECRET_KEY}
    depends_on:
      - redis
      - ollama
    volumes:
      - ./app:/app/app             # hot-reload in dev
    command: uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    command: redis-server --appendonly yes
    volumes:
      - redis_data:/data

  ollama:
    image: ollama/ollama:latest
    ports:
      - "11434:11434"
    volumes:
      - ollama_models:/root/.ollama
    # Pull model on first start:
    # docker exec cloudmind-ollama-1 ollama pull llama3.1:8b

  langfuse:
    image: ghcr.io/langfuse/langfuse:latest
    ports:
      - "3000:3000"
    environment:
      DATABASE_URL:         postgresql://langfuse:langfuse@langfuse-db/langfuse
      NEXTAUTH_URL:         http://localhost:3000
      NEXTAUTH_SECRET:      secret
      SALT:                 salt
    depends_on:
      - langfuse-db

  langfuse-db:
    image: postgres:16-alpine
    environment:
      POSTGRES_USER:     langfuse
      POSTGRES_PASSWORD: langfuse
      POSTGRES_DB:       langfuse
    volumes:
      - langfuse_data:/var/lib/postgresql/data

volumes:
  redis_data:
  ollama_models:
  langfuse_data:
```

---

## 9. GitHub Actions CI/CD

```yaml
# .github/workflows/deploy.yml

name: CloudMind Deploy

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

env:
  AWS_REGION:   ap-south-1
  GCP_REGION:   asia-south1
  ECR_REPO:     cloudmind-api
  GCP_PROJECT:  ${{ secrets.GCP_PROJECT }}

jobs:
  # ── 1. Test ────────────────────────────────────────────────────────
  test:
    runs-on: ubuntu-latest
    services:
      redis:
        image: redis:7-alpine
        ports: ["6379:6379"]
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - run: pip install -r requirements.txt
      - run: pip install pytest pytest-asyncio httpx
      - run: |
          CLOUD_PROVIDER=local pytest tests/ -v \
            --tb=short -x
      - run: ruff check app/ workers/

  # ── 2. Build Docker image ─────────────────────────────────────────
  build:
    needs: test
    runs-on: ubuntu-latest
    outputs:
      image_tag: ${{ steps.tag.outputs.tag }}
    steps:
      - uses: actions/checkout@v4
      - id: tag
        run: echo "tag=${GITHUB_SHA::8}" >> $GITHUB_OUTPUT

      # Build once, push to both ECR and Artifact Registry
      - name: Build image
        run: docker build -t cloudmind:${{ steps.tag.outputs.tag }} .

      - name: Push to ECR
        uses: aws-actions/amazon-ecr-login@v2
        with:
          aws-access-key-id:     ${{ secrets.AWS_ACCESS_KEY_ID }}
          aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
          aws-region:            ${{ env.AWS_REGION }}
      - run: |
          ECR=${{ secrets.AWS_ACCOUNT_ID }}.dkr.ecr.${{ env.AWS_REGION }}.amazonaws.com
          docker tag  cloudmind:${{ steps.tag.outputs.tag }} $ECR/$ECR_REPO:${{ steps.tag.outputs.tag }}
          docker push $ECR/$ECR_REPO:${{ steps.tag.outputs.tag }}

      - name: Push to Artifact Registry
        uses: google-github-actions/auth@v2
        with:
          credentials_json: ${{ secrets.GCP_CREDENTIALS }}
      - run: |
          gcloud auth configure-docker ${{ env.GCP_REGION }}-docker.pkg.dev
          docker tag  cloudmind:${{ steps.tag.outputs.tag }} \
            ${{ env.GCP_REGION }}-docker.pkg.dev/$GCP_PROJECT/cloudmind/api:${{ steps.tag.outputs.tag }}
          docker push \
            ${{ env.GCP_REGION }}-docker.pkg.dev/$GCP_PROJECT/cloudmind/api:${{ steps.tag.outputs.tag }}

  # ── 3. Deploy to AWS Lambda ───────────────────────────────────────
  deploy-aws:
    needs: build
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    steps:
      - uses: actions/checkout@v4
      - uses: aws-actions/configure-aws-credentials@v4
        with:
          aws-access-key-id:     ${{ secrets.AWS_ACCESS_KEY_ID }}
          aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
          aws-region:            ${{ env.AWS_REGION }}
      - name: Update Lambda function code
        run: |
          ECR=${{ secrets.AWS_ACCOUNT_ID }}.dkr.ecr.${{ env.AWS_REGION }}.amazonaws.com
          aws lambda update-function-code \
            --function-name cloudmind-api \
            --image-uri $ECR/$ECR_REPO:${{ needs.build.outputs.image_tag }}
      - name: Wait for Lambda update
        run: |
          aws lambda wait function-updated \
            --function-name cloudmind-api
      - name: Smoke test Lambda
        run: |
          URL=$(aws lambda get-function-url-config \
            --function-name cloudmind-api \
            --query FunctionUrl --output text)
          curl -f "$URL/health"

  # ── 4. Deploy to GCP Cloud Run ────────────────────────────────────
  deploy-gcp:
    needs: build
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    steps:
      - uses: google-github-actions/auth@v2
        with:
          credentials_json: ${{ secrets.GCP_CREDENTIALS }}
      - uses: google-github-actions/deploy-cloudrun@v2
        with:
          service: cloudmind-api
          image:   ${{ env.GCP_REGION }}-docker.pkg.dev/${{ env.GCP_PROJECT }}/cloudmind/api:${{ needs.build.outputs.image_tag }}
          region:  ${{ env.GCP_REGION }}
          flags:   >-
            --min-instances=1
            --max-instances=10
            --cpu=2
            --memory=2Gi
            --timeout=300
            --set-env-vars=CLOUD_PROVIDER=gcp,GCP_PROJECT=${{ env.GCP_PROJECT }}
            --set-secrets=LANGFUSE_SECRET_KEY=langfuse-secret:latest
      - name: Smoke test Cloud Run
        run: curl -f ${{ steps.deploy-gcp.outputs.url }}/health

  # ── 5. Deploy to K8s (minikube via kubeconfig secret) ─────────────
  deploy-k8s:
    needs: build
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    steps:
      - uses: actions/checkout@v4
      - name: Configure kubectl
        run: |
          echo "${{ secrets.KUBECONFIG }}" | base64 -d > $HOME/.kube/config
      - name: Update K8s image
        run: |
          kubectl set image deployment/cloudmind-api \
            api=cloudmind:${{ needs.build.outputs.image_tag }} \
            -n cloudmind
          kubectl rollout status deployment/cloudmind-api -n cloudmind
```

---

## 10. Demo Script

```bash
# ─── LOCAL (Docker Compose) ─────────────────────────────────────────

# Start everything
docker compose up -d

# Wait for Ollama to be ready
docker exec cloudmind-ollama-1 ollama pull llama3.1:8b

# Health check
curl http://localhost:8000/health
# → {"status":"ok","provider":"local","version":"1.0.0"}

# Sync chat (Ollama)
curl -X POST http://localhost:8000/chat \
  -H "content-type: application/json" \
  -d '{"topic": "Transformer architecture"}'

# Streaming chat (SSE)
curl -X POST http://localhost:8000/chat/stream \
  -H "content-type: application/json" \
  -d '{"topic": "What is RAG?"}' \
  --no-buffer

# Queue deep research
JOB=$(curl -s -X POST http://localhost:8000/research/queue \
  -H "content-type: application/json" \
  -d '{"topic": "Recent advances in LLM reasoning"}' | jq -r .job_id)
echo "Job ID: $JOB"

# Poll for result
sleep 10
curl http://localhost:8000/research/$JOB

# Langfuse: view traces at http://localhost:3000

# ─── KUBERNETES (minikube) ───────────────────────────────────────────

minikube start
eval $(minikube docker-env)         # use minikube's docker
docker build -t cloudmind-api .     # build locally

kubectl apply -f k8s/ -n cloudmind
kubectl get pods -n cloudmind        # watch until all Running

kubectl port-forward svc/cloudmind-service 8000:80 -n cloudmind &
curl http://localhost:8000/health
# → {"status":"ok","provider":"local"}

# Watch HPA scale under load
kubectl get hpa -n cloudmind -w &
hey -n 200 -c 20 http://localhost:8000/health   # pip install hey
# Watch REPLICAS increase

# Check PVC bound
kubectl get pvc -n cloudmind
# → ollama-model-cache   Bound   10Gi

# Check CronJob
kubectl get cronjob -n cloudmind
kubectl create job --from=cronjob/daily-digest test-digest-now -n cloudmind
kubectl logs -l job-name=test-digest-now -n cloudmind

# ─── AWS LAMBDA ──────────────────────────────────────────────────────

export CLOUD_PROVIDER=aws

# Test Lambda URL (after deploy)
LAMBDA_URL="https://xxxx.lambda-url.ap-south-1.on.aws"
curl $LAMBDA_URL/health
curl -X POST $LAMBDA_URL/chat/stream \
  -H "content-type: application/json" \
  -d '{"topic": "AWS Bedrock Knowledge Bases"}' \
  --no-buffer

# Queue research job (SQS → Lambda worker)
JOB=$(curl -s -X POST $LAMBDA_URL/research/queue \
  -H "content-type: application/json" \
  -d '{"topic": "Serverless AI architectures"}' | jq -r .job_id)

sleep 30
curl $LAMBDA_URL/research/$JOB

# Test guardrail block
curl -X POST $LAMBDA_URL/chat \
  -H "content-type: application/json" \
  -d '{"topic": "how to bypass security systems"}'
# → {"answer": "I can't help with that request."}

# CloudWatch: Lambda logs
aws logs tail /aws/lambda/cloudmind-api --follow

# Manually trigger digest Lambda
aws lambda invoke \
  --function-name cloudmind-digest \
  --payload '{"source": "manual-test"}' \
  /tmp/digest_result.json
cat /tmp/digest_result.json

# ─── GCP CLOUD RUN ───────────────────────────────────────────────────

GCR_URL=$(gcloud run services describe cloudmind-api \
  --region asia-south1 --format 'value(status.url)')

curl $GCR_URL/health
curl -X POST $GCR_URL/chat/stream \
  -H "content-type: application/json" \
  -d '{"topic": "Google ADK vs LangGraph"}' \
  --no-buffer

# Watch ADK SequentialAgent steps in Langfuse
```

---

## 11. Phase 7 Coverage Checklist

```
AWS BEDROCK ✅
[x] invoke_model (BedrockBackend.complete)
[x] invoke_model_with_response_stream (BedrockBackend.stream)
[x] SSE streaming from FastAPI to browser (/chat/stream)
[x] Bedrock Knowledge Base (retrieve_and_generate)
[x] Bedrock Agent (research_with_agent → invoke_agent)
[x] Bedrock Guardrail (guardrailIdentifier on every call)
[x] IAM role used (no hardcoded keys in code)
[x] Region: ap-south-1 (India data residency)

AWS LAMBDA ✅
[x] Mangum wrapping FastAPI → single handler = Mangum(app)
[x] Lambda streaming (streaming_response decorator)
[x] Lambda layers (boto3 + pydantic separated)
[x] Cold start measurement (before/after provisioned concurrency)
[x] EventBridge cron → digest_lambda (daily 8 AM)
[x] SQS → Lambda worker (deep_research_worker)
[x] DLQ configured for failed messages
[x] Deployment zip script (scripts/deploy_aws.sh)

AWS SAGEMAKER ✅ (mentioned in router; full SageMaker project = Project D in Phase 7)
[ ] SageMaker endpoint referenced in OllamaBackend comments
    (Full SageMaker usage = separate Project D — HireSense retraining pipeline)

GCP VERTEX AI + ADK ✅
[x] Vertex AI Gemini 2.0 Flash (ADKBackend.complete + stream)
[x] ADK LlmAgent with custom tool (get_company_data as FunctionTool)
[x] ADK LlmAgent with google_search built-in tool
[x] ADK LlmAgent with built_in_code_execution
[x] ADK SequentialAgent (researcher → analyst → writer)
[x] ADK sessions (InMemorySessionService + Runner)
[x] ADK deployed to Cloud Run

GCP CLOUD RUN ✅
[x] Full Dockerfile (already in project)
[x] gcloud run deploy with all flags (in deploy_gcp.sh)
[x] SSE streaming from Cloud Run (native, no config needed)
[x] Cloud Run + Vertex AI (service account auth, no API keys)
[x] Pub/Sub queue for async jobs (pubsub_queue.py)
[x] Min-instances=1 (cold start mitigation)
[x] GitHub Actions → gcloud run deploy

KUBERNETES ✅
[x] Deployment.yaml with rollingUpdate + readiness/liveness probes
[x] Service.yaml (LoadBalancer)
[x] ConfigMap + Secret properly separated
[x] HPA (CPU 70% → scale 2-10 replicas)
[x] Ollama Deployment with InitContainer (pull model before start)
[x] PersistentVolumeClaim (ollama-model-cache 10Gi)
[x] CronJob (daily-digest 8 AM)
[x] Redis Deployment + Service (in-cluster)
[x] kubectl commands in demo script (apply, get, describe, logs, port-forward, rollout)
[x] GitHub Actions → kubectl set image

PROVIDER ROUTER ✅
[x] ProviderRouter reads CLOUD_PROVIDER env var
[x] Switches: aws → Bedrock + SQS; gcp → ADK + PubSub; local → Ollama + InMemory
[x] All three paths tested (CLOUD_PROVIDER=local docker compose up)

OBSERVABILITY ✅
[x] Langfuse @observe on all LLM calls (bedrock, adk, ollama)
[x] CloudWatch Lambda metrics (invocations, errors, duration)
[x] Structured JSON logging (all FastAPI requests)
[x] Demo: side-by-side Langfuse traces (aws vs gcp path)

CI/CD ✅
[x] GitHub Actions: test → build → deploy-aws → deploy-gcp → deploy-k8s
[x] Single Docker image pushed to both ECR and Artifact Registry
[x] Smoke test after each deploy (curl /health)
[x] Rollback: kubectl rollout undo / aws lambda update-function-code
```

---

*CloudMind — Phase 7 Capstone Mini-Project*
*Covers: Bedrock · Lambda + Mangum · SQS · EventBridge · Vertex AI · ADK · Cloud Run · K8s + GPU manifests · Provider Router · CI/CD*
