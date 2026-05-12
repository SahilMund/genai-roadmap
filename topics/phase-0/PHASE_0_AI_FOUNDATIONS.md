# 🧠 Phase 0 — AI Foundations
> **Complete Study Notes | Interview Prep | Real-Life Examples | Beginner-Friendly Explanations**  
> Part of: GenAI + LLMOps Engineering Roadmap 2026  
> Estimated time: 25–35 hrs over 2 weeks

---

## 📑 Table of Contents

1. [0.1 — AI Landscape](#01--ai-landscape)
2. [0.2 — Attention Mechanism](#02--attention-mechanism)
3. [0.3 — Transformers](#03--transformers)
4. [0.4 — NLP Essentials](#04--nlp-essentials)
5. [0.5 — Python Patterns for AI](#05--python-patterns-for-ai)
6. [0.6 — Intro to Agents](#06--intro-to-agents)
7. [Interview Cheat Sheet](#-interview-cheat-sheet)
8. [Quick Revision Cards](#-quick-revision-cards)

---

## 0.1 — AI Landscape

### 🧑‍🎓 Explain Like I'm in BTech 2nd Year

Imagine a big family tree:

```
🌳 AI (Artificial Intelligence)
   └── 🌿 ML (Machine Learning)  ← learns from data
         └── 🍃 DL (Deep Learning)  ← uses neural networks
               ├── 🌸 GenAI  ← generates new content
               │     └── 🤖 Agentic AI  ← takes actions on its own
               └── 🖼️ Computer Vision, Speech etc.
```

**Simple analogy:**
- **AI** = teaching a computer to be "smart" (broad)
- **ML** = giving it data and letting it figure out rules (instead of you writing rules)
- **DL** = ML using brain-inspired layers called neural networks
- **GenAI** = DL that creates — text, images, audio, code
- **Agentic AI** = GenAI that doesn't just respond, it *does things* — browses web, writes files, calls APIs

---

### 📖 Formal Definitions

| Term | Definition |
|------|-----------|
| **AI** | Any technique that enables machines to mimic human intelligence |
| **ML** | A subset of AI where systems learn from data without being explicitly programmed |
| **DL** | A subset of ML using multi-layered neural networks to learn hierarchical representations |
| **GenAI** | AI systems that generate new content (text, images, audio, code, video) |
| **Agentic AI** | AI systems that autonomously plan, use tools, and take multi-step actions to achieve goals |
| **LLM** | Large Language Model — a DL model trained on massive text data to understand and generate text |

---

### 🗺️ Visual Hierarchy

> AI ⊃ ML ⊃ DL ⊃ GenAI ⊃ Agentic AI

---

### ⚔️ Traditional AI vs Generative AI

| | Traditional AI | Generative AI |
|--|--|--|
| **Goal** | Classify, predict, detect | Create new content |
| **Output** | Label, number, decision | Text, image, audio, code |
| **Example** | Spam filter, fraud detection | ChatGPT, DALL-E, Copilot |
| **Data needed** | Labelled dataset | Massive unlabelled text/images |
| **Training** | Supervised, task-specific | Self-supervised, foundation model |
| **Flexibility** | One task per model | One model, many tasks |
| **Interpretability** | Easier to explain | Harder (black box) |

**Real-life example:**
- Traditional AI: Gmail's spam filter (classifies: spam or not spam)
- GenAI: Gmail's "Help me write" (generates a full email from a prompt)

---

### 🏭 LLM vs Multimodal Models

| | LLM | Multimodal Model |
|--|--|--|
| **Input** | Text only | Text + image + audio + video |
| **Output** | Text only | Text + image + audio |
| **Examples** | GPT-4 (text), Claude 2 | GPT-4o, Gemini 2.0, Claude 3 |
| **Use case** | Chatbots, code gen, summarisation | Image Q&A, video analysis, voice agents |

---

### 🔄 AI Product Lifecycle

```
1. Discovery     → Is AI the right solution? What problem?
2. Prototype     → Quick POC, use GPT-4 API, see if it works
3. Evaluation    → Measure quality, build eval dataset, RAGAS scores
4. Production    → Docker, cloud deploy, monitoring, guardrails
5. Monitoring    → Track quality drift, cost, latency, user feedback
6. Iteration     → Improve prompts, retrain, new model version
```

**As a full-stack dev, you've done this for regular apps. AI adds eval and monitoring as first-class citizens.**

---

### 💼 Interview Questions — AI Landscape

**Q1: What is the difference between AI, ML, and DL?**

> AI is the broad field of making machines intelligent. ML is a subset where machines learn from data instead of being explicitly programmed. Deep Learning is a subset of ML that uses multi-layered neural networks. Think of it as: AI is the goal, ML is one method, DL is a specific technique within ML.

**Q2: What makes Generative AI different from traditional AI?**

> Traditional AI classifies or predicts — given data, output a label or number. GenAI generates — given a prompt, create new content. The fundamental shift is from discriminative models (what is this?) to generative models (create something new).

**Q3: What is an LLM?**

> A Large Language Model is a deep learning model trained on massive text corpora using self-supervised learning. It learns to predict the next token given context, and through this simple objective, emerges understanding of language, reasoning, and knowledge. GPT-4, Claude, Gemini, and Llama are all LLMs.

**Q4: What is Agentic AI?**

> Agentic AI extends LLMs with the ability to take actions — use tools, call APIs, browse the web, write files, and make decisions across multiple steps. Instead of one response, an agent runs a loop: reason about what to do → take action → observe result → reason again. LangGraph, CrewAI, and AutoGen are frameworks for building agents.

---

### 🌍 Real-Life Use Cases

| Domain | Traditional AI | GenAI | Agentic AI |
|--------|---------------|-------|-----------|
| **Banking** | Fraud detection | Summarise transactions in plain English | Agent that investigates suspicious activity |
| **E-commerce** | Product recommendation | Write product descriptions | Agent that manages inventory & reorders |
| **Healthcare** | X-ray classification | Summarise patient notes | Agent that books appointments & sends reminders |
| **Software** | Bug detection (static analysis) | GitHub Copilot (code gen) | Devin / Claude Code (full task automation) |
| **HR** | Resume screening (score) | Write job descriptions | Agent that screens, schedules, and follows up |

---

## 0.2 — Attention Mechanism

### 🧑‍🎓 Explain Like I'm in BTech 2nd Year

Imagine you're reading this sentence:

> *"The **trophy** didn't fit in the suitcase because **it** was too big."*

What does "it" refer to? The trophy, right? How did you know?

You looked back at the sentence and **paid more attention** to "trophy" than to "suitcase" to resolve "it".

**That's literally what the attention mechanism does** — for every word it generates, it looks back at all other words and decides how much to "pay attention" to each one.

Before attention existed, models had to compress the entire input sentence into a fixed-size vector (like a summary). Imagine compressing a 10,000-word document into one sentence — you lose a lot. Attention solved this by letting the model look at the full input at any point.

---

### 📖 The Problem Attention Solved

**Old way (RNN/LSTM):**
```
"I love eating pizza in Italy"
         ↓
[compress everything into one fixed vector]
         ↓
Translate to French
```
By the time you get to "Italy", the model has mostly forgotten "I". This is the **vanishing gradient problem**.

**With attention:**
```
"I love eating pizza in Italy"
         ↓
When generating each output word,
LOOK BACK at ALL input words
and decide which ones matter most
         ↓
"J'adore manger de la pizza en Italie"
```

---

### 🔢 The Math (Don't Fear It)

Attention uses three vectors for each token:
- **Q (Query)** — what am I looking for?
- **K (Key)** — what do I have to offer?
- **V (Value)** — what information do I actually carry?

**Formula:**
```
Attention(Q, K, V) = softmax(QKᵀ / √d_k) · V
```

**In plain English:**
1. Compute similarity between Query and all Keys (`QKᵀ`)
2. Scale by `√d_k` to prevent huge values that make softmax sharp
3. `softmax` converts scores to probabilities (sum = 1)
4. Multiply probabilities by Values — high-attention tokens contribute more

**Analogy:** Think of a library:
- Query = your search query ("books about Python")
- Keys = book titles in the catalog
- Values = the actual book content
- Attention = match query to titles, fetch the most relevant books

---

### 👁️ Self-Attention vs Cross-Attention

| | Self-Attention | Cross-Attention |
|--|--|--|
| **What** | Tokens attend to other tokens in the SAME sequence | Tokens in one sequence attend to another sequence |
| **Where used** | Encoder (understand input) | Decoder (use encoder info while generating) |
| **Example** | "it" attending to "trophy" in the same sentence | English tokens attending to French tokens during translation |

---

### 🎯 Multi-Head Attention

Instead of one attention function, run H attention functions in parallel:

```
Input
  ├── Head 1 → focuses on syntax (subject-verb relationships)
  ├── Head 2 → focuses on coreference ("it" → "trophy")
  ├── Head 3 → focuses on position (nearby words)
  ├── Head 4 → focuses on semantics (meaning)
  └── ...
  → Concatenate all heads → Linear layer → Output
```

**Why multiple heads?**
Different heads specialise in different types of relationships. One sentence has many kinds of structure happening simultaneously.

---

### 👀 Attention Visualisation

![Attention Heatmap Visualisation](https://jalammar.github.io/images/t/transformer_self-attention_visualization.png)

> In this heatmap, darker cells = more attention. You can see "it" strongly attends to "animal" — the model has learned coreference resolution just from training on text.

---

### 💼 Interview Questions — Attention

**Q1: Why was the attention mechanism invented?**

> RNNs and LSTMs encoded the entire input into a fixed-size vector, which became a bottleneck for long sequences — early tokens got "forgotten". Attention was introduced in the 2015 Bahdanau paper to allow the decoder to look at all encoder hidden states at each generation step, weighted by relevance. This eliminated the fixed-size bottleneck.

**Q2: Explain Q, K, V in attention.**

> Each token is projected into three vectors. The Query represents what the current token is looking for. The Key represents what each token offers. The Value represents the actual content. We compute dot products between Queries and Keys to get attention scores, normalise with softmax, and use those weights to take a weighted sum of Values. It's conceptually like a soft, differentiable dictionary lookup.

**Q3: Why do we divide by √d_k in the attention formula?**

> When d_k (dimension of keys) is large, the dot products QKᵀ grow large in magnitude. Large values push softmax into regions where gradients are extremely small (saturation), making training difficult. Dividing by √d_k keeps the variance of the dot products around 1, stabilising gradients.

**Q4: What is multi-head attention and why is it useful?**

> Multi-head attention runs H attention functions in parallel with different learned projections. Each head can specialise in capturing different types of relationships — one head might focus on syntactic structure, another on coreference, another on positional relationships. The outputs are concatenated and linearly projected. This gives the model richer representational capacity than a single attention function.

---

### 🌍 Real-Life Analogy

**Attention = Google Search**
- Your search query = Q
- Every webpage's title/meta = K
- The actual webpage content = V
- Google's ranking score = attention weights
- What you see in results = weighted combination of V

The difference: attention is differentiable, so it can be trained end-to-end.

---

## 0.3 — Transformers

### 🧑‍🎓 Explain Like I'm in BTech 2nd Year

Before Transformers, AI processed text like reading a book one word at a time, from left to right, keeping notes as it went (that's what RNNs do). The problem? By the time you reach page 100, you've mostly forgotten page 1.

Transformers say: **read the entire book at once, pay attention to every word simultaneously**.

This has two massive advantages:
1. **No forgetting** — every word can directly attend to every other word
2. **Parallelism** — process all words at the same time → GPUs love this → train faster

**Real-world impact:** This one architecture change (2017 paper "Attention Is All You Need") led to GPT, BERT, Claude, Gemini, Llama — literally every major AI system you use today.

---

### 🏗️ The Full Architecture

![Transformer Architecture](https://machinelearningmastery.com/wp-content/uploads/2021/08/attention_research_1.png)

```
INPUT TEXT
    ↓
[Token Embeddings] + [Positional Encoding]
    ↓
┌─────────────────────────────┐
│      ENCODER STACK          │  ← N layers (e.g. 12 in BERT-base)
│  ┌─────────────────────┐    │
│  │ Multi-Head Attention │    │
│  │ Add & LayerNorm      │    │
│  │ Feed Forward         │    │
│  │ Add & LayerNorm      │    │
│  └─────────────────────┘    │
└─────────────────────────────┘
    ↓
┌─────────────────────────────┐
│      DECODER STACK          │  ← N layers
│  ┌─────────────────────┐    │
│  │ Masked Multi-Head    │    │
│  │ Attention            │    │
│  │ Cross-Attention      │    │  ← attends to encoder output
│  │ Feed Forward         │    │
│  └─────────────────────┘    │
└─────────────────────────────┘
    ↓
[Linear] → [Softmax] → OUTPUT TOKEN
```

---

### 📍 Positional Encoding — Why Transformers Need It

**The problem:** Attention doesn't know word order. "Dog bites man" and "Man bites dog" would look the same without positional info.

**The solution:** Add a position signal to each token's embedding before attention.

```python
# Each position gets a unique vector added to its embedding
position_0 = embedding("The") + pos_encoding(0)
position_1 = embedding("cat") + pos_encoding(1)
position_2 = embedding("sat") + pos_encoding(2)
```

**Original paper:** Used sine and cosine functions at different frequencies. Modern models use **Rotary Position Embedding (RoPE)** — used in Llama, Mistral, Gemini.

**Analogy:** Like putting page numbers on a shuffled book. The content (embedding) + page number (position) = full context.

---

### 🏛️ The 3 Architecture Families — CRITICAL

This is the most important thing to know from Phase 0.

#### 1. Encoder-Only (BERT, RoBERTa)

```
Input → [Encoder] → Rich representation of input
```

- Sees the FULL input (bidirectional)
- Good at **understanding** text, not generating
- Use cases: text classification, embeddings for RAG, named entity recognition
- Every token attends to every other token (no masking)

**Think of it as:** A reader who reads the entire book before answering questions about it.

#### 2. Decoder-Only (GPT, Claude, Llama, Gemini, Mistral)

```
[Previous tokens] → [Decoder] → Next token
```

- Only sees past tokens (causal/autoregressive, left-to-right)
- Good at **generating** text
- This is what ALL major LLMs are — GPT-4, Claude, Llama, Gemini
- Each token only attends to tokens before it (masked attention)

**Think of it as:** A writer who generates the next word based only on what's already been written.

#### 3. Encoder-Decoder (T5, BART, Whisper, mT5)

```
Input → [Encoder] → [Decoder] → Output
```

- Encoder reads and understands input
- Decoder generates output, cross-attending to encoder
- Good at **sequence-to-sequence** tasks
- Use cases: translation, summarisation, speech-to-text (Whisper)

**Think of it as:** A translator — reads the full foreign text (encoder), then generates in the target language (decoder).

---

### ⚡ Comparison Table

| Architecture | Direction | Best For | Examples | Used In |
|---|---|---|---|---|
| Encoder-Only | Bidirectional | Understanding, embeddings | BERT, RoBERTa, DeBERTa | RAG embeddings, classification |
| Decoder-Only | Left-to-right | Text generation | GPT-4, Claude, Llama, Gemini | Chatbots, code gen, agents |
| Encoder-Decoder | Both | Seq2seq | T5, BART, Whisper | Translation, summarisation, STT |

---

### 🧠 KV Cache — Why It Matters for Engineers

When a decoder generates token by token:

```
Generate token 1: compute attention over [token 1]
Generate token 2: compute attention over [token 1, token 2]
Generate token 3: compute attention over [token 1, token 2, token 3]
```

Without caching: recompute everything from scratch every step. Slow.

With **KV Cache**: store the Key and Value matrices from previous steps. Only compute for the new token.

**Why you care as an engineer:**
- KV cache = GPU memory. Long context = more KV cache = more VRAM needed
- Bedrock/OpenAI charge less for **cached prompt tokens** (you've already paid for them)
- This is why 1M-context models are expensive — huge KV cache

---

### 🏋️ The 3-Stage Training Pipeline

```
Stage 1: Pre-training
  - Train on 10T+ tokens of internet text
  - Task: predict next token
  - Output: base model (knows language, but not how to chat)
  - Example: Llama 3 base

Stage 2: Instruction Fine-tuning (SFT)
  - Train on (instruction, response) pairs
  - Teaches model to follow instructions
  - Output: instruction-tuned model
  - Example: Llama 3 Instruct

Stage 3: RLHF / RLAIF
  - Human raters rank outputs (RLHF) or AI ranks outputs (RLAIF)
  - Train to prefer better responses
  - Output: aligned model (helpful, harmless, honest)
  - Example: ChatGPT, Claude
```

**Analogy:**
- Pre-training = reading every book ever written
- SFT = taking a customer service training course
- RLHF = getting feedback from your manager and improving

---

### 🤥 What Is Hallucination — Technically

**Simple answer most people give:** "The model makes things up."

**Better technical answer for interviews:**

LLMs are trained to predict the next most probable token given context. They do NOT have a mechanism to verify factual accuracy against a ground truth. When the model's training data is sparse for a topic, it extrapolates based on surface patterns — generating plausible-sounding but incorrect tokens.

Types:
1. **Factual hallucination** — wrong facts ("Einstein won the Nobel Prize in 1925" — it was 1921)
2. **Faithfulness hallucination** — answer not grounded in the source document (RAG failure)
3. **Intrinsic hallucination** — contradicts its own earlier statement

Why RAG helps: forces the model to generate from retrieved evidence, reducing reliance on parametric memory.

---

### 💼 Interview Questions — Transformers

**Q1: What was the key innovation of the "Attention Is All You Need" paper?**

> The paper showed that you can build a powerful sequence model using ONLY attention mechanisms, completely eliminating recurrence (RNNs) and convolutions. This enabled full parallelism during training — all tokens processed simultaneously — and allowed models to scale dramatically on GPUs.

**Q2: Why do decoder-only models use masked attention?**

> During autoregressive generation, the model should only see tokens that have already been generated — it shouldn't "cheat" by looking at future tokens. Masking (setting future positions to -∞ before softmax) ensures each position can only attend to itself and previous positions. This is also called causal attention.

**Q3: What is the difference between BERT and GPT architecturally?**

> BERT is encoder-only and bidirectional — it sees the full input simultaneously. GPT is decoder-only and causal — it processes tokens left-to-right. BERT is better for understanding tasks (classification, embedding). GPT is better for generation tasks (chatbots, code completion).

**Q4: What is the KV cache and why does it matter?**

> During autoregressive decoding, the Key and Value matrices for previous tokens are the same at every step. The KV cache stores these so they don't need to be recomputed. This reduces per-token generation time from O(n²) to O(n). The trade-off is memory — long contexts require large KV caches, which is why long-context models are expensive to serve.

**Q5: What is RLHF?**

> Reinforcement Learning from Human Feedback is a training technique where a reward model is trained on human preference rankings, then used to fine-tune the LLM via PPO (Proximal Policy Optimisation) to generate outputs that receive higher reward scores. It aligns the model with human preferences — making it more helpful, less harmful, more honest.

---

## 0.4 — NLP Essentials

### 🧑‍🎓 Explain Like I'm in BTech 2nd Year

**Tokenisation — why not just split by spaces?**

Try splitting "don't" by spaces → ["don't"]. But the model needs to understand "do" + "n't" separately. And what about "ChatGPT"? Or emojis? Or Chinese characters (no spaces!)?

Tokenisation converts raw text into numbers the model can process.

**Embeddings — what are they really?**

Every word gets converted to a list of numbers (a vector). The magic: similar words get similar numbers.

```
"king"   → [0.2, 0.8, 0.1, 0.9, ...]  (300 numbers)
"queen"  → [0.2, 0.8, 0.2, 0.8, ...]  (very similar!)
"dog"    → [0.9, 0.1, 0.7, 0.3, ...]  (very different)
```

This lets the model understand that "king" and "queen" are related without being programmed to know that.

---

### 🔡 Tokenisation Deep Dive

#### How BPE Works (Byte-Pair Encoding)

BPE is a data compression algorithm adapted for tokenisation:

```
Start with characters: ["h", "e", "l", "l", "o"]

Step 1: Find most common adjacent pair → "l" + "l" = "ll"
        → ["h", "e", "ll", "o"]

Step 2: Find next most common pair → "he" appears often
        → ["he", "ll", "o"]

Step 3: Continue until vocabulary size reached
        → ["hello"] (if frequent enough)
```

**Real example (GPT-4 tokeniser):**

```
"ChatGPT is amazing!"
→ ["Chat", "G", "PT", " is", " amazing", "!"]
→ [  9106, 38,  2898,  374,  8056,     0  ]
```

**Key facts:**
- GPT-4 vocabulary: ~100K tokens
- Common English words: 1 token
- Rare words: split into 2-5 tokens
- Emojis: 1-3 tokens
- Code: usually efficient (keywords = 1 token)


#### Why Token Count Matters for You as an Engineer

```python
import tiktoken
enc = tiktoken.encoding_for_model("gpt-4o")

text = "Hello, my name is Rahul and I am an AI engineer."
tokens = enc.encode(text)
print(len(tokens))  # → 12 tokens

# GPT-4o pricing: $5 per 1M input tokens
# 12 tokens = $0.00006 — tiny, but at scale it adds up fast
```

**Cost example:**
- 1M users × 100 tokens/query = 100M tokens/day
- GPT-4o: $5/1M tokens = $500/day = $15,000/month just for input

This is why prompt compression (LLMLingua) and semantic caching matter.

---

### 🧮 Embeddings — Technical Deep Dive

#### What Is a Vector Embedding?

An embedding maps discrete tokens to continuous vector space:

```
f("king")  → ℝᵈ  (d-dimensional real vector)
f("queen") → ℝᵈ
f("dog")   → ℝᵈ
```

Where d might be 768 (BERT), 1536 (OpenAI text-embedding-3-small), or 3072 (text-embedding-3-large).

#### The Famous King - Man + Woman = Queen

```
embed("king") - embed("man") + embed("woman") ≈ embed("queen")
```

This shows embeddings capture **relationships** as directions in vector space:
- The direction "man → woman" represents gender
- The direction "king → queen" represents the same gender change
- So king + (gender flip direction) = queen

#### Cosine Similarity — How RAG Retrieval Works

```python
import numpy as np

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Embeddings of two sentences
embed_1 = embed("What is the capital of France?")
embed_2 = embed("Paris is the capital city of France.")
embed_3 = embed("I love eating pizza.")

sim_1_2 = cosine_similarity(embed_1, embed_2)  # → 0.94 (very similar)
sim_1_3 = cosine_similarity(embed_1, embed_3)  # → 0.21 (not similar)
```

RAG uses this: embed the user's question, find the most similar document chunks (highest cosine similarity), feed them to the LLM.

**Range:** -1 (opposite) to 1 (identical). In practice, text embeddings rarely go negative.

---

### 🔍 Dense vs Sparse Embeddings

| | Dense (Neural) | Sparse (BM25/TF-IDF) |
|--|--|--|
| **Representation** | Continuous vector (768–3072 dims) | Sparse vector (vocab size, mostly zeros) |
| **Captures** | Semantic meaning | Exact keyword matches |
| **Example** | "automobile" ≈ "car" | "automobile" ≠ "car" |
| **Speed** | Slower (vector math) | Faster (inverted index) |
| **Best for** | Semantic search | Keyword precision |
| **Tools** | OpenAI embeddings, BGE | BM25 (Elasticsearch, Qdrant) |

**Production RAG uses BOTH (hybrid search):**
```
Query: "What's the revenue of Apple in Q3 2024?"

Dense search → finds semantically similar docs
Sparse search → finds docs containing exact terms "Apple", "Q3 2024", "revenue"

RRF (Reciprocal Rank Fusion) → combines both rankings → better results
```

---

### 💼 Interview Questions — NLP Essentials

**Q1: What is tokenisation and why don't we just split text by spaces?**

> Tokenisation converts raw text into sub-word units that balance vocabulary size and coverage. Space-splitting fails for: contractions (don't → do/n't), compound words, multiple languages (Chinese has no spaces), rare words, emojis, and code. BPE and WordPiece use learned merge rules to create tokens that are frequent enough to be useful but small enough to handle rare words by composing them.

**Q2: What is the difference between sparse and dense embeddings?**

> Dense embeddings (from neural models like BERT or OpenAI's API) represent text as continuous low-dimensional vectors that capture semantic meaning — "car" and "automobile" have similar vectors. Sparse embeddings (like TF-IDF or BM25) represent text as high-dimensional vectors where most values are zero, and matching is based on keyword overlap. Dense is better for semantic understanding; sparse is better for exact keyword recall. Production RAG systems use both in hybrid search.

**Q3: How does cosine similarity work for semantic search?**

> Cosine similarity measures the angle between two vectors in embedding space. A score of 1 means identical direction (semantically similar), 0 means orthogonal (unrelated), -1 means opposite. In RAG, we embed the user query and all document chunks, then retrieve the top-k chunks with highest cosine similarity to the query. This enables semantic matching — "What is the capital of France?" finds "Paris is the capital city" even though no words overlap.

---

## 0.5 — Python Patterns for AI Systems

### 🧑‍🎓 Explain Like I'm in BTech 2nd Year

You know Python. You know `requests`. But AI apps have specific patterns you'll use every single day. These aren't optional — they're the patterns that make the difference between a toy and a production system.

---

### ⚡ Async/Await for LLM Calls

**Why async matters:** LLM API calls take 1–30 seconds. If you use synchronous code, your server blocks for every request. With async, handle 100 concurrent users with the same server resources.

```python
import asyncio
import httpx
from openai import AsyncOpenAI

client = AsyncOpenAI()

# ❌ WRONG — synchronous, blocks the server
def get_response_sync(prompt: str) -> str:
    response = client.chat.completions.create(...)  # blocks for 5 seconds
    return response.choices[0].message.content

# ✅ RIGHT — async, non-blocking
async def get_response(prompt: str) -> str:
    response = await client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

# ✅ EVEN BETTER — call 5 LLMs simultaneously
async def compare_models(prompt: str):
    results = await asyncio.gather(
        get_response_openai(prompt),
        get_response_claude(prompt),
        get_response_gemini(prompt),
    )
    return results  # all 3 done in parallel, takes as long as the slowest
```

**Real-life analogy:** Async is like a waiter who takes 5 orders to the kitchen at once. Sync is a waiter who stands at the kitchen watching one order cook before taking another.

---

### 🔄 Generator Functions for Streaming

LLMs stream tokens one by one. Generators are perfect for this:

```python
from openai import AsyncOpenAI

client = AsyncOpenAI()

async def stream_response(prompt: str):
    """Generator that yields tokens as they arrive"""
    stream = await client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        stream=True  # enable streaming
    )
    async for chunk in stream:
        if chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content  # yield each token

# In FastAPI:
from fastapi.responses import StreamingResponse

@app.post("/chat")
async def chat(prompt: str):
    async def generate():
        async for token in stream_response(prompt):
            yield f"data: {token}\n\n"  # SSE format
    return StreamingResponse(generate(), media_type="text/event-stream")
```

---

### 🛡️ Pydantic v2 for LLM Output Validation

LLMs return text. Pydantic turns that text into typed, validated Python objects.

```python
from pydantic import BaseModel, Field
from typing import List
from openai import OpenAI
import instructor  # pip install instructor

client = instructor.from_openai(OpenAI())

# Define the schema you want
class ResumeAnalysis(BaseModel):
    skills: List[str] = Field(description="Technical skills found in resume")
    match_score: float = Field(ge=0, le=100, description="Job match score 0-100")
    gaps: List[str] = Field(description="Skills in JD but missing in resume")
    summary: str = Field(max_length=500)

# instructor automatically retries until output matches schema
result = client.chat.completions.create(
    model="gpt-4o",
    response_model=ResumeAnalysis,  # ← this is the magic
    messages=[
        {"role": "user", "content": f"Analyse this resume: {resume_text}"}
    ]
)

print(result.match_score)  # → 78.5 (actual float, not string)
print(result.skills)       # → ["Python", "FastAPI", "Docker"]
```

---

### 🔌 Context Managers for LLM Clients

```python
from contextlib import asynccontextmanager
from openai import AsyncOpenAI
import httpx

# ✅ Proper resource management — don't create a new client per request
@asynccontextmanager
async def get_llm_client():
    async with httpx.AsyncClient() as http_client:
        client = AsyncOpenAI(http_client=http_client)
        try:
            yield client
        finally:
            pass  # cleanup happens automatically

# Better: singleton pattern for production
class LLMClientManager:
    _client: AsyncOpenAI = None

    @classmethod
    def get_client(cls) -> AsyncOpenAI:
        if cls._client is None:
            cls._client = AsyncOpenAI()
        return cls._client
```

---

## 0.6 — Intro to Agents

### 🧑‍🎓 Explain Like I'm in BTech 2nd Year

A regular chatbot is like a person who answers one question at a time and immediately forgets everything.

An agent is like an intern you give a task to:
- It thinks about what steps are needed
- It uses tools (search, calculator, file system, APIs)
- It checks its own work
- It keeps going until the task is done
- It remembers what it did earlier

**Real example:**
```
You: "Research the top 5 AI startups in India, compare their funding, 
      and create a summary report."

Regular ChatGPT: Makes things up based on training data (might be wrong)

Agent:
  Step 1: Search web for "top AI startups India 2025"
  Step 2: Read each result URL
  Step 3: Extract funding data from Crunchbase
  Step 4: Build comparison table
  Step 5: Write summary report
  Step 6: Check report quality
  Step 7: Return final report with sources
```

---

### 🔄 The ReAct Loop

**ReAct = Reason + Act**

This is the fundamental loop every agent runs:

```
┌─────────────────────────────────────────┐
│                                         │
│  THOUGHT: "I need to search for X"      │
│      ↓                                  │
│  ACTION: search_web(query="X")          │
│      ↓                                  │
│  OBSERVATION: "Results: [...]"          │
│      ↓                                  │
│  THOUGHT: "Based on results, I need Y" │
│      ↓                                  │
│  ACTION: read_url(url="...")            │
│      ↓                                  │
│  OBSERVATION: "Content: [...]"          │
│      ↓                                  │
│  THOUGHT: "I have enough info now"      │
│      ↓                                  │
│  FINAL ANSWER: "[Complete response]"    │
│                                         │
└─────────────────────────────────────────┘
```

---

### ⚙️ Agent vs Workflow vs Pipeline

This distinction matters for architecture decisions:

| | Pipeline | Workflow | Agent |
|--|--|--|--|
| **Flow** | Fixed sequence | Fixed with conditions | Dynamic, LLM decides |
| **Control** | Code | Code + conditions | LLM at each step |
| **Flexibility** | None | Limited | Full |
| **Predictability** | Fully predictable | Mostly predictable | Less predictable |
| **Debug** | Easy | Medium | Hard |
| **Cost** | Lowest | Low | Highest (more LLM calls) |
| **Example** | ETL job | if-else routing | Research agent |

**When to use each:**
- **Pipeline:** scrape → chunk → embed → store (always the same steps)
- **Workflow:** route to billing agent OR tech support (fixed conditions)
- **Agent:** "handle this customer complaint however makes sense"

---

### 🧠 Memory Types

```
┌─────────────────────────────────────────────────────────┐
│                    AGENT MEMORY                         │
│                                                         │
│  In-Context Memory (Short-term)                         │
│  ┌─────────────────────────────┐                        │
│  │ Current conversation        │ ← Lost when context    │
│  │ Tool results this session   │   window fills up      │
│  └─────────────────────────────┘                        │
│                                                         │
│  External Memory (Long-term)                            │
│  ┌─────────────────────────────┐                        │
│  │ Vector DB (Qdrant/Pinecone) │ ← Semantic recall      │
│  │ SQL DB (facts, history)     │ ← Exact recall         │
│  └─────────────────────────────┘                        │
│                                                         │
│  Procedural Memory                                      │
│  ┌─────────────────────────────┐                        │
│  │ System prompt instructions  │ ← How to behave        │
│  │ Tool schemas                │ ← What tools exist     │
│  └─────────────────────────────┘                        │
└─────────────────────────────────────────────────────────┘
```

---

### ⚠️ Why Agents Fail — What You Need to Know

| Failure Mode | Description | Fix |
|---|---|---|
| **Infinite loop** | Agent keeps calling tools, never finishes | Max iteration limit |
| **Tool hallucination** | Agent calls a tool that doesn't exist | Strict tool schema validation |
| **Context overflow** | Conversation too long, early info lost | Summarise + compress history |
| **Wrong tool selection** | Picks wrong tool for the job | Better tool descriptions |
| **Cascading errors** | One wrong step poisons all later steps | Checkpoints + validation |
| **Cost explosion** | 50 LLM calls for a simple task | Max token budget per run |

---

### 💼 Interview Questions — Agents

**Q1: What is the difference between an LLM and an agent?**

> An LLM is a model that takes input and produces output in one forward pass — a single prompt → single response. An agent is a system built around an LLM that operates in a loop: it can use tools (web search, APIs, code execution), observe results, reason about next steps, and iterate until a goal is achieved. The LLM is the brain; the agent is the complete system that enables multi-step, autonomous action.

**Q2: Explain the ReAct framework.**

> ReAct (Reason + Act) is an agent design pattern where the LLM alternates between generating a thought (reasoning about what to do) and taking an action (calling a tool). After the action, the result is observed and fed back into the context. This loop continues until the LLM determines it has enough information to give a final answer. It was introduced in a 2022 paper and is now the basis of most production agent frameworks including LangGraph.

**Q3: What are the main failure modes of AI agents?**

> The most common failures are: infinite loops (no max iteration limit), tool hallucination (calling tools that don't exist or with wrong parameters), context overflow (losing early information as the conversation grows), cascading errors (one wrong tool call corrupts downstream reasoning), and cost explosion (too many LLM calls for simple tasks). Production agents need iteration limits, output validation, context compression, and cost budgets.

**Q4: How is an agent different from a workflow?**

> In a workflow, the control flow is defined by code — conditions, if-else branches, fixed sequences. The LLM is just one node in a programmer-defined graph. In an agent, the LLM itself decides what to do next at each step — it's the orchestrator, not just a node. Agents are more flexible but harder to debug and less predictable. For well-defined tasks, workflows are better; for open-ended tasks, agents are appropriate.

---

## 📝 Interview Cheat Sheet

> Print this. Review before every interview.

### The 10 Questions You Will Always Get

| # | Question | Key Points in Answer |
|---|---|---|
| 1 | AI vs ML vs DL vs GenAI? | Nesting hierarchy, each is a subset |
| 2 | What is an LLM? | DL model, next-token prediction, self-supervised, emergent capabilities |
| 3 | How does attention work? | Q/K/V, similarity scores, weighted sum of values |
| 4 | BERT vs GPT architecture? | Encoder-only bidirectional vs decoder-only causal |
| 5 | What is tokenisation? | Sub-word units, BPE, why not spaces |
| 6 | What are embeddings? | Dense vectors, semantic similarity, cosine distance |
| 7 | What is hallucination? | Next-token prediction without fact verification, types |
| 8 | What is RLHF? | Human preferences → reward model → PPO fine-tuning |
| 9 | What is an agent? | LLM + tool use + loop + memory |
| 10 | Transformer vs RNN? | Parallel vs sequential, attention vs hidden state, no vanishing gradient |

### One-Line Answers for Speed Rounds

- **Positional encoding** → adds position info since attention is order-agnostic
- **Multi-head attention** → H parallel attention heads, each specialises differently
- **KV cache** → stores K,V matrices to avoid recomputation during decoding
- **Context window** → max tokens model can see at once
- **Fine-tuning** → continue training on domain-specific data
- **RAG** → retrieve relevant docs, inject into prompt, reduce hallucination
- **Self-attention** → tokens attend to other tokens in the SAME sequence
- **Cross-attention** → tokens in one sequence attend to ANOTHER sequence
- **Softmax** → converts raw scores to probability distribution summing to 1
- **Temperature** → controls randomness of token selection (0=deterministic, 2=very random)

---

## 🃏 Quick Revision Cards

Cut these out mentally and review daily:

---

**Card 1: Attention Formula**
```
Attention(Q, K, V) = softmax(QKᵀ / √d_k) · V

Q = Query (what am I looking for?)
K = Key  (what do I offer?)
V = Value (what info do I carry?)
√d_k = scaling to prevent gradient saturation
```

---

**Card 2: 3 Transformer Families**
```
Encoder-Only  → BERT → embeddings, classification
Decoder-Only  → GPT/Claude/Llama → generation, chatbots
Encoder-Decoder → T5/Whisper → translation, STT
```

---

**Card 3: Tokenisation Facts**
```
BPE = Byte-Pair Encoding
Common words = 1 token
Rare words = 2-5 tokens
GPT-4 vocab = ~100K tokens
Cost = per token, not per word
```

---

**Card 4: Embedding Geometry**
```
Similar meaning → Similar direction → High cosine similarity
Cosine range: -1 to 1
RAG retrieval: embed query → find nearest chunks
king - man + woman ≈ queen
```

---

**Card 5: ReAct Loop**
```
THOUGHT → ACTION → OBSERVATION → THOUGHT → ...→ ANSWER
Reason what to do → Call tool → See result → Repeat
```

---

**Card 6: Agent Failure Modes**
```
1. Infinite loop → add max iterations
2. Tool hallucination → validate tool calls
3. Context overflow → compress history
4. Cascading errors → validate each step
5. Cost explosion → set token budget
```

---

## 📚 Resources for Phase 0

| Resource | Type | Time | Priority |
|---|---|---|---|
| [Karpathy — Let's build GPT](https://www.youtube.com/watch?v=kCc8FmEb1nY) | Video | 3 hrs | 🔥 Must-watch |
| [Karpathy — Intro to LLMs](https://www.youtube.com/watch?v=zjkBMFhNj_g) | Video | 1 hr | 🔥 Must-watch |
| [Illustrated Transformer (Alammar)](https://jalammar.github.io/illustrated-transformer/) | Blog | 1 hr | 🔥 Must-read |
| [Illustrated BERT (Alammar)](https://jalammar.github.io/illustrated-bert/) | Blog | 45 min | ⭐ Recommended |
| [Attention Is All You Need (paper)](https://arxiv.org/abs/1706.03762) | Paper | 2 hrs | ⭐ Recommended |
| [tiktoken playground](https://platform.openai.com/tokenizer) | Tool | 15 min | ✅ Interactive |
| [Embedding projector](https://projector.tensorflow.org) | Tool | 20 min | ✅ Interactive |
| [HuggingFace NLP Course Ch 1-3](https://huggingface.co/learn/nlp-course/) | Course | 4 hrs | ⭐ Recommended |

---

## ✅ Phase 0 Completion Checklist

```
[ ] Can explain AI vs ML vs DL vs GenAI in 60 seconds without notes
[ ] Can draw the Transformer architecture from memory
[ ] Understand Q, K, V and can explain the attention formula
[ ] Know the 3 architecture families and one use case each
[ ] Understand tokenisation and why token count = cost
[ ] Can compute cosine similarity and explain what it means
[ ] Understand the ReAct agent loop
[ ] Can answer all 10 interview questions above
[ ] Pushed Obsidian/markdown notes to GitHub
[ ] Done: Karpathy GPT video + Illustrated Transformer blog
```

---

*Phase 0 of GenAI + LLMOps Engineering Roadmap 2026*  
*→ Next: Phase 1 — LLM Engineering Core (APIs, Prompting, MCP)*
