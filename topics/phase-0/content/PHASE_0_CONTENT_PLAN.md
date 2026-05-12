# 🎬 Phase 0 — Content Plan & Reel Guide
> **Series:** From Full-Stack to AI Engineer | Building in Public 2026  
> **Platform:** Instagram Reels + YouTube Shorts  
> **Total Reels:** 10 across 2 weeks

---

## 📅 Posting Schedule

```
Week 1
  Monday    → Reel 1: AI vs ML vs DL vs GenAI
  Thursday  → Reel 2: Traditional AI vs GenAI
  Saturday  → Reel 3: Why Attention Was Invented

Week 2
  Monday    → Reel 4: Q, K, V - The Attention Math
  Wednesday → Reel 5: Transformer Architecture
  Thursday  → Reel 6: BERT vs GPT vs T5
  Saturday  → Reel 7: Tokenisation
  
Week 3
  Monday    → Reel 8: Embeddings
  Thursday  → Reel 9: What Is an AI Agent?
  Saturday  → Reel 10: Why Agents Fail
```

**Best posting times (IST):** 7–9 AM or 7–9 PM

---

## 🎯 Series Branding

| Element | Value |
|---|---|
| **Series name** | "From Full-Stack to AI Engineer" |
| **Sub-tagline** | "Building in Public 2026" |
| **Hook formula** | Surprising claim → Why it matters → What you'll learn |
| **CTA rotation** | Save this / Follow for more / Comment X / Tag a dev friend |
| **Hashtag core** | `#aiengineering #buildinpublic #learnai #genai #python` |
| **Visual style** | Dark terminal, 22pt font, 9:16 portrait, code running live |

---

## 📋 All 10 Reels — Quick Reference

| # | Title | Hook | Duration | Notebook Cell |
|---|-------|------|----------|--------------|
| 1 | AI vs ML vs DL vs GenAI | "Everyone uses these terms wrong" | 45–60s | Reel 1 |
| 2 | Traditional AI vs GenAI | "Spam filter and ChatGPT are both AI. But different." | 50–60s | Reel 2 |
| 3 | Why Attention Was Invented | "AI literally forgot the beginning of sentences" | 55–65s | Reel 3 |
| 4 | Q, K, V Math | "3 letters run every AI tool you use" | 60–75s | Reel 4 |
| 5 | Transformer Architecture | "All built on same architecture from 2017" | 60–75s | Reel 5 |
| 6 | BERT vs GPT vs T5 | "All Transformers. Built completely differently." | 50–60s | Reel 6 |
| 7 | Tokenisation | "ChatGPT doesn't read words. It reads tokens." | 45–55s | Reel 7 |
| 8 | Embeddings | "king - man + woman = queen. AI does this math." | 60–70s | Reel 8 |
| 9 | AI Agents | "ChatGPT answers. Agent DOES things." | 60–75s | Reel 9 |
| 10 | Why Agents Fail | "5 ways AI agents break in production" | 55–65s | Reel 10 |

---

## 🎬 Reel 1 — AI vs ML vs DL vs GenAI

### Production Notes
- **Open with:** Nested diagram animation (run Reel 1 code cell)
- **B-roll idea:** Screen record of the hierarchy chart saving as PNG
- **Talking head:** After showing the diagram, look at camera, explain one line per level
- **Text overlay:** Bold each term as you say it

### Full Script
> "So everyone throws these terms around like they mean the same thing. They don't.
>
> AI is the big umbrella — any machine that seems smart. Gmail's spam filter? AI.
>
> ML is a specific approach — instead of you writing rules, you give it data and it figures out the rules. Netflix recommendations? ML.
>
> Deep Learning is ML using neural networks — layers inspired by the human brain. Face ID on your iPhone? Deep Learning.
>
> GenAI is Deep Learning that creates — text, images, code, audio. ChatGPT? GenAI.
>
> And Agentic AI — that's GenAI that doesn't just answer. It plans, uses tools, takes actions. Claude Code writing and running an entire project autonomously? Agentic AI.
>
> Save this. I'm covering all of it in depth over the next few weeks."

### Caption
```
Everyone uses these terms wrong.

The actual hierarchy:
AI -> any smart machine
ML -> learns from data
Deep Learning -> neural network layers
GenAI -> creates new content
Agentic AI -> takes real actions

Save this for the next time someone uses "AI" and "ML" interchangeably.

Documenting my Full-Stack -> AI Engineer journey. Follow along.

#aiengineering #genai #machinelearning #buildinpublic #learnai #python #llm
```

---

## 🎬 Reel 2 — Traditional AI vs GenAI

### Production Notes
- **Open with:** Side by side terminal output (run Reel 2 code)
- **Show:** Spam classifier output left, streaming response right
- **Key moment:** Slow down on "creates" vs "classifies" — this is the aha moment

### Full Script
> "Here's something that trips people up. Gmail's spam filter is AI. ChatGPT is AI. But they work completely differently.
>
> Traditional AI learns to classify — spam or not spam? Fraud or legit? Someone had to label millions of emails before it could learn.
>
> Generative AI creates. It doesn't say yes or no — it produces something entirely new. An email, an image, a piece of code.
>
> The massive shift: traditional AI needs labelled data for every single task. One model, one job.
>
> GenAI is trained once on the entire internet and can handle almost anything. One model, unlimited tasks.
>
> That's the paradigm shift that changed the industry in 2022."

### Caption
```
Gmail's spam filter and ChatGPT are both AI.
Built completely differently.

Traditional AI -> classifies (spam or not spam)
Generative AI -> creates (writes the email for you)

The paradigm shift that changed the entire industry.

Week 1 of my AI Engineering journey. Building in public.

#genai #aiengineering #chatgpt #machinelearning #buildinpublic #tech
```

---

## 🎬 Reel 3 — Why Attention Was Invented

### Production Notes
- **Open with:** The RNN left side of the chart — show memory fading
- **Reveal:** Then show attention right side — direct connection line
- **Key visual:** The green arc from "it" to "trophy" — pause here
- **Energy:** Build tension ("the problem") then resolve it ("the fix")

### Full Script
> "Before 2017, AI read text like this — one word at a time, passing a tiny memory forward.
>
> The problem? By the time it reached word 50, it had mostly forgotten word 1. The memory gets compressed and diluted with every step.
>
> Try translating a 500-word paragraph. The AI compresses everything into one small vector. By the end, the beginning is basically gone.
>
> 'The trophy didn't fit in the suitcase because it was too big.' What does 'it' refer to? The trophy, right?
>
> An RNN by the time it reaches 'it' has mostly forgotten 'trophy.'
>
> Attention fixed this. Instead of a chain, every word looks directly at every other word — simultaneously. No forgetting. No compression.
>
> That single idea, published in 2017, became the foundation of GPT, Claude, Gemini, Llama. Everything."

### Caption
```
Before 2017, AI had a memory problem.

It literally forgot the beginning of long sentences.

"The trophy didn't fit in the suitcase because it was too big"
-> RNN forgets "trophy" by the time it reaches "it"
-> Attention lets "it" look directly at "trophy" -> problem solved

That single fix became the foundation of GPT, Claude, Gemini, Llama.

Next: the actual math (Q, K, V) - simpler than you think.

#attention #transformer #llm #aiengineering #deeplearning #buildinpublic
```

---

## 🎬 Reel 4 — Q, K, V: The Attention Math

### Production Notes
- **Open with:** Print the formula bold on screen before explaining
- **Walk through:** Matrix output step by step — don't rush
- **Show:** Heatmap — point to the bright cells
- **Key moment:** Bar chart — "it" attending most to "trophy" — this is the payoff

### Full Script
> "Q, K, V — three vectors that power every transformer model.
>
> Think of it like a library. Q is your search query. K is every book title. V is the actual book content.
>
> You match your query against all the titles. You get relevance scores. Then you fetch book content weighted by how relevant each book is.
>
> In attention, this happens for every single token, attending to every other token, all at the same time.
>
> The formula: attention equals softmax of Q times K-transpose divided by root d-k, times V.
>
> Don't fear the math. It's just: similarity scores, normalise to probabilities, weighted content retrieval.
>
> Let me show you in actual code."

### Caption
```
3 letters run every AI tool you've ever used.

Q = what am I looking for?
K = what do I have to offer?
V = what information do I carry?

Library analogy:
Q = your search query
K = book titles in catalog
V = actual book content

Formula: Attention(Q,K,V) = softmax(QK^T / sqrt(d_k)) * V

#attention #qkv #transformer #aiengineering #deeplearning #math #python
```

---

## 🎬 Reel 5 — Transformer Architecture

### Production Notes
- **Open with:** The full architecture diagram rendering on screen
- **Walk through:** Point to each component as you name it
- **Highlight:** The two stacks — encoder (green border) and decoder (purple border)
- **Key stat:** "2017 paper, one architecture, powers everything"

### Full Script
> "2017. A paper called 'Attention Is All You Need' changed AI forever.
>
> Before it: RNNs — slow, sequential, and forgetful.
> After it: Transformers — parallel, scalable, and powerful.
>
> Here's what's inside every LLM you use.
>
> Input text gets tokenised and embedded. Positional encoding tells the model where each token is in the sequence.
>
> The encoder stack — 6 identical layers — builds a rich contextual understanding of the entire input.
>
> The decoder stack — 6 identical layers — generates output one token at a time. It attends to the encoder's output via cross-attention.
>
> Every layer uses that Q, K, V attention from the last reel. Residual connections and layer norm keep training stable.
>
> This is the foundation. GPT, Claude, Llama — all variations of this architecture. Everything else builds on this."

### Caption
```
2017. One paper changed AI forever.

"Attention Is All You Need"

Before: RNNs -> slow, sequential, forgetful
After: Transformers -> parallel, scalable, powerful

GPT, Claude, Gemini, Llama.
All variations of this same architecture.

Understanding this = understanding every LLM you will ever use.

#transformer #llm #aiengineering #deeplearning #buildinpublic #gpt
```

---

## 🎬 Reel 6 — BERT vs GPT vs T5

### Production Notes
- **Open with:** The 3-panel diagram side by side
- **Point:** To each one as you name it
- **Key message:** ALL major LLMs are decoder-only — this surprises people
- **Practical:** End with "so when you pick embeddings for RAG, use encoder-only"

### Full Script
> "Three types of Transformers. Same underlying architecture. Built differently for different jobs.
>
> Encoder-only — BERT, RoBERTa. Reads the entire sentence at once, bidirectionally. Best for understanding. In our AI engineering stack, this is what we use for RAG embeddings.
>
> Decoder-only — GPT-4, Claude, Llama, Gemini. Reads left to right, generates one token at a time. This is ALL major LLMs — every chatbot, every code assistant you use.
>
> Encoder-decoder — T5, BART, Whisper. Encoder reads the full input, decoder generates the output. Great for translation and speech-to-text.
>
> The practical takeaway: when you need to embed documents for RAG, use encoder-only like BERT or nomic-embed. When you're building an agent or chatbot, you're using decoder-only. Know which to choose before you start building."

### Caption
```
3 types of Transformers. Each built differently.

Encoder-Only (BERT, RoBERTa)
-> Reads everything bidirectionally
-> Use for: RAG embeddings, classification

Decoder-Only (GPT-4, Claude, Llama, Gemini)
-> Generates left-to-right
-> Use for: chatbots, code gen, ALL major LLMs

Encoder-Decoder (T5, Whisper)
-> Reads then generates
-> Use for: translation, speech-to-text

Know which to use before you build.

#bert #gpt #transformer #llm #aiengineering #rag #nlp
```

---

## 🎬 Reel 7 — Tokenisation

### Production Notes
- **Open with:** tiktoken output loading — let the numbers surprise
- **Key moment:** "antidisestablishmentarianism" = 6 tokens — pause here
- **Show:** The cost table — the numbers are alarming at scale
- **Energy:** "Practical, you need to know this" tone

### Full Script
> "ChatGPT doesn't read words. It reads tokens.
>
> Watch this. 'hello' — 1 token. 'ChatGPT' — 2 tokens. 'antidisestablishmentarianism' — 6 tokens.
>
> Tokenisation is a sub-word splitting algorithm that converts text into numeric IDs the model can process.
>
> Why does this matter to you as an engineer? Because you pay per token, not per word.
>
> A common English word is usually 1 token. A rare word might be 3-6 tokens. An emoji is 1-3 tokens. Chinese characters — multiple tokens each.
>
> At 1 million users, 100 tokens each, per day — that's 100 million tokens. GPT-4o charges $5 per million. That's $500 a day. $15,000 a month.
>
> Claude Haiku? $0.25 per million. Same traffic: $25 a day.
>
> This is why model selection and prompt compression are engineering problems, not just academic ones."

### Caption
```
ChatGPT doesn't read words.
It reads TOKENS.

'hello' = 1 token
'antidisestablishmentarianism' = 6 tokens

Why this matters:
You pay per token, not per word.

1M users x 100 tokens = 100M tokens/day
GPT-4o: $500/day = $15,000/month
Claude Haiku: $25/day = $750/month

Prompt engineering is actually a cost problem.

#tokenization #llm #promptengineering #aiengineering #genai #cost
```

---

## 🎬 Reel 8 — Embeddings

### Production Notes
- **Open with:** The cosine similarity table — green = similar, red = different
- **Big moment:** The cluster plot — words in the same category grouping together
- **Say:** "This is literally how RAG works" — make the connection explicit
- **Energy:** Mind-blowing, slow down on the vector analogy

### Full Script
> "Every word or sentence gets converted to a list of numbers — a vector. Hundreds or thousands of numbers.
>
> The magic: similar meaning gets similar numbers. Similar numbers means close together in space.
>
> 'What is the capital of France?' and 'Tell me the capital city of France' — completely different words, almost identical vectors. Watch the similarity score — 0.94.
>
> 'I love pizza' versus 'What is the capital of France?' — 0.21. Unrelated.
>
> Now watch this cluster plot. I embed 20 words from 4 categories — royalty, tech, food, AI. In 2D space, they cluster together by meaning. The model learned this just from training on text.
>
> This is literally how RAG retrieval works. Embed your question. Find the nearest document chunks. That's it. That's semantic search.
>
> And king minus man plus woman approximately equals queen. The model doesn't know this fact — it learned the relationship structure from patterns in training data."

### Caption
```
king - man + woman = queen

AI actually does this math.

Words with similar meaning -> similar vectors
Similar vectors -> close in space
Close in space -> retrieved together in search

This is how RAG works.
This is how semantic search works.
This is how AI understands language.

Watch the cluster plot - words in the same category group naturally.

#embeddings #rag #semanticsearch #aiengineering #nlp #vectors #python
```

---

## 🎬 Reel 9 — What Is an AI Agent?

### Production Notes
- **Open with:** Show a single ChatGPT-style response vs the agent loop starting
- **Real-time:** Let the agent loop run step by step — don't speed it up
- **Pause:** After each OBSERVE — let the viewer read it
- **End:** Show the final answer panel — the payoff of the loop

### Full Script
> "ChatGPT: you ask a question, it answers. Done. It immediately forgets you.
>
> An AI agent is something different. I give it one task: find the top 3 AI startups in India and their funding.
>
> Watch what happens.
>
> It thinks about what to do first. It searches the web. It reads the results. It thinks again — I need more detail. It gets details. It thinks — I have enough. It compiles the answer.
>
> This loop — Thought, Action, Observe, repeat — is called the ReAct pattern.
>
> The difference: ChatGPT makes one pass. An agent takes as many steps as the task requires.
>
> This is what LangGraph builds in production. That's Phase 4 of my roadmap — but I wanted you to see the concept early because it's what all of this is building toward."

### Caption
```
ChatGPT answers questions.
An AI Agent does your work.

THOUGHT -> "I need to search for this"
ACTION  -> search_web("top AI startups India")
OBSERVE -> results
THOUGHT -> "Now I need more detail"
ACTION  -> get_details("Sarvam AI")
ANSWER  -> Complete research report with sources

This is the ReAct loop.
This is what LangGraph builds.
This is Phase 4 of the roadmap.

#aiagents #langgraph #react #buildinpublic #aiengineering #llm #python
```

---

## 🎬 Reel 10 — Why Agents Fail

### Production Notes
- **Open with:** Warning tone — "before you build an agent, watch this"
- **Pace:** One failure mode per 8-10 seconds
- **Code:** Show the bad code briefly, then the fix — clear contrast
- **End:** Summary table — screenshot-worthy, high save rate

### Full Script
> "AI agents are powerful. But they fail in predictable ways.
>
> I hit every single one of these building production systems. Here's what to watch for.
>
> One: Infinite loops. No exit condition, the agent just keeps calling tools forever. Fix: set a max iterations limit. Return what you have when you hit it.
>
> Two: Tool hallucination. The LLM makes up tool names or calls them with wrong arguments. Fix: validate tool names and args with Pydantic before executing.
>
> Three: Context overflow. The conversation gets too long and early information falls off the context window. Fix: summarise older turns when you approach the limit.
>
> Four: Cost explosion. 50 LLM calls for a task that needed 5. Fix: set a token budget per agent run. Hard limit.
>
> Five: Cascading errors. One wrong step poisons everything downstream. Fix: validate the output of each step before continuing. Add retry with reflection.
>
> Save this. Phase 4 of my roadmap covers all of these in production systems."

### Caption
```
5 ways AI agents fail in production:

1. Infinite loop -> add max iterations
2. Tool hallucination -> validate with Pydantic
3. Context overflow -> summarise older turns
4. Cost explosion -> set token budget
5. Cascading errors -> validate each step

Save this. You will hit every single one.

Week 2 of AI Engineering complete. On to Phase 1.

#aiagents #llmops #aiengineering #langchain #langgraph #buildinpublic #python
```

---

## 🎥 Production Setup

### Equipment
```
Camera:    Phone (portrait, 1080p 60fps)
Mic:       Phone mic is fine to start, upgrade to Rode Wireless Go later
Lighting:  Ring light or natural window light to your face
Screen:    Laptop in portrait crop, or phone screencast
```

### Recording Order (efficient batch)
```
Day 1: Record Reels 1, 2, 7 (concept/comparison reels — no setup needed)
Day 2: Record Reels 3, 4 (attention — needs the charts)
Day 3: Record Reels 5, 6 (transformer — needs architecture diagram)
Day 4: Record Reels 8 (embeddings — needs cluster plot)
Day 5: Record Reels 9, 10 (agents — needs terminal running)
```

### Editing Workflow
```
1. Trim to exact duration (CapCut or DaVinci Resolve free)
2. Auto-captions ON (always — accessibility + reach)
3. Add trending audio at 5-10% volume under your voice
4. Add text overlay for the hook (first 3 seconds)
5. Export 1080x1920 (9:16)
6. Post natively to Instagram (not cross-posted — worse reach)
```

---

## 📊 Metrics to Track

| Metric | Target by Week 4 |
|---|---|
| Average watch time | > 50% |
| Saves per reel | > 30 |
| Profile visits after reel | > 100 |
| Follows per reel | > 20 |
| Comments | Respond to every single one |

**Best performing content for this niche:** Reels 1, 7, 8 (surprising facts), Reel 10 (warning/list format)

---

## 🔗 Notebook + File Map

| File | Purpose |
|---|---|
| `PHASE_0_AI_FOUNDATIONS_REELS.ipynb` | Main recording notebook — run cells live |
| `PHASE_0_AI_FOUNDATIONS.md` | Full study notes with interview prep |
| `PHASE_0_CONTENT_PLAN.md` | This file — scripts, captions, production |
| `reel1_ai_hierarchy.png` | Generated chart — use as thumbnail |
| `reel3_rnn_vs_attention.png` | Generated chart — B-roll |
| `reel4_attention_qkv.png` | Generated heatmap — B-roll |
| `reel5_transformer.png` | Architecture diagram — B-roll |
| `reel8_embeddings.png` | Cluster plot — B-roll |

---

*Phase 0 Content Plan | GenAI + LLMOps Engineering Roadmap 2026*  
*Next: Phase 1 Content Plan — Prompt Engineering + LLM APIs + MCP*
