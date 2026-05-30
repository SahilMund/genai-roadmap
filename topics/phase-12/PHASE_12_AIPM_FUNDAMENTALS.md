# 🎯 Phase 12 — AIPM Fundamentals
> **Interview Prep · Scenario Questions · Product Thinking for AI Engineers**
> Part of: GenAI + LLMOps Engineering Roadmap 2026
> Estimated time: 20–25 hrs | Learn alongside Capstone Projects
> **This phase is NOT a career change. It makes you 2x more effective as an AI engineer and dramatically better in interviews.**

---

## 📑 Table of Contents

1. [Why Engineers Need to Think Like PMs](#1-why-engineers-need-to-think-like-pms)
2. [12.1 — What AIPMs Do: The Engineer's Perspective](#121--what-aipms-do-the-engineers-perspective)
3. [12.2 — AI Product Strategy](#122--ai-product-strategy)
4. [12.3 — Eval Design Thinking](#123--eval-design-thinking)
5. [12.4 — AI Risk, Ethics & Communication](#124--ai-risk-ethics--communication)
6. [12.5 — AI Metrics That Matter](#125--ai-metrics-that-matter)
7. [12.6 — AIPM Artifacts: PRD, Spec, Incident Report](#126--aipm-artifacts-prd-spec-incident-report)
8. [🔥 Scenario-Based Interview Questions — Full Answers](#-scenario-based-interview-questions--full-answers)
9. [🎤 Behavioural Questions for AI Engineers](#-behavioural-questions-for-ai-engineers)
10. [📋 PM-Style Questions in Engineering Interviews](#-pm-style-questions-in-engineering-interviews)
11. [Quick Revision Cards](#-quick-revision-cards)
12. [AIPM Exercises](#-aipm-exercises)

---

## 1. Why Engineers Need to Think Like PMs

### 🧑‍🎓 Explain Like I'm in BTech 2nd Year

Most engineers wait for a PM to tell them what to build. The best AI engineers don't. They understand the product deeply enough to:

- Push back when the PM's plan won't work technically
- Propose solutions the PM didn't think of
- Design evals that measure what actually matters to users
- Communicate trade-offs without being dismissive of product concerns
- Get promoted faster (senior engineers own outcomes, not just tasks)

```
JUNIOR AI ENGINEER (task executor):
  PM: "Build a chatbot that answers questions about our docs."
  Engineer: builds it, ships it, done.
  
  Three months later: users complain it gives wrong answers.
  Engineer: "That's a product problem, not a technical one."
  
SENIOR AI ENGINEER (outcome owner):
  PM: "Build a chatbot that answers questions about our docs."
  Engineer: "What does success look like? What's the acceptable 
             hallucination rate? How will we measure helpfulness?
             Should we use RAG or fine-tuning? What's the 
             escalation path when the bot doesn't know?"
  
  Three months later: system has RAGAS scores, user acceptance
  rate tracked, alert when quality drops, playbook for incidents.
```

### The AIPM ↔ AI Engineer Interface

```
WHERE YOUR WORK MEETS THE PM'S DAILY:

PM's concern          →  Engineer's translation
──────────────────────────────────────────────────────────────────
"Will it hallucinate?" → faithfulness score in RAGAS eval
"How accurate is it?"  → eval on golden dataset, report pass %
"Can we ship it?"      → eval gate passes + A/B test planned
"Why is it wrong?"     → Langfuse trace → which context was retrieved
"Users are unhappy"    → acceptance rate dropped → prompt regression
"Competitor is better" → pairwise eval: our model vs theirs, score it
"Can we afford it?"    → cost per query × DAU × plan limits
"Is it safe?"          → guardrail coverage, red-team results, bias report
"How do we improve it?"→ golden dataset expansion + prompt A/B cycle
```

---

## 12.1 — What AIPMs Do: The Engineer's Perspective

### AIPM vs Regular PM — What Changes

```
REGULAR PM:
  Defines features → writes user stories → coordinates with engineering
  "Done" = feature ships, bugs are fixed
  Quality: deterministic (button works or doesn't)
  Metrics: DAU, retention, NPS, revenue

AIPM:
  Defines AI capabilities → writes evals → manages model lifecycle
  "Done" = eval passes threshold + monitored in production
  Quality: probabilistic (works 87% of the time, fails 13%)
  Metrics: task completion rate, acceptance rate, hallucination rate, cost/query

KEY ADDITIONAL RESPONSIBILITIES OF AN AIPM:
  1. Eval ownership:   who defines what "good" means and measures it
  2. Model lifecycle:  prompt changes, model upgrades, rollback decisions
  3. Risk calibration: what failure modes are acceptable, which are not
  4. Trust design:     how to show users when AI is confident vs uncertain
  5. Cost governance:  LLM cost is variable, must be tracked and budgeted

AS AN ENGINEER, YOU OFTEN DO THESE EVEN WITHOUT A PM:
  At startups, there's no dedicated AIPM — engineers own eval design.
  Even at big companies, engineers who understand PM thinking get more
  autonomy and move faster.
```

### AI Product Lifecycle — Where You Plug In

```
DISCOVERY
  PM: user research, problem definition
  Engineer: feasibility assessment, "can an LLM actually solve this?"
  Your output: technical feasibility doc with constraints + alternatives

PROTOTYPE
  PM: prototype → user feedback → pivot or proceed
  Engineer: build minimal version, instrument for evals
  Your output: working prototype + initial eval dataset (20 examples)

EVAL DESIGN
  PM: define success criteria in user terms
  Engineer: translate to measurable technical metrics
  Your output: golden dataset, RAGAS/DeepEval metrics, pass/fail thresholds

LAUNCH
  PM: phased rollout, feature flags
  Engineer: eval gate in CI/CD, A/B test routing, circuit breakers
  Your output: CI/CD pipeline with eval gate, rollback playbook

MONITOR
  PM: reads dashboards, flags issues
  Engineer: builds dashboards, sets alerts, investigates quality drops
  Your output: Langfuse dashboard, CloudWatch alarms, weekly quality report

ITERATE
  PM: prompt change based on user feedback
  Engineer: A/B test the prompt change, eval before promoting
  Your output: prompt versioning system, A/B experiment results
```

---

## 12.2 — AI Product Strategy

### Build vs Buy vs Fine-Tune — The Framework

```
THREE OPTIONS FOR EVERY AI FEATURE:

BUY (use external API as-is):
  Cost:     per-token, variable
  Speed:    days to ship
  Quality:  general-purpose model quality
  Control:  none — provider controls the model
  Use when: standard task, don't need customisation, low volume
  Example:  use GPT-4o for a one-off report generation feature
  
BUILD/FINE-TUNE (train/fine-tune your own model):
  Cost:     upfront training + serving infrastructure
  Speed:    weeks to months
  Quality:  domain-specific, can beat general models significantly
  Control:  full — you own the weights
  Use when: high volume, domain-specific needs, compliance, cost at scale
  Example:  fine-tune Llama 3.2 3B on 5K support tickets for CSA agent

RAG (augment general model with your data):
  Cost:     moderate (retrieval + LLM generation)
  Speed:    1-2 weeks
  Quality:  grounded in your data, reduces hallucination
  Control:  you control the knowledge base
  Use when: factual Q&A over proprietary docs, knowledge changes frequently
  Example:  SynapseIQ chatbot over uploaded documents

DECISION MATRIX:
  Volume    < 10K/day:          Buy
  Volume    10K-100K/day:       Buy with caching + routing
  Volume    > 100K/day:         Build/Fine-tune
  
  Domain specificity low:       Buy
  Domain specificity medium:    RAG
  Domain specificity high:      Fine-tune
  
  Compliance (data can't leave):Build/Fine-tune (self-hosted)
  
  Time to market < 2 weeks:     Buy
  Time to market < 2 months:    RAG
  Time to market 2-6 months:    Fine-tune
```

### Competitive Differentiation — AI Moats

```
WHY "WE USE GPT-4O" IS NOT A MOAT:
  Your competitor can use GPT-4o too.
  Model capabilities commoditise fast.
  In 6 months, GPT-4o-mini will outperform today's GPT-4o.

REAL AI MOATS (sustainable advantages):

1. PROPRIETARY DATA:
   If you have data competitors can't get, you can train better models.
   Example: Zepto has 2 years of Indian grocery order data.
             No competitor can buy or scrape that dataset.
             A demand prediction model trained on it is a real moat.

2. FEEDBACK LOOPS:
   More users → more feedback → better evals → better model → more users.
   Example: GitHub Copilot: millions of accept/reject signals per day
             → trains better autocomplete → more users prefer it.

3. PROPRIETARY KNOWLEDGE BASE:
   If you've spent 6 months ingesting, cleaning, and curating your
   knowledge base, competitors can't replicate that in a week.
   It's not the RAG system — it's the data curation behind it.

4. INTEGRATION DEPTH:
   Deep integration into user workflow = switching cost.
   Example: Cursor AI inside the IDE = users retrain habits.
             A generic chatbot = easy to replace.

5. DOMAIN FINE-TUNING + EVALS:
   A fine-tuned model with 200 domain-specific evals is hard to replicate.
   Competitors need the model + the evals + the training data + the infra.
   
AS AN ENGINEER:
   Push your team toward proprietary data collection from day 1.
   Every user interaction that generates labelled data is a moat asset.
   Build feedback collection into your AI features: thumbs up/down, edits, accepts.
```

### AI Feature Prioritisation — Effort vs Impact

```
EFFORT vs IMPACT MATRIX for AI features:
(Plot each feature on this 2×2 — prioritise top-right)

                    LOW EFFORT          HIGH EFFORT
                    (days-weeks)        (months)
                    ─────────────────────────────────────────
HIGH IMPACT         QUICK WINS          BIG BETS
                    Ship immediately    Plan carefully, de-risk
                    (semantic cache,    (fine-tuned model,
                     better prompt,      multimodal RAG,
                     RAGAS eval gate)    voice agent)

LOW IMPACT          FILLERS             AVOID
                    If time allows      Don't build
                    (UI polish,         (complex agent for
                     minor edge cases)   low-frequency task)

ESTIMATING AI EFFORT (specific to AI):
  Prompt engineering only:       1-3 days (existing LLM + new prompt)
  RAG pipeline from scratch:     1-2 weeks
  Semantic cache + routing:      3-5 days
  Fine-tuning pipeline:          4-6 weeks (data + train + eval + serve)
  LangGraph agent (1-2 tools):   1 week
  LangGraph agent (5+ tools):    3-4 weeks
  Voice pipeline (Pipecat):      2-3 weeks
  Full CI/CD with eval gate:     1 week
  
ESTIMATING AI IMPACT (harder — always tie to a metric):
  "Better answers" is not a metric. 
  "Task completion rate from 54% to 70%" is a metric.
  Always ask: which north star metric does this move? By how much?
```

---

## 12.3 — Eval Design Thinking

### Start With Evals — The Engineer's Responsibility

```
WRONG ORDER (most teams):
  1. Build the AI feature
  2. Test manually a few times
  3. Ship it
  4. Wait for user complaints
  5. Try to fix it without a baseline

RIGHT ORDER (what you should do):
  1. Define success criteria with PM ("80% of queries answered correctly")
  2. Build a golden dataset of 50 examples BEFORE building the feature
  3. Build the feature
  4. Run evals — if it doesn't pass, it doesn't ship
  5. Ship with monitoring (the evals run continuously in prod)

WHY THIS MATTERS:
  You can't improve what you can't measure.
  "The model got worse after the prompt change" — did it? By how much?
  Without a baseline, you're guessing.
  
  The golden dataset is also your regression test suite.
  Every time you change the prompt, the system prompt, the model,
  the chunking strategy, or the retrieval — run the golden dataset.
  If scores drop → don't ship.
```

### What "Good" Looks Like per Product Type

```
RAG CHATBOT (SynapseIQ):
  faithfulness:        > 0.75  (answer supported by retrieved context)
  answer_relevancy:    > 0.80  (answered what was actually asked)
  context_precision:   > 0.65  (retrieved chunks are relevant)
  context_recall:      > 0.70  (retrieved all relevant info)
  latency P95:         < 2s    (user-facing)
  user acceptance rate:> 65%   (thumbs up on generated answers)
  
AI-POWERED SEARCH:
  precision@5:         > 0.75  (5 of top-5 results are relevant)
  recall@10:           > 0.85  (relevant items found in top 10)
  MRR (Mean Reciprocal Rank): > 0.70 (relevant result appears near top)
  query latency:       < 200ms P95
  
CODE ASSISTANT:
  acceptance rate:     > 30%   (% of suggestions accepted by developer)
  correctness on test: > 80%   (accepted suggestions pass unit tests)
  latency to first token: < 500ms (inline suggestion must feel instant)
  hallucination rate (wrong API calls): < 5%
  
CONTENT GENERATION:
  factuality score (LLM-as-judge): > 0.80
  brand voice match (LLM-as-judge): > 0.75
  human edit rate:     < 40%   (how often human edits before publishing)
  
VOICE AGENT (VoiceIQ):
  first-call resolution: > 70%
  call completion rate:  > 80%  (customer stays on the call)
  word error rate:       < 10%  (STT accuracy)
  latency P95:          < 1.5s  (STT → LLM → TTS round trip)
  opt-out rate:         < 5%   (customers who ask to speak to human)
  
INTERVIEW AGENT (HireSense):
  interviewer-AI correlation: > 0.75 (AI score matches human score)
  completion rate:      > 80%   (candidates finish the interview)
  question quality (LLM-as-judge): > 0.80
  false positive rate:  < 15%   (strong_yes candidates who fail next round)
```

### User-Facing Quality vs Technical Metrics

```
COMMON MISTAKE:
  Engineers optimise technical metrics that don't correlate with user satisfaction.

EXAMPLE:
  Technical metric: RAGAS faithfulness = 0.88 (excellent)
  User metric:      NPS = 32, users say "answers are too long and confusing"
  
  The system is technically accurate but practically useless.
  Faithfulness measures grounding, not communication quality.

BRIDGING THE GAP:
  For every technical metric, ask: "What user behaviour does this predict?"
  
  faithfulness:    → reduces user complaints about wrong answers
  answer_relevancy:→ increases task completion rate
  context_precision:→ reduces latency (fewer irrelevant chunks)
  latency P95:     → increases retention (slow AI = user leaves)
  
  Add user-facing metrics alongside technical ones:
    thumbs up/down on every response
    edit rate (how often user edits the AI output before using it)
    re-query rate (user asked again → first answer wasn't good enough)
    session completion rate (user accomplished their goal this session)

HOW TO INSTRUMENT YOUR AI APP:
  Every response: attach a unique response_id
  Show: 👍 👎 buttons → log to Postgres + Langfuse
  Capture: did user copy the answer? did user edit it? did user re-ask?
  These signals become your training data for the next fine-tuning cycle.
```

### A/B Testing AI Features — Statistical Rigor

```
HOW TO A/B TEST A PROMPT CHANGE:

SETUP:
  Control: current prompt (v3) → 80% traffic
  Treatment: new prompt (v4) → 20% traffic
  Assignment: hash(user_id + experiment_id) → deterministic, no switching
  Duration: minimum 1 week (enough to see weekly usage patterns)
  
SAMPLE SIZE CALCULATION:
  Baseline acceptance rate: 65%
  Minimum detectable effect: 5% (want to detect 65% → 70%)
  Power: 80% (standard)
  Significance: 0.05 (standard)
  Required sample: ~1,200 per group
  At 500 queries/day on 20% traffic: 100 in treatment/day → 12 days
  
METRICS TO TRACK:
  Primary:    user acceptance rate (thumbs up / total queries)
  Secondary:  RAGAS faithfulness, answer_relevancy (from nightly eval run)
  Guardrail:  latency P95 (must not increase > 20%)
              error rate (must not increase)
  
DECISION RULES:
  Promote v4 if:  acceptance rate improvement is statistically significant (p < 0.05)
                  AND RAGAS scores don't drop > 0.02
                  AND latency doesn't increase > 20%
  
  Roll back if:   acceptance rate drops
                  OR error rate increases
                  OR latency increases > 50%
  
COMMON MISTAKES:
  Peeking: checking results daily and stopping when you like what you see
           → inflates false positive rate
           Fix: pre-commit to run duration before starting
  
  Multiple testing: testing 5 prompt variations simultaneously
                    → 5x higher chance of false positive
                    Fix: test one change at a time, or use Bonferroni correction

INTERVIEW ANSWER on A/B testing an AI feature:
  "Before A/B testing, we define our primary metric (acceptance rate),
   calculate required sample size for 80% power, and pre-commit to run
   duration. We never stop early just because one variant looks better.
   We also track guardrail metrics — if the 'better' prompt causes latency
   to spike, we don't promote it regardless of the primary metric win."
```

---

## 12.4 — AI Risk, Ethics & Communication

### Communicating AI Limitations to Stakeholders

```
THE CARDINAL RULE:
  Never say "the AI will be X% accurate."
  You don't know. The model is probabilistic. Users will hold you to it.

WHAT TO SAY INSTEAD:

"Will it hallucinate?"
  ❌ "No, we've prevented hallucination."
  ✅ "On our golden dataset, the system answers correctly 87% of the time.
      For the remaining 13%, it either says it doesn't know (8%)
      or gives a partially incorrect answer (5%). We have human review
      as a fallback for high-stakes queries."

"Is it ready to ship?"
  ❌ "Yes, it works well."
  ✅ "It passes our eval gate: RAGAS faithfulness > 0.75, answer relevancy > 0.80.
      We've run a 500-query red-team test with 91% block rate on adversarial inputs.
      We recommend A/B testing at 10% traffic for 2 weeks before full rollout."

"Why did it give a wrong answer?"
  ❌ "The model made a mistake."
  ✅ "Looking at the Langfuse trace: the retrieved context didn't contain
      the relevant information [show the trace]. The retriever ranked an
      older version of the document higher. We can fix this by updating
      the knowledge base and adding a context quality gate."

"Can we use this for [sensitive use case]?"
  ❌ "Sure, the model is smart enough."
  ✅ "That use case may fall under the EU AI Act's high-risk category.
      We'd need: human oversight mechanism, audit logs, bias testing,
      and potentially a DPIA. I'd recommend we schedule a risk review
      with legal before proceeding."
```

### AI Incident Handling

```
SCENARIO: Users report the chatbot gave confident but wrong legal advice.

IMMEDIATE RESPONSE (first 2 hours):
  1. Kill switch: disable the feature or route to human agents
     (never argue about whether it's really wrong — act first)
  2. Scope: how many users were affected? Pull Langfuse traces for last 24h
  3. Contain: identify if this is a systematic failure or isolated
  4. Communicate: internal Slack to product, legal, leadership within 1 hour

ROOT CAUSE ANALYSIS (next 24 hours):
  Pull Langfuse traces for the bad responses
  Questions:
    Was it hallucinating (saying things not in context)?
    Was the context wrong (retriever pulled old/incorrect doc)?
    Was the prompt too permissive ("answer any question confidently")?
    Was there a knowledge base update that introduced bad data?
    Was this a jailbreak? (user manipulated the model)
  
FIXES (in priority order):
  1. Knowledge base: remove/correct incorrect documents
  2. Prompt: add "Only provide general information, not legal advice.
              Recommend consulting a qualified lawyer for legal matters."
  3. Guardrail: add "legal advice" to RestrictToTopic blocked categories
  4. Eval: add the bad query to golden dataset (negative example)
  5. Monitor: set CloudWatch alarm if "legal" appears in response text

POSTMORTEM TEMPLATE:
  What happened:   chatbot gave incorrect legal interpretation to ~120 users
  Impact:          potential reputational risk, no confirmed user harm
  Root cause:      RAG retrieved outdated compliance document (2023 version)
                   Knowledge base update process had no version check
  Timeline:        [incident] [detected] [contained] [fixed]
  Fix applied:     updated KB, added legal advice guardrail, improved KB versioning
  Preventive:      add "document date" to all KB chunks, alert on outdated docs
  Metrics after:   0 legal advice responses in 500-query test post-fix
```

### Bias as a Product Risk

```
THREE TYPES OF AI BIAS THAT BECOME PRODUCT PROBLEMS:

1. REPRESENTATION BIAS:
   Model trained on English data → poor quality on Hindi queries
   Impact: product doesn't work for Hindi-speaking users
   Fix: eval in multiple languages, fine-tune on Hindi data, use multilingual models

2. DEMOGRAPHIC BIAS IN DECISIONS:
   HireSense resume screener scores women lower in STEM roles
   Impact: legal liability, reputational damage, regulatory investigation
   Fix: SHAP analysis to verify gender not correlated with score,
        demographic parity check (selection rate similar across groups)

3. CONFIRMATION BIAS IN AI:
   Recommendation AI keeps showing users what they've seen before
   Impact: filter bubble, user trust erodes over time
   Fix: inject diversity, track long-term engagement not just click-through

HOW TO COMMUNICATE BIAS RISK TO A PM:
  "Our resume screener has a 12% selection rate for women vs 18% for men
   in software engineering roles. This is a 6 percentage point disparity.
   Under EU AI Act (high-risk AI), this requires a corrective action before
   we can deploy for hiring decisions. I recommend we retrain with balanced
   data and add a fairness constraint to the model."

HOW TO NOT COMMUNICATE IT:
  "The model is biased" → PM panics, feature gets cancelled
  "We're working on fairness" → PM doesn't understand the urgency
```

### User Trust Design

```
APPROPRIATE TRUST CALIBRATION:
  Over-trust:  user believes AI without question → dangerous
  Under-trust: user ignores AI → why build it?
  
  Goal: user trusts AI for what it's good at,
        appropriately sceptical where it might be wrong

DESIGN PATTERNS FOR APPROPRIATE TRUST:

1. CONFIDENCE INDICATORS:
   Show when AI is uncertain: "I'm not certain about this — please verify."
   Don't fake confidence: never "The answer is X" when RAGAS score is 0.55

2. SOURCE CITATIONS:
   "According to [Return Policy 2026, Section 3.2]..."
   User can verify → builds trust through transparency

3. GRACEFUL FAILURE:
   Better to say "I don't know" than give a wrong answer confidently
   Add clear escalation: "Would you like to speak to a human agent?"

4. AUDIT TRAIL (for professional users):
   "View the context used to generate this answer"
   Link to Langfuse trace (internal) or simplified context view (external)

5. AI DISCLOSURE:
   EU AI Act: chatbots must disclose they are AI
   Best practice: always disclose, even if not legally required
   Design: "I'm an AI assistant. For complex issues, I'll connect you to a human."
```

---

## 12.5 — AI Metrics That Matter

### North Star Metrics for AI Products

```
METRIC 1: TASK COMPLETION RATE
  Definition: % of sessions where user accomplished their stated goal with AI help
  Why it matters: the only metric that proves the AI is actually useful
  How to measure: track conversation → end state → did user get what they needed?
  
  Implementation:
    After each session: optional micro-survey "Did you find what you were looking for?"
    Or: infer from behaviour — did they re-ask the same question? (no = success)
    Or: track next action — did user proceed to checkout after product question? (yes = success)
  
  Target: > 65% for a new AI feature (most queries resolved without human)
  Red flag: < 40% → AI is not providing value, reconsider the use case

METRIC 2: AI ACCEPTANCE RATE
  Definition: for suggestions/completions/drafts — what % does the user accept as-is?
  Why it matters: measures real-world quality better than any technical metric
  
  Code assistant: accept without modification → 30%+ is good
  Email draft:    send with < 20% edits → 50%+ is good
  RAG answer:     thumbs up → 65%+ is good
  
  Low acceptance rate signal: answer is wrong, too long, wrong tone, or irrelevant

METRIC 3: TIME-TO-VALUE
  Definition: how quickly does the AI produce useful output? (not just latency)
  Why it matters: P95 latency is technical; time-to-value is business
  
  Examples:
    Old: analyst spends 3 hours writing a report → New: AI draft in 5 minutes
    Old: customer waits 8 minutes for support → New: AI resolves in 45 seconds
    
  Measure: compare task duration with AI vs without
  Report: "AI reduced average support resolution time from 8 min to 45 sec (89%)"

METRIC 4: COST PER SUCCESSFUL INTERACTION
  Definition: total LLM cost ÷ number of successful completions (task completion = true)
  Why it matters: cost per query is only half the story — if AI fails 40%, cost is misleading
  
  Example:
    LLM cost per query: $0.005
    Task completion rate: 60%
    Cost per successful interaction: $0.005 / 0.60 = $0.0083
    
    After improvement: completion rate → 80%
    Cost per successful interaction: $0.005 / 0.80 = $0.0063 (cheaper!)
    
  This shows that improving quality often improves unit economics too.

METRIC 5: ESCALATION RATE
  Definition: % of sessions where AI fails and human takes over
  Why it matters: AI escalation = AI failure = human cost you're paying for AI's shortcomings
  
  Formula: escalations / total AI sessions × 100
  Target: < 20% for general support chatbot
         < 10% for well-scoped AI feature
  
  High escalation rate signals:
    Scope too broad (AI asked to handle things it can't)
    Knowledge base missing key information
    Guardrails too aggressive (blocks legitimate queries)
```

### Leading vs Lagging Indicators

```
LAGGING INDICATORS (what happened — react after the fact):
  NPS, CSAT, user churn, revenue impact
  Problem: by the time you see it, 10,000 users had a bad experience
  
LEADING INDICATORS (what's about to happen — act before it does):
  RAGAS faithfulness score (quality leading indicator)
  P95 latency (performance leading indicator)
  Escalation rate (satisfaction leading indicator)
  Daily A/B test acceptance rate (quality leading indicator)
  
EXAMPLE LEADING INDICATOR SYSTEM:
  
  Day 1: RAGAS faithfulness drops from 0.84 to 0.76
  → Alert fires (threshold: < 0.78)
  → Investigate: which queries started failing?
  → Find: knowledge base update 2 days ago introduced bad document
  → Fix: remove document, re-eval
  → Day 3: faithfulness back to 0.84
  
  Without leading indicators:
  Day 1: bad document enters KB
  Day 7: user NPS drops, complaints spike
  Day 14: leadership asks why AI quality dropped
  Day 21: root cause found
  Damage: 2 weeks of bad AI answers reaching users
  
HOW TO BUILD LEADING INDICATOR DASHBOARD:
  CloudWatch: P50/P95/P99 latency (alarm if P95 > 3s for 2 periods)
  Langfuse:   RAGAS scores (run nightly, alert if < threshold)
  Redis:      acceptance rate (compute rolling 4hr window, alert if drops 10%)
  Postgres:   escalation rate (count per hour, alert if spikes)
  Sentry:     error rate (immediate alert on new error types)
```

---

## 12.6 — AIPM Artifacts: PRD, Spec, Incident Report

### PRD Template for an AI Feature

```markdown
# AI Feature PRD: [Feature Name]

## Problem Statement
What user pain are we solving? What does the user currently have to do?
Quantify: X% of users do Y manually, taking Z hours per week.

## Proposed Solution
One sentence: "An AI [feature type] that [does what] for [who] so they can [outcome]."

## Success Metrics
Primary:   Task completion rate → target X%  (baseline: Y%)
Secondary: AI acceptance rate   → target X%
           Latency P95          → target Xms
           Cost per interaction → target $X
Guardrail: Error rate           → must not exceed X%
           Escalation rate      → must not exceed X%

## Non-Goals (explicitly out of scope)
- [Thing people will ask for but we're not building]
- [Adjacent feature that sounds related but is separate]

## Constraints
- Latency budget: < 2s P95 (user-facing, synchronous)
- Cost budget: < $0.01 per query
- Compliance: PII must not be sent to external LLM APIs
- Data residency: EU users' data must stay in eu-west-1

## AI Approach
Model: GPT-4o-mini / Claude Haiku / fine-tuned Llama 3.2
Method: RAG / fine-tuning / prompt engineering
Why this approach: [1-2 sentences justifying the choice]
Alternatives considered: [and why rejected]

## Eval Plan
Golden dataset: 50 (question, context, ground_truth) triples by [date]
Metrics: RAGAS faithfulness > 0.75, answer_relevancy > 0.80
Eval gate: CI/CD blocks deploy if metrics drop below threshold
A/B test: 10% traffic → full rollout after 2 weeks if primary metric passes

## Rollback Plan
Trigger: acceptance rate drops > 10% in any 4-hour window
Action: revert to previous prompt version (PromptRegistry meta.json)
Time to rollback: < 5 minutes (automated via feature flag)

## Open Questions
- [ ] [Question for PM to resolve before engineering starts]
- [ ] [Risk to validate in prototype phase]
```

### AI Feature Spec (Engineer's Version of PRD)

```markdown
# AI Feature Spec: [Feature Name]
> For: Engineering team | By: [Name] | Date: [Date]

## What We're Building
[One paragraph — what the system does, not how]

## Success Definition
MUST HAVE (ship is blocked if these fail):
  - RAGAS faithfulness > 0.75 on 50-example golden dataset
  - P95 latency < 2s
  - Guardrail block rate > 90% on red-team suite

NICE TO HAVE (target but not blocking):
  - User acceptance rate > 65%
  - Context precision > 0.70

## Technical Approach
  LLM:          [model + provider + why]
  Retrieval:    [vector store + chunking + embedding]
  Guardrails:   [input validators + output validators]
  Caching:      [L1 exact + L2 semantic — estimated hit rate X%]
  Async:        [what goes in queue vs synchronous]

## Eval Plan (written BEFORE building)
  Golden dataset: [who builds it, how many examples, by when]
  RAGAS eval:     [script path, run in CI/CD]
  LLM-as-judge:   [custom metric for domain-specific quality]
  Red-team:       [attack types to test, target block rate]

## Data Flow Diagram
  User input → Guardrail → Retrieval → LLM → Output Guardrail → User

## Monitoring
  Langfuse: trace every LLM call (model, tokens, cost, latency)
  CloudWatch: P95 alert > 3s, error rate alert > 2%
  Redis:    acceptance rate rolling 4h window, alert if drops 10%

## Rollback Plan
  Prompt change: revert meta.json → immediate effect (no redeploy)
  Model change: ECS rollback → previous task definition (< 5 min)
  Full disable: feature flag → graceful degradation message

## Risks
  [Risk]: [Mitigation]
  Hallucination on edge cases: add edge cases to golden dataset
  Latency spikes: circuit breaker + fallback to Groq
  Cost overrun: per-user daily limit + budget alarm
```

### Prompt Changelog — Document Every Change

```markdown
# Prompt Changelog: rag_system

## v4 — 2026-03-15 — Sahil Mund
  Status:    AB testing (20% traffic)
  Changed:   Added explicit citation format requirement
             Added multilingual support (Hindi + English)
             Reduced word limit from 300 to 200 words
  Why:       Users complained answers were too long (CSAT feedback 2026-03-10)
             Hindi users got English responses despite Hindi queries
  Eval:      faithfulness 0.84 → 0.86 (+0.02), answer_relevancy 0.87 → 0.85 (-0.02)
  Decision:  Monitoring A/B. Acceptance rate data needed before promoting.

## v3 — 2026-01-20 — Sahil Mund  [CURRENT]
  Status:    Production (80% traffic)
  Changed:   Added context quality gate ("If not in context: say so")
             Added source citation requirement
  Why:       faithfulness was 0.71 (below 0.75 threshold) in v2
  Eval:      faithfulness 0.71 → 0.84 (+0.13) — passed gate
  Decision:  Promoted to production.

## v2 — 2025-12-10  [DEPRECATED]
  Status:    Deprecated
  Issue:     faithfulness 0.71 (below threshold). Hallucinated on ambiguous queries.

## v1 — 2025-11-01  [DEPRECATED]
  Status:    Deprecated
  Issue:     No citation requirement. Users couldn't verify answers.
```

### AI Incident Report Template

```markdown
# AI Incident Report: [Incident Name]
  Severity:   P1 (user harm) / P2 (quality degradation) / P3 (performance)
  Date:       [Date]
  Duration:   [Start] → [Resolved]
  Author:     [Name]

## What Happened (2-3 sentences, non-technical)
The AI chatbot provided incorrect refund eligibility information to approximately
240 users between 14:00 and 16:30 on March 12th. Users were told they could
return sale items, when our policy explicitly excludes them. No financial harm
occurred — no refunds were processed based on this incorrect information.

## Impact
  Users affected:  ~240 (estimated from Langfuse trace count)
  Financial:       No direct financial impact (no refunds processed)
  Reputational:    3 Twitter complaints, 1 Trustpilot review
  Customer support: 18 escalation calls received

## Timeline
  14:00:  Knowledge base updated with new sale policy document
  14:05:  Chatbot starts serving incorrect information
  16:15:  Support team notices spike in refund-related calls
  16:22:  Alert fired: escalation rate > 20%
  16:25:  Engineering paged
  16:30:  Feature disabled, human agents handling all queries
  17:45:  Root cause identified
  18:30:  Fix deployed, feature re-enabled
  18:45:  Verified 0 incorrect responses in 100-query test

## Root Cause
  The knowledge base update process did not validate document consistency.
  The new "Sale Terms 2026" document contradicted the existing "Return Policy 2026"
  document. The retriever served both documents for return-related queries, and the
  LLM resolved the contradiction in favour of the newer (incorrect) document.

## Why We Didn't Catch It
  The eval suite did not include a test case for "sale item return policy."
  The knowledge base update had no automated conflict detection.
  The escalation rate alert threshold was 25% (should have been 15%).

## Fix Applied
  Immediate:  Removed contradictory sale terms document from KB
  Short-term: Added 5 sale item return test cases to golden dataset
  Long-term:  Built KB conflict detector (new doc vs existing on same topic)
              Lowered escalation rate alert threshold from 25% to 15%
              Added "knowledge base validation" step to update pipeline

## Preventive Measures
  [ ] Automated KB conflict detection on every document update
  [ ] Eval suite expanded by 15 sale-related test cases
  [ ] Alert threshold lowered: escalation rate > 15% → page immediately
  [ ] Runbook created: "KB update causing incorrect answers" → steps to fix
```

---

## 🔥 Scenario-Based Interview Questions — Full Answers

These are the questions companies actually ask in AI PM / Senior AI Engineer interviews. Read each question, think for 60 seconds, then read the answer.

---

### SCENARIO 1 — The Quality Drop

> **"Your RAG chatbot's CSAT score dropped from 4.2 to 3.7 over the past two weeks. You don't have Langfuse set up. How do you diagnose and fix it?"**

```
APPROACH: structured debugging without instrumentation, then fix the process.

STEP 1: GATHER EVIDENCE
  Pull user feedback/support tickets from those 2 weeks.
  Group into categories: "wrong answer", "too long", "off-topic", "slow".
  Also: what changed 2 weeks ago? (prompt change? model upgrade? knowledge base update?)
  Look at git log for prompts/ directory and KB update logs.

STEP 2: REPRODUCE
  Take the top 20 complaints and manually run those exact queries.
  Do you see the same bad behaviour? If yes: it's systematic.
  If no: it may be edge cases or user expectation mismatch.

STEP 3: IDENTIFY ROOT CAUSE
  Wrong answer → faithfulness issue → RAG retrieval problem or hallucination
  Too long     → output constraint removed or model changed
  Off-topic    → guardrail regressed or user base changed
  Slow         → provider latency spike or new bottleneck

STEP 4: QUICK FIX
  Whatever changed 2 weeks ago → revert it → check if CSAT recovers.
  If nothing changed → something external changed (model version silently updated?
  Provider changed default parameters? Knowledge base naturally drifted?)

STEP 5: FIX THE PROCESS (so this never happens again)
  Set up Langfuse TODAY — before doing anything else.
  Create a 50-example golden dataset from the complaint queries.
  Add RAGAS eval to CI/CD — blocks any future change that drops quality.
  Set up daily RAGAS dashboard — you'll catch this in day 1 next time.

THE INTERVIEWER IS ALSO TESTING:
  Do you jump to "fix it" without diagnosing? (bad sign — guessing)
  Do you recognise this is a process failure, not just a product failure?
  Do you understand that without observability you're flying blind?

ANSWER SUMMARY: "First I'd establish a timeline — what changed two weeks ago.
Then reproduce the bad behaviour manually. Then fix the immediate issue.
But the real fix is preventing recurrence: Langfuse setup, golden dataset,
eval gate in CI/CD. The CSAT drop is the symptom — the missing observability
is the disease."
```

---

### SCENARIO 2 — The Stakeholder Ask

> **"A VP comes to you and says: 'I want our AI to be 100% accurate — no hallucinations.' How do you respond?"**

```
DON'T SAY: "That's not possible." (confrontational, unhelpful)
DON'T SAY: "Sure, we'll work on it." (impossible promise)

SAY THIS:

"I completely understand why 100% accuracy is the goal — every wrong answer
 erodes user trust and creates support costs. Let me explain how we think
 about this technically, and then propose a practical path forward.

LLMs are probabilistic — they don't work like a database where you get
exactly what you stored. Our current system answers correctly 87% of the time
based on our eval suite. For the remaining 13%:
  8% of the time it correctly says 'I don't know'
  5% of the time it gives a partially correct answer

For the 5% that are wrong, we have three strategies:

1. REDUCE IT: expand the knowledge base with more documents,
   improve retrieval quality, add a context quality gate that
   refuses to answer when retrieved context is too weak.
   This can get us from 5% wrong to maybe 2%.

2. CONTAIN IT: every answer shows source citations.
   Users can verify. For high-stakes decisions, we show
   'Always verify this with a qualified [professional].'

3. MONITOR AND CATCH IT: with our current eval system,
   if wrong answer rates spike, we get alerted in 4 hours.
   We can disable the feature faster than the damage compounds.

What I'd push back on: '100% accurate AI' is a very different
product than a fast, helpful AI. Achieving near-100% would require
human review of every response, removing most of the speed advantage.

What I'd recommend: define which categories of errors are unacceptable
(legal advice, medical info, financial decisions) and put hard guardrails
on those. For general Q&A, target 95%+ correct and monitor continuously.

Would that framing work as a starting point for our roadmap?"

WHY THIS ANSWER WORKS:
  Shows you understand the technical reality without dismissing the business concern
  Offers a specific number (87% current baseline)
  Proposes three concrete paths forward
  Asks for clarification on which errors matter most (shows product thinking)
  Doesn't promise something you can't deliver
```

---

### SCENARIO 3 — The Build Decision

> **"The team wants to build a document Q&A chatbot. Should you use RAG or fine-tune a model?"**

```
ANSWER: It depends — but here's how to decide in under 5 minutes.

ASK THESE QUESTIONS:

1. How often does the knowledge change?
   Changes daily/weekly → RAG (knowledge base is easy to update)
   Mostly static (style guide, brand voice) → fine-tune possible

2. How large is the knowledge base?
   < 1000 pages:  RAG (fast to set up, sufficient)
   > 10K pages:   RAG (fine-tuning can't hold all facts in weights anyway)
   Fine-tuning doesn't store facts well — it learns patterns, not specific answers

3. What's the primary failure mode?
   "Answers with information not in our documents" → RAG is the fix
   "Answers in wrong tone/format" → fine-tune or prompt engineering
   "Answers incorrectly about our specific domain" → RAG + more data in KB

4. What's the timeline and budget?
   2 weeks, < $5K → RAG (minimal infra, use existing LLM)
   3 months, $50K+ → fine-tuning feasible if justified

5. Compliance?
   Can't send documents to external API → self-hosted fine-tuned model
   Otherwise → RAG with external LLM (cheaper, faster, better quality)

FOR DOCUMENT Q&A SPECIFICALLY:
  RAG wins 95% of the time because:
  • Documents change (new policies, updated products)
  • Fine-tuned model can't easily "forget" old facts when docs update
  • RAG is grounded — cites source, reduces hallucination
  • RAG is cheaper at inference time (no fine-tuned model to host)

WHEN TO CONSIDER FINE-TUNING ON TOP OF RAG:
  Style/tone mismatch: model doesn't write in our brand voice
  Format issues: model doesn't follow our response format even when prompted
  Latency: fine-tuned small model + RAG can be faster than large model + RAG
  Cost: at 1M queries/day, fine-tuned small model is dramatically cheaper

RECOMMENDATION STRUCTURE FOR INTERVIEW:
  "I'd start with RAG because [reasons]. We validate it meets quality
   thresholds. If we see [specific failure mode], we'd add fine-tuning
   for [specific behaviour]. My decision criteria are: knowledge update
   frequency, compliance constraints, timeline, and volume."
```

---

### SCENARIO 4 — The Regression

> **"You upgraded from GPT-3.5 to GPT-4o to improve answer quality. User acceptance rate dropped from 68% to 55%. What happened and what do you do?"**

```
COUNTERINTUITIVE BUT COMMON SCENARIO.
GPT-4o is "smarter" but slower and more verbose — both can hurt UX.

POSSIBLE CAUSES:

1. LATENCY REGRESSION:
   GPT-3.5: P95 800ms. GPT-4o: P95 2.8s.
   Users see a 3-second wait where they used to see < 1 second.
   Even if answers are better, the perceived slowness hurts satisfaction.
   Check: compare P95 latency before and after upgrade.

2. VERBOSITY:
   GPT-4o gives longer, more nuanced answers.
   For a support chatbot, a 400-word response is worse than a 80-word one.
   Users wanted the quick answer, not the comprehensive one.
   Check: average output token count before vs after.

3. TONE SHIFT:
   GPT-4o has a slightly different personality (more formal, more caveats).
   Support chatbot that was friendly is now clinical.
   Check: human review 20 before vs 20 after responses side by side.

4. OVER-REFUSALS:
   GPT-4o has stricter content policy — may refuse or hedge more.
   "I'm not able to provide specific legal/medical/financial advice..."
   Users wanted a direct answer, got a disclaimer.
   Check: count responses containing "I'm not able to" before vs after.

WHAT TO DO:

Immediate: A/B test GPT-3.5 vs GPT-4o on 50% traffic.
           Measure acceptance rate and latency side by side.

If latency is the issue:
  → Keep GPT-4o but add streaming (users see first tokens in 400ms)
  → Or: use GPT-4o-mini (faster, cheaper, often comparable quality)
  → Or: use Groq as LLM provider for GPT-compatible API at lower latency

If verbosity is the issue:
  → Add explicit output constraint: "Answer in 2-3 sentences maximum."
  → Set max_tokens=150

If tone is the issue:
  → Update system prompt to specify tone explicitly

If over-refusals are the issue:
  → Adjust system prompt to be more permissive for your use case
  → Add examples of desired refusal behaviour in few-shot

LESSON:
  Model upgrades are not automatically improvements for your product.
  Always eval on your specific use case before upgrading in production.
  "Smarter" model ≠ better product metrics.
  
  "Before upgrading from GPT-3.5 to GPT-4o, I would have run a silent
   A/B test — 10% traffic to GPT-4o, 90% on GPT-3.5, measure acceptance
   rate and latency for 1 week. If metrics matched or improved, promote.
   We didn't do that here, so now we're diagnosing in production."
```

---

### SCENARIO 5 — The Cost Problem

> **"Your LLM bill is $45,000 this month. Your budget was $8,000. The CEO is asking you to explain it in your next meeting. Walk me through your approach."**

```
STRUCTURE: understand → diagnose → fix → prevent.

BEFORE THE MEETING (30 minutes of analysis):

1. UNDERSTAND THE SCALE:
   $45K / 30 days = $1,500/day
   Budget: $8K / 30 = $267/day
   We're spending 5.6x budget. That's not a pricing miscalculation — that's a bug or abuse.

2. PULL THE DATA:
   From CloudWatch / cost tracking: cost per day chart
   Did it start high and stay high? (launch without cost controls)
   Did it spike suddenly mid-month? (bug, abuse, new feature gone wrong)
   From your cost tracker: breakdown by org_id, feature, model

3. COMMON CULPRITS AT 5X:
   A: No per-user limits — one enterprise customer ran a batch job that
      called the API 500K times over a weekend
   B: A background process was running in a loop (bug → infinite retries)
   C: A new feature launched without cost estimation — report generation
      feature sends 15K tokens per request, 5x the estimated 3K
   D: No caching — every identical query hits the LLM (semantic cache would save 35%)
   E: Wrong model — using GPT-4o for classification tasks that need GPT-4o-mini

IN THE CEO MEETING:

"Here's what we found: our cost was $45K vs $8K budget.
After analysis, 80% of the overage came from two sources:
  1. Our largest enterprise client ran an automated batch job over the weekend
     with no rate limiting — 300K API calls at $0.011 each = $3,300 from one org.
  2. The new report generation feature averages 14K tokens per report instead
     of the projected 3K — 4.7x our estimate.

Both are fixable immediately:
  Rate limits: per-org daily spending caps deploying today.
               Enterprise client will be capped at $500/day.
  Report cost: switching report generation to Claude Haiku + prompt compression
               → reduce per-report cost from $0.17 to $0.03 (82% reduction).

Going forward:
  Every feature launch requires a cost estimate sign-off.
  I'm adding a CloudWatch alarm: alert at 50% of monthly budget.
  Daily cost breakdown by org and feature in our internal dashboard."

DON'T SAY IN THE MEETING:
  "LLMs are expensive, this is normal." (No — $45K vs $8K is not normal)
  "We don't know why." (Unacceptable — dig deeper before the meeting)
  "We'll look into it." (Look into it first, then have the meeting)
```

---

### SCENARIO 6 — The Feature Request

> **"A PM asks you to add 'AI autocomplete' to your company's search bar. How do you respond?"**

```
THIS IS NOT A YES/NO QUESTION. It's a conversation.

YOUR FIRST RESPONSE:
"Before we say yes or no, I have a few questions to make sure
 we're solving the right problem in the right way."

THE 5 QUESTIONS:

1. WHAT PROBLEM ARE WE SOLVING?
   "Is the search bar currently returning irrelevant results?
    Or are users abandoning search because they don't know how to phrase their query?
    Or is there a conversion drop between search and actually finding something?"
   (The PM might not actually want autocomplete — they want to fix a specific drop-off)

2. WHAT DOES SUCCESS LOOK LIKE?
   "What metric improves if we ship this?
    Query reformulation rate? Search-to-click rate? Conversion from search page?"
   (Without a metric, you can't know if it worked)

3. WHO USES THE SEARCH BAR AND HOW?
   "What's the current query length? Do users type partial words or full questions?
    What's the p50 query length?"
   (If p50 query = 2 words, autocomplete won't help much — queries are already short)

4. WHAT'S THE TECHNICAL APPROACH?
   "I see 3 options:
    A: Query completion (predict next word, fast, simple, deterministic)
    B: Query suggestion (suggest alternative phrasings using LLM, slower, more useful)
    C: Semantic search improvement (don't change the UI, improve the backend)
    Option C might solve the same problem with no UI change and 2x less effort."

5. WHAT ARE THE RISKS?
   "Autocomplete surfaces queries users haven't thought of — some might be weird or
    inappropriate. Do we have guardrails on what autocomplete suggests?
    Also: latency. Autocomplete must respond in < 100ms or users type faster
    than the suggestions appear. How do we handle that?"

THEN GIVE YOUR RECOMMENDATION:
"Given the search-to-click rate is the main problem, I'd recommend we first
 look at improving search relevance (semantic search improvements) since that
 addresses the root cause without UI complexity. If the PM's main goal is
 discoverability — helping users who don't know what to search for — then
 query suggestion is the right feature. Let me estimate both and we can decide."

WHY THIS ANSWER WINS:
  Shows product thinking (what problem, what metric, what user)
  Challenges the frame without dismissing the idea
  Proposes alternatives (shows you're thinking beyond execution)
  Identifies risks before they become incidents
  Ends with a concrete next step
```

---

### SCENARIO 7 — Hallucination Reached a User

> **"A user posts on Twitter: 'Your AI told me I had 60-day returns, I tried to return something after 45 days and was denied. This is fraud.' How do you handle this?"**

```
THIS REQUIRES: public response + internal investigation + fix + prevention.

IMMEDIATE (within 1 hour):

1. TWITTER RESPONSE (PM + legal draft, you validate technically):
   "We're really sorry this happened. We're investigating right now.
    Can you DM us your order details? We want to make this right for you
    and understand what went wrong with our AI."
   (Don't admit fraud, don't deny. Empathise, investigate, resolve)

2. INTERNAL INVESTIGATION:
   Pull Langfuse trace for the conversation (search by user ID or session time)
   Was the AI given context that said "60 days"? → knowledge base error
   Did it say 60 days without any context? → hallucination
   Was this a one-off or systematic? (search all traces for "60 day return")

3. CUSTOMER RESOLUTION:
   Regardless of technical fault — make the customer whole.
   If they were denied a return they expected: process the return.
   Cost: one refund. Value: customer goodwill, avoid PR escalation.

WITHIN 24 HOURS — ROOT CAUSE:

If hallucination:
  Add to golden dataset as negative example
  Add output guardrail: never state specific return windows without citation
  Update eval: any response with "X-day return" must cite a document source

If knowledge base error:
  Old document said 60 days (it was updated to 30 days)
  KB update didn't include old document removal
  Fix: add "valid_from/valid_to" dates on all policy documents
       Add KB validation: when new policy uploaded, flag conflicting old docs

PREVENTION (next sprint):
  Add eval: "What is the return policy?" on golden dataset — must cite correct days
  Add guardrail: responses about return periods must include document citation
  Add monitoring: alert if any response mentions a return period not matching current policy (regex)

PUBLIC FOLLOW-UP (3 days later):
  "We've identified that our AI incorrectly stated a 60-day return policy.
   This was caused by [brief non-technical explanation]. We've resolved it
   for [customer name], updated our AI system, and added new safeguards
   to prevent this from happening again. We take AI accuracy very seriously."

IN THE INTERVIEW:
  "The customer is made whole immediately — technical investigation is secondary.
   The fix is specific: either remove the hallucinated response pattern or fix
   the KB. The prevention is systemic: eval and monitoring so next time this
   pattern fires an alert instead of a tweet."
```

---

### SCENARIO 8 — The Prioritisation Question

> **"You have one engineer and three AI feature requests from the PM: (A) improve RAG retrieval quality, (B) add voice mode to the chatbot, (C) build a usage analytics dashboard for customers. You can only ship one in Q1. Which do you pick and why?"**

```
FRAMEWORK: Impact × Confidence ÷ Effort, anchored to the company's north star.

FIRST: ASK WHAT'S THE CURRENT BIGGEST PROBLEM
  "Before I pick, what's our main metric we're trying to move in Q1?
   Is it retention, conversion, NPS, or revenue?"
  (The right answer depends entirely on context)

THEN REASON THROUGH EACH:

A: IMPROVE RAG RETRIEVAL QUALITY
  Impact:     Potentially high — if users are getting wrong answers, fixing this
              directly improves retention and reduces support costs
  Confidence: Medium-high — RAGAS will tell us if we improved
  Effort:     Medium — 1-2 weeks for hybrid search + reranking
  Signal to prioritise: if RAGAS faithfulness is below 0.75, or users
              are complaining about wrong answers. Foundation must work first.

B: ADD VOICE MODE
  Impact:     High ceiling, uncertain in practice
  Confidence: Low — don't know if users actually want voice
  Effort:     High — Pipecat + Twilio + STT + TTS = 4-6 weeks minimum
  Risk:       Latency-sensitive, many failure points
  Signal to prioritise: only if users have explicitly requested voice AND
              it's a key differentiator from competitors

C: USAGE ANALYTICS DASHBOARD
  Impact:     Medium — customers love it, helps retention, enables upsell
  Confidence: High — we know what metrics they want
  Effort:     Medium — 1-2 weeks with standard charting library
  Unique value: turns a cost (AI infra) into a product feature (insights)
  Signal to prioritise: if sales team says "we're losing deals because competitors 
              show this data" or if churn is high among customers who don't see value

MY RECOMMENDATION STRUCTURE:
  "If current RAGAS scores are below threshold → A first. Foundation before features.
   If quality is acceptable (> 0.75 faithfulness) → C.
   A dashboard (C) has high confidence, medium effort, and drives sales cycles.
   Voice (B) is a big bet with 4-6 weeks of effort and uncertain user demand.
   I'd put voice in Q2 after a lightweight user research spike to validate demand."

THE ANSWER THE INTERVIEWER WANTS:
  Not "it depends" with no decision
  Not picking without reasoning
  A clear recommendation with explicitly stated assumptions
  Awareness that your answer could change if one assumption changes
  ("If you tell me NPS is tanking because of wrong answers, I'd pick A immediately.")
```

---

## 🎤 Behavioural Questions for AI Engineers

### "Tell me about a time your AI system failed in production."

```
STRUCTURE: Situation → What failed → How you diagnosed → Fix → What you changed permanently

STRONG ANSWER TEMPLATE:
"While working on [project], we had [specific failure].
 I detected it because [monitoring / user complaint / alert].
 I diagnosed it by [Langfuse trace / looking at eval scores / checking KB].
 The immediate fix was [specific action].
 The permanent fix was [systemic change: eval, monitoring, process].
 Now we [new practice] so this category of failure alerts within [time]."

THINGS THAT SCORE WELL:
  Specific numbers (affected X users, quality dropped from Y to Z)
  Showing you dug into root cause (not just "model was wrong")
  Systemic fix (not just patching the incident)
  Personal ownership ("I should have caught this with an eval")

AVOID:
  Vague ("the model gave some wrong answers, we fixed the prompt")
  Blaming the model ("GPT-4o just hallucinated, nothing we could do")
  No learnings ("we fixed it and moved on")
```

### "Tell me about a time you disagreed with a PM or stakeholder."

```
STRONG ANSWER TEMPLATE:
"The PM wanted to ship [feature] with [specific approach].
 My concern was [specific technical/quality risk].
 I raised it by [showing eval data / writing a short doc / bringing a proposal].
 We resolved it by [compromise / data proved one side / ran an experiment].
 The outcome was [what happened, what we learned]."

EXAMPLE:
"The PM wanted to ship our RAG chatbot after a week of manual testing.
 I was concerned because we had no baseline — we didn't know if tomorrow's
 prompt change would improve or degrade quality. I proposed one more week
 to build a 50-example golden dataset and RAGAS eval before shipping.
 The PM agreed once I showed that this would prevent future incidents
 and take only 1 engineer-week. When we ran the first eval, we found
 faithfulness was 0.62 — below our 0.75 target. We fixed it before
 users ever saw it. The PM later said this was the right call."

SHOWS: you push back with data, not opinion. You propose, not just object.
```

### "How do you stay current with AI developments?"

```
DON'T SAY: "I follow AI Twitter and read papers."
TOO VAGUE. Every candidate says this.

BETTER ANSWER (specific, with examples):
"I have a deliberate system:
 Weekly: I read the Anthropic, OpenAI, and Google DeepMind release notes
         directly — I care about API changes more than research papers.
 Monthly: I prototype one new model or tool that's getting attention.
          Last month it was Google ADK — built a 3-agent research pipeline
          in a weekend to understand it properly.
 Quarterly: I audit our stack — are we still using the right models?
            Is there a cheaper/faster option that meets our quality bar?
 
 I also have a 'tech radar' — a simple doc with Adopt / Trial / Assess / Hold
 for every tool in our AI stack. I update it whenever I evaluate something new."
```

---

## 📋 PM-Style Questions in Engineering Interviews

These appear in senior AI engineer interviews. The company is checking if you can own outcomes, not just execute tasks.

### "How would you measure the success of this AI feature?"

```
FRAMEWORK — answer with 5 dimensions:
  Primary:   one north star metric the feature exists to move
  Secondary: 2-3 supporting metrics that explain the primary
  Guardrail: 1-2 metrics that must not regress
  Leading:   1 metric you can see every day (not monthly)
  Baseline:  current value of the primary metric

EXAMPLE for a support chatbot:
  Primary:   task completion rate (target: > 65%)
  Secondary: AI acceptance rate, RAGAS faithfulness
  Guardrail: escalation rate (must stay < 20%), latency P95 (< 2s)
  Leading:   daily acceptance rate rolling 4h window
  Baseline:  current: support ticket volume 200/day, avg resolve time 8 min
```

### "What would you do differently if you were rebuilding this from scratch?"

```
TEMPLATE:
  "What I'd keep: [what works well and why]
   What I'd change: [specific technical decision I'd make differently]
   What I'd add from day 1: [monitoring/eval/process I wished we had earlier]"

SHOWS: you reflect on your work, not just execute. You learn from experience.
```

### "How would you explain [technical AI concept] to a non-technical executive?"

```
TESTED CONCEPTS:
  RAG: "Instead of teaching the AI everything, we give it the right book
        to look up answers from, right before it responds."
  
  Hallucination: "The AI generates plausible text, but plausible isn't
                  always true. It's like a very confident intern who
                  sometimes makes up facts. Our guardrails are the fact-checker."
  
  RAGAS eval: "We have 50 test questions with correct answers. Every time
               we change anything, we check all 50. If the AI gets fewer
               right, we don't ship the change."
  
  Fine-tuning: "Instead of giving the AI all its knowledge every time
                it responds, we teach it once and it remembers. Like
                training a new employee vs giving them a manual each shift."

RULE: one analogy, max two sentences. Never use acronyms with executives.
```

---

## 🃏 Quick Revision Cards

```
CARD 1: AIPM vs Regular PM
  AIPM adds: eval ownership, model lifecycle, risk calibration, trust design
  "Done" for AI: eval gate passes, not just "it ships"
  Quality: probabilistic (87% correct), not binary
  The PM defines what "good" means in user terms
  The engineer translates it to measurable technical metrics

CARD 2: Build vs Buy vs Fine-Tune
  Buy:        standard task, low volume, fast time-to-market
  RAG:        factual Q&A over proprietary docs, knowledge changes often
  Fine-tune:  high volume, compliance, domain-specific style/format, cost at scale
  Document Q&A → almost always RAG first
  Decide on: update frequency, volume, compliance, timeline, budget

CARD 3: Eval Design
  Write success metric BEFORE building the feature
  Golden dataset: 50 human-verified examples minimum
  RAGAS thresholds: faithfulness > 0.75, answer_relevancy > 0.80
  A/B test: minimum 1 week, pre-commit to duration, don't peek
  Eval gate in CI/CD: blocks deploy if metrics drop

CARD 4: North Star Metrics
  Task completion rate:        user accomplished goal with AI? (> 65%)
  AI acceptance rate:         user accepted suggestion as-is? (> 30-65%)
  Time-to-value:              AI saved X minutes vs baseline
  Cost per successful interaction: LLM cost ÷ completion rate
  Escalation rate:            AI failed, human took over (< 20%)

CARD 5: AI Incident Response
  Immediate: kill switch → scope → contain → communicate (< 1 hour)
  Investigation: Langfuse traces → root cause → systematic or isolated?
  Fix: specific (KB fix, prompt fix, guardrail addition)
  Prevention: add to golden dataset, add monitoring, lower alert threshold
  Postmortem: what happened, why missed, fix, preventive measures

CARD 6: Communicating AI Limitations
  Never: "100% accurate" or "no hallucinations"
  Always: give current baseline ("87% correct on our eval suite")
  Offer: three paths (reduce, contain, monitor)
  Frame: which error categories are unacceptable vs acceptable?
  Ask: "Help me understand which failures matter most to you"

CARD 7: Leading vs Lagging Indicators
  Lagging:  CSAT, NPS, churn (you see the damage after it happened)
  Leading:  RAGAS score, P95 latency, acceptance rate, escalation rate
  Build:    daily dashboard of leading indicators + alerts
  Rule:     if you're finding out about quality drops from user complaints,
            your leading indicator system is missing or misconfigured

CARD 8: A/B Testing AI Features
  Assign by: hash(user_id + experiment_id) → deterministic
  Duration:  minimum 1 week (capture weekly patterns)
  Sample:    calculate required N before starting (power analysis)
  Metrics:   primary (acceptance rate) + guardrail (latency, errors)
  Decision:  promote if primary improves AND guardrails don't regress
  Never:     stop early because you like what you see (peeking)

CARD 9: AI Competitive Moats
  NOT a moat: "we use GPT-4o" (anyone can)
  Real moats: proprietary data, feedback loops, curated KB,
              integration depth, domain fine-tuning + evals
  Engineer's role: build feedback collection from day 1
                   every thumbs up/down is a training example
                   proprietary labelled data = defensible position

CARD 10: Scenario Interview Structure
  Quality drop:    timeline → reproduce → root cause → fix → prevent
  Stakeholder ask: current baseline → options → trade-offs → recommendation
  Build decision:  clarifying questions → criteria → recommendation → if-then
  Cost problem:    understand scale → diagnose by org/feature → fix → prevent
  Feature request: 5 clarifying Qs → alternatives → recommendation → next step
  Incident:        customer first → investigate → specific fix → systemic prevention
```

---

## 🏗️ AIPM Exercises

These are not coding exercises. They are thinking exercises. Do them in a doc, then read them aloud to yourself or a friend.

### Exercise 1 — Write a PRD for Your RAG Chatbot
```
Using the template in Section 12.6, write a 1-page PRD for SynapseIQ's RAG feature.
Fill in every section: problem statement, success metrics, non-goals, constraints,
AI approach, eval plan, rollback plan, open questions.

When you're done, ask yourself:
  Could an engineer build exactly this from your spec?
  Would a PM approve this without follow-up questions?
  What would you change if the budget was 3x larger? 3x smaller?
```

### Exercise 2 — Define Evals Before Building
```
Pick any feature from your capstone projects that you haven't started yet.
Before writing a single line of code:
  Write 20 golden dataset examples (question, context, ground_truth)
  Define your success thresholds (which RAGAS metrics, which values)
  Write the LLM-as-judge rubric for your custom quality criterion
  
Then build it. Compare your pre-build expectations to actual eval results.
What surprised you?
```

### Exercise 3 — Map a Competitor's AI Feature
```
Pick any AI product you use (Notion AI, Perplexity, Cursor, GitHub Copilot).
Answer:
  What model are they likely using? (infer from speed, quality, output style)
  What's their eval likely measuring? (infer from what they're good/bad at)
  What are their obvious failure modes? (test them — prompt injection, edge cases)
  What's their moat? (data, integration, brand?)
  If you were rebuilding this, what would you do differently?

Write a 1-page competitive analysis. This is exactly what AIPMs do.
```

### Exercise 4 — Incident Report for a Real Failure
```
Think of a time your AI feature (in any project) gave a wrong answer.
Write a full AI incident report using the template in Section 12.6:
  What happened? Impact? Timeline? Root cause? Why missed? Fix? Prevention?

If you haven't had a real failure: invent a plausible one for SynapseIQ.
"A user asked about the return policy and the chatbot cited a 2024 document
after a 2026 update. Write the full incident report."
```

---

*Phase 12 — AIPM Fundamentals | GenAI + LLMOps Engineering Roadmap 2026*
*Covers: Product Strategy · Eval Design · AI Risk & Ethics · North Star Metrics · 8 Full Scenario Questions · Behavioural Questions · PM-Style Engineering Questions*
*This phase makes you 2x more effective as an AI engineer and dramatically stronger in senior-level interviews.*
