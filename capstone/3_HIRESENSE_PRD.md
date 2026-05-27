# HireSense — AI Hiring Intelligence Platform
> **Type:** End-to-End AI-Powered Recruitment System
> **Stack:** React 18 + Vite + Tailwind · FastAPI · PostgreSQL · Redis · Qdrant · Scikit-learn · LangGraph · Celery · AWS · Clerk · Langfuse
> **Unique Differentiator:** Classical ML (resume screening) + GenAI (interview agent) + LLMOps/MLOps in one production system
> **Timeline:** 16–18 weeks

---

## 📑 Table of Contents

1. [The Origin Story](#1-the-origin-story)
2. [What HireSense Does](#2-what-hiresense-does)
3. [The Three Pillars](#3-the-three-pillars)
4. [Why This Is The Perfect Portfolio Project](#4-why-this-is-the-perfect-portfolio-project)
5. [System Architecture](#5-system-architecture)
6. [Technology Stack](#6-technology-stack)
7. [Pillar 1 — Classical ML Resume Screener](#7-pillar-1--classical-ml-resume-screener)
8. [Pillar 2 — LLM Interview Agent (LangGraph)](#8-pillar-2--llm-interview-agent-langgraph)
9. [Pillar 3 — MLOps + LLMOps + DevOps Layer](#9-pillar-3--mlops--llmops--devops-layer)
10. [Custom Job Configuration](#10-custom-job-configuration)
11. [Candidate Portal](#11-candidate-portal)
12. [Recruiter Dashboard](#12-recruiter-dashboard)
13. [Database Schema](#13-database-schema)
14. [API Reference](#14-api-reference)
15. [Design Patterns Used](#15-design-patterns-used)
16. [Guardrails](#16-guardrails)
17. [Phase-by-Phase Build Plan](#17-phase-by-phase-build-plan)
18. [Folder Structure](#18-folder-structure)
19. [Resume Deliverables + Interview Story](#19-resume-deliverables--interview-story)

---

## 1. The Origin Story

### The Problem Space

Every growing company hits the same wall at some point.

A job posting goes live. 400 resumes arrive in 48 hours. A recruiter opens each one, spends 90 seconds skimming it, and decides yes or no. By resume 80, fatigue sets in. By resume 200, they're going on pattern-matching autopilot. Implicit biases creep in. Good candidates get filtered out. Weak candidates slip through. Two weeks later, 12 people are shortlisted — and half shouldn't be there.

Then comes the phone screen. The recruiter calls each shortlisted candidate to verify basics: "Tell me about yourself. Why are you interested in this role? What are your salary expectations?" These are 20-minute calls to verify things that could be established asynchronously. 12 candidates × 20 minutes = 4 hours of recruiter time per job posting, just to filter further.

Then the technical screen. Scheduled. Postponed. Rescheduled. A calendar puzzle that takes a week to resolve. The candidate had already accepted another offer.

Total time from posting to first real technical interview: 3–4 weeks. Best candidates stay available for 1 week.

### The Insight

Three separate tools exist for parts of this problem:
- ATS systems (Greenhouse, Lever) handle pipeline tracking but no AI screening
- Resume parsers (Affinda, Sovren) extract structured data but don't score or rank
- Interview schedulers (Calendly, HireVue) handle logistics but not intelligence

Nobody has connected classical ML (for objective screening) + conversational AI (for initial interview) + a proper MLOps layer (so the system actually improves over time) in a single, configurable platform.

**HireSense is that platform.**

### Origin for Your Portfolio

You're an agency that started building this for a specific client: a bootstrapped EdTech startup hiring 40 engineers in 6 months. They had 2 recruiters, 400+ applications per role, and couldn't afford to hire more recruiters. You built HireSense as their internal tool. It worked. You're now turning it into a product.

```
Before HireSense (at client):
  Step 1: 400 resumes received
  Step 2: 2 recruiters spend 3 days screening manually
  Step 3: 40 phone screens scheduled over 2 weeks
  Step 4: 18 proceed to technical interview
  Time to technical interview: 3-4 weeks
  Cost: ~₹80,000 in recruiter time

After HireSense:
  Step 1: 400 resumes uploaded
  Step 2: ML model scores in 12 minutes (automated)
  Step 3: Top 60 invited to AI interview (async, 24/7)
  Step 4: AI interview scores + recordings reviewed in 2 hours
  Step 5: 18 proceed to technical interview
  Time to technical interview: 3 days
  Cost: ~₹8,000 in recruiter time + infra
```

---

## 2. What HireSense Does

```
RESUME INTAKE (→ Classical ML Screening)
  Upload resumes (PDF, DOCX, bulk ZIP)
  Parse structured data (experience, skills, education, projects)
  Run ML classifier → Shortlisted / Not Shortlisted
  Score 0-100 + feature importance explanation
  Recruiter reviews ML decision, can override

CANDIDATE NOTIFICATION (→ AI Interview Invitation)
  Shortlisted candidates receive email/SMS
  Custom interview link with job-specific configuration
  Candidate picks time window (or async anytime)

AI INTERVIEW (→ LangGraph Voice/Text Agent)
  Candidate arrives at interview portal
  AI conducts structured interview (30 min)
  Custom question bank per job role
  Real-time transcription + answer evaluation
  Follow-up questions based on answers
  Scores: technical depth, communication, role fit

RECRUITER REVIEW (→ Dashboard)
  Side-by-side: ML screening score + AI interview score
  Full transcript + recording per candidate
  AI-generated candidate summary
  One-click: shortlist / reject / move to next round
  Bulk actions + calendar invite generation

MLOPS + LLMOPS (→ Continuous Improvement)
  Recruiter overrides feed back into ML model retraining
  Prompt versioning with A/B testing on interview quality
  RAGAS eval on AI interview answer assessments
  Langfuse traces on every LLM call
  Drift detection on ML model performance
  CI/CD pipeline with eval gate before deploy
```

---

## 3. The Three Pillars

```
PILLAR 1: CLASSICAL ML (Resume Screening)
  ┌────────────────────────────────────────────────┐
  │  Resume Parser (PDFMiner + spaCy)              │
  │      ↓                                         │
  │  Feature Engineering                           │
  │  (skills match, experience years, edu score,   │
  │   project relevance, keyword density)          │
  │      ↓                                         │
  │  ML Classifier (RandomForest + XGBoost)        │
  │      ↓                                         │
  │  Score 0-100 + SHAP explanation                │
  │      ↓                                         │
  │  Shortlisted / Rejected + Reason               │
  └────────────────────────────────────────────────┘
  
  Why Classical ML (not LLM for this step)?
  → Faster: 400 resumes in 12 minutes (LLM would take hours + cost $)
  → Explainable: SHAP shows WHY a candidate scored high/low
  → Retrainable: recruiter feedback → model improves over time
  → Auditable: classic ML is easier to audit for bias than LLM
  → Interview question: "Why not use GPT-4 for screening?"
    → "Cost, speed, explainability, and retrainability"

PILLAR 2: LLM INTERVIEW AGENT (Conversational)
  ┌────────────────────────────────────────────────┐
  │  Candidate enters interview portal             │
  │      ↓                                         │
  │  LangGraph Interview Graph                     │
  │  ┌──────────────────────────────────────────┐  │
  │  │ intro_node (greet, set expectations)     │  │
  │  │     ↓                                    │  │
  │  │ question_node (structured question)      │  │
  │  │     ↓                                    │  │
  │  │ evaluate_node (score answer, decide:     │  │
  │  │   follow_up? next_question? conclude?)   │  │
  │  │     ↓ (loop N times)                     │  │
  │  │ conclude_node (wrap up, thank candidate) │  │
  │  └──────────────────────────────────────────┘  │
  │      ↓                                         │
  │  Generate: transcript + answer scores +        │
  │           candidate summary + recommendation   │
  └────────────────────────────────────────────────┘

PILLAR 3: MLOPS + LLMOPS + DEVOPS
  ┌────────────────────────────────────────────────┐
  │  MLOps (Classical ML pipeline)                 │
  │  • Feature store (Redis)                       │
  │  • Model registry (MLflow)                     │
  │  • Drift detection (Evidently AI)              │
  │  • Retraining pipeline (Celery + DVC)          │
  │  • Model versioning + rollback                 │
  │                                                │
  │  LLMOps (Interview Agent)                      │
  │  • Prompt registry (versioned .txt files)      │
  │  • A/B prompt testing                          │
  │  • RAGAS eval on answer assessments            │
  │  • Langfuse traces (cost, latency, quality)    │
  │  • CI/CD eval gate (must pass RAGAS threshold) │
  │                                                │
  │  DevOps                                        │
  │  • Docker Compose (local)                      │
  │  • ECS Fargate + RDS + ElastiCache (prod)      │
  │  • GitHub Actions CI/CD                        │
  │  • CloudWatch + Sentry alerts                  │
  └────────────────────────────────────────────────┘
```

---

## 4. Why This Is The Perfect Portfolio Project

```
SHOWS CLASSICAL ML + GENAI IN ONE SYSTEM:
  Most candidates have one or the other.
  HireSense has both, with a clear architectural reason for each.
  Interview question: "When do you use ML vs LLM?"
  → Objective, fast, scalable tasks → ML
  → Subjective, conversational, contextual tasks → LLM

SHOWS MLOPS END-TO-END:
  Feature engineering → training → evaluation → serving
  Drift detection → retraining trigger → new model deployed
  MLflow model registry → versioning → rollback
  The full MLOps loop in one project.

SHOWS LLMOPS END-TO-END:
  Prompt versioning → A/B test → eval (RAGAS) → deploy
  Langfuse traces → cost tracking → quality monitoring
  CI/CD gate: eval must pass before prompt goes to production

SHOWS DEVOPS:
  Docker Compose (local multi-container)
  GitHub Actions (CI test → eval gate → deploy)
  ECS Fargate (production deployment)
  CloudWatch alarms, Sentry error tracking

SHOWS SYSTEM DESIGN:
  Why Redis for feature store?
  Why Celery for retraining pipeline?
  Why SHAP for explainability?
  Why Qdrant for semantic resume search?
  Every decision is defensible in an interview.

SHOWS PRODUCT THINKING:
  Custom job configuration (the PM angle)
  Bias audit report (the responsible AI angle)
  Candidate experience (the UX angle)
  Cost tracking per hire (the business angle)
```

---

## 5. System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        HireSense Platform                           │
│                                                                     │
│  ┌─────────────────┐    ┌──────────────────────────────────────┐   │
│  │  Recruiter UI   │    │  Candidate Portal                    │   │
│  │  (React 18)     │    │  (React 18 — separate subdomain)     │   │
│  └────────┬────────┘    └──────────────────┬───────────────────┘   │
│           │                                │                        │
│  ┌────────▼────────────────────────────────▼───────────────────┐   │
│  │                   FastAPI Backend                           │   │
│  │                                                             │   │
│  │  /api/jobs     /api/resumes    /api/interviews  /api/ml     │   │
│  └─────┬──────────────┬────────────────┬─────────────┬─────────┘   │
│        │              │                │             │              │
│  ┌─────▼──────┐ ┌─────▼──────┐ ┌──────▼──────┐ ┌───▼───────────┐  │
│  │  Job Conf  │ │  Resume    │ │  Interview  │ │  ML Service   │  │
│  │  Service   │ │  Pipeline  │ │  Agent      │ │  + MLflow     │  │
│  └────────────┘ └─────┬──────┘ │  (LangGraph)│ └───────────────┘  │
│                        │       └──────┬──────┘                     │
│  ┌─────────────────────▼──────────────▼───────────────────────┐   │
│  │              Celery Workers (async processing)              │   │
│  │   • Resume parsing + feature extraction                     │   │
│  │   • ML model inference (batch scoring)                      │   │
│  │   • ML model retraining pipeline                            │   │
│  │   • Interview transcript processing                         │   │
│  │   • Email + notification dispatch                           │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ │
│  │Postgres  │ │  Redis   │ │  Qdrant  │ │  S3      │ │  MLflow  │ │
│  │(primary) │ │(cache+Q) │ │(semantic)│ │(resumes) │ │(models)  │ │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────┘ │
│                                                                     │
│  ┌──────────┐ ┌──────────┐ ┌──────────────────────────────────┐   │
│  │ Langfuse │ │ Evidently│ │  GitHub Actions CI/CD            │   │
│  │(LLM obs) │ │(ML drift)│ │  (test → eval gate → deploy)    │   │
│  └──────────┘ └──────────┘ └──────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

### Data Flow — Hiring Pipeline

```
                  RECRUITER CREATES JOB
                          │
                  Custom configuration set
                  (questions, weights, thresholds)
                          │
             ┌────────────▼─────────────┐
             │    RESUME INTAKE         │
             │  Upload → S3 → Celery   │
             └────────────┬─────────────┘
                          │
             ┌────────────▼─────────────┐
             │  PARSE + EXTRACT         │
             │  PDFMiner → spaCy NER   │
             │  Structured JSON output  │
             └────────────┬─────────────┘
                          │
             ┌────────────▼─────────────┐
             │  FEATURE ENGINEERING     │
             │  skills_match_score      │
             │  experience_relevance    │
             │  education_tier          │
             │  project_quality_score   │
             │  keyword_density         │
             └────────────┬─────────────┘
                          │
             ┌────────────▼─────────────┐
             │  ML CLASSIFIER           │
             │  RandomForest + XGBoost  │
             │  Score: 0-100            │
             │  Label: shortlisted/not  │
             │  SHAP: why this score    │
             └────────────┬─────────────┘
                          │
             ┌────────────▼─────────────┐
             │  RECRUITER REVIEW        │
             │  Approve / Override      │
             │  Override → feedback DB  │──→ Retraining trigger
             └────────────┬─────────────┘
                          │ (shortlisted candidates)
             ┌────────────▼─────────────┐
             │  AI INTERVIEW INVITATION │
             │  Email + unique link     │
             └────────────┬─────────────┘
                          │
             ┌────────────▼─────────────┐
             │  AI INTERVIEW (LangGraph)│
             │  Text or voice mode      │
             │  Structured questions    │
             │  Dynamic follow-ups      │
             │  Answer scoring          │
             └────────────┬─────────────┘
                          │
             ┌────────────▼─────────────┐
             │  INTERVIEW REPORT        │
             │  Transcript + scores     │
             │  AI summary              │
             │  Recommendation          │
             └────────────┬─────────────┘
                          │
             ┌────────────▼─────────────┐
             │  RECRUITER FINAL REVIEW  │
             │  ML score + AI score     │
             │  Move to next round OR   │
             │  Reject with reason      │
             └──────────────────────────┘
```

---

## 6. Technology Stack

```
FRONTEND:
  React 18 + Vite + Tailwind CSS v4
  Recruiter dashboard (job management, pipeline, analytics)
  Candidate portal (interview interface, separate subdomain)
  React Query (server state), Zustand (UI state)

BACKEND:
  FastAPI (main API server)
  Celery + Redis (async task queue for parsing, scoring, retraining)
  asyncpg (Postgres async driver)

CLASSICAL ML:
  scikit-learn:   RandomForestClassifier, LogisticRegression, TfidfVectorizer
  XGBoost:        gradient boosting classifier (ensemble with RF)
  spaCy:          NER for resume entity extraction
  PDFMiner:       resume PDF text extraction
  SHAP:           model explainability (why this score)
  imbalanced-learn: SMOTE for class imbalance (more rejects than accepts in training)

MLOPS:
  MLflow:         model registry, experiment tracking, artifact storage
  Evidently AI:   data drift detection on resume feature distributions
  DVC:            data versioning for training datasets
  Celery:         retraining pipeline orchestration
  Great Expectations: training data quality validation

GENAI / LLMOPS:
  LangGraph:      interview agent state machine
  OpenAI GPT-4o:  interview question generation + answer evaluation
  Groq:           fast inference for simpler evaluations
  instructor:     structured output from LLM
  RAGAS:          evaluation of answer quality assessments
  Langfuse:       LLM observability (traces, cost, latency)
  Prompt Registry: versioned .txt files per interview type

STORAGE:
  PostgreSQL:     jobs, candidates, resumes, interviews, outcomes
  Redis:          Celery broker, feature cache, session state, rate limiting
  Qdrant:         semantic resume search (find similar past candidates)
  S3:             resume files, interview recordings, model artifacts

AUTH:
  Clerk:          recruiter auth + org management
  JWT:            candidate portal auth (stateless, short-lived)

DEVOPS:
  Docker Compose: local dev (all services)
  GitHub Actions: CI/CD (test → lint → eval gate → build → deploy)
  AWS ECS Fargate: production backend
  AWS RDS Postgres: production database
  AWS ElastiCache: production Redis
  Sentry:         error tracking
  CloudWatch:     infrastructure metrics + alarms
```

---

## 7. Pillar 1 — Classical ML Resume Screener

### 7.1 Why Classical ML for Resume Screening

```
WHY NOT USE GPT-4O TO SCREEN RESUMES?

Cost:
  400 resumes × 3000 tokens avg × $2.50/1M = $3.00 per job posting
  At scale: 100 job postings × $3.00 = $300/month just for screening
  Classical ML: $0 inference cost after training

Speed:
  GPT-4o: 400 resumes × ~2s = 13 minutes (if not rate limited)
  Classical ML: 400 resumes in ~10 seconds (all in-memory vectorised)
  
Explainability:
  GPT-4o: "This candidate looks strong" (opaque)
  SHAP:   "Score 73 because: skills_match=+28, experience=+19, education=+14,
           project_quality=+12 — but missing React.js (-8, required)"
  Recruiters can audit and challenge this. LLM output cannot be audited.

Retrainability:
  Every recruiter override is a labelled example.
  Classical ML retrains in minutes on new feedback.
  GPT-4o requires expensive fine-tuning.

Bias Auditing:
  Classical ML: run fairness metrics (demographic parity, equalized odds)
  SHAP: confirm protected attributes not used as features
  LLM: black-box bias is much harder to detect and prove in court

INTERVIEW ANSWER:
  "We use classical ML for resume screening because it's 1000x faster,
   10x cheaper, explainable via SHAP, and retrainable on recruiter feedback.
   We use LLM for the interview because that's a conversational, contextual,
   open-ended task that genuinely requires LLM reasoning."
```

### 7.2 Resume Parser

```python
# app/ml/resume_parser.py

import spacy
import pdfminer
from pdfminer.high_level import extract_text
from typing import TypedDict

nlp = spacy.load("en_core_web_md")

class ParsedResume(TypedDict):
    raw_text:        str
    name:            str | None
    email:           str | None
    phone:           str | None
    years_experience:float
    skills:          list[str]
    education:       list[dict]   # [{degree, institution, year, tier}]
    work_history:    list[dict]   # [{company, title, duration_months, description}]
    projects:        list[dict]   # [{name, tech_stack, description}]
    certifications:  list[str]
    languages:       list[str]
    github_url:      str | None
    linkedin_url:    str | None

class ResumeParser:
    """Parse PDF/DOCX resumes into structured JSON for ML feature extraction"""

    EDUCATION_TIERS = {
        "iit":       5, "iim":  5, "nit":      4, "bits":      4,
        "vit":       3, "srm":  3, "manipal":  3, "anna":      3,
        "state":     2, "open": 1, "distance": 1,
    }

    SKILL_ALIASES = {
        "js":           "javascript",
        "ts":           "typescript",
        "py":           "python",
        "ml":           "machine learning",
        "dl":           "deep learning",
        "k8s":          "kubernetes",
        "pg":           "postgresql",
        "mongo":        "mongodb",
    }

    def parse_pdf(self, path: str) -> ParsedResume:
        raw_text = extract_text(path)
        return self._extract_structure(raw_text)

    def parse_text(self, text: str) -> ParsedResume:
        return self._extract_structure(text)

    def _extract_structure(self, text: str) -> ParsedResume:
        doc = nlp(text)

        return ParsedResume(
            raw_text=text,
            name=self._extract_name(doc),
            email=self._extract_email(text),
            phone=self._extract_phone(text),
            years_experience=self._extract_experience_years(text),
            skills=self._extract_skills(text),
            education=self._extract_education(text),
            work_history=self._extract_work_history(doc),
            projects=self._extract_projects(text),
            certifications=self._extract_certifications(text),
            languages=self._extract_languages(text),
            github_url=self._extract_url(text, "github.com"),
            linkedin_url=self._extract_url(text, "linkedin.com"),
        )

    def _extract_skills(self, text: str) -> list[str]:
        """Extract and normalise skills from resume text"""
        import re

        # Common skill section patterns
        SKILL_SECTION = re.compile(
            r"(?:skills|technologies|tech stack|tools)[:\s]*(.*?)(?:\n\n|\Z)",
            re.IGNORECASE | re.DOTALL
        )
        skills = set()
        match = SKILL_SECTION.search(text)
        if match:
            raw = match.group(1)
            # Split on common delimiters
            tokens = re.split(r"[,•|/\n·]", raw)
            for token in tokens:
                skill = token.strip().lower()
                # Normalise aliases
                skill = self.SKILL_ALIASES.get(skill, skill)
                if 2 < len(skill) < 30:
                    skills.add(skill)
        return list(skills)

    def _extract_experience_years(self, text: str) -> float:
        """Estimate years of experience from work history dates"""
        import re
        from datetime import datetime

        YEAR_RANGE = re.compile(r"(20\d{2})\s*[-–]\s*(20\d{2}|present|current)", re.IGNORECASE)
        total_months = 0
        for match in YEAR_RANGE.finditer(text):
            start = int(match.group(1))
            end_str = match.group(2).lower()
            end = datetime.now().year if end_str in ["present", "current"] else int(end_str)
            total_months += (end - start) * 12
        return round(total_months / 12, 1)

    def _education_tier(self, institution: str) -> int:
        inst_lower = institution.lower()
        for keyword, tier in self.EDUCATION_TIERS.items():
            if keyword in inst_lower:
                return tier
        return 2   # default: decent institution
```

### 7.3 Feature Engineering

```python
# app/ml/feature_engineer.py

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

class FeatureEngineer:
    """
    Converts ParsedResume + JobConfig into a numeric feature vector for ML.
    
    Feature vector structure (20 features):
    
    [0]  skills_match_score         — % of required skills present
    [1]  bonus_skills_count         — count of preferred skills present
    [2]  years_experience           — raw years
    [3]  experience_in_range        — 1 if within min_exp/max_exp, else 0
    [4]  education_tier             — 1-5 institution quality score
    [5]  degree_level               — 0=none,1=diploma,2=bachelors,3=masters,4=phd
    [6]  relevant_degree            — 1 if degree field matches job domain
    [7]  project_count              — number of listed projects
    [8]  project_relevance_score    — avg % of required tech in project stacks
    [9]  github_present             — 1 if GitHub URL found
    [10] keyword_density            — TF-IDF similarity of resume to job description
    [11] job_title_match            — string similarity of past titles to target
    [12] certification_relevance    — count of relevant certifications
    [13] career_progression         — 1 if titles show upward trajectory
    [14] company_tier_avg           — avg company size/known-ness score
    [15] gap_years                  — total years of employment gaps
    [16] avg_tenure_months          — average months spent per job
    [17] location_match             — 1 if location matches job location
    [18] language_match             — 1 if required languages present
    [19] custom_score_0             — job-specific custom rule (from JobConfig)
    """

    def __init__(self, job_config: "JobConfig"):
        self.config  = job_config
        self.tfidf   = TfidfVectorizer(ngram_range=(1, 2), max_features=500)
        self._fitted = False

    def extract(self, parsed: "ParsedResume") -> np.ndarray:
        return np.array([
            self._skills_match_score(parsed),
            self._bonus_skills_count(parsed),
            parsed["years_experience"],
            self._experience_in_range(parsed),
            self._education_tier(parsed),
            self._degree_level(parsed),
            self._relevant_degree(parsed),
            len(parsed["projects"]),
            self._project_relevance(parsed),
            1.0 if parsed["github_url"] else 0.0,
            self._keyword_density(parsed),
            self._job_title_match(parsed),
            self._certification_relevance(parsed),
            self._career_progression(parsed),
            self._company_tier_avg(parsed),
            self._gap_years(parsed),
            self._avg_tenure(parsed),
            self._location_match(parsed),
            self._language_match(parsed),
            self._custom_score(parsed),
        ], dtype=np.float32)

    def _skills_match_score(self, parsed: "ParsedResume") -> float:
        """What % of required skills does the candidate have?"""
        required = set(s.lower() for s in self.config.required_skills)
        present  = set(s.lower() for s in parsed["skills"])
        if not required:
            return 1.0
        return len(required & present) / len(required)

    def _keyword_density(self, parsed: "ParsedResume") -> float:
        """TF-IDF cosine similarity between resume text and job description"""
        from sklearn.metrics.pairwise import cosine_similarity
        if not self._fitted:
            self.tfidf.fit([self.config.job_description])
            self._fitted = True
        job_vec    = self.tfidf.transform([self.config.job_description])
        resume_vec = self.tfidf.transform([parsed["raw_text"]])
        return float(cosine_similarity(job_vec, resume_vec)[0][0])
```

### 7.4 ML Model — Training + Serving

```python
# app/ml/model.py

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

FEATURE_NAMES = [
    "skills_match_score", "bonus_skills_count", "years_experience",
    "experience_in_range", "education_tier", "degree_level",
    "relevant_degree", "project_count", "project_relevance_score",
    "github_present", "keyword_density", "job_title_match",
    "certification_relevance", "career_progression", "company_tier_avg",
    "gap_years", "avg_tenure_months", "location_match",
    "language_match", "custom_score",
]

class ResumeScreenerModel:
    """
    Ensemble of RandomForest + XGBoost + LogisticRegression.
    Trained per job role (or global model for generic screening).
    Versioned in MLflow. SHAP for explainability.
    """

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,   # 1=shortlisted, 0=rejected
        job_id: str,
        experiment_name: str = "resume_screener",
    ) -> str:
        """Train, evaluate, and register model. Returns run_id."""

        with mlflow.start_run(experiment_id=self._get_experiment(experiment_name)) as run:
            mlflow.set_tag("job_id", job_id)
            mlflow.log_param("features", FEATURE_NAMES)
            mlflow.log_param("training_samples", len(X))
            mlflow.log_param("positive_rate", y.mean())

            # Handle class imbalance — resumes are usually 80% rejected
            smote    = SMOTE(random_state=42)
            X_res, y_res = smote.fit_resample(X, y)
            mlflow.log_param("smote_samples", len(X_res))

            # Ensemble model
            rf  = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
            xgb = XGBClassifier(n_estimators=200, learning_rate=0.1, random_state=42)
            lr  = LogisticRegression(C=1.0, max_iter=1000)

            ensemble = VotingClassifier(
                estimators=[("rf", rf), ("xgb", xgb), ("lr", lr)],
                voting="soft",
                weights=[3, 3, 1],  # RF + XGB weighted higher
            )

            pipeline = Pipeline([
                ("scaler", StandardScaler()),
                ("model",  ensemble),
            ])

            # Stratified K-fold evaluation
            cv_scores = cross_val_score(pipeline, X_res, y_res,
                                         cv=StratifiedKFold(n_splits=5),
                                         scoring="f1")
            mlflow.log_metric("cv_f1_mean", cv_scores.mean())
            mlflow.log_metric("cv_f1_std",  cv_scores.std())

            # Train final model
            pipeline.fit(X_res, y_res)

            # Log SHAP feature importance
            explainer    = shap.TreeExplainer(rf.estimators_[0] if hasattr(rf, 'estimators_') else rf)
            shap_values  = explainer.shap_values(X[:100])
            mean_abs_shap= np.abs(shap_values).mean(axis=0)
            for i, name in enumerate(FEATURE_NAMES):
                mlflow.log_metric(f"shap_{name}", mean_abs_shap[i] if len(mean_abs_shap.shape) == 1 else mean_abs_shap[0][i])

            # Register in MLflow model registry
            mlflow.sklearn.log_model(
                pipeline,
                artifact_path="model",
                registered_model_name=f"resume_screener_{job_id}",
            )

            self._pipeline = pipeline
            return run.info.run_id

    def predict(self, X: np.ndarray) -> dict:
        """Returns score 0-100 + shortlisted bool + SHAP explanations"""
        proba     = self._pipeline.predict_proba(X)[0]
        score     = int(proba[1] * 100)
        decision  = score >= self._threshold

        # SHAP for this individual prediction
        explanation = self._explain(X)

        return {
            "score":       score,
            "shortlisted": decision,
            "confidence":  round(float(proba[1]), 4),
            "explanation": explanation,   # top 5 positive/negative factors
        }

    def _explain(self, X: np.ndarray) -> list[dict]:
        """SHAP explanation for one prediction — top 5 factors"""
        rf_step  = self._pipeline.named_steps["model"].estimators_[0][1]
        explainer= shap.TreeExplainer(rf_step)
        shap_vals= explainer.shap_values(X)

        contributions = []
        for i, name in enumerate(FEATURE_NAMES):
            val = shap_vals[1][0][i] if len(shap_vals) > 1 else shap_vals[0][i]
            contributions.append({
                "feature":      name,
                "value":        float(X[0][i]),
                "contribution": float(val),
                "direction":    "positive" if val > 0 else "negative",
            })

        # Return top 3 positive + top 2 negative factors
        positives = sorted([c for c in contributions if c["contribution"] > 0],
                           key=lambda x: x["contribution"], reverse=True)[:3]
        negatives = sorted([c for c in contributions if c["contribution"] < 0],
                           key=lambda x: x["contribution"])[:2]
        return positives + negatives
```

### 7.5 Drift Detection + Retraining

```python
# app/ml/drift_monitor.py

from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset
from evidently import ColumnMapping

class DriftMonitor:
    """
    Detect when the current model's predictions degrade.
    Two triggers for retraining:
    1. Data drift: incoming resumes look different from training data
    2. Label drift: recruiter overrides exceed threshold (model is wrong)
    """

    OVERRIDE_THRESHOLD = 0.15    # retrain if >15% of recruiter decisions differ from model

    async def check_data_drift(
        self,
        reference_df: pd.DataFrame,  # training data features
        current_df: pd.DataFrame,    # last 200 scored resumes
    ) -> dict:
        report = Report(metrics=[DataDriftPreset()])
        report.run(reference_data=reference_df, current_data=current_df,
                   column_mapping=ColumnMapping(target="shortlisted"))

        result    = report.as_dict()
        drifted   = result["metrics"][0]["result"]["dataset_drift"]
        n_drifted = result["metrics"][0]["result"]["number_of_drifted_columns"]

        return {
            "dataset_drifted":    drifted,
            "drifted_features":   n_drifted,
            "should_retrain":     drifted,
        }

    async def check_label_drift(self, job_id: str, window_days: int = 7) -> dict:
        """Check how often recruiters override the model's decision"""
        from_date = datetime.utcnow() - timedelta(days=window_days)
        total, overrides = await ResumeRepository.get_override_rate(job_id, from_date)
        override_rate    = overrides / max(total, 1)
        return {
            "override_rate":   override_rate,
            "total_scored":    total,
            "overrides":       overrides,
            "should_retrain":  override_rate >= self.OVERRIDE_THRESHOLD,
        }

# app/ml/retrain_pipeline.py — Celery task
@celery_app.task(name="retrain_model")
def retrain_model_task(job_id: str, trigger: str):
    """
    Celery task triggered by:
    1. Drift monitor (scheduled check every 24h)
    2. Manual trigger from recruiter dashboard
    3. After N new labelled examples accumulated
    """
    # 1. Load new training data (original + recruiter overrides)
    X, y = FeatureRepository.get_training_data(job_id)

    # 2. Validate data quality
    validator = DataQualityValidator()
    if not validator.validate(X, y):
        logger.error(f"Training data quality check failed for job {job_id}")
        return {"status": "failed", "reason": "data_quality"}

    # 3. Train new model
    model = ResumeScreenerModel(threshold=50)
    run_id = model.train(X, y, job_id=job_id)

    # 4. Compare with current model (A/B on holdout set)
    new_metrics = evaluate_on_holdout(model, job_id)
    old_metrics = ModelRegistry.get_current_metrics(job_id)

    # 5. Only deploy if new model is better
    if new_metrics["f1"] > old_metrics["f1"]:
        ModelRegistry.promote(run_id, job_id)
        logger.info(f"New model deployed for {job_id}: F1 {old_metrics['f1']:.3f} → {new_metrics['f1']:.3f}")
        return {"status": "deployed", "run_id": run_id, "f1": new_metrics["f1"]}
    else:
        logger.info(f"New model NOT deployed (no improvement) for {job_id}")
        return {"status": "skipped", "run_id": run_id}
```

---

## 8. Pillar 2 — LLM Interview Agent (LangGraph)

### 8.1 Interview Types

```
INTERVIEW TYPE 1 — Behavioural (all roles)
  Purpose: assess communication, past behaviour, culture fit
  Questions: "Tell me about a time you..." style
  Evaluation: STAR method adherence, communication clarity
  Duration: 15-20 minutes, 5-6 questions

INTERVIEW TYPE 2 — Technical Screening (engineering roles)
  Purpose: verify technical knowledge before live coding round
  Questions: concept questions, trade-off discussions, architecture
  Evaluation: depth, accuracy, ability to admit unknowns
  Duration: 20-25 minutes, 6-8 questions

INTERVIEW TYPE 3 — Role-Specific (custom per job config)
  Purpose: job-specific competencies
  Examples:
    Sales:   "Walk me through how you'd handle a procurement manager..."
    Design:  "How do you approach user research when time is constrained?"
    PM:      "Prioritise these 5 features with conflicting stakeholders"
  Evaluation: custom rubric per job
  Duration: 20-30 minutes, configurable

INTERVIEW TYPE 4 — Quick Qualifier (high-volume roles)
  Purpose: fast yes/no on basic fit
  Questions: 3 fixed questions, no follow-ups
  Evaluation: simple scoring rubric
  Duration: 10 minutes
```

### 8.2 LangGraph Interview State

```python
# app/agent/interview/state.py

from typing import TypedDict, Annotated, Literal
from operator import add

class InterviewState(TypedDict):
    # ── Session ───────────────────────────────────────────────────────
    session_id:          str
    candidate_id:        str
    job_id:              str
    interview_type:      Literal["behavioural", "technical", "role_specific", "quick"]

    # ── Job context (loaded at start) ─────────────────────────────────
    job_title:           str
    job_description:     str
    required_skills:     list[str]
    question_bank:       list[dict]    # [{question, category, weight, rubric}]
    evaluation_rubric:   dict          # custom rubric from JobConfig

    # ── Candidate context (from resume ML stage) ──────────────────────
    candidate_name:      str
    resume_summary:      str           # LLM-generated summary of their resume
    ml_score:            int
    ml_highlights:       list[str]     # top skills/experience to probe deeper

    # ── Conversation ──────────────────────────────────────────────────
    messages:            Annotated[list[dict], add]
    current_question_idx:int
    questions_asked:     list[str]
    follow_up_depth:     int           # how many follow-ups on current question

    # ── Evaluation ───────────────────────────────────────────────────
    answer_scores:       Annotated[list[dict], add]   # per-question scores
    running_score:       float
    red_flags:           list[str]
    green_flags:         list[str]

    # ── Flow control ──────────────────────────────────────────────────
    should_follow_up:    bool
    should_conclude:     bool
    max_questions:       int
    max_follow_ups:      int           # per question (default: 2)

    # ── Output ────────────────────────────────────────────────────────
    final_score:         float | None
    recommendation:      str | None   # "strong_yes" | "yes" | "no" | "strong_no"
    ai_summary:          str | None
    areas_of_concern:    list[str]
    areas_of_strength:   list[str]
```

### 8.3 Interview Graph

```python
# app/agent/interview/graph.py

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.redis import RedisSaver

def build_interview_graph() -> CompiledGraph:
    builder = StateGraph(InterviewState)

    builder.add_node("intro",            intro_node)
    builder.add_node("ask_question",     ask_question_node)
    builder.add_node("evaluate_answer",  evaluate_answer_node)
    builder.add_node("ask_follow_up",    ask_follow_up_node)
    builder.add_node("conclude",         conclude_node)
    builder.add_node("generate_report",  generate_report_node)

    builder.set_entry_point("intro")
    builder.add_edge("intro", "ask_question")

    # After asking a question, wait for candidate response (this is
    # the "human turn" — handled by the frontend sending a message back)
    # The graph is invoked again per candidate response

    builder.add_edge("ask_question", END)   # wait for candidate

    # When candidate responds: evaluate their answer
    builder.add_conditional_edges(
        "evaluate_answer",
        route_after_evaluation,
        {
            "follow_up":      "ask_follow_up",
            "next_question":  "ask_question",
            "conclude":       "conclude",
        },
    )

    builder.add_edge("ask_follow_up", END)   # wait for candidate
    builder.add_edge("conclude", "generate_report")
    builder.add_edge("generate_report", END)

    return builder.compile(
        checkpointer=RedisSaver(redis_client)
    )

def route_after_evaluation(state: InterviewState) -> str:
    """Decide: ask follow-up, move on, or conclude?"""
    current_q = state["current_question_idx"]
    n_follow_ups = state["follow_up_depth"]

    # Conclude if: enough questions asked OR score clearly established
    if current_q >= state["max_questions"]:
        return "conclude"
    if state["running_score"] < 20 and current_q >= 3:
        return "conclude"   # candidate clearly not a fit — don't waste their time

    # Follow up if: incomplete answer or interesting thread to pull
    if state["should_follow_up"] and n_follow_ups < state["max_follow_ups"]:
        return "follow_up"

    # Otherwise: next question
    return "next_question"
```

### 8.4 Interview Nodes

```python
# app/agent/interview/nodes.py

INTERVIEWER_SYSTEM_PROMPT = """
You are a professional interviewer conducting a {interview_type} interview
for the role of {job_title}.

Candidate context:
  Name: {candidate_name}
  Resume highlights: {resume_summary}
  Skills to probe: {ml_highlights}

Interview rules:
  1. Ask ONE question at a time. Never two questions in one message.
  2. Listen actively — reference what the candidate said in follow-ups.
  3. Be warm but professional. This is not a test, it's a conversation.
  4. For technical questions: don't reveal the answer if they're wrong.
     Probe: "Can you walk me through your reasoning?"
  5. For vague answers: probe for specifics: "Can you give me a concrete example?"
  6. Never evaluate aloud. Just ask questions.
  7. Max 2 follow-ups per question, then move on.
  8. Conclude naturally when all main questions are covered.

Current progress: {current_q}/{max_q} questions asked.
"""

async def ask_question_node(state: InterviewState) -> dict:
    """Select next question from bank and ask it"""
    q_bank   = state["question_bank"]
    idx      = state["current_question_idx"]
    question = q_bank[idx]

    messages = list(state["messages"]) + [{
        "role":    "user",
        "content": f"Ask question #{idx+1} from the bank: {question['question']}\n"
                   f"Category: {question['category']}\n"
                   f"Tone guidance: {question.get('tone_guidance', 'professional')}",
    }]

    response = await openai.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "system", "content": build_system_prompt(state)}] + messages,
        max_tokens=150,
        temperature=0.3,
    )
    agent_question = response.choices[0].message.content

    return {
        "messages":              [{"role": "assistant", "content": agent_question}],
        "questions_asked":       [question["question"]],
        "current_question_idx":  idx,   # not incremented yet — wait for answer
    }

class AnswerEvaluation(BaseModel):
    score:           float    # 0-10 for this answer
    reasoning:       str      # why this score
    should_follow_up:bool     # is a follow-up warranted?
    follow_up_hook:  str | None  # what specifically to follow up on
    red_flags:       list[str]
    green_flags:     list[str]

async def evaluate_answer_node(state: InterviewState) -> dict:
    """Evaluate the candidate's latest answer against the rubric"""
    last_answer = state["messages"][-1]["content"]
    current_q   = state["question_bank"][state["current_question_idx"]]

    evaluation: AnswerEvaluation = await groq_instructor.chat.completions.create(
        model="llama-3.3-70b-versatile",    # cheaper model for eval
        response_model=AnswerEvaluation,
        messages=[{
            "role": "user",
            "content": f"""Evaluate this interview answer.

Question: {current_q['question']}
Category: {current_q['category']}
Evaluation rubric: {current_q['rubric']}

Candidate's answer:
{last_answer}

Score 0-10. Be strict but fair. Look for:
  Specificity (vague vs concrete examples)
  Depth (surface vs demonstrated understanding)
  Relevance (answered what was asked)
  Communication (clarity, structure)""",
        }],
    )

    new_score    = (state["running_score"] * state["current_question_idx"] +
                    evaluation.score) / (state["current_question_idx"] + 1)

    return {
        "answer_scores": [{
            "question":          current_q["question"],
            "answer":            last_answer,
            "score":             evaluation.score,
            "reasoning":         evaluation.reasoning,
            "category":          current_q["category"],
        }],
        "running_score":          new_score,
        "should_follow_up":       evaluation.should_follow_up,
        "red_flags":              evaluation.red_flags,
        "green_flags":            evaluation.green_flags,
        "current_question_idx":   state["current_question_idx"] + 1,
        "follow_up_depth":        0,   # reset follow-up depth for new question
    }

class InterviewReport(BaseModel):
    overall_score:      float         # 0-100
    recommendation:     Literal["strong_yes", "yes", "maybe", "no", "strong_no"]
    executive_summary:  str           # 2-3 sentence overview for recruiter
    strengths:          list[str]     # top 3 demonstrated strengths
    areas_of_concern:   list[str]     # top 3 concerns
    notable_quotes:     list[str]     # 2-3 memorable things the candidate said
    suggested_probes:   list[str]     # questions for the next interview round
    hiring_risk:        Literal["low", "medium", "high"]

async def generate_report_node(state: InterviewState) -> dict:
    """Generate final report after interview concludes"""
    full_transcript = "\n".join(
        f"{m['role'].upper()}: {m['content']}"
        for m in state["messages"]
    )

    report: InterviewReport = await openai_instructor.chat.completions.create(
        model="gpt-4o",
        response_model=InterviewReport,
        messages=[{
            "role": "user",
            "content": f"""Generate a comprehensive interview assessment report.

Job: {state['job_title']}
Interview type: {state['interview_type']}
ML screening score: {state['ml_score']}/100

Full transcript:
{full_transcript}

Per-question scores:
{json.dumps(state['answer_scores'], indent=2)}

Generate an honest, specific assessment. The recruiter trusts this report
to make hiring decisions. Do not be generous to avoid hurting feelings.""",
        }],
    )

    return {
        "final_score":       report.overall_score,
        "recommendation":    report.recommendation,
        "ai_summary":        report.executive_summary,
        "areas_of_strength": report.strengths,
        "areas_of_concern":  report.areas_of_concern,
    }
```

### 8.5 Interview Portal — Candidate UX

```
CANDIDATE FLOW:

1. Receive email with unique interview link
   Link contains: job_id, candidate_id, token (24hr expiry)

2. Land on interview portal
   - Company logo + role title
   - "What to expect" screen (duration, type, tips)
   - Mic/camera check (if voice mode enabled)
   - Accept terms: "This interview is AI-conducted and recorded"

3. Interview begins
   - AI introduces itself
   - Questions appear in chat or voice
   - Candidate types (text mode) or speaks (voice mode)
   - Progress indicator: "Question 3 of 7"

4. Interview ends
   - Thank you screen
   - "Your response has been reviewed. You'll hear back in 2-3 days."

5. Post-interview (optional)
   - Candidate receives summary of their own performance (if enabled)
   - Feedback on areas to improve (recruiter decision)

TEXT MODE vs VOICE MODE:
  Text mode:  candidate types answers, AI responds in text
  Voice mode: Deepgram STT → LangGraph → OpenAI TTS (same Pipecat stack as VoiceIQ)
  Configurable per job: some roles prefer voice (sales, customer success)
```

---

## 9. Pillar 3 — MLOps + LLMOps + DevOps Layer

### 9.1 MLOps Pipeline

```
┌─────────────────────────────────────────────────────────────────────┐
│                         MLOps Pipeline                              │
│                                                                     │
│  DATA COLLECTION          TRAINING           SERVING                │
│  ┌───────────┐      ┌────────────────┐  ┌────────────────┐         │
│  │  Resumes  │      │ Feature Eng    │  │  FastAPI       │         │
│  │  +labels  │─────▶│ + SMOTE        │─▶│  /ml/score     │         │
│  │  (Postgres│      │ + Train/eval   │  │  (loads from   │         │
│  │  +S3)     │      │ + MLflow log   │  │   MLflow)      │         │
│  └───────────┘      └────────────────┘  └───────┬────────┘         │
│                              │                   │                  │
│  MONITORING                  │         ┌─────────▼──────────┐      │
│  ┌───────────┐               │         │  Drift Monitor     │      │
│  │ Evidently │◀──────────────┘         │  (Evidently AI)    │      │
│  │ AI drift  │                         │  Check every 24h   │      │
│  │ dashboard │                         └─────────┬──────────┘      │
│  └───────────┘                                   │                  │
│                                                   │ drift detected   │
│  RETRAINING                              ┌────────▼──────────┐     │
│  ┌───────────────────────────────────┐   │  Celery retrain   │     │
│  │  celery beat → check_drift_task  │◀──│  task queued      │     │
│  │  → if drift: trigger retrain     │   └───────────────────┘     │
│  │  → train new model               │                              │
│  │  → evaluate on holdout           │                              │
│  │  → if better: promote in MLflow  │                              │
│  └───────────────────────────────────┘                             │
│                                                                     │
│  DVC: version training data in S3                                   │
│  MLflow: version models + experiments                               │
│  Great Expectations: validate data quality before training          │
└─────────────────────────────────────────────────────────────────────┘
```

### 9.2 LLMOps Pipeline

```
┌─────────────────────────────────────────────────────────────────────┐
│                        LLMOps Pipeline                              │
│                                                                     │
│  PROMPT MANAGEMENT                                                  │
│  prompts/                                                           │
│    interview/                                                       │
│      behavioural/                                                   │
│        system_v1.txt     ← current production                       │
│        system_v2.txt     ← candidate for A/B test                   │
│        meta.json         ← {current: v1, ab_test: {v2: 0.20}}      │
│      technical/                                                     │
│        system_v1.txt                                                │
│      evaluator/                                                     │
│        answer_eval_v1.txt                                           │
│        answer_eval_v2.txt   ← improved rubric                       │
│                                                                     │
│  EVAL PIPELINE (RAGAS)                                              │
│  ┌────────────────────────────────────────────────────────────┐    │
│  │  Golden dataset: 50 sample Q&A pairs with human scores    │    │
│  │      ↓                                                     │    │
│  │  Run evaluator prompt on each pair                         │    │
│  │      ↓                                                     │    │
│  │  Compare AI score vs human score                          │    │
│  │      ↓                                                     │    │
│  │  Metrics: RMSE, correlation, agreement rate               │    │
│  │      ↓                                                     │    │
│  │  Threshold: correlation > 0.75 to pass CI gate            │    │
│  └────────────────────────────────────────────────────────────┘    │
│                                                                     │
│  LANGFUSE OBSERVABILITY                                             │
│  Every LLM call traces: prompt version, input, output,             │
│  tokens used, cost, latency, answer score                           │
│                                                                     │
│  A/B TESTING                                                        │
│  20% of interviews use system_v2, 80% use system_v1                │
│  Track: completion rate, avg score, recruiter acceptance rate       │
│  Promote v2 if all metrics ≥ v1                                    │
└─────────────────────────────────────────────────────────────────────┘
```

### 9.3 CI/CD Pipeline

```yaml
# .github/workflows/deploy.yml

name: HireSense CI/CD

on:
  push:
    branches: [main, staging]

jobs:
  # ── Step 1: Unit + Integration Tests ─────────────────────────────
  test:
    runs-on: ubuntu-latest
    services:
      postgres: { image: postgres:16 }
      redis:    { image: redis:7 }
    steps:
      - uses: actions/checkout@v4
      - run: pip install -r requirements.txt
      - run: pytest tests/ -v --cov=app --cov-report=xml
      - run: ruff check app/      # linting
      - run: mypy app/            # type checking

  # ── Step 2: ML Model Eval Gate ───────────────────────────────────
  ml_eval:
    needs: test
    runs-on: ubuntu-latest
    steps:
      - name: Run ML model evaluation on holdout set
        run: python scripts/eval_ml_model.py
        # Fails if F1 < 0.75 on holdout set
        # Fails if bias metrics exceed thresholds
        env:
          MLFLOW_TRACKING_URI: ${{ secrets.MLFLOW_URI }}

  # ── Step 3: LLMOps Eval Gate ─────────────────────────────────────
  llm_eval:
    needs: test
    runs-on: ubuntu-latest
    steps:
      - name: Run RAGAS eval on interview evaluator prompt
        run: python scripts/eval_interview_prompt.py
        # Fails if evaluator correlation < 0.75
        # Fails if answer quality regression detected
        env:
          OPENAI_API_KEY:  ${{ secrets.OPENAI_API_KEY }}
          LANGFUSE_SECRET: ${{ secrets.LANGFUSE_SECRET }}

  # ── Step 4: Build + Push Docker Image ────────────────────────────
  build:
    needs: [ml_eval, llm_eval]
    runs-on: ubuntu-latest
    steps:
      - run: docker build -t hiresense-api:${{ github.sha }} .
      - run: docker push $ECR_REGISTRY/hiresense-api:${{ github.sha }}

  # ── Step 5: Deploy to ECS ─────────────────────────────────────────
  deploy:
    needs: build
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    steps:
      - name: Update ECS service
        run: |
          aws ecs update-service \
            --cluster hiresense-prod \
            --service hiresense-api \
            --force-new-deployment
      - name: Wait for deployment
        run: aws ecs wait services-stable --cluster hiresense-prod --services hiresense-api
```

### 9.4 Bias Audit Report

```python
# app/ml/bias_audit.py

class BiasAuditor:
    """
    Ensure ML model doesn't discriminate on protected attributes.
    Run: after every model training + on-demand from dashboard.
    Report: stored in S3 + displayed in recruiter dashboard.
    """

    PROTECTED_ATTRIBUTES = ["gender", "age_group", "institution_tier", "location"]

    def audit(
        self,
        model: ResumeScreenerModel,
        X: np.ndarray,
        y_true: np.ndarray,
        metadata_df: pd.DataFrame,  # contains protected attributes
    ) -> dict:
        results = {}
        y_pred = model.predict_batch(X)

        for attr in self.PROTECTED_ATTRIBUTES:
            if attr not in metadata_df.columns:
                continue

            groups = metadata_df[attr].unique()
            group_metrics = {}

            for group in groups:
                mask   = metadata_df[attr] == group
                acc    = accuracy_score(y_true[mask], y_pred[mask])
                pct_pos= y_pred[mask].mean()  # selection rate
                group_metrics[str(group)] = {
                    "n":            int(mask.sum()),
                    "accuracy":     round(float(acc), 4),
                    "selection_pct":round(float(pct_pos * 100), 1),
                }

            # Demographic parity check: selection rate should be similar across groups
            selection_rates = [m["selection_pct"] for m in group_metrics.values()]
            disparity = max(selection_rates) - min(selection_rates)
            results[attr] = {
                "groups":           group_metrics,
                "disparity_pct":    round(disparity, 1),
                "passes_audit":     disparity < 10.0,  # <10% disparity = acceptable
            }

        return {
            "audit_date":     datetime.utcnow().isoformat(),
            "overall_passes": all(r["passes_audit"] for r in results.values()),
            "by_attribute":   results,
        }
```

---

## 10. Custom Job Configuration

The most important product decision in HireSense. Every job is different. A startup hiring its first engineering lead needs completely different criteria than an enterprise hiring junior support staff.

### 10.1 JobConfig Schema

```python
# app/models/job_config.py

from pydantic import BaseModel, Field
from typing import Literal

class QuestionConfig(BaseModel):
    question:        str
    category:        Literal["technical", "behavioural", "situational", "role_specific"]
    weight:          float = 1.0    # relative importance in scoring
    rubric:          str            # what a good answer looks like
    tone_guidance:   str = "professional"
    follow_up_depth: int = 2        # max follow-ups for this question

class MLWeightConfig(BaseModel):
    """Custom weights for the ML feature vector per job"""
    skills_match:      float = 1.0
    experience_years:  float = 0.8
    education_tier:    float = 0.5
    project_quality:   float = 0.7
    keyword_density:   float = 0.6
    github_present:    float = 0.3

class JobConfig(BaseModel):
    # ── Basic Info ────────────────────────────────────────────────────
    job_id:          str
    title:           str
    department:      str
    job_description: str
    company_id:      str

    # ── ML Screening Config ───────────────────────────────────────────
    required_skills:      list[str]    # must-have skills (hard filter)
    preferred_skills:     list[str]    # nice-to-have (soft score)
    min_experience_years: float = 0.0
    max_experience_years: float = 30.0
    required_education:   Literal["none", "diploma", "bachelors", "masters", "phd"] = "bachelors"
    ml_threshold:         int = 60     # score >= this to be shortlisted
    ml_weights:           MLWeightConfig = Field(default_factory=MLWeightConfig)
    custom_ml_rule:       str | None   # Python expression evaluated on ParsedResume

    # ── Interview Config ──────────────────────────────────────────────
    interview_type:       Literal["behavioural", "technical", "role_specific", "quick"]
    interview_mode:       Literal["text", "voice", "either"] = "text"
    max_questions:        int = 6
    max_follow_ups:       int = 2
    interview_duration_min: int = 20
    question_bank:        list[QuestionConfig]
    evaluation_rubric:    dict         # custom rubric for report generation

    # ── Workflow Config ───────────────────────────────────────────────
    auto_invite_shortlisted:    bool = True   # auto-send interview link to shortlisted
    recruiter_review_required:  bool = True   # require human review before invite?
    show_candidate_report:      bool = False  # show candidate their own score?
    allow_retake:               bool = False  # can candidate retake the interview?
    interview_window_days:      int = 7       # days candidate has to complete interview

    # ── Notifications ─────────────────────────────────────────────────
    notify_on_completion:   list[str]   # recruiter emails to notify
    candidate_sms:          bool = True
    candidate_email:        bool = True

    # ── Compliance ────────────────────────────────────────────────────
    require_consent:        bool = True
    record_interview:       bool = True
    retention_days:         int = 90

# Example: Software Engineer role
EXAMPLE_SE_CONFIG = JobConfig(
    job_id="job_se_2026_01",
    title="Senior Software Engineer — Backend",
    department="Engineering",
    job_description="We're building the next generation of...",

    required_skills=["python", "fastapi", "postgresql", "redis", "docker"],
    preferred_skills=["kubernetes", "aws", "langchain", "celery"],
    min_experience_years=3.0,
    max_experience_years=10.0,
    ml_threshold=65,
    ml_weights=MLWeightConfig(
        skills_match=1.5,       # skills more important for SE
        experience_years=0.8,
        education_tier=0.4,     # education less important
        project_quality=1.2,    # projects matter a lot
        github_present=0.8,
    ),

    interview_type="technical",
    max_questions=7,
    question_bank=[
        QuestionConfig(
            question="Tell me about the most complex system you've designed. What were the key trade-offs?",
            category="technical",
            weight=2.0,
            rubric="Look for: specific scale/complexity, clear trade-off reasoning, awareness of distributed systems challenges",
            follow_up_depth=2,
        ),
        QuestionConfig(
            question="How do you approach database performance issues in production?",
            category="technical",
            weight=1.5,
            rubric="Look for: query optimisation, indexing strategy, caching, monitoring approach",
        ),
        QuestionConfig(
            question="Describe a time when you disagreed with a technical decision. How did you handle it?",
            category="behavioural",
            weight=1.0,
            rubric="Look for: constructive communication, data-driven argument, outcome orientation",
        ),
    ],
    evaluation_rubric={
        "technical_depth":  "deep > surface level answers with specific examples",
        "system_thinking":  "considers scale, failure modes, trade-offs",
        "communication":    "explains complex topics clearly",
    },
)
```

### 10.2 Recruiter Configuration UI

```
JOB CONFIGURATION WIZARD (Recruiter UI):

Step 1 — Basic Info
  Job title, department, description (rich text editor)
  Number of positions, deadline

Step 2 — ML Screening Settings
  Required skills (autocomplete from skill database)
  Preferred skills
  Experience range (slider: 0-20 years)
  Education requirement (dropdown)
  Screening threshold (slider: 40-90, default 60)
  Custom rule (code editor with syntax highlighting)
    Example: "years_experience > 3 and 'python' in skills"

Step 3 — ML Weight Tuner
  6 sliders: skills / experience / education / projects / github / keywords
  Preview: "A candidate with Python + 4yr exp + IIT + 3 projects would score 78"

Step 4 — Interview Design
  Interview type selector (behavioural / technical / role-specific / quick)
  Mode: text or voice
  Duration + question count

Step 5 — Question Bank Builder
  Add questions from: template library | AI-generated | custom
  AI generates questions from job description automatically
  Per-question: category, weight, rubric, follow-up depth
  Drag to reorder

Step 6 — Workflow Settings
  Auto-invite toggle
  Review required toggle
  Interview window
  Notification recipients

Step 7 — Preview + Launch
  Simulate: "If we run this config, a candidate with these skills would score X"
  ML model selection: use global model or train job-specific model
  Launch → job goes live
```

---

## 11. Candidate Portal

```
DESIGN PRINCIPLES:
  Mobile-first (candidates apply on phone)
  No account required (link-based access, JWT)
  Accessible (WCAG 2.1 AA compliance)
  Available 24/7 (async, no scheduling)

FLOW:
  /interview/{token}
    → Verify token (job_id, candidate_id, expiry)
    → Load job context
    → Check: already completed? Return "You've already completed this"
    → Check: expired? Return "This interview link has expired"

  /interview/{token}/intro
    → Company branding (logo, colours from company config)
    → Role: {job_title} at {company}
    → What to expect: duration, question count, recording notice
    → Consent checkbox (required)
    → "Start Interview" button

  /interview/{token}/session
    → WebSocket connection opened
    → LangGraph session initialised in Redis
    → Real-time interview (text or voice mode)
    → Progress bar: "Question 3 of 6"
    → Each message timestamped in transcript

  /interview/{token}/done
    → "Thank you for completing the interview"
    → If show_candidate_report=true: show score + feedback
    → "You'll hear back within {N} days"
    → Option: "Download your transcript"

VOICE MODE (optional):
  Browser microphone access via Web Audio API
  Audio streamed to FastAPI WebSocket
  Pipecat pipeline (same as VoiceIQ) handles STT + LLM + TTS
  Voice is always available as text fallback
```

---

## 12. Recruiter Dashboard

```
PAGES:

/dashboard
  Active jobs (cards): candidates in pipeline, last activity
  Today's summary: new resumes scored, interviews completed
  Action items: resumes awaiting review, interviews to review

/jobs
  All jobs table: status, applications, shortlisted %, interviews, hires
  Create new job (wizard)
  
/jobs/{job_id}
  Pipeline kanban:
    [Applied] → [ML Screened] → [Interview Invited] → [Interview Done] → [Next Round] → [Hired]
  Bulk actions: invite selected, reject selected, export CSV

/jobs/{job_id}/candidates/{candidate_id}
  Left panel:
    Resume PDF viewer
    ML score: 78/100
    SHAP explanation: "Skills: +28 | Experience: +19 | Education: +14 | Missing: React (-8)"
    Override button: [Shortlist] or [Reject] with reason
  Right panel (if interview done):
    AI interview score: 7.2/10
    Recommendation: strong_yes
    Transcript (full, searchable)
    AI summary paragraph
    Top 3 strengths + 2 areas of concern
    Notable quotes
    Suggested probes for next round

/ml-ops
  Current model: v3.2 | F1: 0.81 | Deployed: 2 days ago
  Drift report: last check 6 hours ago | No drift detected
  Override rate this week: 8% (below 15% threshold — model healthy)
  Training history: chart of F1 over model versions
  [Trigger manual retrain] button
  Bias audit report: all attributes pass

/llm-ops
  Active prompts: interview_system_v2, evaluator_v3
  A/B tests running: [system v1 (80%) vs v2 (20%)]
    v1: avg score 6.8, recruiter acceptance 74%
    v2: avg score 7.1, recruiter acceptance 79% ← looking good
  Cost tracking: ₹X/interview this week
  Langfuse link → full trace dashboard

/analytics
  Time to shortlist: 2.1 days avg
  Time to first interview: 4.3 days avg
  ML accuracy (vs final hires): 71%
  Interview completion rate: 84%
  Top dropout reasons: tech issues (4%), too long (8%), expired (4%)
  Cost per hire: ₹8,400
  Funnel: 400 applied → 89 shortlisted → 67 interviews → 23 next round → 4 hires
```

---

## 13. Database Schema

```sql
-- ── Companies ─────────────────────────────────────────────────────────
CREATE TABLE companies (
    id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name         TEXT NOT NULL,
    logo_url     TEXT,
    primary_color TEXT DEFAULT '#6366F1',
    clerk_org_id TEXT UNIQUE,
    plan         TEXT DEFAULT 'starter',  -- starter | pro | enterprise
    created_at   TIMESTAMPTZ DEFAULT NOW()
);

-- ── Jobs ──────────────────────────────────────────────────────────────
CREATE TABLE jobs (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    company_id          UUID REFERENCES companies(id) ON DELETE CASCADE,
    title               TEXT NOT NULL,
    department          TEXT,
    description         TEXT,
    status              TEXT DEFAULT 'draft',  -- draft | active | paused | closed
    config              JSONB NOT NULL,        -- full JobConfig serialised
    ml_model_version    TEXT,                  -- MLflow run_id of active model
    created_by          TEXT,                  -- Clerk user_id
    created_at          TIMESTAMPTZ DEFAULT NOW(),
    updated_at          TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX ON jobs (company_id, status);

-- ── Candidates ────────────────────────────────────────────────────────
CREATE TABLE candidates (
    id                UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    company_id        UUID REFERENCES companies(id),
    job_id            UUID REFERENCES jobs(id),
    name              TEXT,
    email             TEXT,
    phone             TEXT,
    resume_s3_key     TEXT,                -- S3 path to original file
    parsed_resume     JSONB,               -- ParsedResume structured output
    feature_vector    FLOAT[] ,            -- extracted ML features
    applied_at        TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX ON candidates (job_id);
CREATE INDEX ON candidates (email, company_id);

-- ── ML Screening Results ──────────────────────────────────────────────
CREATE TABLE ml_screenings (
    id                UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    candidate_id      UUID REFERENCES candidates(id),
    job_id            UUID REFERENCES jobs(id),
    model_version     TEXT,               -- MLflow run_id used
    ml_score          INTEGER,            -- 0-100
    shortlisted       BOOLEAN,
    confidence        FLOAT,
    shap_explanation  JSONB,              -- list of {feature, contribution, direction}
    processed_at      TIMESTAMPTZ DEFAULT NOW()
);

-- ── Recruiter Overrides (feeds ML retraining) ─────────────────────────
CREATE TABLE recruiter_overrides (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    screening_id    UUID REFERENCES ml_screenings(id),
    candidate_id    UUID REFERENCES candidates(id),
    job_id          UUID REFERENCES jobs(id),
    ml_decision     BOOLEAN,         -- what the model said
    recruiter_decision BOOLEAN,      -- what the recruiter decided
    reason          TEXT,
    recruiter_id    TEXT,            -- Clerk user_id
    overridden_at   TIMESTAMPTZ DEFAULT NOW()
);
-- This table is the gold mine — every row is a labelled training example

-- ── Interview Sessions ────────────────────────────────────────────────
CREATE TABLE interview_sessions (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    candidate_id        UUID REFERENCES candidates(id),
    job_id              UUID REFERENCES jobs(id),
    invite_token        TEXT UNIQUE NOT NULL,
    status              TEXT DEFAULT 'invited',  -- invited | in_progress | completed | expired | abandoned
    interview_type      TEXT,
    interview_mode      TEXT DEFAULT 'text',
    started_at          TIMESTAMPTZ,
    completed_at        TIMESTAMPTZ,
    duration_seconds    INTEGER,
    consent_given       BOOLEAN DEFAULT FALSE,
    recording_s3_key    TEXT,
    langraph_thread_id  TEXT,     -- LangGraph session key
    created_at          TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX ON interview_sessions (invite_token);
CREATE INDEX ON interview_sessions (candidate_id, status);

-- ── Interview Turns ───────────────────────────────────────────────────
CREATE TABLE interview_turns (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id      UUID REFERENCES interview_sessions(id),
    turn_number     INTEGER,
    speaker         TEXT,       -- "agent" | "candidate"
    content         TEXT,
    question_category TEXT,
    answer_score    FLOAT,
    tokens_used     INTEGER,
    cost_usd        NUMERIC(10,6),
    latency_ms      INTEGER,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX ON interview_turns (session_id);

-- ── Interview Reports ─────────────────────────────────────────────────
CREATE TABLE interview_reports (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id          UUID REFERENCES interview_sessions(id) UNIQUE,
    candidate_id        UUID REFERENCES candidates(id),
    job_id              UUID REFERENCES jobs(id),
    overall_score       FLOAT,           -- 0-100
    recommendation      TEXT,            -- strong_yes | yes | maybe | no | strong_no
    executive_summary   TEXT,
    strengths           TEXT[],
    areas_of_concern    TEXT[],
    notable_quotes      TEXT[],
    suggested_probes    TEXT[],
    hiring_risk         TEXT,
    prompt_version      TEXT,            -- which prompt was used
    model_used          TEXT,
    generated_at        TIMESTAMPTZ DEFAULT NOW()
);

-- ── ML Model Registry (mirrors MLflow) ───────────────────────────────
CREATE TABLE ml_model_versions (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    job_id          UUID REFERENCES jobs(id),
    mlflow_run_id   TEXT UNIQUE,
    version_tag     TEXT,          -- "v1", "v2", etc.
    f1_score        FLOAT,
    precision_score FLOAT,
    recall_score    FLOAT,
    auc_roc         FLOAT,
    training_samples INTEGER,
    override_rate   FLOAT,         -- override rate that triggered this retrain
    is_production   BOOLEAN DEFAULT FALSE,
    bias_passes     BOOLEAN,
    trained_at      TIMESTAMPTZ DEFAULT NOW()
);

-- ── Prompt Versions ────────────────────────────────────────────────────
CREATE TABLE prompt_versions (
    id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name          TEXT,          -- "interview_system" | "answer_evaluator"
    version       TEXT,          -- "v1", "v2"
    content       TEXT,
    eval_score    FLOAT,         -- RAGAS / correlation score
    is_production BOOLEAN DEFAULT FALSE,
    ab_weight     FLOAT DEFAULT 0.0,    -- % of traffic (A/B testing)
    deployed_at   TIMESTAMPTZ,
    created_at    TIMESTAMPTZ DEFAULT NOW()
);

-- ── Pipeline Stage ─────────────────────────────────────────────────────
CREATE TABLE pipeline_stages (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    candidate_id    UUID REFERENCES candidates(id),
    job_id          UUID REFERENCES jobs(id),
    stage           TEXT,    -- applied | ml_screened | interview_invited |
                             -- interview_done | next_round | offer | hired | rejected
    moved_by        TEXT,    -- "system" | recruiter user_id
    moved_at        TIMESTAMPTZ DEFAULT NOW(),
    notes           TEXT
);
-- Full stage history — gives us the full audit trail + funnel analytics
```

---

## 14. API Reference

```
── JOB MANAGEMENT ────────────────────────────────────────────────────
POST   /api/jobs                          Create job with config
GET    /api/jobs                          List all jobs (company)
GET    /api/jobs/{job_id}                 Get job details + config
PATCH  /api/jobs/{job_id}                 Update config
PATCH  /api/jobs/{job_id}/status          Activate / pause / close
GET    /api/jobs/{job_id}/config/preview  Simulate: score a sample resume

── RESUME INTAKE ──────────────────────────────────────────────────────
POST   /api/jobs/{job_id}/resumes         Upload single resume (PDF/DOCX)
POST   /api/jobs/{job_id}/resumes/bulk    Upload ZIP of resumes (async Celery)
GET    /api/jobs/{job_id}/resumes         List candidates in pipeline
GET    /api/jobs/{job_id}/resumes/{id}    Get candidate details + ML score

── ML SCREENING ──────────────────────────────────────────────────────
POST   /api/jobs/{job_id}/screen/{id}     Re-score a single resume
POST   /api/jobs/{job_id}/overrides/{id}  Recruiter override (feeds retraining)
GET    /api/jobs/{job_id}/ml/metrics      Current model performance

── INTERVIEWS ────────────────────────────────────────────────────────
POST   /api/interviews/invite             Send interview invite to candidate(s)
GET    /api/interviews/{session_id}       Get session status + transcript
GET    /api/interviews/{session_id}/report  Get AI report
PATCH  /api/interviews/{session_id}/decision  Recruiter decision on interview

── INTERVIEW PORTAL (candidate-facing, JWT auth) ─────────────────────
GET    /portal/verify/{token}             Verify invite token
POST   /portal/session/start             Start interview session
WS     /portal/ws/{session_id}           WebSocket for real-time interview
GET    /portal/session/{session_id}/status  Check session status

── MLOPS ─────────────────────────────────────────────────────────────
GET    /api/mlops/models/{job_id}        Model version history
POST   /api/mlops/retrain/{job_id}       Trigger manual retraining
GET    /api/mlops/drift/{job_id}         Latest drift report
GET    /api/mlops/bias/{job_id}          Bias audit report

── LLMOPS ────────────────────────────────────────────────────────────
GET    /api/llmops/prompts               All prompt versions + eval scores
POST   /api/llmops/prompts/{name}/ab    Set A/B test split
POST   /api/llmops/eval/run             Trigger RAGAS eval on prompt version
GET    /api/llmops/cost                  LLM cost breakdown (by job, by type)

── ANALYTICS ─────────────────────────────────────────────────────────
GET    /api/analytics/funnel/{job_id}   Conversion funnel
GET    /api/analytics/time-to-hire       Time metrics
GET    /api/analytics/cost-per-hire      Cost breakdown
GET    /api/analytics/model-accuracy     ML accuracy vs final hires
```

---

## 15. Design Patterns Used

```
FACTORY PATTERN (LLM provider)
  InterviewAgentFactory.create(job_config)
  → returns correct LLM client based on job_config.interview_type
  → technical interviews use GPT-4o; quick interviews use Groq 8b

STRATEGY PATTERN (resume parsing)
  ParserStrategy: PDFParser | DOCXParser | TextParser
  Swapped based on file extension — pipeline doesn't know which

CHAIN OF RESPONSIBILITY (resume guardrails)
  DuplicateCheckGuard → BlacklistGuard → ContentGuard → FormatGuard
  Each can reject or pass through to next

OBSERVER PATTERN (pipeline events)
  EventBus: ResumeScored | InterviewCompleted | DriftDetected
  Observers: NotificationObserver | AnalyticsObserver | SlackObserver

REPOSITORY PATTERN (data access)
  CandidateRepository, JobRepository, InterviewRepository
  FakeRepository for tests — no DB required in unit tests

DECORATOR PATTERN (ML inference)
  @with_retry(max_retries=3)
  @with_cost_tracking(task="ml_inference")
  @with_langfuse_trace
  async def score_resume(...)

PROMPT REGISTRY (LLMOps)
  prompts/{name}/{version}.txt + meta.json
  PromptRegistry.get("interview_system") → always current
  A/B testing via ab_weight in meta.json

CIRCUIT BREAKER (LLM failover)
  GPT-4o → fallback to Groq 70b → fallback to cached response
  Opens after 5 failures, half-opens after 60s timeout
```

---

## 16. Guardrails

```
RESUME SCREENING GUARDRAILS:
  ✅ Protected attributes NOT used as features (name, gender inferred from name, age)
  ✅ SHAP audit: confirm bias doesn't flow through correlated features
  ✅ Minimum candidate count before ML: use rule-based fallback if < 50 training examples
  ✅ Score band review: flag if >90% of candidates score < 40 (config issue)

INTERVIEW AGENT GUARDRAILS:
  ✅ No discriminatory questions (age, family status, religion, nationality)
  ❌ Trigger keywords: "how old are you", "are you married", "do you have kids"
  → If candidate mentions protected attribute: acknowledge, redirect to job-relevant topic
  
  ✅ Candidate distress detection
  → Signs: "this is unfair", "I'm frustrated", very short answers 3 turns in a row
  → Action: acknowledge warmly, offer to end + reschedule
  
  ✅ No hallucinated job details
  → Agent only knows what's in job_description + question_bank
  → "Is there a bonus structure?" → "For specific compensation questions, our recruiter will follow up"
  
  ✅ Time limit enforcement
  → Warn at 80% of duration: "I have a couple more questions"
  → Hard stop at 100%: conclude gracefully

OUTPUT GUARDRAILS (interview report):
  ✅ Report must cite evidence from transcript
  → "Candidate demonstrates strong system design thinking (example: their answer
     about the caching strategy at question 3)"
  ✅ Recommendation must align with score
  → strong_yes: score >= 80
  → yes: score 65-79
  → maybe: score 50-64
  → no: score 35-49
  → strong_no: score < 35
  ✅ No language reflecting protected attributes in report
  → Post-process: check for gender pronouns, age references, name-based inferences
```

---

## 17. Phase-by-Phase Build Plan

### Phase 0 — Foundation (4 days)
```
Tasks:
  [ ] FastAPI project + Docker Compose
  [ ] Postgres + Redis + Qdrant + MLflow containers
  [ ] Clerk auth (recruiter login + org)
  [ ] S3 bucket setup
  [ ] Celery + Redis broker working
  [ ] Sentry + Langfuse connected
  [ ] React frontend scaffolded (Vite + Tailwind)
  [ ] GitHub Actions: lint + test on PR

Showable:
  Recruiter logs in with Clerk
  Company dashboard loads
  Database migrations applied cleanly
```

### Phase 1 — Job Configuration (5 days)
```
Tasks:
  [ ] JobConfig Pydantic model (full schema)
  [ ] Job creation API (POST /api/jobs)
  [ ] Job configuration wizard UI (all 7 steps)
  [ ] Question bank builder with AI question generation
  [ ] ML weight tuner UI (sliders + preview)
  [ ] Jobs list + detail page

Showable:
  Create a job with full config in UI
  AI generates 10 interview questions from job description
  Custom weights visualised in preview
```

### Phase 2 — Resume Parsing + Feature Engineering (6 days)
```
Tasks:
  [ ] PDF/DOCX parser (PDFMiner + python-docx)
  [ ] spaCy NER integration (name, email, skills extraction)
  [ ] FeatureEngineer (all 20 features)
  [ ] Celery task: parse_resume_task
  [ ] Bulk upload endpoint (ZIP → Celery → parse each)
  [ ] Candidate list UI (upload + see parsed results)

Showable:
  Upload 10 PDFs
  Watch them parse in real-time (WebSocket progress)
  See structured JSON: name, skills, experience, education
  Feature vector visible in candidate detail
```

### Phase 3 — ML Screening (8 days)
```
Tasks:
  [ ] ResumeScreenerModel (RF + XGBoost + LR ensemble)
  [ ] MLflow tracking (experiments, metrics, artifacts)
  [ ] SMOTE for class imbalance
  [ ] SHAP explanation per candidate
  [ ] Training script + evaluation on holdout
  [ ] Model registry: promote best model to production
  [ ] POST /api/jobs/{id}/screen (inference endpoint)
  [ ] Celery task: score_all_resumes_task
  [ ] ML score + SHAP UI in candidate detail
  [ ] Global model + job-specific model support

Showable:
  Upload 50 test resumes
  Run screening → all scored in < 30 seconds
  See: "Score 78 | Shortlisted | Because: Python (+28), 4yr exp (+19), Missing: React (-8)"
  Recruiter override → logged as training example
```

### Phase 4 — Interview Agent (LangGraph) (8 days)
```
Tasks:
  [ ] InterviewState TypedDict
  [ ] Interview graph (intro → ask → evaluate → follow_up / next / conclude)
  [ ] Nodes: ask_question, evaluate_answer, conclude, generate_report
  [ ] LangGraph Redis checkpointer
  [ ] Candidate portal (invite token, consent, interview UI)
  [ ] WebSocket handler for real-time interview
  [ ] Interview report generation
  [ ] Langfuse tracing on all LLM calls

Showable:
  Click "Invite to Interview" → candidate receives email
  Candidate opens link → AI interview starts
  Ask 5 questions with follow-ups
  Report generated: score + recommendation + summary
  Full transcript visible in recruiter dashboard
```

### Phase 5 — MLOps (5 days)
```
Tasks:
  [ ] Evidently AI drift monitoring
  [ ] Celery beat: check_drift_task (every 24h)
  [ ] Override rate monitor → retrain trigger
  [ ] Retrain pipeline: load data → validate → train → eval → compare → promote
  [ ] Great Expectations: data quality on training set
  [ ] DVC: version training datasets in S3
  [ ] Bias audit report (BiasAuditor)
  [ ] MLOps dashboard page in recruiter UI

Showable:
  Manually trigger retraining from dashboard
  See: F1 before and after, bias report passes
  Override 20% of decisions → drift triggered → auto retrain
```

### Phase 6 — LLMOps (4 days)
```
Tasks:
  [ ] Prompt registry (versioned .txt files + meta.json)
  [ ] Golden dataset: 50 interview Q&A pairs with human scores
  [ ] RAGAS eval script (correlation with human scores)
  [ ] CI/CD gate: eval must pass before deploy
  [ ] A/B testing: route 20% of interviews to new prompt
  [ ] LLMOps dashboard: prompt versions, A/B results, cost tracking
  [ ] Langfuse dashboard embedded (iframe or API)

Showable:
  Create prompt v2 → run RAGAS eval → pass gate → deploy
  A/B test running: v1 vs v2 with live traffic split
  Cost per interview visible ($X total, $Y/interview)
```

### Phase 7 — Recruiter Dashboard + Analytics (5 days)
```
Tasks:
  [ ] Pipeline kanban (drag and drop stages)
  [ ] Bulk actions (invite, reject, export)
  [ ] Side-by-side: resume PDF + ML score + interview report
  [ ] Analytics: funnel, time-to-hire, cost-per-hire, model accuracy
  [ ] Export: CSV of candidates + scores
  [ ] Email templates (invite, rejection, next-round)
  [ ] Notification system (Celery + SES)

Showable:
  Full recruiter workflow end-to-end
  Pipeline kanban with drag
  Analytics dashboard with real data
```

### Phase 8 — Production (5 days)
```
Tasks:
  [ ] ECS Fargate deployment (API + Celery)
  [ ] RDS Postgres (Multi-AZ)
  [ ] ElastiCache Redis
  [ ] S3 + CloudFront for frontend
  [ ] GitHub Actions: full CI/CD with eval gate
  [ ] CloudWatch alarms: API errors, Celery queue depth, ML drift
  [ ] Rate limiting (per company, per job)
  [ ] Candidate portal on subdomain (interview.hiresense.com)

Showable:
  Live on real domain
  End-to-end: job posted → 50 resumes → screened → interviewed → reports
  CloudWatch dashboard live
  CI/CD: push to main → tests → eval gate → auto deploy
```

---

## 18. Folder Structure

```
hiresense/
├── api/                          ← FastAPI application
│   ├── main.py
│   ├── routes/
│   │   ├── jobs.py
│   │   ├── resumes.py
│   │   ├── interviews.py
│   │   ├── mlops.py
│   │   ├── llmops.py
│   │   └── analytics.py
│   ├── models/
│   │   ├── job_config.py         ← JobConfig, QuestionConfig, MLWeightConfig
│   │   ├── candidates.py
│   │   └── interviews.py
│   ├── services/
│   │   ├── job_service.py
│   │   ├── notification_service.py
│   │   └── analytics_service.py
│   └── core/
│       ├── config.py
│       ├── database.py
│       ├── cache.py
│       └── auth.py               ← Clerk JWT verification
│
├── ml/                           ← Classical ML pipeline
│   ├── resume_parser.py          ← PDF/DOCX parsing + spaCy NER
│   ├── feature_engineer.py       ← 20-feature vector extraction
│   ├── model.py                  ← ResumeScreenerModel (RF + XGB + LR ensemble)
│   ├── drift_monitor.py          ← Evidently AI drift detection
│   ├── retrain_pipeline.py       ← Celery retraining task
│   ├── bias_auditor.py           ← Fairness metrics + SHAP audit
│   └── model_registry.py         ← MLflow wrapper
│
├── agent/                        ← LangGraph interview agent
│   ├── state.py                  ← InterviewState TypedDict
│   ├── graph.py                  ← LangGraph graph definition
│   ├── nodes/
│   │   ├── intro.py
│   │   ├── ask_question.py
│   │   ├── evaluate_answer.py
│   │   ├── conclude.py
│   │   └── generate_report.py
│   └── guardrails.py             ← Interview safety checks
│
├── prompts/                      ← LLMOps: versioned prompt files
│   ├── interview/
│   │   ├── behavioural/
│   │   │   ├── system_v1.txt
│   │   │   ├── system_v2.txt
│   │   │   └── meta.json
│   │   └── technical/
│   │       ├── system_v1.txt
│   │       └── meta.json
│   └── evaluator/
│       ├── answer_eval_v1.txt
│       ├── answer_eval_v2.txt
│       └── meta.json
│
├── workers/                      ← Celery tasks
│   ├── celery_app.py
│   ├── resume_tasks.py           ← parse_resume, score_resume, bulk_score
│   ├── ml_tasks.py               ← check_drift, retrain_model, run_bias_audit
│   ├── interview_tasks.py        ← process_transcript, generate_report
│   └── notification_tasks.py     ← send_invite, send_decision
│
├── evals/                        ← LLMOps evaluation
│   ├── golden_dataset.json       ← 50 interview Q&A pairs with human scores
│   ├── eval_interview_prompt.py  ← RAGAS eval script
│   └── eval_ml_model.py          ← ML holdout eval + bias check
│
├── frontend/                     ← React app (monorepo)
│   ├── recruiter/                ← main dashboard
│   │   ├── src/pages/
│   │   │   ├── Dashboard.tsx
│   │   │   ├── Jobs.tsx
│   │   │   ├── JobConfig.tsx     ← 7-step wizard
│   │   │   ├── Candidate.tsx
│   │   │   ├── MLOps.tsx
│   │   │   └── LLMOps.tsx
│   └── candidate/               ← interview portal
│       └── src/pages/
│           ├── Verify.tsx
│           ├── Intro.tsx
│           ├── Interview.tsx     ← WebSocket chat interface
│           └── Done.tsx
│
├── scripts/
│   ├── seed_demo_data.py         ← create demo job + 100 sample resumes
│   ├── train_initial_model.py    ← bootstrap training on seed data
│   └── generate_golden_dataset.py
│
├── tests/
│   ├── test_parser.py
│   ├── test_feature_engineer.py
│   ├── test_ml_model.py
│   ├── test_interview_graph.py
│   └── test_guardrails.py
│
├── .github/workflows/
│   └── deploy.yml                ← full CI/CD with eval gate
│
├── docker-compose.yml
├── Dockerfile
├── pyproject.toml                ← ruff + mypy config
└── mlflow/                       ← MLflow tracking server config
```

---

## 19. Resume Deliverables + Interview Story

### Deliverables

| Deliverable | Detail |
|---|---|
| Live demo | End-to-end: upload resumes → ML screened → interview → report |
| GitHub repo | Clean code + architecture diagram + README |
| ML model card | F1 score, precision/recall, SHAP importance chart, bias report |
| LLMOps report | RAGAS eval results, A/B test outcomes, cost per interview |
| MLflow dashboard | Screenshot of experiment tracking + model versions |
| Langfuse dashboard | LLM traces with cost + latency per interview |
| CI/CD pipeline | GitHub Actions with eval gate visible in logs |
| Architecture diagram | System design drawing (interviewable) |

### Key Numbers (Target)

```
ML model F1:             0.78+ on holdout set
Resume scoring speed:    400 resumes in < 30 seconds
Interview completion:    > 80% of invited candidates complete
AI-human score correlation: > 0.75 (RAGAS gate)
Time to shortlist:       3 days (vs 3-4 weeks manual)
Cost per screened candidate: < ₹0.50 (ML inference only)
Cost per interview:      < ₹35 (LLM calls)
```

### The 2-Minute Interview Story

> *"The project I'm most proud of is HireSense — an end-to-end AI hiring platform I built for an EdTech client who needed to hire 40 engineers but had only 2 recruiters.*
>
> *There are three technical layers. First: classical ML for resume screening. I chose scikit-learn and XGBoost — not GPT-4 — because it's 1,000x faster, the SHAP explanations are auditable, and recruiter overrides feed directly into model retraining. The ML pipeline is fully MLOps: MLflow for model versioning, Evidently AI for drift detection, and an automated retraining Celery task that triggers when override rate exceeds 15%.*
>
> *Second: a LangGraph interview agent. Candidates click a link, complete a 20-minute AI interview — text or voice — and we generate a full report: score, recommendation, transcript, key quotes, suggested probes for the next round. The agent adapts: it asks follow-ups when an answer is incomplete, moves on when it's satisfied, and concludes early if the candidate clearly isn't a fit.*
>
> *Third: the LLMOps layer. All interview prompts are versioned files. Before any prompt change deploys, it must pass a RAGAS evaluation against 50 human-scored Q&A pairs — correlation must exceed 0.75. The CI/CD pipeline blocks the deploy otherwise. We do A/B testing on prompt versions in production.*
>
> *The result: 400 resumes screened in 12 minutes. 67 AI interviews completed asynchronously over 2 days. Recruiters spent 2 hours reviewing reports instead of 3 weeks doing phone screens. Time to technical interview went from 4 weeks to 3 days."*

---

*HireSense PRD v1.0 | AI Hiring Intelligence Platform*
*Classical ML (scikit-learn + XGBoost) + LLM Interview Agent (LangGraph) + MLOps (MLflow + Evidently) + LLMOps (Langfuse + RAGAS) + DevOps (GitHub Actions + ECS)*
