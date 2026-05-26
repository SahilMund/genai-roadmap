# 🦜 LangChain — Intro to Advanced
> **Complete Study Notes | Theory + Code + Use Cases | Interview Prep**
> Part of: GenAI + LLMOps Engineering Roadmap 2026 — Phase 4
> Estimated time: 12–15 hrs
> Note: We do NOT use LangChain in SynapseIQ (direct SDKs preferred).
> But you MUST know it — every job description mentions it, every interview asks about it.

---

## 📑 Table of Contents

1. [What is LangChain — Theory](#1-what-is-langchain--theory)
2. [Core Concepts](#2-core-concepts)
3. [Chains — Building Pipelines](#3-chains--building-pipelines)
4. [Memory in LangChain](#4-memory-in-langchain)
5. [Tools and Agents](#5-tools-and-agents)
6. [Document Loaders + Text Splitters](#6-document-loaders--text-splitters)
7. [Embeddings + Vector Stores](#7-embeddings--vector-stores)
8. [Retrieval Augmented Generation (RAG) with LangChain](#8-retrieval-augmented-generation-rag-with-langchain)
9. [Advanced — LangChain Expression Language (LCEL)](#9-advanced--langchain-expression-language-lcel)
10. [Advanced — Callbacks and Streaming](#10-advanced--callbacks-and-streaming)
11. [Advanced — Output Parsers](#11-advanced--output-parsers)
12. [LangChain vs Direct SDKs — When to Use What](#12-langchain-vs-direct-sdks--when-to-use-what)
13. [Real-World Use Cases](#13-real-world-use-cases)
14. [Interview Cheat Sheet](#14-interview-cheat-sheet)
15. [Quick Revision Cards](#15-quick-revision-cards)

---

## 1. What is LangChain — Theory

### 🧑‍🎓 Explain Like I'm in BTech 2nd Year

Imagine you're building a robot that can:
1. Read your company's PDF documents
2. Answer questions about them
3. Remember what you asked before
4. Use Google Search when it doesn't know the answer
5. Format the answer as a JSON report

Without LangChain, you'd wire all of this together yourself — write the PDF reader, connect it to OpenAI, build the memory system, add the search tool, write the output formatter. Each piece is custom code.

**LangChain is like LEGO blocks for LLM apps.** Each block is a pre-built component — document loaders, text splitters, embeddings, vector stores, memory, tools, output parsers. You snap them together into a pipeline (called a "chain") instead of building each piece from scratch.

```
Without LangChain:                    With LangChain:
  PDF reader (custom)                   PDFLoader ─┐
  + Text splitter (custom)              Splitter  ─┤
  + Embedding (OpenAI SDK)              Embeddings─┤→ RAGChain → Answer
  + Vector store (custom)               Chroma    ─┤
  + LLM call (custom)                   ChatOpenAI─┘
  + Memory (custom)
  + Output parser (custom)
  = 500 lines of boilerplate            = 50 lines
```

### The Official Definition

> LangChain is an open-source framework for building applications powered by language models. It provides standard interfaces for chains, agents, and memory — enabling developers to create context-aware, reasoning applications.

**Released:** October 2022 by Harrison Chase at Robust Intelligence
**Language:** Python + JavaScript/TypeScript
**GitHub:** 90K+ stars (one of the fastest-growing repos in history)

### Why LangChain Became Popular (2022-2024)

```
Timeline:
  2022: GPT-3.5 released. Everyone wants to build LLM apps.
        OpenAI SDK is great but bare. No memory, no tools, no RAG.
        LangChain fills the gap with pre-built components.

  2023: GPT-4 + Claude + Gemini all released.
        LangChain abstracts away provider differences.
        "One framework to rule them all" moment.
        Became the standard for LLM app development.

  2024: LLM SDKs matured. Direct SDK calls became simpler.
        LangChain's abstraction became a liability (too many layers).
        LangGraph launched as a better agent framework.
        Many teams moved away from LangChain for complex apps.

  2026: LangChain still dominant in:
        - Job descriptions (legacy codebases)
        - Quick prototypes
        - Tutorials and learning
        - Teams that started in 2022-2023
```

### The Architecture

```
LangChain has 4 main packages:

langchain-core        → Base abstractions (interfaces, schema)
                         Runnable interface (everything is composable)
                         No LLM integrations — just the contracts

langchain             → Chains, agents, memory
                         The "brain" of the framework

langchain-community   → 500+ integrations
                         Document loaders, vector stores, tools
                         PDF loaders, Chroma, Pinecone, Wikipedia, etc.

langchain-openai      → Official OpenAI integration
langchain-anthropic   → Official Anthropic integration
langchain-groq        → Official Groq integration
(one package per provider)
```

```python
# Installation
pip install langchain langchain-openai langchain-community
pip install langchain-groq langchain-anthropic  # as needed

# Core imports pattern
from langchain_openai import ChatOpenAI             # LLM
from langchain.prompts import ChatPromptTemplate    # Prompts
from langchain.chains import LLMChain               # Chain (old style)
from langchain_core.runnables import RunnablePassthrough  # LCEL (new style)
```

---

## 2. Core Concepts

### 2.1 Models — The LLM Interface

LangChain wraps every LLM behind a standard interface. You swap providers by changing one line.

```python
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_groq import ChatGroq

# All have the same interface — .invoke(), .stream(), .batch()
openai_llm    = ChatOpenAI(model="gpt-4o", temperature=0)
anthropic_llm = ChatAnthropic(model="claude-sonnet-4-20250514")
groq_llm      = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

# Same call regardless of provider
response = openai_llm.invoke("What is RAG?")
response = anthropic_llm.invoke("What is RAG?")
response = groq_llm.invoke("What is RAG?")

print(response.content)  # always .content — standardised
print(response.usage_metadata)  # tokens used
```

**Two types of models:**
```python
# 1. LLMs (text-in, text-out) — older style
from langchain_openai import OpenAI
llm = OpenAI(model="gpt-3.5-turbo-instruct")
result = llm.invoke("Translate to French: Hello")
# result is a string

# 2. Chat Models (messages-in, message-out) — modern style
from langchain_openai import ChatOpenAI
chat = ChatOpenAI(model="gpt-4o")
from langchain_core.messages import HumanMessage, SystemMessage
result = chat.invoke([
    SystemMessage(content="You are a helpful translator"),
    HumanMessage(content="Translate to French: Hello"),
])
# result is an AIMessage object with .content
```

---

### 2.2 Prompts — Templates for LLMs

```python
from langchain.prompts import (
    PromptTemplate,           # simple string template
    ChatPromptTemplate,       # chat message template
    FewShotPromptTemplate,    # includes examples
    MessagesPlaceholder,      # slot for conversation history
)

# Simple prompt template
template = PromptTemplate(
    input_variables=["topic", "tone"],
    template="Write a {tone} explanation of {topic} for a college student.",
)
formatted = template.format(topic="neural networks", tone="humorous")
# → "Write a humorous explanation of neural networks for a college student."

# Chat prompt template (most common in production)
chat_template = ChatPromptTemplate.from_messages([
    ("system", "You are an expert {domain} tutor. Always give examples."),
    ("human",  "Explain {concept} in simple terms."),
])
messages = chat_template.format_messages(
    domain="machine learning",
    concept="gradient descent",
)
# → [SystemMessage(...), HumanMessage(...)]

# With conversation history placeholder
chat_template_with_history = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    MessagesPlaceholder(variable_name="history"),  # inject past messages here
    ("human", "{question}"),
])
```

**Why templates matter:**
```
Without templates:
  f"Answer this: {user_question}"  ← no system prompt, inconsistent

With templates:
  Centralised prompt management
  Variable injection with validation
  Reusable across your codebase
  Easy to version (store templates in files)
```

---

### 2.3 Messages — The Currency of Chat

```python
from langchain_core.messages import (
    SystemMessage,    # from the developer (instructions)
    HumanMessage,     # from the user
    AIMessage,        # from the LLM
    ToolMessage,      # result of a tool call
    FunctionMessage,  # legacy, use ToolMessage instead
)

messages = [
    SystemMessage(content="You are a helpful data analyst."),
    HumanMessage(content="What is the average revenue?"),
    AIMessage(content="I need to check the data. Let me query the database."),
    ToolMessage(content="Average revenue: ₹45,230", tool_call_id="call_123"),
    HumanMessage(content="What about last quarter?"),
]

# Pass to any LLM
response = groq_llm.invoke(messages)
```

---

### 2.4 Output Parsers — Structure the LLM Response

```python
from langchain.output_parsers import (
    StrOutputParser,         # raw string
    JsonOutputParser,        # parse JSON from LLM output
    PydanticOutputParser,    # validate against Pydantic model
    CommaSeparatedListOutputParser,
)
from pydantic import BaseModel

# String parser (simplest)
parser = StrOutputParser()
chain  = groq_llm | parser
result = chain.invoke("What is 2+2?")
# result is a clean string, not an AIMessage object

# Pydantic parser (most powerful)
class AnalysisResult(BaseModel):
    sentiment:  str   # "positive" | "negative" | "neutral"
    confidence: float
    key_points: list[str]
    summary:    str

pydantic_parser = PydanticOutputParser(pydantic_object=AnalysisResult)

prompt = ChatPromptTemplate.from_messages([
    ("system", "Analyse the text. {format_instructions}"),
    ("human", "{text}"),
])

chain = prompt | groq_llm | pydantic_parser
result: AnalysisResult = chain.invoke({
    "text": "The product is amazing, best purchase of the year!",
    "format_instructions": pydantic_parser.get_format_instructions(),
})
print(result.sentiment)    # "positive"
print(result.confidence)   # 0.95
print(result.key_points)   # ["amazing", "best purchase"]
```

---

## 3. Chains — Building Pipelines

### 🧑‍🎓 Explain Like I'm in BTech 2nd Year

A chain is a recipe. You define the steps once, then run it with different inputs.

```
Recipe for making tea:
  Step 1: Boil water
  Step 2: Add tea bag
  Step 3: Add sugar
  Step 4: Add milk

LangChain Chain for Q&A:
  Step 1: Format user question into prompt
  Step 2: Send to LLM
  Step 3: Parse output
  Step 4: Return clean answer
```

### 3.1 The Old Way — LLMChain (Legacy, Still in Many Codebases)

```python
from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

llm = ChatGroq(model="llama-3.3-70b-versatile")
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert SQL developer."),
    ("human",  "Write a SQL query to: {task}"),
])

chain = LLMChain(llm=llm, prompt=prompt)
result = chain.invoke({"task": "find top 10 customers by revenue"})
print(result["text"])  # the SQL query
```

### 3.2 The New Way — LCEL (LangChain Expression Language)

LCEL uses the pipe operator `|` to compose components. Think of it like Unix pipes:
`cat file.txt | grep "error" | head -20`

```python
# LCEL — clean, composable, lazy evaluation
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate

llm    = ChatGroq(model="llama-3.3-70b-versatile")
prompt = ChatPromptTemplate.from_template("Explain {topic} in one sentence.")
parser = StrOutputParser()

# Build chain with pipe operator
chain = prompt | llm | parser

# Run it
result = chain.invoke({"topic": "transformers"})
print(result)  # clean string output

# Chain is lazy — nothing runs until .invoke() / .stream() / .batch()
# This means: you can build chains without executing them
#             pass chains as arguments to other functions
#             test the structure before running
```

**LCEL with multiple steps:**
```python
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

# Step 1: Translate question
translate_chain = (
    ChatPromptTemplate.from_template("Translate to English: {text}")
    | llm
    | StrOutputParser()
)

# Step 2: Answer the translated question
answer_chain = (
    ChatPromptTemplate.from_template("Answer this clearly: {question}")
    | llm
    | StrOutputParser()
)

# Compose: translate THEN answer
full_chain = (
    {"question": translate_chain}  # output of translate becomes input for answer
    | answer_chain
)

result = full_chain.invoke({"text": "Transformer kya hota hai?"})
# 1. Translates Hindi → English: "What is a transformer?"
# 2. Answers: "A transformer is a neural network architecture..."
```

### 3.3 Sequential Chain — Step by Step

```python
# Real use case: document analysis pipeline
# Step 1: Summarise document
# Step 2: Extract key entities
# Step 3: Generate action items

summarise_prompt = ChatPromptTemplate.from_template(
    "Summarise this document in 3 bullet points:\n\n{document}"
)
entity_prompt = ChatPromptTemplate.from_template(
    "Extract all company names, people, and dates from:\n\n{summary}"
)
action_prompt = ChatPromptTemplate.from_template(
    "Based on this summary, list 3 action items:\n\n{summary}"
)

# Build sequential chain
summarise_chain = summarise_prompt | llm | StrOutputParser()

# Parallel: extract entities AND action items from the same summary
from langchain_core.runnables import RunnableParallel

analysis_chain = RunnableParallel(
    summary=RunnablePassthrough(),   # pass summary through unchanged
    entities=entity_prompt | llm | StrOutputParser(),
    actions=action_prompt | llm | StrOutputParser(),
)

full_pipeline = summarise_chain | analysis_chain

result = full_pipeline.invoke({"document": "Reliance Industries acquired..."})
print(result["summary"])   # 3 bullet points
print(result["entities"])  # companies, people, dates
print(result["actions"])   # 3 action items
```

### 3.4 Router Chain — Different Paths for Different Inputs

```python
from langchain_core.runnables import RunnableBranch

# Route queries to different chains based on topic
sql_chain  = sql_prompt  | llm | StrOutputParser()
rag_chain  = rag_prompt  | llm | StrOutputParser()
chat_chain = chat_prompt | llm | StrOutputParser()

def classify_query(query: str) -> str:
    """Classify query type using LLM"""
    classifier = (
        ChatPromptTemplate.from_template(
            "Classify as exactly one of: sql, document, chat\nQuery: {query}"
        )
        | llm
        | StrOutputParser()
    )
    return classifier.invoke({"query": query}).strip().lower()

# Router branch
router = RunnableBranch(
    (lambda x: "sql"  in x["type"], sql_chain),
    (lambda x: "doc"  in x["type"], rag_chain),
    chat_chain,  # default
)

def route(query: str):
    query_type = classify_query(query)
    return router.invoke({"type": query_type, "query": query})

result = route("Show me top 10 customers by revenue")   # → sql_chain
result = route("What does our return policy say?")       # → rag_chain
result = route("How are you doing today?")               # → chat_chain
```

---

## 4. Memory in LangChain

### 🧑‍🎓 Explain Like I'm in BTech 2nd Year

LLMs have no memory by default. Every call is stateless — like talking to someone who has amnesia and forgets everything after each sentence.

Memory in LangChain is like giving that person a notepad. They write down what you said, read it before responding, and feel "continuous."

```
Without memory:
  You:  "My name is Rahul"
  LLM:  "Nice to meet you!"
  You:  "What's my name?"
  LLM:  "I don't know your name."   ← forgot immediately

With memory:
  You:  "My name is Rahul"
  LLM:  "Nice to meet you, Rahul!"
  You:  "What's my name?"
  LLM:  "Your name is Rahul."       ← remembered from notepad
```

### 4.1 Types of Memory

```python
# 1. ConversationBufferMemory — stores everything (simple, gets expensive)
from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory(
    memory_key="history",     # key injected into prompt
    return_messages=True,     # return as Message objects (not string)
)
memory.save_context(
    {"input": "My name is Rahul"},
    {"output": "Nice to meet you, Rahul!"},
)
print(memory.load_memory_variables({}))
# {'history': [HumanMessage("My name is Rahul"), AIMessage("Nice to meet you, Rahul!")]}

# 2. ConversationBufferWindowMemory — only last N exchanges (cost-controlled)
from langchain.memory import ConversationBufferWindowMemory

memory = ConversationBufferWindowMemory(k=5)  # only last 5 exchanges
# Older exchanges are dropped automatically

# 3. ConversationSummaryMemory — summarises old messages (best for long convos)
from langchain.memory import ConversationSummaryMemory

memory = ConversationSummaryMemory(llm=llm)
# Automatically summarises old messages using the LLM
# Keeps token count low while preserving context

# 4. ConversationSummaryBufferMemory — hybrid (keeps recent + summarises old)
from langchain.memory import ConversationSummaryBufferMemory

memory = ConversationSummaryBufferMemory(
    llm=llm,
    max_token_limit=1000,  # when exceeded, start summarising
)
```

### 4.2 Memory with Chains

```python
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferWindowMemory
from langchain_groq import ChatGroq

llm    = ChatGroq(model="llama-3.3-70b-versatile")
memory = ConversationBufferWindowMemory(k=10, return_messages=True)

conversation = ConversationChain(
    llm=llm,
    memory=memory,
    verbose=True,  # shows what's happening internally (good for learning)
)

# Multi-turn conversation
r1 = conversation.predict(input="Hi, I'm Sahil. I work at an AI startup.")
r2 = conversation.predict(input="We're building a RAG system for enterprise clients.")
r3 = conversation.predict(input="What do you think about my background for this project?")
# LLM remembers Sahil's name and job context from earlier turns
```

### 4.3 Redis-Backed Memory (Production)

```python
# In production, memory must survive server restarts
# Store in Redis, not in-process memory

from langchain.memory import RedisChatMessageHistory
from langchain.memory import ConversationBufferMemory

# Each conversation has a unique session_id
session_id = "user_abc_conv_123"

history = RedisChatMessageHistory(
    session_id=session_id,
    url="redis://localhost:6379",
    ttl=86400,  # expire after 24 hours
)

memory = ConversationBufferMemory(
    chat_memory=history,
    return_messages=True,
    memory_key="history",
)

# Now memory persists in Redis
# If server restarts: same session_id → same history loaded from Redis
```

---

## 5. Tools and Agents

### 🧑‍🎓 Explain Like I'm in BTech 2nd Year

A chain follows a fixed path: Step 1 → Step 2 → Step 3. You decide the path.

An agent is different. It has a set of tools (like a Swiss Army knife) and decides FOR ITSELF which tool to use and when, based on the task.

```
Chain (fixed path):
  You: "Translate and summarise this French article"
  Chain: [always translates first] → [always summarises second]
  Works great. But what if the article is already in English?
  Chain still translates it (wastes money, time).

Agent (decides dynamically):
  You: "Help me research and write about India's GDP growth"
  Agent: "Let me think...
    Step 1: I'll search the web for recent GDP data (uses search tool)
    Step 2: I'll read these 3 articles (uses web reader tool)
    Step 3: I'll query the World Bank database (uses API tool)
    Step 4: I'll synthesise all this and write the report"
  
  YOU didn't specify these steps. The agent decided.
```

### 5.1 Creating Tools

```python
from langchain.tools import tool, StructuredTool
from langchain_community.tools import DuckDuckGoSearchRun, WikipediaQueryRun

# Method 1: @tool decorator (simplest)
@tool
def get_stock_price(ticker: str) -> str:
    """
    Get the current stock price for a given ticker symbol.
    Use this when the user asks about stock prices or market data.

    Args:
        ticker: Stock ticker symbol like 'RELIANCE.NS' or 'INFY.NS'
    """
    # Real implementation would call an API
    return f"Current price of {ticker}: ₹2,340.50 (+1.2% today)"

@tool
def calculate_compound_interest(
    principal: float,
    rate: float,
    years: int,
) -> str:
    """
    Calculate compound interest.

    Args:
        principal: Initial investment amount in rupees
        rate: Annual interest rate as percentage (e.g., 8.5 for 8.5%)
        years: Number of years
    """
    amount = principal * (1 + rate/100) ** years
    profit = amount - principal
    return f"Principal: ₹{principal:,.0f} → Amount after {years} years: ₹{amount:,.0f} (Profit: ₹{profit:,.0f})"

# Method 2: Community tools (pre-built)
search_tool    = DuckDuckGoSearchRun()
wikipedia_tool = WikipediaQueryRun()

# All tools in a list
tools = [
    get_stock_price,
    calculate_compound_interest,
    search_tool,
    wikipedia_tool,
]
```

**Critical insight about tool descriptions:**
```python
# ❌ BAD — vague description, agent won't know when to use it
@tool
def get_data(query: str) -> str:
    """Get some data"""    # ← too vague
    ...

# ✅ GOOD — specific description with use cases
@tool
def get_customer_orders(customer_id: str) -> str:
    """
    Retrieve all orders for a specific customer from the database.
    Use this tool ONLY when:
    - User asks about a specific customer's order history
    - You need order status, items, or payment info for a customer
    
    Do NOT use for:
    - General order statistics (use get_order_analytics instead)
    - Product information (use get_product_details instead)
    
    Args:
        customer_id: Customer ID in format 'CUST-XXXXX' (e.g., 'CUST-10042')
    """
    ...

# The quality of your tool description = quality of agent decisions
# Bad description → agent uses wrong tool → bad results
```

### 5.2 ReAct Agent — The Standard Pattern

ReAct = **Re**asoning + **Act**ing. The agent reasons about what to do, acts by calling a tool, observes the result, and repeats.

```python
from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub

# Load standard ReAct prompt from LangChain Hub
react_prompt = hub.pull("hwchase17/react")

# Create agent (brain) — knows about tools, can reason about which to use
agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=react_prompt,
)

# AgentExecutor (body) — actually runs the agent loop
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,    # shows each Thought/Action/Observation step
    max_iterations=10,
    handle_parsing_errors=True,  # don't crash on malformed LLM output
)

# Run
result = agent_executor.invoke({
    "input": "What is the current stock price of Infosys, and if I invested ₹1,00,000 ten years ago at the current price appreciation rate, how much would it be worth today?"
})

# The agent will:
# THOUGHT: "I need the current Infosys stock price first"
# ACTION: get_stock_price("INFY.NS")
# OBSERVATION: "Current price: ₹1,540 (+0.8% today)"
# THOUGHT: "Now I need to calculate compound interest with this appreciation rate"
# ACTION: calculate_compound_interest(100000, 0.8, 10)  ← uses observed data
# OBSERVATION: "Amount after 10 years: ₹1,08,307"
# THOUGHT: "I have all the information needed"
# FINAL ANSWER: "Infosys is currently trading at ₹1,540..."
```

### 5.3 OpenAI Functions Agent (Modern, More Reliable)

```python
from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

# More reliable than ReAct — uses structured function calling
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful financial analyst. Use tools to get accurate data."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),  # where agent reasoning goes
])

agent = create_openai_functions_agent(
    llm=ChatOpenAI(model="gpt-4o"),  # OpenAI functions work best with GPT
    tools=tools,
    prompt=prompt,
)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    memory=ConversationBufferWindowMemory(
        memory_key="chat_history",
        return_messages=True,
        k=10,
    ),
)

result = agent_executor.invoke({"input": "Compare Reliance and Infosys stock prices"})
```

---

## 6. Document Loaders + Text Splitters

### 6.1 Document Loaders

```python
from langchain_community.document_loaders import (
    PyPDFLoader,           # PDF files
    CSVLoader,             # CSV files
    TextLoader,            # .txt files
    WebBaseLoader,         # web pages
    DirectoryLoader,       # entire folder
    NotionDBLoader,        # Notion databases
    GoogleDriveLoader,     # Google Drive
    UnstructuredPDFLoader, # complex PDFs (scanned, tables, images)
)

# PDF Loader
loader   = PyPDFLoader("policy_document.pdf")
pages    = loader.load()          # list of Document objects
# pages[0].page_content → text content of page 1
# pages[0].metadata     → {"source": "policy.pdf", "page": 0}

# Web page loader
loader  = WebBaseLoader("https://docs.python.org/3/library/asyncio.html")
docs    = loader.load()

# Load entire directory of PDFs
loader = DirectoryLoader(
    "documents/",
    glob="**/*.pdf",         # match all PDFs recursively
    loader_cls=PyPDFLoader,
    show_progress=True,
)
all_docs = loader.load()
print(f"Loaded {len(all_docs)} pages from {len(set(d.metadata['source'] for d in all_docs))} files")
```

### 6.2 Text Splitters

```python
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,  # best general purpose
    CharacterTextSplitter,            # simple split by character
    TokenTextSplitter,                # split by actual token count
    MarkdownHeaderTextSplitter,       # respects markdown structure
    PythonCodeTextSplitter,           # respects Python code structure
)

# RecursiveCharacterTextSplitter — the go-to choice
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,        # target chunk size (chars)
    chunk_overlap=200,      # overlap between chunks (maintains context)
    length_function=len,    # how to measure length
    separators=["\n\n", "\n", ". ", " ", ""],  # try these in order
)
# Tries to split on \n\n first (paragraphs)
# If chunks still too big → splits on \n (lines)
# If still too big → splits on ". " (sentences)
# Last resort: splits on spaces or characters

chunks = splitter.split_documents(pages)  # takes list[Document] → list[Document]
print(f"Split {len(pages)} pages → {len(chunks)} chunks")
print(f"Avg chunk size: {sum(len(c.page_content) for c in chunks)//len(chunks)} chars")
print(chunks[0].page_content)    # first chunk text
print(chunks[0].metadata)        # inherited from original document

# Token-based splitting (respects actual LLM token limits)
from langchain.text_splitter import TokenTextSplitter

token_splitter = TokenTextSplitter(
    encoding_name="cl100k_base",   # OpenAI encoding
    chunk_size=200,                # 200 tokens per chunk
    chunk_overlap=20,
)

# Markdown-aware splitting (preserves section structure)
from langchain.text_splitter import MarkdownHeaderTextSplitter

md_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=[
        ("#",   "h1"),
        ("##",  "h2"),
        ("###", "h3"),
    ]
)
md_chunks = md_splitter.split_text(markdown_content)
# Each chunk includes header metadata:
# {"h1": "Installation", "h2": "Requirements", "h3": "Python Setup"}
```

---

## 7. Embeddings + Vector Stores

### 7.1 Embeddings

```python
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings

# OpenAI embeddings (best quality, costs money)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vector = embeddings.embed_query("What is machine learning?")
print(len(vector))   # 1536 dimensions

# Embed multiple texts efficiently
vectors = embeddings.embed_documents([
    "Machine learning is a subset of AI",
    "Deep learning uses neural networks",
    "Transformers use attention mechanisms",
])
print(len(vectors))    # 3
print(len(vectors[0])) # 1536

# HuggingFace embeddings (free, runs locally)
hf_embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3",  # excellent multilingual model
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)
```

### 7.2 Vector Stores

```python
from langchain_community.vectorstores import Chroma, FAISS, Qdrant
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Chroma (easy local dev, persistent storage)
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db",  # saves to disk
    collection_name="my_docs",
)
# Reload later:
vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings,
    collection_name="my_docs",
)

# FAISS (in-memory, blazing fast, no persistence by default)
vectorstore = FAISS.from_documents(
    documents=chunks,
    embedding=embeddings,
)
# Save/load manually:
vectorstore.save_local("faiss_index")
vectorstore = FAISS.load_local("faiss_index", embeddings)

# Similarity search
query = "What is the return policy for electronics?"
results = vectorstore.similarity_search(query, k=5)
# results: list[Document] — top 5 most similar chunks

# With scores
results_with_scores = vectorstore.similarity_search_with_score(query, k=5)
for doc, score in results_with_scores:
    print(f"Score: {score:.4f} | {doc.page_content[:100]}")
# Lower score = more similar (L2 distance for FAISS)

# MMR — Maximum Marginal Relevance (diversifies results)
results_mmr = vectorstore.max_marginal_relevance_search(
    query,
    k=5,             # return 5 results
    fetch_k=20,      # consider top 20 candidates
    lambda_mult=0.5, # 0=max diversity, 1=max relevance
)
# Avoids returning 5 nearly-identical chunks
```

---

## 8. Retrieval Augmented Generation (RAG) with LangChain

### 8.1 Basic RAG Pipeline

```python
# The full RAG pipeline in LangChain
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate

# Setup
llm        = ChatGroq(model="llama-3.3-70b-versatile")
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore= Chroma(persist_directory="./db", embedding_function=embeddings)

# RetrievalQA chain (old but still common in production)
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",   # "stuff" = put all retrieved docs in one prompt
    retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
    return_source_documents=True,  # include which docs were used
)

result = qa_chain.invoke({"query": "What is the warranty period?"})
print(result["result"])           # the answer
print(result["source_documents"]) # which chunks were retrieved
```

### 8.2 LCEL RAG Pipeline (Modern Approach)

```python
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Custom RAG prompt — grounding the LLM in the retrieved context
rag_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful assistant.
Answer ONLY based on the provided context.
If the answer is not in the context, say "I don't have that information."
Never make up information.

Context:
{context}
"""),
    ("human", "{question}"),
])

def format_docs(docs):
    """Format retrieved docs into a single context string with source info"""
    return "\n\n---\n\n".join(
        f"Source: {doc.metadata.get('source', 'Unknown')}, "
        f"Page: {doc.metadata.get('page', 'N/A')}\n"
        f"{doc.page_content}"
        for doc in docs
    )

retriever = vectorstore.as_retriever(
    search_type="mmr",           # MMR for diversity
    search_kwargs={"k": 5, "fetch_k": 20},
)

# LCEL RAG chain
rag_chain = (
    {
        "context":  retriever | format_docs,   # retrieve + format
        "question": RunnablePassthrough(),     # pass question through unchanged
    }
    | rag_prompt
    | llm
    | StrOutputParser()
)

answer = rag_chain.invoke("What is the return policy for electronic items?")
print(answer)
```

### 8.3 RAG with Source Citations

```python
from langchain_core.runnables import RunnableParallel

# Chain that returns BOTH answer and source documents
rag_chain_with_sources = RunnableParallel({
    "answer":  rag_chain,
    "sources": retriever,  # also return the retrieved documents
})

result = rag_chain_with_sources.invoke("What are the warranty terms?")

print("Answer:", result["answer"])
print("\nSources used:")
for doc in result["sources"]:
    print(f"  - {doc.metadata['source']}, page {doc.metadata.get('page', '?')}")
```

### 8.4 Conversational RAG (Multi-Turn)

```python
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    output_key="answer",  # which output to store in memory
)

conversational_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory,
    return_source_documents=True,
    verbose=True,
)

# First question
r1 = conversational_chain.invoke({"question": "What is the return policy?"})
print(r1["answer"])

# Follow-up — chain remembers the context
r2 = conversational_chain.invoke({"question": "What about for electronics specifically?"})
# LLM understands "What about" refers to return policy from previous turn
print(r2["answer"])

# Third follow-up
r3 = conversational_chain.invoke({"question": "And what about software products?"})
print(r3["answer"])
```

---

## 9. Advanced — LangChain Expression Language (LCEL)

### 9.1 Runnable Interface — Everything is Composable

```python
# In LCEL, everything implements the Runnable interface:
# .invoke(input)          → single input, synchronous
# .ainvoke(input)         → single input, async
# .batch([input1, input2]) → multiple inputs, parallel
# .abatch([...])           → multiple inputs, async parallel
# .stream(input)           → streaming output
# .astream(input)          → async streaming

from langchain_groq import ChatGroq
llm = ChatGroq(model="llama-3.3-70b-versatile")

# All of these work on any chain/component:
result  = llm.invoke("What is 2+2?")
results = llm.batch(["What is 2+2?", "What is 3+3?", "What is 4+4?"])
# batch is parallelised automatically

async def async_example():
    result = await llm.ainvoke("What is 2+2?")
    results = await llm.abatch(["Q1", "Q2", "Q3"])
    async for chunk in llm.astream("Write a story"):
        print(chunk.content, end="", flush=True)
```

### 9.2 RunnablePassthrough — Pass Data Through

```python
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

# RunnablePassthrough: pass input unchanged (useful for routing data)
chain = (
    {
        "original_question": RunnablePassthrough(),   # keep original
        "rewritten_question": rewrite_chain,          # also rewrite it
    }
    | answer_chain
)

# RunnableLambda: wrap any Python function as a Runnable
def add_metadata(text: str) -> dict:
    return {"text": text, "word_count": len(text.split()), "processed": True}

metadata_step = RunnableLambda(add_metadata)
result = metadata_step.invoke("Hello world")
# → {"text": "Hello world", "word_count": 2, "processed": True}
```

### 9.3 RunnableParallel — Fan-Out Pattern

```python
from langchain_core.runnables import RunnableParallel

# Run multiple chains on the SAME input simultaneously
multi_analysis = RunnableParallel(
    sentiment=sentiment_chain,      # runs concurrently
    summary=summary_chain,          # runs concurrently
    key_entities=entity_chain,      # runs concurrently
    action_items=action_chain,      # runs concurrently
)

# All 4 chains run in parallel — total time = slowest chain, not sum
result = multi_analysis.invoke({"text": "Quarterly report content..."})
# result["sentiment"]    → "positive"
# result["summary"]      → "Revenue grew 15%..."
# result["key_entities"] → ["CEO John", "Q3 2025", "Mumbai"]
# result["action_items"] → ["Review supply chain", "Expand to Tier 2 cities"]
```

### 9.4 Retries and Fallbacks

```python
# Retry on failure
llm_with_retry = llm.with_retry(
    retry_if_exception_type=(Exception,),
    stop_after_attempt=3,
    wait_exponential_jitter=True,
)

# Fallback to different model on failure
from langchain_openai import ChatOpenAI

primary_llm  = ChatGroq(model="llama-3.3-70b-versatile")
fallback_llm = ChatOpenAI(model="gpt-4o-mini")

llm_with_fallback = primary_llm.with_fallbacks(
    [fallback_llm],  # try these in order if primary fails
    exceptions_to_handle=(Exception,),
)

# If Groq fails → automatically tries OpenAI
result = llm_with_fallback.invoke("What is RAG?")
```

### 9.5 Configurable Chains

```python
from langchain_core.runnables import ConfigurableField

# Make the LLM swappable at runtime
configurable_llm = llm.configurable_alternatives(
    ConfigurableField(id="llm"),
    default_key="groq",
    openai=ChatOpenAI(model="gpt-4o"),
    claude=ChatAnthropic(model="claude-sonnet-4-20250514"),
)

chain = configurable_llm | StrOutputParser()

# Use default (Groq)
result = chain.invoke("Explain RAG")

# Switch to OpenAI at runtime — no code change
result = chain.with_config(configurable={"llm": "openai"}).invoke("Explain RAG")

# Switch to Claude
result = chain.with_config(configurable={"llm": "claude"}).invoke("Explain RAG")
```

---

## 10. Advanced — Callbacks and Streaming

### 10.1 Callbacks — Hook Into Any Step

```python
from langchain.callbacks.base import BaseCallbackHandler
from langchain_core.outputs import LLMResult

class CustomCallbackHandler(BaseCallbackHandler):
    """Hook into every step of the LangChain pipeline"""

    def on_llm_start(self, serialized, prompts, **kwargs):
        """Called when LLM starts generating"""
        print(f"LLM started. Prompt length: {len(prompts[0])} chars")

    def on_llm_end(self, response: LLMResult, **kwargs):
        """Called when LLM finishes"""
        tokens = response.llm_output.get("token_usage", {})
        print(f"Input tokens: {tokens.get('prompt_tokens', 0)}")
        print(f"Output tokens: {tokens.get('completion_tokens', 0)}")

    def on_llm_error(self, error: Exception, **kwargs):
        """Called on LLM error"""
        print(f"LLM Error: {error}")
        # Alert Sentry, log to monitoring system

    def on_tool_start(self, serialized, input_str, **kwargs):
        """Called when an agent tool starts"""
        print(f"Tool started: {serialized['name']} with input: {input_str[:100]}")

    def on_tool_end(self, output, **kwargs):
        """Called when tool finishes"""
        print(f"Tool result: {output[:100]}")

    def on_agent_action(self, action, **kwargs):
        """Called when agent decides on an action"""
        print(f"Agent action: {action.tool} | Input: {action.tool_input}")

    def on_chain_end(self, outputs, **kwargs):
        """Called when a chain completes"""
        print(f"Chain finished. Output keys: {list(outputs.keys())}")

# Use callback
handler = CustomCallbackHandler()
result = chain.invoke("What is RAG?", config={"callbacks": [handler]})

# Or set globally on a chain
chain_with_callbacks = chain.with_config({"callbacks": [handler]})
```

### 10.2 Streaming Tokens

```python
# Stream tokens as they're generated
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

chain = (
    ChatPromptTemplate.from_template("Write a story about {topic}")
    | ChatGroq(model="llama-3.3-70b-versatile", streaming=True)
    | StrOutputParser()
)

# Synchronous streaming
print("Streaming output:")
for chunk in chain.stream({"topic": "an AI robot learning to cook"}):
    print(chunk, end="", flush=True)
print()  # newline at end

# Async streaming (for FastAPI endpoints)
async def stream_response(topic: str):
    async for chunk in chain.astream({"topic": topic}):
        yield f"data: {chunk}\n\n"  # SSE format

# FastAPI endpoint with streaming
from fastapi.responses import StreamingResponse

@router.post("/chat/stream")
async def chat_stream(body: ChatRequest):
    return StreamingResponse(
        stream_response(body.topic),
        media_type="text/event-stream",
    )
```

---

## 11. Advanced — Output Parsers

### 11.1 Structured Output with Pydantic

```python
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List

class CompanyAnalysis(BaseModel):
    company_name:    str         = Field(description="Full company name")
    sector:          str         = Field(description="Industry sector")
    strengths:       List[str]   = Field(description="List of company strengths")
    weaknesses:      List[str]   = Field(description="List of company weaknesses")
    investment_score:float       = Field(description="Investment score from 0-10", ge=0, le=10)
    recommendation:  str         = Field(description="BUY, HOLD, or SELL")

parser = PydanticOutputParser(pydantic_object=CompanyAnalysis)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a financial analyst. {format_instructions}"),
    ("human", "Analyse this company: {company_info}"),
])

chain = prompt | llm | parser

result: CompanyAnalysis = chain.invoke({
    "company_info": "Reliance Industries Q3 FY2026 results: revenue ₹2.34L crore...",
    "format_instructions": parser.get_format_instructions(),
})
print(result.recommendation)     # "BUY"
print(result.investment_score)   # 8.5
print(result.strengths)          # ["Diversified portfolio", "Strong retail growth"]
```

### 11.2 Auto-Fixing Parser (Handles LLM Mistakes)

```python
from langchain.output_parsers import OutputFixingParser

# Wraps any parser with auto-retry using LLM to fix malformed output
fixing_parser = OutputFixingParser.from_llm(
    parser=PydanticOutputParser(pydantic_object=CompanyAnalysis),
    llm=llm,
)

# If LLM returns invalid JSON → fixing_parser automatically asks LLM to fix it
# Prevents crashes on malformed output
```

---

## 12. LangChain vs Direct SDKs — When to Use What

### The Honest Truth

```
USE LANGCHAIN WHEN:
  ✅ Prototyping quickly — 50 lines instead of 200
  ✅ Learning — great abstractions for understanding concepts
  ✅ Working with legacy codebases that already use it
  ✅ Need many integrations (100+ loaders, vector stores, tools)
  ✅ Team is unfamiliar with LLM APIs directly
  ✅ Simple RAG pipelines without custom logic

DON'T USE LANGCHAIN WHEN:
  ❌ Complex custom logic — abstractions fight you
  ❌ Performance matters — extra layers add latency
  ❌ Debugging is critical — 5-deep stack traces are painful
  ❌ Production stability — breaking changes between versions
  ❌ You know the LLM APIs well — direct SDKs are cleaner
  ❌ Building agents — LangGraph is better for this now

THE RULE:
  If you're spending more time fighting LangChain than building features
  → switch to direct SDKs + LangGraph

  LangChain is scaffolding — useful to go fast initially,
  remove it when it gets in the way.
```

### Side-by-Side Comparison

```python
# Simple RAG — LangChain wins (fewer lines, more readable)

# === LangChain ===
from langchain.chains import RetrievalQA
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
result = qa_chain.invoke({"query": "What is the return policy?"})

# === Direct SDKs ===
# Must write: retrieve → format → prompt → call LLM → parse
# ~30 more lines but full control

# ---

# Custom logic — Direct SDKs wins (no abstraction fighting)

# === LangChain ===
# Want to: retry differently based on error type + log cost + add custom metadata
# Fighting LangChain's abstractions for every custom step

# === Direct SDKs ===
async def custom_llm_call(messages, retries=3):
    for attempt in range(retries):
        try:
            result = await groq_client.chat.completions.create(...)
            await track_cost(result.usage)   # custom cost tracking
            return result                    # full control
        except RateLimitError:
            await asyncio.sleep(60 * attempt)
        except APIError as e:
            if e.status_code == 400:
                raise  # don't retry on bad request
```

---

## 12.5 — Pros, Cons & Product Decision Framework

### Full Pros and Cons

```
PROS of LangChain:
┌─────────────────────────────────────────────────────────────────┐
│ ✅ Speed to prototype                                            │
│    50 lines instead of 200. RAG pipeline in an afternoon.       │
│    Perfect for validating an idea before investing in infra.    │
│                                                                  │
│ ✅ 500+ integrations out of the box                              │
│    Loaders: PDF, CSV, Notion, Confluence, S3, Google Drive...   │
│    Vector stores: Chroma, Pinecone, Qdrant, Weaviate, FAISS...  │
│    Tools: Wikipedia, Google Search, Arxiv, SQL, Pandas...       │
│    Saves weeks of integration work.                             │
│                                                                  │
│ ✅ Standardised interfaces                                        │
│    Swap OpenAI → Groq → Anthropic by changing one line.         │
│    Same .invoke() API for every model, every loader, every store.│
│                                                                  │
│ ✅ LCEL composition (genuinely good)                              │
│    pipe operator makes complex pipelines readable.              │
│    Async, streaming, batching all built into every component.   │
│                                                                  │
│ ✅ Learning resource                                              │
│    Most tutorials use it. Job descriptions mention it.          │
│    Understanding LangChain = understanding LLM app patterns.    │
│                                                                  │
│ ✅ Active ecosystem                                               │
│    Regular updates. Large community. Many solved problems.       │
└─────────────────────────────────────────────────────────────────┘

CONS of LangChain:
┌─────────────────────────────────────────────────────────────────┐
│ ❌ Abstraction tax                                                │
│    Simple: OpenAI SDK = 3 lines. LangChain = 8 lines.           │
│    Complex: fighting the abstraction adds 50% more code.        │
│    "I spent 2 hours making LangChain do what 10 lines would."   │
│                                                                  │
│ ❌ Breaking changes (historically bad)                            │
│    2022-2024: major breaking changes every few versions.        │
│    Code that worked 3 months ago silently breaks.               │
│    Community frustration: 1000+ GitHub issues on this.          │
│    Better in 2025 but reputation remains.                       │
│                                                                  │
│ ❌ Hard to debug                                                  │
│    Stack trace is 8 layers deep. Which layer broke?             │
│    Internal logic is hidden. What exactly is being sent to LLM? │
│    Logging is non-trivial to add to internals.                  │
│                                                                  │
│ ❌ Performance overhead                                           │
│    Extra abstraction layers add latency (~50-200ms per call).   │
│    Heavy import time (~3 seconds cold start on Lambda).         │
│    Memory footprint large (langchain + community = 500MB+).     │
│                                                                  │
│ ❌ Overshoots for simple tasks                                    │
│    One LLM call doesn't need a framework.                       │
│    A 20-line prompt call becomes a 50-line LangChain setup.     │
│                                                                  │
│ ❌ Wrong tool for complex agents                                   │
│    AgentExecutor is a black box — can't customise the loop.     │
│    LangGraph (same team) is better for this.                    │
│    Many teams: used LangChain agents in 2023 → migrated to LG.  │
└─────────────────────────────────────────────────────────────────┘
```

---

### The Product Manager Perspective — When to Pick LangChain

Think of this as a build vs buy decision. LangChain is "buying" pre-built components. Direct SDKs is "building" from scratch. The right answer depends on what stage you're at, what your team knows, and what you're building.

```
STAGE 1 — EXPLORATION (0–2 weeks)
  Goal:       Does this idea even work with AI?
  Team state: Small, figuring things out
  Pick:       LangChain ✅

  Why: You need to validate fast. LangChain's 500+ integrations mean
  you can connect any data source, any LLM, any vector store in hours.
  The abstraction overhead doesn't matter — you'll throw this away anyway.
  
  Example: "We want to see if RAG can answer questions from our 50 PDFs
           before we invest 3 months building it properly."
  LangChain: working prototype in 2 days ✅
  Direct SDK: working prototype in 1 week ❌ (too slow for exploration)

STAGE 2 — VALIDATION (2–8 weeks)
  Goal:       Does the product work for real users?
  Team state: Small, building first real version
  Pick:       LangChain for simple features, LangGraph for agents

  Why: You're moving fast. LangChain for the simple stuff
  (document loading, embedding, basic RAG chains).
  LangGraph for anything with loops, state, or HITL.
  Don't refactor yet — validate first.

STAGE 3 — PRODUCTION (3+ months, real users, real money)
  Goal:       Scale, reliability, cost control, debuggability
  Team state: Dedicated AI team, real SLAs
  Pick:       Direct SDKs + LangGraph (remove LangChain)

  Why: At this stage, the abstraction becomes a liability.
  You need: full control over retry logic, streaming, error handling.
  You need: predictable performance (no hidden LangChain latency).
  You need: clear stack traces when things break at 3 AM.
  You need: cost control (every extra abstraction = tokens).
  
  Migration path:
    Keep: LangGraph (it IS the right tool for agents)
    Remove: LangChain chains → replace with direct SDK calls
    Remove: LangChain agents → replace with LangGraph graphs
    Keep: LangChain document loaders IF they save time
           (often worth keeping — they're thin wrappers)
```

---

### Decision Matrix — LangChain vs Direct SDKs vs LangGraph

```
                    LangChain    Direct SDKs    LangGraph
                    ─────────    ───────────    ─────────
Prototype speed      ⭐⭐⭐⭐⭐      ⭐⭐⭐           ⭐⭐⭐
Production reliability ⭐⭐⭐       ⭐⭐⭐⭐⭐         ⭐⭐⭐⭐⭐
Debuggability         ⭐⭐         ⭐⭐⭐⭐⭐         ⭐⭐⭐⭐
Customisability       ⭐⭐         ⭐⭐⭐⭐⭐         ⭐⭐⭐⭐⭐
Learning curve        ⭐⭐⭐⭐       ⭐⭐⭐⭐⭐         ⭐⭐
Performance           ⭐⭐⭐        ⭐⭐⭐⭐⭐         ⭐⭐⭐⭐
State management      ⭐           ⭐⭐⭐           ⭐⭐⭐⭐⭐
HITL support          ⭐           ⭐⭐            ⭐⭐⭐⭐⭐
Multi-agent           ⭐⭐          ⭐⭐⭐           ⭐⭐⭐⭐⭐
Integration breadth   ⭐⭐⭐⭐⭐      ⭐⭐⭐           ⭐⭐⭐
Cold start (Lambda)   ⭐⭐          ⭐⭐⭐⭐⭐         ⭐⭐⭐⭐
```

---

### Real-World Team Scenarios

**Scenario 1 — Early-Stage Startup (2-5 engineers)**
```
Context:
  - 2 engineers, one has LLM experience
  - Need to ship a RAG chatbot in 4 weeks
  - Investors want to see a demo next month
  - No dedicated DevOps

Recommendation: LangChain + LangGraph
  LangChain: document loading, basic RAG chain (saves time)
  LangGraph: any agent logic (future-proofs architecture)
  
  Why not direct SDKs: you don't have 4 weeks to build from scratch
  Why LangGraph over LangChain agents: you'll need HITL and state later
  
  Plan: use LangChain now, migrate chains to direct SDKs at Series A
  when you can hire a dedicated AI engineer who has time to refactor.
```

**Scenario 2 — Enterprise SaaS Team (10–50 engineers)**
```
Context:
  - Dedicated AI team (3-4 engineers)
  - Building AI features in existing product
  - SLA: 99.5% uptime, < 3s P95 latency
  - Compliance: full audit trail required

Recommendation: Direct SDKs + LangGraph only
  Direct SDKs: litellm + instructor + direct provider SDKs
  LangGraph:   all agent orchestration
  No LangChain: too much black-box behaviour for compliance
  
  Why: at this scale, you need to control every LLM call.
  Audit trail requires knowing exactly what was sent/received.
  LangChain's internal logging is not production-grade.
  LangGraph's checkpointer gives you full state history.
```

**Scenario 3 — Agency / Consulting (building for clients)**
```
Context:
  - Building 5+ different AI products simultaneously
  - Different clients, different requirements
  - Time-to-delivery is the key metric
  - Clients will maintain the code after handoff

Recommendation: LangChain for standard features, LangGraph for agents
  
  Why: clients can find LangChain developers easily.
  LangChain is well-documented — clients can maintain it.
  LangGraph for complex agents — explain it's the "enterprise" option.
  
  Avoid: over-engineering with direct SDKs for simple chatbots.
  The client will need to understand and maintain the code.
  LangChain's readability is an asset here.
```

**Scenario 4 — Internal AI Tools (non-product team)**
```
Context:
  - Data team building internal analytics tools
  - 1-2 engineers, primarily data scientists not software engineers
  - Users are internal (analysts, managers)
  - Reliability is nice-to-have, not critical

Recommendation: LangChain + Streamlit/Gradio
  
  Why: data scientists are most comfortable with LangChain.
  Internal tools can tolerate occasional issues.
  Streamlit makes frontend trivial.
  Ship fast, iterate based on internal feedback.
```

---

### The Honest Migration Path

```
Reality: most teams START with LangChain, OUTGROW it, and MIGRATE.
This is normal. Not a mistake. The right sequence.

Phase 1 (Month 1-3):  Use LangChain for everything
Phase 2 (Month 3-6):  Notice pain points: debugging, customisation, performance
Phase 3 (Month 6-12): Migrate piece by piece:
  Step 1: Replace LangChain chains with LCEL + direct SDKs (keep LCEL)
  Step 2: Replace AgentExecutor with LangGraph graphs
  Step 3: Keep LangChain document loaders (thin wrappers, still useful)
  Step 4: Direct SDK calls everywhere else
Phase 4 (Month 12+):  Only LangGraph remains. Direct SDKs for everything else.

Signs it's time to migrate:
  "I spent 2 hours debugging a LangChain error I don't understand"
  "We can't add custom retry logic without fighting the framework"
  "Our P95 latency is unacceptably high and LangChain overhead is part of it"
  "The LangChain version upgrade broke 5 things"
  "I need to know exactly what's being sent to the LLM and I can't easily"
```

---

## 13. Real-World Use Cases

### Use Case 1 — Customer Support Chatbot

```python
"""
Real use case: E-commerce customer support bot
- Knows about products (RAG over product catalog)
- Can look up orders (tool: get_order_status)
- Maintains conversation history (Redis memory)
- Escalates to human when needed (tool: create_support_ticket)
"""

from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain.memory import ConversationBufferWindowMemory

@tool
def get_order_status(order_id: str) -> str:
    """Look up the status of a customer's order by order ID.
    Use when customer asks about their order, delivery, or tracking."""
    # Query database
    return f"Order {order_id}: Shipped, expected delivery: 2 days"

@tool
def create_support_ticket(issue: str, priority: str = "normal") -> str:
    """Create a support ticket for issues that cannot be resolved automatically.
    Use when: refund requests, complaints, complex issues requiring human review.
    Args:
        issue: Detailed description of the customer's issue
        priority: 'urgent' for payment issues, 'normal' for general issues
    """
    ticket_id = f"TKT-{hash(issue) % 100000:05d}"
    return f"Support ticket {ticket_id} created. Team will respond within 24 hours."

tools = [
    get_order_status,
    create_support_ticket,
    vectorstore.as_retriever(),  # for product/policy questions
]

support_agent = AgentExecutor(
    agent=create_openai_functions_agent(llm, tools, support_prompt),
    tools=tools,
    memory=ConversationBufferWindowMemory(memory_key="chat_history", k=10, return_messages=True),
    max_iterations=5,
)
```

### Use Case 2 — Document Intelligence Pipeline

```python
"""
Real use case: Legal document analyser
- Load and chunk 100s of contracts
- Answer questions about specific clauses
- Extract structured data (parties, dates, amounts)
- Compare across multiple documents
"""

# Load all contracts
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

loader = DirectoryLoader("contracts/", glob="**/*.pdf", loader_cls=PyPDFLoader)
all_contracts = loader.load()

# Split with metadata preserved
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    add_start_index=True,   # adds "start_index" to metadata (useful for finding exact location)
)
chunks = splitter.split_documents(all_contracts)

# Add custom metadata
for chunk in chunks:
    contract_name = chunk.metadata["source"].split("/")[-1]
    chunk.metadata["contract"] = contract_name
    chunk.metadata["type"] = "legal_contract"

# Build vector store
vectorstore = Chroma.from_documents(chunks, embeddings, collection_name="contracts")

# Metadata-filtered retrieval
retriever = vectorstore.as_retriever(
    search_kwargs={
        "k": 5,
        "filter": {"contract": "vendor_agreement_2026.pdf"},  # only search this contract
    }
)

# Extract structured data
class ContractKeyTerms(BaseModel):
    parties:        List[str]  = Field(description="All parties in the contract")
    effective_date: str        = Field(description="Contract effective date")
    value:          str        = Field(description="Contract monetary value if mentioned")
    termination:    str        = Field(description="Termination clause summary")
    jurisdiction:   str        = Field(description="Governing jurisdiction/law")

extraction_chain = (
    ChatPromptTemplate.from_template(
        "Extract key terms from this contract excerpt. {format_instructions}\n\n{text}"
    )
    | llm
    | PydanticOutputParser(pydantic_object=ContractKeyTerms)
)
```

### Use Case 3 — Code Review Agent

```python
"""
Real use case: Automated code review
- Reads PR diff
- Checks: bugs, security issues, performance, style
- Provides specific inline comments
- Rates severity (critical / major / minor)
"""

from langchain.tools import tool

@tool
def run_static_analysis(code: str, language: str) -> str:
    """Run static code analysis on provided code snippet.
    Returns list of issues with line numbers."""
    # In reality: call pylint, eslint, etc.
    return "Line 42: Unused variable 'x'. Line 67: Potential SQL injection."

@tool
def search_security_database(pattern: str) -> str:
    """Search CVE/security database for known vulnerabilities matching a code pattern.
    Use when you detect potential security issues."""
    return f"CVE-2024-1234: SQL injection via string formatting. Severity: HIGH"

code_review_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a senior software engineer doing code review.
For each issue found:
- Specify the exact line number
- Explain what's wrong
- Suggest the fix
- Rate severity: CRITICAL / MAJOR / MINOR
Use tools to check for security issues."""),
    ("human", "Review this {language} code:\n\n```\n{code}\n```"),
])

review_agent = AgentExecutor(
    agent=create_openai_functions_agent(llm, [run_static_analysis, search_security_database], code_review_prompt),
    tools=[run_static_analysis, search_security_database],
    max_iterations=8,
)

result = review_agent.invoke({
    "language": "Python",
    "code": open("new_feature.py").read(),
})
```

---

## 14. Interview Cheat Sheet

**Q: What is LangChain and why was it created?**
```
Framework for building LLM applications. Created because:
1. Raw LLM SDKs had no memory, no tools, no RAG support
2. Every provider had different interfaces — LangChain standardised them
3. Common patterns (load PDF → embed → retrieve → generate) needed to be reusable

Today: still relevant but direct SDKs + LangGraph often preferred for production.
```

**Q: What is LCEL and why does it matter?**
```
LangChain Expression Language — pipe operator | to compose components.
Replaces the old chain classes (LLMChain, SequentialChain, etc.)

Benefits:
- Lazy evaluation (build chain without running it)
- Async support built in (.ainvoke, .astream)
- Batch processing built in (.batch)
- Streaming built in (.stream)
- All components implement same Runnable interface

chain = prompt | llm | parser
result = chain.invoke({"topic": "RAG"})
```

**Q: What types of memory does LangChain support?**
```
4 main types:
1. ConversationBufferMemory — stores everything (expensive for long convos)
2. ConversationBufferWindowMemory — last K exchanges only (cost-controlled)
3. ConversationSummaryMemory — summarises old messages (preserves context cheaply)
4. ConversationSummaryBufferMemory — hybrid (recent=full, older=summarised)

For production: always use external memory (Redis) for durability.
RedisChatMessageHistory + ConversationBufferMemory.
```

**Q: ReAct vs OpenAI Functions Agent — difference?**
```
ReAct: general purpose, works with any LLM
  Thought → Action → Observation loop in plain text
  Less reliable — LLM must format action as text

OpenAI Functions: works with GPT models + Groq
  Uses structured function calling (JSON schema)
  More reliable — structured output, fewer parsing errors
  Preferred for production use
```

**Q: When would you NOT use LangChain?**
```
1. Complex custom logic — abstractions fight you
2. High-performance requirements — extra layers add latency
3. Complex agents — LangGraph is purpose-built for this
4. You need full debuggability — stack traces are deep
5. Breaking changes between versions are a risk

Use LangChain for: quick prototypes, simple RAG, many integrations needed.
Use direct SDKs for: production agents, custom logic, performance-critical paths.
```

---

## 15. Quick Revision Cards

```
CARD 1: LangChain Core Components
  Models: ChatOpenAI, ChatGroq, ChatAnthropic (all same interface)
  Prompts: PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
  Chains: LLMChain (old) | LCEL pipe operator | (new)
  Memory: Buffer, Window, Summary, SummaryBuffer
  Agents: ReAct, OpenAI Functions, Structured Output
  Tools: @tool decorator, StructuredTool, community tools

CARD 2: LCEL Pipe Operator
  chain = prompt | llm | parser
  All Runnables support: .invoke, .ainvoke, .batch, .abatch, .stream, .astream
  RunnablePassthrough: pass data through unchanged
  RunnableParallel: run multiple chains on same input simultaneously
  RunnableLambda: wrap any Python function as Runnable

CARD 3: Memory Types
  Buffer:          stores all messages (infinite, expensive)
  Window(k=5):     last 5 messages only (fixed cost)
  Summary:         LLM summarises old messages (cheap + context preserved)
  SummaryBuffer:   hybrid — recent messages in full + old ones summarised
  Redis:           production — persists across server restarts

CARD 4: Agent Types
  ReAct:           any LLM, plain text reasoning, less reliable
  OpenAI Functions: GPT/Groq, structured JSON, more reliable
  Tool choice: describe tools precisely → agent quality directly depends on descriptions

CARD 5: Document Processing Pipeline
  Load:  DirectoryLoader/PyPDFLoader/WebBaseLoader → list[Document]
  Split: RecursiveCharacterTextSplitter(chunk_size=1000, overlap=200)
  Embed: OpenAIEmbeddings / HuggingFaceEmbeddings → vectors
  Store: Chroma (local) / FAISS (in-memory) / Qdrant (production)
  Query: vectorstore.similarity_search(query, k=5)

CARD 6: RAG Chain Pattern (LCEL)
  chain = {
    "context":  retriever | format_docs,
    "question": RunnablePassthrough(),
  } | rag_prompt | llm | StrOutputParser()
  
  Always ground LLM: "Answer ONLY based on provided context"
  Always cite sources: include doc.metadata["source"] in format_docs

CARD 7: Output Parser Types
  StrOutputParser:              raw string from AIMessage
  JsonOutputParser:             parse JSON string
  PydanticOutputParser:         validate against Pydantic model
  OutputFixingParser:           auto-fix malformed output using LLM
  CommaSeparatedListOutputParser: "a, b, c" → ["a", "b", "c"]

CARD 8: Callbacks
  on_llm_start:    called before LLM generates
  on_llm_end:      called after LLM generates (token counts here)
  on_tool_start:   called before tool executes
  on_tool_end:     called after tool executes
  on_agent_action: called when agent chooses an action
  Use for: logging, cost tracking, monitoring, debugging

CARD 9: LangChain vs Direct SDK
  Prototype fast:     LangChain ✅
  Many integrations:  LangChain ✅
  Complex agents:     LangGraph ✅
  Custom logic:       Direct SDKs ✅
  Production agents:  LangGraph ✅
  High performance:   Direct SDKs ✅

CARD 10: Fallbacks and Retries
  llm.with_retry(stop_after_attempt=3)    → retry on failure
  llm.with_fallbacks([backup_llm])        → try backup on failure
  OutputFixingParser.from_llm(parser, llm) → auto-fix bad output
```

---

## ✅ LangChain Completion Checklist

```
THEORY
[ ] Understand what LangChain is and why it was created
[ ] Know the 4 main packages (core, langchain, community, provider-specific)
[ ] Understand the Runnable interface (invoke, stream, batch, async variants)
[ ] Know 4 memory types and when to use each
[ ] Know difference between ReAct and OpenAI Functions agents
[ ] Understand why tool descriptions are critical for agent quality
[ ] Can explain LCEL and the pipe operator

CODE
[ ] Built a basic LLM chain (prompt | llm | parser)
[ ] Built a RAG pipeline with LCEL (retriever + LLM)
[ ] Implemented ConversationBufferWindowMemory with Redis persistence
[ ] Created a custom tool with @tool decorator and clear description
[ ] Built a ReAct agent with 3+ tools
[ ] Built an OpenAI Functions agent with memory
[ ] Used RunnableParallel for fan-out pattern
[ ] Implemented custom callback handler for logging
[ ] Built streaming endpoint with LCEL

ADVANCED
[ ] Used configurable chains (swap models at runtime)
[ ] Implemented with_retry and with_fallbacks
[ ] Used PydanticOutputParser with complex nested model
[ ] Used OutputFixingParser for auto-correction
[ ] Built conversational RAG with ConversationalRetrievalChain
```

---

*LangChain Study Notes | GenAI + LLMOps Engineering Roadmap 2026 — Phase 4*
*Next: LangGraph Deep Dive → phase-4/LANGGRAPH.md*
