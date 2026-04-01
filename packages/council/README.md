# Task-Aware LLM Council

A multi-model orchestration system that leverages multiple Large Language Models (LLMs) to generate, evaluate, and converge on high-quality responses using a council-style consensus mechanism.

---

## 🚀 Overview

This project builds on top of `llm_gateway`, a reusable client layer for interacting with different LLM providers, and introduces a **Council System** that:

- Queries multiple models in parallel
- Aggregates responses
- Applies majority voting
- Uses LLM-based evaluation when consensus is not reached

---

## 🧠 Key Features

- Multi-LLM orchestration (OpenAI, Groq, HuggingFace)
- Parallel inference using async calls
- Majority voting system
- LLM-based evaluation (judge mode)
- Retry, backoff, and error handling via `llm_gateway`
- Config-driven setup (no hardcoding)
- Fault-tolerant (handles model failures gracefully)

---

## 🏗️ Architecture

```
User Prompt
     ↓
Council Layer (orchestrator)
     ↓
llm_gateway (client abstraction)
     ↓
LLM Providers 
     ↓
Responses → Voting → Final Answer
```

---

## 📁 Project Structure

```
Task-Aware-LLM-Council/
│
├── packages/
│   ├── llm_gateway/       # LLM client abstraction layer
│   ├── council/           # Council logic
│         
└── README.md
```

---

## ⚙️ Setup

### 1. Install dependencies

```bash
uv sync
```

---

### 2. Create `.env` file

```bash
touch .env
```

---

### 3. Add configuration

```
COUNCIL_PROVIDERS=[{"provider":"openai-compatible","api_base":"https://api.groq.com/openai/v1/chat/completions","api_key_env":"GROQ_API_KEY","model":"llama-3.1-8b-instant"},{"provider":"huggingface","model":"Qwen/Qwen3-32B"}]

GROQ_API_KEY=your_key
HUGGINGFACE_API_KEY=your_key
```

---

### ⚠️ Important

- Do NOT commit `.env`
- Add `.env` to `.gitignore`

---

## ▶️ Running the Project

```bash
uv run -m council.run
```

---

## 🔄 How It Works

### Step 1: Multi-model generation

Each model generates a response:

```
Model A → Response A
Model B → Response B
Model C → Response C
```

---

### Step 2: Majority check

- If agreement exists → return result
- Else → proceed to voting

---

### Step 3: Voting (LLM-as-judge)

All models evaluate:

```
Which answer is best?
→ return index
```

---

### Step 4: Convergence

Responses are refined until:
- Majority is reached
- Max rounds exceeded

---

## 🧪 Example

**Input:**
```
What is the capital of France?
```

**Output:**
```
Paris
```

---


## ⚠️ Known Limitations

- No streaming support yet
- No tool/function calling
- Semantic similarity not implemented (string matching used)
- Model availability depends on provider

---




## 📜 License

MIT License