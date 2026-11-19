# RAG and Agentic AI: A Hands-On Learning Repository

Welcome to **RAG_and_AgenticAI**, a structured, beginner-friendly collection of Python examples, notebooks, and small applications for learning **Generative AI**, **Retrieval-Augmented Generation (RAG)**, **vector databases**, and **agentic AI** through hands-on code.

## Learning Objectives

After working through this repository, you should be able to:

- Use **prompt templates**, **LangChain Expression Language (LCEL)**, and **output parsers** with a local LLM.
- Understand **embeddings**, **similarity metrics**, and **retrieval** over text.
- Build **vector stores** with **Chroma** and **FAISS**, and wire them into **RAG** pipelines.
- Apply **advanced RAG** patterns: chunking, retrievers, and context-grounded generation.
- Work with **multimodal** flows: speech-to-text, text-to-speech, and vision-language analysis.
- Design **AI agents** with **tools**, tool binding, and multi-step reasoning.
- Orchestrate workflows with **LangGraph** (state, nodes, routing, reflection, ReAct).
- Coordinate **multi-agent crews** with **CrewAI**.
- Connect LLMs to external capabilities using the **Model Context Protocol (MCP)**.
- Assemble **practical apps**: PDF Q&A, YouTube transcript Q&A, and nutrition analysis from images.

Topics are taught through the code that exists in each folder; some examples require you to install models, provide PDFs, or verify package versions on your machine.


## Learning Roadmap

Follow these eight modules in order for a logical path from foundations to agents and MCP.

| # | Module | Directory | Description |
|---|--------|-----------|-------------|
| 1 | Foundations of Generative AI and Prompt Engineering | [FoundationsOfGenAIAndPromptEng](FoundationsOfGenAIAndPromptEng/) | Prompt templates, LCEL, chaining notebook, parsers, tools, tool binding, LangChain agents | [README](FoundationsOfGenAIAndPromptEng/README.md) |
| 2 | Vector Databases for RAG | [VectorDatabasesForRAG](VectorDatabasesForRAG/) | Embeddings with Ollama, similarity metrics, TF-IDF + LLM recommendations | [README](VectorDatabasesForRAG/README.md) |
| 3 | Advanced RAG Concepts | [AdvancedRAGConcepts](AdvancedRAGConcepts/) | FAISS semantic search, Chroma retriever, chunking, grounded Q&A | [README](AdvancedRAGConcepts/README.md) |
| 4 | Multimodal Generative AI | [MultiModalGenAI](MultiModalGenAI/) | Whisper transcription, meeting assistant, gTTS, LLM storytelling | [README](MultiModalGenAI/README.md) |
| 5 | Foundations of AI Agents | [FoundationsOfAIAgents](FoundationsOfAIAgents/) | Tool-calling agents, data analysis agent, pandas visualization agent | [README](FoundationsOfAIAgents/README.md) |
| 6 | Agentic AI with LangGraph | [AgenticAIWithLangGraph](AgenticAIWithLangGraph/) | StateGraph basics, ReAct, reflection, reflexion, workflow and design patterns | [README](AgenticAIWithLangGraph/README.md) |
| 7 | Agentic AI with CrewAI | [AgenticAIWithCrewAI](AgenticAIWithCrewAI/) | Agents, tasks, sequential crews, structured Pydantic outputs | [README](AgenticAIWithCrewAI/README.md) |
| 8 | Model Context Protocol | [ModelContextProtocol](ModelContextProtocol/) | FastMCP servers/clients, HTTP MCP, security and permissions, LangGraph + MCP | [README](ModelContextProtocol/README.md) |

## Hands-On Projects

These projects apply concepts from the modules above. Each has its own setup and dependencies.

| Project | Directory | Purpose | README |
|---------|-----------|---------|--------|
| **AI Nutrition Coach** | [AI_NUTRITION_Coach](AI_NUTRITION_Coach/) | Upload a food image; local vision model estimates nutrition via Ollama | [README](AI_NUTRITION_Coach/README.md) |
| **PDF Q&A Bot** | [PDF_QA_Bot](PDF_QA_Bot/) | RAG over PDFs with Chroma, Ollama embeddings, and Gradio UI | [README](PDF_QA_Bot/README.md) |
| **YouTube RAG Bot** | [YOUTUBE_RAG_Bot](YOUTUBE_RAG_Bot/) | Fetch transcripts, index with FAISS, answer questions in Streamlit | [README](YOUTUBE_RAG_Bot/README.md) |


## Prerequisites and Setup

### Recommended environment

- **Python**: Examples in the included Jupyter notebook reference **Python 3.11**. Use Python 3.10+ unless a specific module fails on your system.
- **[Ollama](https://ollama.com/)**: Most modules call local models through Ollama. Pull models as needed (see below).
- **Node.js / npx**: Required for several MCP examples that spawn npm packages (`@upstash/context7-mcp`, `metmuseum-mcp`).
- **Git**: To clone this repository.

### Clone the repository

```bash
git clone https://github.com/<your-org>/RAG_and_AgenticAI.git
cd RAG_and_AgenticAI
```

Replace the URL with your fork or upstream remote if different.

### Virtual environment

There is **no single root `requirements.txt`**. Create a virtual environment per module or project:

```bash
python -m venv .venv
```

**Windows (PowerShell):**

```powershell
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux:**

```bash
source .venv/bin/activate
```

Install packages based on the imports in that folder (each module README lists suggested packages).

### Ollama models (commonly used)

| Model | Used for |
|-------|----------|
| `llama3.1:8b` | Chat / agents / RAG generation |
| `nomic-embed-text` | Text embeddings |
| `qwen2.5vl:7b` | Vision + text (AI Nutrition Coach) |

```bash
ollama pull llama3.1:8b
ollama pull nomic-embed-text
ollama pull qwen2.5vl:7b
```

Ensure the Ollama service is running (default API: `http://127.0.0.1:11434`).

### Environment variables

- A root [`.gitignore`](.gitignore) excludes `.env` files. Individual scripts do not define a shared env template in the repo.
- Some files set `OLLAMA_LLM_LIBRARY=cpu` locally (for example in visualization and reflexion examples) to force CPU inference, adjust for your hardware if needed.
- **Do not commit API keys.** MCP HTTP examples use public endpoints (for example Context7); verify terms of use and availability before running.

### `chroma_db/`

The [`.gitignore`](.gitignore) ignores `chroma_db/`. The [PDF Q&A Bot](PDF_QA_Bot/local_RAG_app.py) can persist embeddings to `./chroma_db` at the **repository root** when you run that script from the root directory. This folder is **generated data**, not a learning module, do not treat it as source code.


## Repository Structure

```text
RAG_and_AgenticAI/
├── FoundationsOfGenAIAndPromptEng/   # Module 1: prompts, LCEL, parsers, tools, agents
├── VectorDatabasesForRAG/            # Module 2: embeddings & similarity
├── AdvancedRAGConcepts/              # Module 3: FAISS, Chroma retriever RAG
├── MultiModalGenAI/                  # Module 4: speech & TTS pipelines
├── FoundationsOfAIAgents/            # Module 5: Streamlit tool agents
├── AgenticAIWithLangGraph/           # Module 6: LangGraph workflows & patterns
├── AgenticAIWithCrewAI/              # Module 7: CrewAI multi-agent crews
├── ModelContextProtocol/             # Module 8: MCP servers, clients, security labs
├── AI_NUTRITION_Coach/               # Project: vision-based nutrition coach
├── PDF_QA_Bot/                       # Project: PDF RAG with Gradio
├── YOUTUBE_RAG_Bot/                  # Project: YouTube transcript RAG
├── chroma_db/                        # (optional, gitignored) Chroma persistence for PDF bot
└── .gitignore
```


## Recommended Learning Sequence

1. Complete **Module 1** before Modules 2–3 (LangChain primitives used everywhere).
2. Do **Module 2**, then **Module 3**, before the **PDF** or **YouTube** projects (RAG pipelines).
3. **Module 4** can follow Module 1; it does not require vector databases.
4. **Module 5** builds on tool concepts from Module 1 (`7_Agent.py`, tools binding).
5. **Module 6** (LangGraph) and **Module 7** (CrewAI) are parallel tracks after Module 5; LangGraph is used again in MCP (`MCP_Application.py`).
6. **Module 8** assumes comfort with async Python and agents.
7. **Projects**: attempt **PDF Q&A Bot** after Module 3; **YouTube RAG Bot** after Modules 2–3; **AI Nutrition Coach** after Module 4 (multimodal).

---


## Learning and Contribution Guidelines

- **Explore module-by-module**: open the folder README, install only what that folder needs, and run one script at a time.
- **Experiment**: change prompts, `chunk_size`, `k` in retrievers, or temperature on `ChatOllama` and observe outputs.
- **Extend**: add documents to RAG examples, new tools in agent demos, or an extra node in a LangGraph workflow.
- **Issues and improvements**: if you use GitHub, open issues for broken instructions, missing dependencies, or unclear examples. This repo does not define a formal contribution policy in code, follow standard fork-and-pull-request practice if you contribute.

---

## Limitations and Disclaimer

This is an **educational** repository, not a production system.

- Examples rely on **local LLMs**; quality and speed depend on your hardware and Ollama setup.
- Some scripts use **`eval` for calculators** (ReAct demo), acceptable for learning only, not for production.
- **gTTS** and external MCP HTTP services require network access.
- **PDF Q&A** references a sample PDF path that is **not included** in the repository, you must supply your own file (see project README).
- Package names and LangChain APIs evolve; pin versions if you need reproducibility.

---

## Navigation

### Module READMEs

1. [FoundationsOfGenAIAndPromptEng/README.md](FoundationsOfGenAIAndPromptEng/README.md)
2. [VectorDatabasesForRAG/README.md](VectorDatabasesForRAG/README.md)
3. [AdvancedRAGConcepts/README.md](AdvancedRAGConcepts/README.md)
4. [MultiModalGenAI/README.md](MultiModalGenAI/README.md)
5. [FoundationsOfAIAgents/README.md](FoundationsOfAIAgents/README.md)
6. [AgenticAIWithLangGraph/README.md](AgenticAIWithLangGraph/README.md)
7. [AgenticAIWithCrewAI/README.md](AgenticAIWithCrewAI/README.md)
8. [ModelContextProtocol/README.md](ModelContextProtocol/README.md)

### Project READMEs

- [AI_NUTRITION_Coach/README.md](AI_NUTRITION_Coach/README.md)
- [PDF_QA_Bot/README.md](PDF_QA_Bot/README.md)
- [YOUTUBE_RAG_Bot/README.md](YOUTUBE_RAG_Bot/README.md)
