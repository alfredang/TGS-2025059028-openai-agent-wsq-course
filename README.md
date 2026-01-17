# OpenAI Agent WSQ Course

This repository contains three Streamlit applications demonstrating OpenAI-powered chatbots and multi-agent systems.

**Live Demo:** [Invoice RAG Chat](https://alfredang-tgs-2025059028-openai-agent-wsq-appinvoice-rag-q0goip.streamlit.app/)

## Applications

### 1. Simple Chat (`app/simple_chat.py`)

A simple chatbot agent built with the OpenAI Agents SDK that can search the internet for current information.

**Features:**
- Conversational chat interface using Streamlit
- Internet search capability via Tavily API
- Uses GPT-4.1-mini model
- Chat history management with "New Chat" button

**Required Environment Variables:**
- `OPENAI_API_KEY` - Your OpenAI API key
- `TAVILY_API_KEY` - Your Tavily API key for web search

**Run:**
```bash
streamlit run app/simple_chat.py
```

---

### 2. Invoice RAG Chat (`app/invoice_rag.py`)

A Retrieval-Augmented Generation (RAG) application for querying invoice PDFs using natural language.

**Features:**
- Upload and index multiple PDF invoices
- Extracts text from PDFs and chunks it for embedding
- Stores embeddings in ChromaDB vector database
- Semantic search to find relevant invoice context
- Chat interface to ask questions about your invoices
- Uses OpenAI's `text-embedding-3-large` for embeddings
- Uses GPT-4.1-mini for generating responses

**Required Environment Variables:**
- `OPENAI_API_KEY` - Your OpenAI API key

**Run:**
```bash
streamlit run app/invoice_rag.py
```

---

### 3. Multi-Agent Trip Planner (`app/trip_planner.py`)

A multi-agent AI system that orchestrates multiple specialized agents to plan trips. This demonstrates real-world agentic AI patterns used in enterprise automation.

**Features:**
- Multiple specialized AI agents working together:
  - 🧠 **Planner Agent** - Creates day-by-day itineraries (uses Tavily search for attractions)
  - 💰 **Budget Agent** - Estimates and tracks trip costs (uses Tavily search for prices)
  - 🍣 **Local Guide Agent** - Recommends food and local tips (uses Tavily search for restaurants)
- ✈️ **Travel Agent Orchestrator** - Coordinates all agents using `.as_tool()` pattern
- **Tavily Search Integration** - Each agent searches for real-time, accurate information
- Interactive UI for entering trip details
- Structured output using Pydantic models
- Tabbed results display for organized output
- Uses GPT-4.1-mini model for all agents

**Required Environment Variables:**
- `OPENAI_API_KEY` - Your OpenAI API key
- `TAVILY_API_KEY` - Your Tavily API key for real-time search

**Run:**
```bash
streamlit run app/trip_planner.py
```

**Activity Guide:** See [ACTIVITY_MULTI_AGENT_TRIP_PLANNER.md](ACTIVITY_MULTI_AGENT_TRIP_PLANNER.md) for the full tutorial.

---

## Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/alfredang/TGS-2025059028-openai-agent-wsq-course.git
   cd TGS-2025059028-openai-agent-wsq-course
   ```

2. Install dependencies:
   ```bash
   pip install streamlit openai python-dotenv pypdf chromadb tavily-python openai-agents
   ```

3. Create a `.env` file in the project root:
   ```
   OPENAI_API_KEY=your_openai_api_key
   TAVILY_API_KEY=your_tavily_api_key
   ```

4. Run either application using the commands above.
