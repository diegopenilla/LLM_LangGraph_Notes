# LangGraph Serve: Deployment Guide

**LangGraph Serve** provides methods for deploying and serving LangGraph applications as production-ready APIs.

## Table of Contents

- [Deployment Options](#deployment-options)
  - [LangServe (REST API)](#langserve-rest-api)
  - [LangGraph Cloud](#langgraph-cloud)
  - [OpenAI-Compatible API](#openai-compatible-api)
- [API Endpoints](#api-endpoints)
- [Configuration](#configuration)
- [Resources](#resources)

---

## Deployment Options

### LangServe (REST API)

Deploy LangGraph applications as REST APIs using FastAPI.

#### LangServe Installation

```bash
pip install langserve
```

#### LangServe Setup

```python
from fastapi import FastAPI
from langserve import add_routes
from langgraph.graph import StateGraph, MessagesState, START, END
from langchain_openai import ChatOpenAI

# Define your graph
def assistant(state: MessagesState):
    llm = ChatOpenAI(model="gpt-3.5-turbo")
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

graph = StateGraph(MessagesState)
graph.add_node("assistant", assistant)
graph.add_edge(START, "assistant")
graph.add_edge("assistant", END)

compiled_graph = graph.compile()

# Create FastAPI app
app = FastAPI(title="LangGraph Assistant")
add_routes(app, compiled_graph, path="/assistant")
```

#### Running the Server

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
# Or: langchain serve
```

#### Async Execution

For better concurrency and performance, use async node functions. LangServe automatically exposes async endpoints when your graph uses async nodes.

**Async Node Functions:**

```python
from fastapi import FastAPI
from langserve import add_routes
from langgraph.graph import StateGraph, MessagesState, START, END
from langchain_openai import ChatOpenAI

# Define async node functions
async def assistant(state: MessagesState):
    llm = ChatOpenAI(model="gpt-3.5-turbo")
    # Use ainvoke for async LLM calls
    response = await llm.ainvoke(state["messages"])
    return {"messages": [response]}

graph = StateGraph(MessagesState)
graph.add_node("assistant", assistant)
graph.add_edge(START, "assistant")
graph.add_edge("assistant", END)

compiled_graph = graph.compile()

app = FastAPI(title="LangGraph Assistant")
add_routes(app, compiled_graph, path="/assistant")
```

**Key Points:**

- Use `async def` for node functions
- Use `await llm.ainvoke()` instead of `llm.invoke()` for async LLM calls
- LangServe automatically creates `/assistant/ainvoke` endpoint for async execution
- Async execution allows a single worker to handle hundreds of concurrent requests efficiently

**Using Async Endpoints:**

```python
import httpx

# Async invoke endpoint
async with httpx.AsyncClient() as client:
    response = await client.post(
        "http://localhost:8000/assistant/ainvoke",
        json={
            "input": {
                "messages": [{"role": "user", "content": "Hello!"}]
            }
        }
    )
    print(response.json())
```

#### Features

- Automatic schema inference
- Built-in tracing (LangSmith)
- Streaming support
- Batch processing
- Async endpoint support (when using async nodes)

---

### LangGraph Cloud

Managed platform for deploying LangGraph applications.

#### Configuration (`langgraph.json`)

```json
{
  "graphs": {
    "research_assistant": "./research_assistant.py:graph"
  },
  "env": "./.env",
  "python_version": "3.11",
  "dependencies": ["."]
}
```

#### Deployment

```bash
pip install langgraph-cli
langgraph login
langgraph deploy
```

#### Using Deployed Graphs

```python
from langgraph_sdk import get_client

client = get_client(url="https://your-graph-url.langgraph.cloud")
result = await client.runs.create(
    assistant_id="research_assistant",
    input={"messages": [{"role": "user", "content": "Hello!"}]}
)
```

---

### OpenAI-Compatible API

Expose LangGraph instances through an OpenAI-compatible API interface.

#### OpenAI-Compatible API Installation

```bash
pip install langgraph-openai-serve
```

#### OpenAI-Compatible API Setup

```python
from langgraph_openai_serve import serve
from langgraph.graph import StateGraph, MessagesState

graph = StateGraph(MessagesState)
# ... build your graph ...
compiled_graph = graph.compile()

serve(compiled_graph, host="0.0.0.0", port=8000, model_name="my-assistant")
```

#### Usage

```python
import openai

client = openai.OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed"
)

response = client.chat.completions.create(
    model="my-assistant",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

---

## API Endpoints

### LangServe Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/graph/invoke` | POST | Synchronous graph execution |
| `/graph/ainvoke` | POST | Async graph execution (when nodes are async) |
| `/graph/stream` | POST | Stream graph execution |
| `/graph/astream` | POST | Async stream execution (when nodes are async) |
| `/graph/batch` | POST | Batch execution |
| `/graph/input_schema` | GET | Get input schema |
| `/graph/output_schema` | GET | Get output schema |
| `/docs` | GET | Interactive API documentation |

**Note:** Async endpoints (`/ainvoke`, `/astream`) are automatically available when your graph nodes use `async def`. Use async execution for better concurrency with I/O-bound operations (API calls, database queries).

### Request Format

```json
{
  "input": {
    "messages": [{"role": "user", "content": "Hello!"}]
  },
  "config": {
    "configurable": {
      "thread_id": "user-123"
    }
  }
}
```

---

## Configuration

### Production Checkpoints

```python
from langgraph.checkpoint.postgres import PostgresSaver

checkpointer = PostgresSaver.from_conn_string(
    os.getenv("DATABASE_URL")
)
graph = builder.compile(checkpointer=checkpointer)
```

### CORS Configuration

```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://your-frontend.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### Environment Variables

Use `.env` file for sensitive configuration:

```bash
OPENAI_API_KEY=sk-...
DATABASE_URL=postgresql://...
```

---

## Resources

### Official Documentation

- **LangServe Docs**: [python.langchain.com/docs/langserve](https://python.langchain.com/docs/langserve)
- **LangGraph Cloud**: [cloud.langchain.com](https://cloud.langchain.com)
- **LangGraph SDK**: [github.com/langchain-ai/langgraph-sdk](https://github.com/langchain-ai/langgraph-sdk)

### Related Packages

- **langserve** → REST API deployment
- **langgraph-cli** → CLI tools for LangGraph Cloud
- **langgraph-openai-serve** → OpenAI-compatible API
- **langgraph-sdk** → Python client for deployed graphs

---

**Last Updated:** January 2025  
**LangGraph Version:** 1.0+
