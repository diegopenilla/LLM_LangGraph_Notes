# LangGraph Study Notes

**LangGraph** is a low-level orchestration framework designed for building, managing, and deploying **long-running, stateful agents**. Unlike traditional sequential workflows, LangGraph introduces a **graph-based architecture**, enabling greater control over the flow of logic, enhanced parallel processing, and improved interpretability of AI-driven workflows.

## Table of Contents

- [Version Information](#version-information)
- [Key Features](#key-features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Core Concepts](#core-concepts)
  - [Chat Models](#1-chat-models)
  - [Message State Management](#2-message-state-management)
  - [Graph Representation](#3-graph-representation)
  - [State Management](#4-state-management)
  - [Agents](#5-agents)
  - [Tool Nodes](#6-tool-nodes)
  - [Reducers](#7-reducers)
  - [Input-Output Schema](#8-input-output-schema)
- [Advanced Features](#advanced-features)
  - [Checkpoints and Persistence](#checkpoints-and-persistence)
  - [Streaming](#streaming)
  - [Batch Processing](#batch-processing)
  - [Async Execution](#async-execution)
  - [Human-in-the-Loop](#human-in-the-loop)
  - [Conditional Routing](#conditional-routing)
  - [Command Pattern (LLM-Driven Routing)](#command-pattern-llm-driven-routing)
  - [Parallel Execution](#parallel-execution)
- [Common Patterns](#common-patterns)
- [Best Practices](#best-practices)
- [Resources](#resources)

---

## Version Information

- **LangGraph Version:** 1.0+ (compatible)
- **Last Updated:** January 2025
- **Note:** This repository uses LangGraph 1.0+ patterns. The deprecated `langgraph.prebuilt` module has been migrated to `langchain.agents`.

---

## Key Features

LangGraph provides several production-ready capabilities:

- **🔄 Durable Execution**: Agents can persist through failures and resume from checkpoints, ensuring reliability in long-running processes.
- **👤 Human-in-the-Loop**: Incorporate human oversight by allowing inspection and modification of agent states during execution.
- **🧠 Comprehensive Memory**: Support both short-term working memory and long-term memory across sessions, facilitating rich, personalized interactions.
- **🚀 Production-Ready Deployment**: Deploy sophisticated agent systems confidently with scalable infrastructure designed for stateful, long-running workflows.
- **📊 Streaming Support**: Real-time output streaming for interactive applications.
- **🔍 Observability**: Integration with LangSmith for agent evaluations and monitoring.

---

## Installation

```bash
pip install -U langgraph
```

For full functionality, you may also need:

```bash
pip install langchain-core langchain-openai langchain-community
```

For checkpoint persistence (optional):

```bash
# In-memory checkpoints (default)
# Already included in langgraph

# PostgreSQL checkpoints
pip install langgraph-checkpoint-postgres

# SQLite checkpoints  
pip install langgraph-checkpoint-sqlite
```

---

## Quick Start

Here's a minimal example to get started:

```python
from langgraph.graph import START, END, StateGraph, MessagesState
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

# Define a simple node
def assistant(state: MessagesState):
    llm = ChatOpenAI(model="gpt-3.5-turbo")
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

# Build the graph
graph = StateGraph(MessagesState)
graph.add_node("assistant", assistant)
graph.add_edge(START, "assistant")
graph.add_edge("assistant", END)

# Compile and run
compiled_graph = graph.compile()
result = compiled_graph.invoke({"messages": [HumanMessage(content="Hello!")]})
print(result["messages"][-1].content)
```

---

## Core Concepts

### **1. Chat Models**

LangGraph integrates with **LLMs (Large Language Models)** through **Chat Models**. A Chat Model represents a structured interface between a model and the graph-based system.

#### **Core Parameters**

- **Model Name** (`model`) → Specifies the underlying LLM (e.g., `gpt-4`, `claude-2`, `gpt-3.5-turbo`).
- **Temperature (`T`)** → Controls LLM output randomness:
  - `T = 0` → Deterministic and fact-driven responses.
  - `T = 1` → Highly creative and variable responses.
- **System Role** → Establishes context and behavior for the model.
- **Streaming Capabilities** → Supports real-time output streaming for interactive applications.

Chat Models serve as **nodes in the computation graph**, processing input messages and generating structured responses.

**Example:**

```python
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="What is LangGraph?")
]
response = llm.invoke(messages)
```

---

### **2. Message State Management**

LangGraph introduces **stateful message tracking**, ensuring continuity in conversations and AI agent interactions.

#### **Structure of a Message**

Each message is structured as:

```json
{
  "role": "user",
  "content": "What is the capital of France?",
  "response_metadata": {}
}
```

- **`role`** → Specifies message origin (`"user"`, `"assistant"`, `"system"`, `"tool"`).
- **`content`** → The actual text or multimodal input.
- **`response_metadata`** → Logs additional data (e.g., token usage, log probabilities).

Messages are stored in **Message State**, a mutable structure that dynamically updates as the agent interacts with the environment.

**MessagesState** is a built-in state class that manages conversation history:

```python
from langgraph.graph import MessagesState

# MessagesState automatically handles message accumulation
def node(state: MessagesState):
    # Access all messages
    all_messages = state["messages"]
    # Add new message
    return {"messages": [new_message]}
```

---

### **3. Graph Representation**

A **graph-based workflow** forms the foundation of LangGraph. It enables complex logic execution by defining nodes (operations) and edges (data flow).

A **graph** in LangGraph consists of:

- **Nodes** $N = n_1, n_2, ..., n_k$ → Represent **computational units** (functions, models, or decision points).
- **Edges** $E =(n_i, n_j)$ → Define **execution order** and dependencies between nodes.

**Basic Graph Construction:**

```python
from langgraph.graph import START, END, StateGraph, MessagesState

def process_input(state: MessagesState):
    # Process input
    return {"messages": [processed_message]}

def llm_node(state: MessagesState):
    # LLM processing
    return {"messages": [llm_response]}

# Build the graph
builder = StateGraph(MessagesState)
builder.add_node("process_input", process_input)
builder.add_node("llm", llm_node)
builder.add_edge(START, "process_input")
builder.add_edge("process_input", "llm")
builder.add_edge("llm", END)

graph = builder.compile()
```

This structure allows **conditional routing, parallel execution, and adaptive workflows**.

---

### **4. State Management**

LangGraph uses **TypedDict** for type-safe state definitions. States can be simple or complex, with support for reducers to handle state updates.

#### **Custom State Definition**

```python
from typing import TypedDict, Annotated
from typing_extensions import Annotated
import operator

class MyState(TypedDict):
    messages: list  # Simple list
    counter: int    # Simple value
    items: Annotated[list, operator.add]  # Reducer for accumulation
```

#### **State Reducers**

Reducers define how state fields are updated when multiple nodes modify them:

- **`operator.add`** → Concatenates lists or adds numbers
- **`operator.or_`** → Merges dictionaries
- **Custom reducers** → Define your own update logic

**Example with Reducer:**

```python
from typing import Annotated
import operator

class State(TypedDict):
    context: Annotated[list, operator.add]  # Accumulates items

def node_a(state: State):
    return {"context": ["item1"]}

def node_b(state: State):
    return {"context": ["item2"]}

# After both nodes run, context = ["item1", "item2"]
```

---

### **5. Agents**

An **Agent** in LangGraph is an entity that interacts with the system by executing a sequence of actions, deciding on tool usage, and dynamically adjusting its state.

#### **Agent Components**

- **Memory State** → Retains past interactions.
- **Decision Policy** → Determines the next best action.
- **Tool Invocation** → Calls external tools or functions.

Agents operate within a graph, allowing flexibility in AI-driven applications, such as task automation and intelligent decision-making.

**Basic Agent Pattern:**

```python
from langgraph.graph import MessagesState, StateGraph, START, END
from langchain.agents import ToolNode, tools_condition

def agent(state: MessagesState):
    llm_with_tools = llm.bind_tools(tools)
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}

builder = StateGraph(MessagesState)
builder.add_node("agent", agent)
builder.add_node("tools", ToolNode(tools))
builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", tools_condition)  # Routes to tools if needed
builder.add_edge("tools", "agent")  # Loop back after tool execution
```

---

### **6. Tool Nodes**

LangGraph enables the use of **Tool Nodes**, which represent external function calls that augment model capabilities.

#### **Defining Tools**

```python
from langchain_core.tools import tool

@tool
def search_tool(query: str) -> str:
    """Search for information about a query."""
    return f"Searching for {query}..."

@tool
def calculator(a: float, b: float, operation: str) -> float:
    """Perform arithmetic operations."""
    if operation == "add":
        return a + b
    elif operation == "multiply":
        return a * b
    # ...

tools = [search_tool, calculator]
```

#### **Using ToolNode**

```python
from langchain.agents import ToolNode

# ToolNode automatically handles tool execution
tool_node = ToolNode(tools)

builder.add_node("tools", tool_node)
```

Tool Nodes integrate with agents, allowing them to **execute API calls, database queries, or computations dynamically**.

---

### **7. Reducers**

A **Reducer** aggregates multiple outputs into a single consolidated result. This is useful in **multi-step, multi-agent, or parallel workflows**.

#### **Common Reducers**

- **Concatenation Reducer** (`operator.add`) → Merges outputs into a single list.
- **Dictionary Merge** (`operator.or_`) → Combines dictionaries.
- **Custom Reducers** → Define your own aggregation logic.

**Example:**

```python
from typing import Annotated
import operator

class State(TypedDict):
    responses: Annotated[list, operator.add]  # Accumulates responses

def parallel_node_1(state: State):
    return {"responses": ["Response 1"]}

def parallel_node_2(state: State):
    return {"responses": ["Response 2"]}

# After both nodes: responses = ["Response 1", "Response 2"]
```

**Custom Reducer Example:**

```python
def custom_reducer(left: list, right: list) -> list:
    """Custom logic to merge lists."""
    return left + [f"Processed: {item}" for item in right]

class State(TypedDict):
    items: Annotated[list, custom_reducer]
```

---

### **8. Input-Output Schema**

To ensure consistency and structured data processing, LangGraph enforces **Input-Output Schemas** using TypedDict.

#### **Schema Definition**

```python
from typing import TypedDict, Optional

class InputSchema(TypedDict):
    user_query: str
    context: Optional[str]

class OutputSchema(TypedDict):
    response: str
    confidence: float
```

Schemas ensure that **each node receives and outputs data in a well-defined format**, making debugging and scaling much easier.

**Using with StateGraph:**

```python
class MyState(TypedDict):
    query: str
    results: list
    metadata: dict

builder = StateGraph(MyState)
# All nodes must conform to MyState schema
```

---

## Advanced Features

### Checkpoints and Persistence

LangGraph supports checkpointing to persist agent state, enabling durable execution and resumable workflows.

#### **In-Memory Checkpoints**

```python
from langgraph.checkpoint.memory import MemorySaver

checkpointer = MemorySaver()
graph = builder.compile(checkpointer=checkpointer)

# Use with thread_id for multi-user scenarios
config = {"configurable": {"thread_id": "user-123"}}
result = graph.invoke(state, config=config)
```

#### **PostgreSQL Checkpoints**

```python
from langgraph.checkpoint.postgres import PostgresSaver

checkpointer = PostgresSaver.from_conn_string("postgresql://...")
graph = builder.compile(checkpointer=checkpointer)
```

#### **SQLite Checkpoints**

```python
from langgraph.checkpoint.sqlite import SqliteSaver

checkpointer = SqliteSaver.from_conn_string("sqlite:///checkpoints.db")
graph = builder.compile(checkpointer=checkpointer)
```

**Benefits:**

- **Resume from failures**: Agents can recover from crashes
- **Multi-user support**: Isolated state per user/session
- **State inspection**: Debug and monitor agent state
- **Human review**: Pause and inspect state before continuing

---

### Streaming

Stream intermediate steps as they execute for real-time feedback.

#### **Basic Streaming**

```python
for event in graph.stream(state, config=config):
    # Process each step
    print(event)
```

#### **Streaming Specific Nodes**

```python
# Stream only specific nodes
for event in graph.stream(state, config=config, stream_mode="updates"):
    if "assistant" in event:
        print("Assistant:", event["assistant"])
```

#### **Streaming Messages**

```python
# Stream message updates
for event in graph.stream(state, config=config, stream_mode="messages"):
    if event:
        print(event[-1].content)
```

**Stream Modes:**

- `"values"` → Full state after each step
- `"updates"` → Only changed state fields
- `"messages"` → Only message updates

---

### Batch Processing

Process multiple inputs efficiently in a single call. Batch processing is essential for handling multiple requests simultaneously, improving throughput and resource utilization.

#### **Synchronous Batch Processing**

```python
# Process multiple inputs at once
inputs = [
    {"messages": [HumanMessage(content="What is Python?")]},
    {"messages": [HumanMessage(content="What is JavaScript?")]},
    {"messages": [HumanMessage(content="What is Rust?")]},
]

# All inputs processed sequentially but efficiently
results = graph.batch(inputs, config=config)

# Access individual results
for i, result in enumerate(results):
    print(f"Result {i}: {result['messages'][-1].content}")
```

**Use Cases:**
- Processing multiple user queries in a single request
- Batch data processing pipelines
- ETL workflows with multiple inputs
- When you need all results before proceeding

**Benefits:**
- **Efficient resource usage**: Better utilization of LLM API rate limits
- **Simplified code**: Single call instead of loops
- **Consistent configuration**: Same config applied to all inputs
- **Error handling**: Can handle errors per input without stopping entire batch

#### **Async Batch Processing**

For I/O-bound operations (API calls, database queries), async batch processing allows concurrent execution, dramatically improving performance:

```python
import asyncio

# Process multiple inputs concurrently
async def process_batch():
    inputs = [
        {"messages": [HumanMessage(content="Query 1")]},
        {"messages": [HumanMessage(content="Query 2")]},
        {"messages": [HumanMessage(content="Query 3")]},
    ]
    
    # All inputs processed concurrently (non-blocking)
    results = await graph.abatch(inputs, config=config)
    
    return results

# Run async batch
results = asyncio.run(process_batch())
```

**When to Use `abatch()` vs `batch()`:**

- **Use `abatch()`** when:
  - Processing many inputs (10+)
  - Operations are I/O-bound (API calls, database queries)
  - You want maximum throughput
  - You're already in an async context

- **Use `batch()`** when:
  - Processing few inputs (< 10)
  - You need synchronous execution
  - Simple scripts or synchronous codebases
  - Debugging (easier to debug sync code)

**Performance Comparison:**

```python
# Synchronous: Processes one at a time
# Time: ~3 seconds for 3 queries (1 sec each)
results = graph.batch(inputs, config=config)

# Async: Processes concurrently
# Time: ~1 second for 3 queries (all at once)
results = await graph.abatch(inputs, config=config)
```

**Batch with Different Configs:**

```python
# Each input can have its own config (e.g., different thread_ids)
configs = [
    {"configurable": {"thread_id": "user-1"}},
    {"configurable": {"thread_id": "user-2"}},
    {"configurable": {"thread_id": "user-3"}},
]

results = graph.batch(inputs, config=configs)
```

**Error Handling in Batches:**

```python
from langgraph.errors import GraphRecursionError

try:
    results = graph.batch(inputs, config=config)
except Exception as e:
    # Handle batch-level errors
    print(f"Batch failed: {e}")

# Or handle per-input errors
results = []
for input_state in inputs:
    try:
        result = graph.invoke(input_state, config=config)
        results.append(result)
    except Exception as e:
        print(f"Input failed: {e}")
        results.append(None)  # Or handle error state
```

---

### Async Execution

LangGraph supports async execution for improved performance in I/O-bound operations. Use async methods when building web servers, APIs, or any application that needs to handle multiple concurrent requests.

#### **Async Node Functions**

Define nodes as async functions to enable non-blocking execution:

```python
async def async_assistant(state: MessagesState):
    llm = ChatOpenAI(model="gpt-3.5-turbo")
    # Use ainvoke for async LLM calls
    response = await llm.ainvoke(state["messages"])
    return {"messages": [response]}

# Graph automatically handles async nodes
builder.add_node("assistant", async_assistant)
graph = builder.compile()
```

#### **Async Invoke**

Execute a single graph run asynchronously:

```python
import asyncio

async def main():
    config = {"configurable": {"thread_id": "user-123"}}
    result = await graph.ainvoke(
        {"messages": [HumanMessage(content="Hello!")]},
        config=config
    )
    return result

result = asyncio.run(main())
```

#### **Async Stream**

Stream results asynchronously for real-time updates:

```python
async def stream_results():
    config = {"configurable": {"thread_id": "user-123"}}
    
    async for event in graph.astream(
        {"messages": [HumanMessage(content="Hello!")]},
        config=config
    ):
        # Process each event as it arrives
        print(event)

asyncio.run(stream_results())
```

#### **Async Batch**

Process multiple inputs concurrently (see [Batch Processing](#batch-processing) for details):

```python
async def process_multiple():
    inputs = [
        {"messages": [HumanMessage(content="Query 1")]},
        {"messages": [HumanMessage(content="Query 2")]},
    ]
    
    # All inputs processed concurrently
    results = await graph.abatch(inputs, config=config)
    return results
```

**When to Use Async:**

- ✅ **Web servers/APIs**: Handle multiple requests concurrently
- ✅ **High-throughput applications**: Process many inputs simultaneously  
- ✅ **I/O-bound operations**: API calls, database queries, file operations
- ✅ **Real-time applications**: Streaming responses to multiple clients

**When NOT to Use Async:**

- ❌ **Simple scripts**: Synchronous code is easier to debug
- ❌ **CPU-bound operations**: Async doesn't help with CPU-intensive tasks
- ❌ **Single requests**: No benefit for one-off executions

---

### Human-in-the-Loop

Add interrupts for human review and intervention during agent execution.

#### **Basic Interrupt**

```python
from langgraph.graph import interrupt

def review_node(state: MessagesState):
    # This will pause execution for human input
    interrupt("Please review the response before continuing")
    return state

builder.add_node("review", review_node)
```

#### **Using Interrupts**

```python
# When graph reaches interrupt, execution pauses
config = {"configurable": {"thread_id": "user-123"}}
result = graph.invoke(state, config=config)

# Resume after human review
updated_state = modify_state(result)  # Human modifies state
graph.invoke(updated_state, config=config)
```

**Use Cases:**

- **Approval workflows**: Require human approval before proceeding
- **Error handling**: Pause on errors for manual intervention
- **Quality control**: Review outputs before finalizing

---

### Conditional Routing

Route execution based on state or node output.

#### **Simple Conditional**

```python
def should_continue(state: MessagesState) -> str:
    if len(state["messages"]) > 10:
        return "summarize"
    return "continue"

builder.add_conditional_edges(
    "agent",
    should_continue,
    {
        "summarize": "summarizer",
        "continue": "agent"
    }
)
```

#### **Using tools_condition**

```python
from langchain.agents import tools_condition

# Automatically routes based on tool calls
builder.add_conditional_edges("agent", tools_condition)
builder.add_edge("tools", "agent")
```

#### **Multiple Routes**

```python
def route(state: State) -> list[str]:
    # Can return multiple next nodes for parallel execution
    if state["parallel"]:
        return ["node_a", "node_b"]
    return ["node_c"]

builder.add_conditional_edges("start", route)
```

---

### Command Pattern (LLM-Driven Routing)

**TLDR:** The Command pattern enables **LLM-driven dynamic routing** - where the LLM itself intelligently decides the next step, rather than relying on static conditional logic. This is essential for multi-agent systems and adaptive workflows.

#### **Why Command Pattern?**

**Without Command (Static Routing):**
```python
# ❌ Static conditional - YOU decide the logic
def route(state: MessagesState) -> str:
    if "hotel" in state["messages"][-1].content.lower():
        return "hotel_advisor"
    elif "travel" in state["messages"][-1].content.lower():
        return "travel_advisor"
    return "general_agent"

builder.add_conditional_edges("agent", route)
```

**With Command (LLM-Driven Routing):**
```python
# ✅ LLM decides - More intelligent, flexible
from langgraph.types import Command
from typing_extensions import Literal

def travel_advisor(state: MessagesState) -> Command[Literal["hotel_advisor", "__end__"]]:
    ai_msg = model.bind_tools([transfer_to_hotel_advisor]).invoke(state["messages"])
    
    if len(ai_msg.tool_calls) > 0:
        # LLM intelligently decided it needs hotel expertise
        return Command(goto="hotel_advisor", update={"messages": [ai_msg]})
    
    return {"messages": [ai_msg]}  # LLM decided it can handle it
```

#### **Key Benefits**

- **Intelligent Routing**: LLM understands context and intent, not just keywords
- **Natural Language**: Handles variations ("hotel", "accommodation", "place to stay")
- **Multi-Agent Collaboration**: Agents can intelligently hand off to each other
- **Adaptive**: No need to update code for new scenarios
- **State Updates**: Can update state while routing

#### **Complete Example: Multi-Agent System**

```python
from langgraph.graph import MessagesState, StateGraph, START
from langgraph.types import Command
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from typing_extensions import Literal

model = ChatOpenAI(model="gpt-3.5-turbo")

# Define transfer tools
@tool
def transfer_to_hotel_advisor():
    """Ask the hotel advisor agent for help with hotel recommendations."""
    return

@tool
def transfer_to_travel_advisor():
    """Ask the travel advisor agent for help with destinations."""
    return

# Travel Advisor Agent
def travel_advisor(state: MessagesState) -> Command[Literal["hotel_advisor", "__end__"]]:
    """Provides travel destination advice. Can hand off to hotel advisor if needed."""
    system_prompt = (
        "You are a travel expert. Recommend destinations. "
        "If user asks about hotels, transfer to hotel_advisor."
    )
    messages = [{"role": "system", "content": system_prompt}] + state["messages"]
    
    # LLM decides: Do I need hotel expertise?
    ai_msg = model.bind_tools([transfer_to_hotel_advisor]).invoke(messages)
    
    if len(ai_msg.tool_calls) > 0:
        # YES - LLM intelligently decided to hand off
        tool_msg = {
            "role": "tool",
            "content": "Transferring to hotel advisor",
            "tool_call_id": ai_msg.tool_calls[-1]["id"]
        }
        return Command(
            goto="hotel_advisor",
            update={"messages": [ai_msg, tool_msg]}
        )
    
    # NO - I can handle this myself
    return {"messages": [ai_msg]}

# Hotel Advisor Agent
def hotel_advisor(state: MessagesState) -> Command[Literal["travel_advisor", "__end__"]]:
    """Provides hotel recommendations. Can hand off back to travel advisor if needed."""
    system_prompt = (
        "You are a hotel expert. Provide hotel recommendations. "
        "If you need destination info, ask travel_advisor."
    )
    messages = [{"role": "system", "content": system_prompt}] + state["messages"]
    
    ai_msg = model.bind_tools([transfer_to_travel_advisor]).invoke(messages)
    
    if len(ai_msg.tool_calls) > 0:
        tool_msg = {
            "role": "tool",
            "content": "Transferring to travel advisor",
            "tool_call_id": ai_msg.tool_calls[-1]["id"]
        }
        return Command(
            goto="travel_advisor",
            update={"messages": [ai_msg, tool_msg]}
        )
    
    return {"messages": [ai_msg]}

# Build Graph
builder = StateGraph(MessagesState)
builder.add_node("travel_advisor", travel_advisor)
builder.add_node("hotel_advisor", hotel_advisor)
builder.add_edge(START, "travel_advisor")
graph = builder.compile()

# Usage
result = graph.invoke({
    "messages": [{"role": "user", "content": "I want to visit Paris and need hotels"}]
})
# Travel advisor → [LLM analyzes] → Hands off to hotel_advisor → Response
```

#### **When to Use Command vs Conditional Edges**

| Use Case | Use Command ✅ | Use Conditional Edges ✅ |
|----------|---------------|-------------------------|
| **LLM should decide routing** | Yes | No |
| **Multi-agent collaboration** | Yes | No |
| **Natural language understanding** | Yes | No |
| **Simple boolean conditions** | No | Yes |
| **Performance-critical (no LLM call)** | No | Yes |
| **Predefined workflow paths** | No | Yes |

#### **Command with State Updates**

```python
def agent(state: MessagesState) -> Command[Literal["specialist", "__end__"]]:
    ai_msg = model.bind_tools([transfer_tool]).invoke(state["messages"])
    
    if ai_msg.tool_calls:
        return Command(
            goto="specialist",
            update={
                "messages": [ai_msg],
                "context": {"previous_agent": "general"},  # Add metadata
                "priority": "high"  # Update state
            }
        )
    
    return {"messages": [ai_msg]}
```

#### **Real-World Use Cases**

**1. Customer Support Escalation**
```python
def support_agent(state: MessagesState) -> Command[Literal["billing", "technical", "manager", "__end__"]]:
    tools = [transfer_to_billing, transfer_to_tech, transfer_to_manager]
    ai_msg = model.bind_tools(tools).invoke(state["messages"])
    
    if ai_msg.tool_calls:
        # LLM intelligently chooses: billing, tech, or manager
        return Command(goto=chosen_agent, update={"messages": [ai_msg]})
    
    return {"messages": [ai_msg]}  # Handle it myself
```

**2. Research Workflow**
```python
def research_agent(state: MessagesState) -> Command[Literal["web_search", "database", "expert", "__end__"]]:
    # LLM analyzes query and decides best next step
    # - Simple fact? → database
    # - Recent event? → web_search
    # - Complex topic? → expert_interview
    return Command(goto=llm_decided_route)
```

---

### Parallel Execution

Execute multiple nodes simultaneously for improved performance.

#### **Parallel Nodes**

```python
# Both nodes execute in parallel
builder.add_edge(START, "node_a")
builder.add_edge(START, "node_b")
builder.add_edge("node_a", "merge")
builder.add_edge("node_b", "merge")
```

#### **Using Send for Dynamic Parallelism**

```python
from langgraph.types import Send

def fan_out(state: State) -> list[Send]:
    # Dynamically create parallel executions
    return [Send("process", {"item": item}) for item in state["items"]]

builder.add_conditional_edges("start", fan_out)
```

**Benefits:**

- **Performance**: Execute independent operations simultaneously
- **Scalability**: Handle multiple items concurrently
- **Efficiency**: Reduce total execution time

---

## Common Patterns

### **1. ReAct Agent Pattern**

```python
# Reason + Act loop
builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", tools_condition)
builder.add_edge("tools", "agent")  # Loop until done
```

### **2. Map-Reduce Pattern**

```python
# Map: Process items in parallel
# Reduce: Aggregate results
def map_step(state): ...
def reduce_step(state): ...

builder.add_conditional_edges("start", fan_out_to_map_nodes)
builder.add_edge("map_nodes", "reduce")
```

### **3. Human-in-the-Loop Pattern**

```python
# Agent → Review → Continue/Modify
builder.add_edge("agent", "review")
builder.add_conditional_edges("review", check_approval)
```

### **4. Memory-Augmented Agent**

```python
# Load memory → Process → Update memory
builder.add_edge(START, "load_memory")
builder.add_edge("load_memory", "agent")
builder.add_edge("agent", "update_memory")
```

---

## Best Practices

### **1. State Design**

- Use **TypedDict** for type safety
- Use **reducers** for accumulating state
- Keep state minimal and focused
- Document state schema clearly

### **2. Node Design**

- Keep nodes **focused and single-purpose**
- Make nodes **idempotent** when possible
- Handle errors gracefully
- Use type hints for clarity

### **3. Graph Structure**

- Use **conditional edges** for dynamic routing
- Leverage **parallel execution** for performance
- Design for **resumability** with checkpoints
- Add **interrupts** for human oversight

### **4. Error Handling**

```python
def robust_node(state: State):
    try:
        # Node logic
        return {"result": value}
    except Exception as e:
        # Handle error
        return {"error": str(e)}
```

### **5. Testing**

- Test nodes in isolation
- Use mock checkpoints for testing
- Verify state transitions
- Test error scenarios

### **6. Performance**

- Use **streaming** for long-running tasks
- Leverage **parallel execution** where possible
- **Use async graph execution** (`await graph.ainvoke()`, `await graph.abatch()`) instead of blocking calls - LangGraph operations are I/O-bound (API calls, database queries), so async allows a single worker to handle hundreds of concurrent requests efficiently, rather than wasting CPU cycles waiting for I/O
- **Use batch processing** (`batch()` or `abatch()`) when processing multiple inputs - Much more efficient than looping over `invoke()` calls
- Implement **checkpointing** for durability
- Monitor with **LangSmith**

---

## Tips

### **1. Prevent Message State Bloat**
`MessagesState` accumulates messages automatically - summarize or truncate periodically to prevent memory issues:

```python
# Summarize when messages exceed threshold
if len(state["messages"]) > 50:
    summary = summarize_messages(state["messages"][:-10])
    return {"messages": [summary] + state["messages"][-10:]}
```

### **2. Always Add Timeouts**
LLM API calls can hang indefinitely - add timeouts:

```python
llm = ChatOpenAI(model="gpt-4", timeout=30.0, max_retries=2)
# Or wrap: await asyncio.wait_for(llm.ainvoke(messages), timeout=30.0)
```

### **3. Use Structured Outputs**
Avoid parsing text - use `with_structured_output()` for type-safe responses:

```python
class Response(BaseModel):
    answer: str
    confidence: float
response = llm.with_structured_output(Response).invoke(messages)
```

### **4. Manage Thread IDs Properly**
Use unique thread IDs per user/session to avoid state collisions:

```python
thread_id = f"{user_id}:{session_id}"  # ✅ Good
# thread_id = "default"  # ❌ Bad - causes collisions
```

### **5. Track and Optimize Costs**
Monitor token usage and costs:

```python
from langchain.callbacks import get_openai_callback
with get_openai_callback() as cb:
    result = graph.invoke(input)
    print(f"Tokens: {cb.total_tokens}, Cost: ${cb.total_cost:.4f}")
```

### **6. Validate Inputs Before Execution**
Validate inputs before expensive graph execution:

```python
class GraphInput(BaseModel):
    messages: list
    @validator('messages')
    def validate_messages(cls, v):
        if len(v) > 100:
            raise ValueError("Too many messages")
        return v
```

### **7. Implement Graceful Degradation**
Add fallback strategies when tools/APIs fail:

```python
try:
    result = external_api.call(state["query"])
except Exception:
    result = simple_fallback(state["query"])  # Fallback
```

### **8. Visualize Graphs for Debugging**
Use graph visualization to understand flow:

```python
graph_image = graph.get_graph().draw_mermaid_png()
# Or use LangGraph Studio for interactive debugging
```

---

## Resources

### **Official Documentation**

- **LangGraph Docs**: [docs.langchain.com/langgraph](https://docs.langchain.com/langgraph)
- **API Reference**: [langchain-ai.github.io/langgraph](https://langchain-ai.github.io/langgraph/reference/)
- **GitHub Repository**: [github.com/langchain-ai/langgraph](https://github.com/langchain-ai/langgraph)

### **Tools and Integrations**

- **LangGraph Studio**: Visual debugging and development tool
- **LangSmith**: Observability and evaluation platform
- **LangChain**: Comprehensive LLM application framework

### **Learning Resources**

- **Recipes**: See [recipes/README.md](./recipes/README.md) for practical examples
- **Tutorials**: Check official documentation for step-by-step guides
- **Community**: Join LangChain Discord for discussions
