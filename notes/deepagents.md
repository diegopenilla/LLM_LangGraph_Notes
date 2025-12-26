# DeepAgents Study Notes

**DeepAgents** is an agent harness developed by LangChain for building **autonomous, long-running agents** capable of handling complex, multi-step tasks over extended periods. It combines LangChain's agent abstractions with LangGraph's runtime to provide a high-level framework with built-in planning, context management, and memory capabilities.

## Table of Contents

- [Architecture & Relationship](#architecture--relationship)
- [Core Features](#core-features)
  - [Planning & Task Decomposition](#1-planning--task-decomposition)
  - [Context Management](#2-context-management)
  - [Subagent Spawning](#3-subagent-spawning)
  - [Long-term Memory](#4-long-term-memory)
- [Advanced Features](#advanced-features)
  - [Pluggable Backends](#pluggable-backends)
  - [Composite Backends](#composite-backends)
  - [Middleware Architecture](#middleware-architecture)
  - [Sandboxed Execution](#sandboxed-execution)
  - [DeepAgents CLI](#deepagents-cli)
- [Customization](#customization)
- [Quick Start](#quick-start)
- [Best Practices](#best-practices)
- [Resources](#resources)

---

## Architecture & Relationship

DeepAgents sits on top of the LangChain ecosystem:

```
┌─────────────────────────────────────┐
│   DeepAgents (Agent Harness)        │  ← High-level framework
│   - Built-in planning tools         │
│   - Filesystem tools                │
│   - Subagent spawning               │
│   - Long-term memory                │
└─────────────────────────────────────┘
              ↓ Uses both
    ┌─────────┴─────────┐
    ↓                   ↓
┌──────────────┐  ┌──────────────┐
│  LangChain   │  │  LangGraph   │
│  (Framework) │  │  (Runtime)   │
│              │  │              │
│ - Agent      │  │ - Graph      │
│   prompts    │  │   execution  │
│ - Tools      │  │ - State      │
│ - Abstractions│  │   management│
└──────────────┘  └──────────────┘
```

**Key Points:**
- **LangGraph**: Provides graph execution and state management (runtime)
- **LangChain**: Provides agent framework, tools, and model integrations
- **DeepAgents**: Combines both to offer a complete agent harness with built-in capabilities

**When to Use Each:**
- **LangGraph**: Fine-grained control, custom workflows, production systems
- **LangChain**: Rapid agent development with standard patterns
- **DeepAgents**: Complex, open-ended tasks requiring planning, context management, and long-term memory

---

## Core Features

### **1. Planning & Task Decomposition**

DeepAgents includes a built-in `write_todos` tool that enables agents to:

- **Break down complex tasks** into discrete, manageable steps
- **Track progress** through visual todo lists
- **Adapt plans** dynamically as new information emerges

**How It Works:**
```python
# Agent automatically uses write_todos tool
# Example: "Build a web scraper"
# Agent creates:
# - [ ] Set up project structure
# - [ ] Implement HTTP client
# - [ ] Add parsing logic
# - [ ] Handle errors
# - [ ] Write tests
```

**Benefits:**
- Structured approach to complex tasks
- Progress tracking and visibility
- Ability to modify plans mid-execution

---

### **2. Context Management**

DeepAgents provides **filesystem tools** to manage large contexts without overwhelming the LLM's context window:

**Available Tools:**
- `ls` - List directory contents
- `read_file` - Read file contents
- `write_file` - Create or write files
- `edit_file` - Modify existing files
- `glob` - Match filenames using patterns
- `grep` - Search file contents for patterns

**How It Works:**
```python
# Instead of loading entire codebase into context:
# ❌ context = read_entire_codebase()  # 100k tokens!

# Agent uses filesystem tools:
# ✅ agent.read_file("src/utils.py")
# ✅ agent.grep("def calculate", "src/")
# ✅ agent.write_file("output.txt", results)
```

**Benefits:**
- **Prevents context overflow**: Offload large data to virtual filesystem
- **Variable-length tool results**: Handle outputs of any size
- **Efficient context management**: Only load what's needed

**Advanced:**
- Large tool results are automatically evicted to filesystem when exceeding token limits
- Conversation history can be summarized to manage token usage

---

### **3. Subagent Spawning**

DeepAgents features a `task` tool that enables agents to **spawn specialized subagents** for context isolation.

**How It Works:**
```python
# Main agent delegates to subagent
main_agent.task(
    description="Research Python async patterns",
    subagent_prompt="You are a Python expert focusing on async/await..."
)

# Subagent runs in isolated context
# - Doesn't pollute main agent's context
# - Can focus deeply on specific subtask
# - Returns results to main agent
```

**Benefits:**
- **Context isolation**: Keep main agent's context clean
- **Specialized expertise**: Subagents can have domain-specific prompts
- **Modularity**: Break complex tasks into focused sub-tasks
- **Scalability**: Handle tasks that require deep dives

**Use Cases:**
- Research tasks requiring focused attention
- Code analysis in specific domains
- Multi-step workflows with distinct phases

---

### **4. Long-term Memory**

DeepAgents leverages **LangGraph's Store** to provide persistent memory across threads and conversations.

**How It Works:**
```python
# Agent saves information to persistent store
agent.save_to_memory("project_conventions", {...})

# Later, in a different conversation/thread:
agent.retrieve_from_memory("project_conventions")
# Returns: {...}
```

**Benefits:**
- **Cross-conversation persistence**: Remember information across sessions
- **Thread isolation**: Each thread has its own memory space
- **Knowledge accumulation**: Build up context over time
- **Continuity**: Maintain project conventions and learned patterns

**Storage Options:**
- LangGraph Store (default) - Cross-thread persistence
- Local filesystem - Per-thread storage
- Custom backends - S3, databases, etc.

---

## Advanced Features

### **Pluggable Backends**

DeepAgents 0.2+ introduced **pluggable backends** for flexible storage solutions.

**Built-in Backends:**
- **Local Filesystem**: Direct file system access
- **LangGraph State**: Per-thread state storage
- **LangGraph Store**: Cross-thread persistent storage

**Example:**
```python
from deepagents import DeepAgent
from deepagents.backends import LocalFilesystemBackend, LangGraphStoreBackend

# Use local filesystem
agent = DeepAgent(backend=LocalFilesystemBackend("/path/to/storage"))

# Use LangGraph Store for persistence
agent = DeepAgent(backend=LangGraphStoreBackend(store=my_store))
```

---

### **Composite Backends**

Mix and match storage systems by mapping different backends to specific subdirectories.

**Example:**
```python
from deepagents.backends import CompositeBackend, LocalFilesystemBackend, S3Backend

# Base: local filesystem
# /memories/ → S3 for long-term persistence
backend = CompositeBackend(
    base=LocalFilesystemBackend("/local/storage"),
    mappings={
        "/memories/": S3Backend(bucket="agent-memories")
    }
)

agent = DeepAgent(backend=backend)
```

**Use Cases:**
- Local development files + cloud-persisted memories
- Temporary files locally + important data in database
- Hybrid storage strategies

---

### **Middleware Architecture**

DeepAgents uses a **modular middleware system** for extensibility.

**Built-in Middleware:**
- **TodoListMiddleware**: Provides `write_todos` tool
- **FilesystemMiddleware**: Provides filesystem tools (`ls`, `read_file`, etc.)
- **SubAgentMiddleware**: Enables subagent spawning via `task` tool

**Custom Middleware:**
```python
from deepagents import DeepAgent
from deepagents.middleware import Middleware

class CustomMiddleware(Middleware):
    def get_tools(self):
        return [my_custom_tool]
    
    def modify_prompt(self, prompt):
        return prompt + "\n\nAdditional instructions..."
    
    def on_agent_start(self, state):
        # Hook into agent lifecycle
        pass

agent = DeepAgent(middleware=[CustomMiddleware()])
```

**Benefits:**
- **Extensibility**: Add custom tools, modify prompts, hook into lifecycle
- **Composability**: Mix and match middleware components
- **Modularity**: Each middleware handles specific concerns

---

### **Sandboxed Execution**

DeepAgents can execute code inside **remote sandboxes** for safe, isolated execution.

**Features:**
- **Safety**: Isolated execution environment
- **Configurability**: Custom environments and dependencies
- **Parallel execution**: Run multiple sandboxes concurrently
- **Long-running tasks**: Support for extended execution
- **Reproducibility**: Consistent environments

**Use Cases:**
- Code execution in untrusted environments
- Testing and validation
- Isolated computation tasks

---

### **DeepAgents CLI**

Command-line interface for building agents with persistent memory.

**Installation:**
```bash
pip install deepagents-cli
```

**Usage:**
```bash
# Set API keys
export ANTHROPIC_API_KEY=your_key
export OPENAI_API_KEY=your_key
export TAVILY_API_KEY=your_key  # For web search

# Launch CLI
deepagents
```

**CLI Features:**
- **File Operations**: Read, write, edit project files
- **Shell Commands**: Execute commands with human approval
- **Web Search**: Search for up-to-date information (Tavily)
- **HTTP Requests**: Interact with APIs
- **Task Planning**: Visual todo lists
- **Memory Management**: Store/retrieve across sessions
- **Human-in-the-Loop**: Approval for sensitive operations

**Example Session:**
```bash
You: Add type hints to all functions in src/utils.py

Agent: Reading src/utils.py...
Agent: Found 5 functions without type hints
Agent: Proposing changes:
  - def calculate_total(items) → def calculate_total(items: list[float]) -> float
  ...
Approve changes? [y/n]: y
Agent: Writing changes...
Agent: Done!
```

---

## Customization

### **Model Selection**

```python
from deepagents import DeepAgent
from langchain_openai import ChatOpenAI

# Default: claude-sonnet-4-5-20250929
agent = DeepAgent()

# Custom model
agent = DeepAgent(model="gpt-4o")
# Or
agent = DeepAgent(model=ChatOpenAI(model="gpt-4-turbo"))
```

### **System Prompt**

```python
custom_prompt = """
You are a specialized code review agent.
Focus on:
- Security vulnerabilities
- Performance optimizations
- Code style consistency
"""

agent = DeepAgent(system_prompt=custom_prompt)
```

### **Custom Tools**

```python
from langchain_core.tools import tool

@tool
def custom_api_call(query: str) -> str:
    """Call custom API endpoint."""
    return api_client.get(query)

agent = DeepAgent(tools=[custom_api_call])
```

### **Custom Subagents**

```python
from deepagents import DeepAgent

subagent = DeepAgent(
    system_prompt="You are a Python async expert...",
    tools=[async_specific_tools]
)

main_agent = DeepAgent(subagents={"async_expert": subagent})
```

### **Backend Configuration**

```python
from deepagents.backends import LocalFilesystemBackend

backend = LocalFilesystemBackend("/custom/path")
agent = DeepAgent(backend=backend)
```

---

## Quick Start

### **Installation**

```bash
pip install deepagents-cli
# Or
pip install deepagents  # Library only
```

### **Basic Usage**

```python
from deepagents import DeepAgent

# Initialize agent
agent = DeepAgent()

# Run a task
result = agent.run("Analyze the codebase and suggest improvements")
print(result)
```

### **With Custom Configuration**

```python
from deepagents import DeepAgent
from deepagents.backends import LangGraphStoreBackend
from langgraph.checkpoint.postgres import PostgresSaver

# Set up persistent storage
checkpointer = PostgresSaver.from_conn_string("postgresql://...")
store = LangGraphStoreBackend(checkpointer=checkpointer)

# Create agent with persistence
agent = DeepAgent(
    model="gpt-4o",
    backend=store,
    system_prompt="You are a helpful coding assistant."
)

# Use with thread_id for multi-user scenarios
config = {"configurable": {"thread_id": "user-123"}}
result = agent.run("Task here", config=config)
```

### **CLI Usage**

```bash
# Set environment variables
export ANTHROPIC_API_KEY=your_key
export OPENAI_API_KEY=your_key

# Launch
deepagents

# In CLI:
You: Build a REST API with FastAPI
Agent: [Creates todo list, starts implementing...]
```

---

## Best Practices

### **1. Context Management**

- **Use filesystem tools** instead of loading entire codebases into context
- **Leverage grep/glob** to find relevant files before reading
- **Write intermediate results** to files to avoid context bloat

### **2. Task Planning**

- **Break down complex tasks** using `write_todos` before starting
- **Review and approve** plans before execution
- **Adapt plans** as new information emerges

### **3. Subagent Usage**

- **Spawn subagents** for focused, isolated tasks
- **Use specialized prompts** for subagents
- **Keep main agent context clean** by delegating deep dives

### **4. Memory Management**

- **Save important conventions** to memory for reuse
- **Use thread IDs** properly for multi-user scenarios
- **Choose appropriate backends** based on persistence needs

### **5. Backend Selection**

- **Local filesystem**: Development, single-user scenarios
- **LangGraph Store**: Multi-user, cross-thread persistence
- **Composite backends**: Hybrid strategies for different data types

### **6. Error Handling**

- **Use human-in-the-loop** for sensitive operations
- **Implement approval workflows** for file writes and shell commands
- **Monitor with LangSmith** for observability

### **7. Performance**

- **Use async execution** for concurrent operations
- **Leverage sandboxes** for parallel code execution
- **Optimize context usage** to reduce token costs

---

## Resources

### **Official Documentation**

- **DeepAgents Docs**: [docs.langchain.com/oss/python/deepagents](https://docs.langchain.com/oss/python/deepagents/overview)
- **GitHub Repository**: [github.com/langchain-ai/deepagents](https://github.com/langchain-ai/deepagents)
- **CLI Documentation**: [docs.langchain.com/oss/python/deepagents/cli](https://docs.langchain.com/oss/python/deepagents/cli)

### **Related Resources**

- **LangGraph Docs**: [docs.langchain.com/langgraph](https://docs.langchain.com/langgraph)
- **LangChain Docs**: [docs.langchain.com](https://docs.langchain.com)
- **LangSmith**: Observability and deployment platform

### **Blog Posts & Announcements**

- **Introducing DeepAgents CLI**: [blog.langchain.com/introducing-deepagents-cli](https://blog.langchain.com/introducing-deepagents-cli/)
- **DeepAgents 0.2 Release**: [changelog.langchain.com/announcements/deepagents-0-2-release](https://changelog.langchain.com/announcements/deepagents-0-2-release-for-more-autonomous-agents)
- **Sandboxes for Deep Agents**: [changelog.langchain.com/announcements/sandboxes-for-deep-agents](https://changelog.langchain.com/announcements/sandboxes-for-deep-agents)

---

## Summary

**DeepAgents** provides a comprehensive agent harness for building autonomous, long-running agents. Key capabilities include:

✅ **Planning**: Built-in task decomposition and tracking  
✅ **Context Management**: Filesystem tools for handling large contexts  
✅ **Subagents**: Spawn specialized agents for context isolation  
✅ **Memory**: Persistent storage across conversations  
✅ **Flexibility**: Pluggable backends, middleware, and customization  
✅ **CLI**: Terminal interface for agent development  

**Best For:**
- Complex, multi-step tasks
- Long-running autonomous agents
- Tasks requiring context management
- Projects needing persistent memory
- Code generation and analysis workflows

**Not Best For:**
- Simple, single-step tasks (use LangChain directly)
- Highly custom workflows (use LangGraph directly)
- Real-time, low-latency applications


