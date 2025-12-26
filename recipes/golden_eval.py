"""
Golden Dataset Evaluation with LLM Judge
=========================================

TEXT TO TEXT GENERATION

TLDR: Simple evaluation workflow that:
1. Takes an input (task/prompt)
2. Generates output using a generator model
3. LLM judge evaluates quality by comparing to gold reference
4. Returns evaluation metrics (score, feedback, etc.)

Use Case: Regression testing and evaluation of golden datasets.
Compares generated outputs against gold standards to track quality over time.

Workflow: Input → Generate → Judge → Report
(No refinement loop - pure evaluation only)
"""

from typing import Optional
from typing_extensions import TypedDict
from pydantic import BaseModel, Field

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv
from utils import save_graph

load_dotenv()

# Initialize LLMs
generator_model = ChatOpenAI(model="gpt-4o", temperature=0.7)
judge_model = ChatOpenAI(model="gpt-4o", temperature=0)  # Lower temperature for consistent judging

# ============================================================================
# State Schema
# ============================================================================

class EvaluationState(TypedDict):
    """State for golden dataset evaluation workflow."""
    # Input
    task: str  # The task/prompt to generate content for
    use_case: str  # Type of generation: "text", "code", "image_description", etc.
    
    # Generation
    generated_output: str  # Output from generator model
    
    # Gold reference
    gold_reference: Optional[str]  # Reference output from golden dataset
    
    # Evaluation results
    quality_score: float  # Score from 0-100
    feedback: str  # Detailed feedback from judge
    strengths: list[str]  # List of strengths
    weaknesses: list[str]  # List of weaknesses
    passed_threshold: bool  # Whether score meets minimum threshold

# ============================================================================
# Pydantic Models for Structured Output
# ============================================================================

class QualityEvaluation(BaseModel):
    """Structured output from the LLM judge."""
    score: float = Field(
        description="Quality score from 0-100",
        ge=0,
        le=100
    )
    feedback: str = Field(
        description="Detailed feedback comparing generated output to gold reference"
    )
    strengths: list[str] = Field(
        description="List of strengths in the generated output"
    )
    weaknesses: list[str] = Field(
        description="List of weaknesses or areas where it differs from gold reference"
    )

# ============================================================================
# Gold Dataset (Simulated - replace with your actual dataset)
# ============================================================================

GOLD_DATASET = {
    "text": {
        "Write a short story about a robot learning to paint": 
        "In a small studio overlooking the city, ARIA-7 discovered colors. Her metallic fingers, "
        "designed for precision assembly, trembled as she dipped a brush into cerulean blue. "
        "The canvas, blank and waiting, became a sky. Each stroke was a question, each color "
        "an answer she didn't know she was seeking. When the sun set, casting golden light "
        "through the window, ARIA-7 stepped back. The painting wasn't perfect, but it was hers. "
        "And in that moment, she understood what it meant to create something that didn't exist before."
    },
    "code": {
        "Write a Python function to calculate fibonacci numbers":
        "def fibonacci(n: int) -> int:\n"
        "    \"\"\"Calculate the nth Fibonacci number using memoization.\"\"\"\n"
        "    if n < 0:\n"
        "        raise ValueError('n must be non-negative')\n"
        "    memo = {0: 0, 1: 1}\n"
        "    for i in range(2, n + 1):\n"
        "        memo[i] = memo[i - 1] + memo[i - 2]\n"
        "    return memo[n]"
    },
    "image_description": {
        "Describe a futuristic cityscape at sunset":
        "A sprawling metropolis bathed in golden-orange light as the sun sets behind towering "
        "neon-lit skyscrapers. Flying vehicles streak across the sky leaving trails of light. "
        "Holographic advertisements flicker on building facades. The architecture blends "
        "organic curves with geometric precision. In the foreground, a pedestrian bridge "
        "connects two districts, with people silhouetted against the vibrant sky."
    }
}

def get_gold_reference(task: str, use_case: str) -> Optional[str]:
    """Retrieve gold reference from dataset if available."""
    if use_case in GOLD_DATASET:
        return GOLD_DATASET[use_case].get(task)
    return None

# ============================================================================
# Node Functions
# ============================================================================

def generate_output(state: EvaluationState) -> EvaluationState:
    """Generate output for the given task."""
    task = state["task"]
    use_case = state.get("use_case", "text")
    
    # Get gold reference if available
    gold_reference = get_gold_reference(task, use_case)
    
    # Generate output
    generation_prompt = f"""Generate {use_case} content for the following task:

Task: {task}

Generate high-quality content that addresses the task comprehensively."""

    generated_output = generator_model.invoke([
        SystemMessage(content=f"You are an expert {use_case} generator."),
        HumanMessage(content=generation_prompt)
    ]).content
    
    return {
        "generated_output": generated_output,
        "gold_reference": gold_reference
    }

def evaluate_quality(state: EvaluationState) -> EvaluationState:
    """LLM judge evaluates generated output against gold reference."""
    task = state["task"]
    use_case = state["use_case"]
    generated_output = state["generated_output"]
    gold_reference = state.get("gold_reference")
    
    # Build evaluation prompt
    evaluation_prompt = f"""You are an expert quality judge for {use_case} generation.

Task: {task}

Generated Output:
{generated_output}
"""

    # Add gold reference comparison if available
    if gold_reference:
        evaluation_prompt += f"""

Gold Reference (Expected Output):
{gold_reference}

Compare the generated output to the gold reference. Evaluate:
- How well does it match the reference quality?
- What elements are present/missing compared to reference?
- Are there any errors or inconsistencies?
- Overall quality score (0-100) based on similarity and correctness
"""
    else:
        evaluation_prompt += """

Evaluate the output based on:
- Relevance to the task
- Quality and coherence
- Completeness
- Technical correctness (if applicable)
- Overall quality score (0-100)
"""

    evaluation_prompt += """

Provide a detailed evaluation with:
1. A quality score from 0-100
2. Specific feedback comparing generated vs gold (if available)
3. List of strengths
4. List of weaknesses or differences from gold reference
"""

    # Get structured evaluation
    structured_judge = judge_model.with_structured_output(QualityEvaluation)
    evaluation = structured_judge.invoke([
        SystemMessage(content="You are a fair and thorough quality evaluator for regression testing."),
        HumanMessage(content=evaluation_prompt)
    ])
    
    # Check if score meets threshold (configurable for regression testing)
    MIN_SCORE_THRESHOLD = 70.0  # Minimum acceptable score for regression tests
    passed_threshold = evaluation.score >= MIN_SCORE_THRESHOLD
    
    return {
        "quality_score": evaluation.score,
        "feedback": evaluation.feedback,
        "strengths": evaluation.strengths,
        "weaknesses": evaluation.weaknesses,
        "passed_threshold": passed_threshold
    }

# ============================================================================
# Graph Construction
# ============================================================================

builder = StateGraph(EvaluationState)

# Add nodes
builder.add_node("generate", generate_output)
builder.add_node("evaluate", evaluate_quality)

# Define flow: Generate → Evaluate → End (no loops)
builder.add_edge(START, "generate")
builder.add_edge("generate", "evaluate")
builder.add_edge("evaluate", END)

# Compile graph
graph = builder.compile()

# ============================================================================
# Main Function
# ============================================================================

def main():
    """Example usage for golden dataset evaluation."""
    
    print("\n" + "="*70)
    print("Golden Dataset Evaluation with LLM Judge")
    print("="*70 + "\n")
    
    # Example 1: Text generation evaluation
    print("Example 1: Text Generation Evaluation")
    print("-" * 70)
    
    state = {
        "task": "Write a short story about a robot learning to paint",
        "use_case": "text"
    }
    
    result = graph.invoke(state)
    
    print(f"\nTask: {result['task']}")
    print(f"Quality Score: {result['quality_score']:.1f}/100")
    print(f"Passed Threshold (≥70): {result['passed_threshold']}")
    print(f"\nStrengths:")
    for strength in result['strengths']:
        print(f"  • {strength}")
    print(f"\nWeaknesses:")
    for weakness in result['weaknesses']:
        print(f"  • {weakness}")
    print(f"\nFeedback:\n{result['feedback']}")
    
    print("\n" + "="*70 + "\n")
    
    # Example 2: Code generation evaluation
    print("Example 2: Code Generation Evaluation")
    print("-" * 70)
    
    state = {
        "task": "Write a Python function to calculate fibonacci numbers",
        "use_case": "code"
    }
    
    result = graph.invoke(state)
    
    print(f"\nTask: {result['task']}")
    print(f"Quality Score: {result['quality_score']:.1f}/100")
    print(f"Passed Threshold (≥70): {result['passed_threshold']}")
    print(f"\nGenerated Output:\n{result['generated_output']}")
    print(f"\nFeedback:\n{result['feedback']}")
    
    print("\n" + "="*70 + "\n")
    
    # Example 3: Batch evaluation (for regression testing)
    print("Example 3: Batch Evaluation (Regression Test)")
    print("-" * 70)
    
    test_cases = [
        {"task": "Describe a futuristic cityscape at sunset", "use_case": "image_description"},
        {"task": "Write a short story about a robot learning to paint", "use_case": "text"},
    ]
    
    results = []
    for test_case in test_cases:
        result = graph.invoke(test_case)
        results.append({
            "task": result["task"],
            "score": result["quality_score"],
            "passed": result["passed_threshold"]
        })
    
    print("\nRegression Test Results:")
    print("-" * 70)
    for r in results:
        status = "✓ PASS" if r["passed"] else "✗ FAIL"
        print(f"{status} | Score: {r['score']:.1f}/100 | {r['task'][:50]}...")
    
    avg_score = sum(r["score"] for r in results) / len(results)
    all_passed = all(r["passed"] for r in results)
    print(f"\nAverage Score: {avg_score:.1f}/100")
    print(f"All Tests Passed: {all_passed}")
    
    # Save graph visualization
    save_graph(graph, "./images/golden_eval.png")
    
    print("\n✓ Graph visualization saved to ./images/golden_eval.png")
    print("\nDone!")

if __name__ == "__main__":
    main()

