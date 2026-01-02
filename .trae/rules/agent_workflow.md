# AI Agent Workflow Rules

These rules guide the AI Assistant (Trae) on how to utilize the internal Agent System for development tasks.

## When to use the Agent System
If the user requests a complex feature implementation, bug fix, or documentation task that can be automated, consider using the `DevPipeline` in `workflows/pipeline.py` to generate the initial artifacts.

## How to use
You can write a script to invoke the pipeline:

```python
from workflows.pipeline import DevPipeline

pipeline = DevPipeline()
result = pipeline.implement_feature("Description of the task")
```

## Available Agents
- **CodingAgent**: For generating new code.
- **DebuggingAgent**: For analyzing errors.
- **TestingAgent**: For creating test suites.
- **DocumentationAgent**: For writing docs.

## Rule Enforcement
Always check code against `rules.engine.RuleEngine` before committing it to the codebase if you are generating it programmatically.
