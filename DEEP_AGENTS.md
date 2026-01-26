# Deep Agents Integration

Soliplex integrates with [pydantic-deep](https://github.com/pydantic/pydantic-ai) to provide autonomous agent capabilities with planning, filesystem operations, subagent delegation, and code execution.

## Overview

Deep agents extend the standard Soliplex agent model with:

- **Todo Planning** - Task breakdown and progress tracking
- **Filesystem Operations** - Read, write, search, and organize files
- **Subagent Delegation** - Spawn specialized agents for subtasks
- **Skills** - Load modular skill packages for specialized tasks
- **Code Execution** - Run Python code in isolated sandboxes

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Soliplex Server                         │
├─────────────────────────────────────────────────────────────┤
│  Room Config (room_config.yaml)                             │
│    └─ agent: kind: "deep"                                   │
│         └─ DeepAgentConfig                                  │
│              └─ create_deep_agent_from_config()             │
│                   └─ pydantic-deep Agent                    │
│                        ├─ Todo Toolset                      │
│                        ├─ Console/Filesystem Toolset        │
│                        ├─ Subagent Toolset                  │
│                        ├─ Skills Toolset                    │
│                        └─ Backend (State/Filesystem)        │
└─────────────────────────────────────────────────────────────┘
```

## Configuration

### Installation-Level Setup

Register the `DeepAgentConfig` in `installation.yaml`:

```yaml
meta:
  agent_configs:
    - "soliplex.deepagents.config.DeepAgentConfig"
```

Configure default backend for all deep agent rooms:

```yaml
deep_agents:
  default_backend_kind: "filesystem"  # or "state" for in-memory
  state_root: "./deep-state"          # Root directory for file persistence
```

### Room Configuration

Create a room with `kind: "deep"` in `room_config.yaml`:

```yaml
id: "my_deep_agent"
name: "My Deep Agent"
description: "An autonomous agent with planning and file capabilities"

agent:
  kind: "deep"
  model_name: "gpt-oss:latest"
  system_prompt: "./prompt.txt"

  # Feature toggles
  include_todo: true          # Task planning and tracking
  include_filesystem: true    # File read/write/search
  include_subagents: true     # Delegate to specialized agents
  include_skills: false       # Modular skill packages
  include_execute: true       # Python code execution (sandbox)

  # Backend for state persistence
  backend_kind: "filesystem"  # "state" (in-memory) or "filesystem" (persistent)

  # Subagent definitions (when include_subagents: true)
  subagents:
    - name: "researcher"
      description: "Researches topics and gathers information"
      instructions: |
        You are a research specialist.
        - Gather comprehensive information
        - Cite sources when possible
        - Summarize findings clearly

    - name: "coder"
      description: "Writes and reviews code"
      instructions: |
        You are a senior software engineer.
        - Write clean, documented code
        - Follow best practices
        - Include type hints

  # Approval requirements for sensitive operations
  interrupt_on:
    write_file: false    # Require approval before writing files
    edit_file: false     # Require approval before editing files
    execute: true        # Require approval before code execution

  # Subagent nesting depth (0 = subagents can't spawn sub-subagents)
  max_nesting_depth: 0

allow_mcp: false
```

### Configuration Options Reference

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `kind` | string | - | Must be `"deep"` for deep agents |
| `model_name` | string | from env | LLM model to use |
| `system_prompt` | string | - | Inline prompt or `"./path.txt"` |
| `include_todo` | bool | `true` | Enable task planning toolset |
| `include_filesystem` | bool | `true` | Enable file operations toolset |
| `include_subagents` | bool | `true` | Enable subagent delegation |
| `include_skills` | bool | `false` | Enable skills toolset |
| `include_execute` | bool | auto | Enable code execution (auto-detects sandbox) |
| `backend_kind` | string | `"state"` | `"state"`, `"filesystem"`, or `"docker"` |
| `backend_root` | string | auto | Directory for filesystem backend |
| `docker_config` | dict | `{}` | Docker sandbox settings (image, work_dir) |
| `subagents` | list | `[]` | Subagent configurations |
| `skill_directories` | list | `[]` | Paths to discover skills |
| `interrupt_on` | dict | `{}` | Tools requiring approval |
| `max_nesting_depth` | int | `0` | Subagent nesting limit |

## Capability Tiers

Deep agent rooms are organized into capability tiers:

### Tier 1: Foundation
Basic capabilities without file operations.

- `deep_chat` - Pure conversation, no tools

### Tier 2: Planning + Filesystem
Todo planning combined with file operations.

- `deep_notepad` - File operations only
- `deep_researcher` - Research with todo + files
- `deep_coder` - Code generation with todo + files
- `deep_writer` - Document drafting
- `deep_persistent` - Persistent state across sessions
- `deep_planner` - Project planning
- `wedding_planner_todo_fs` - Wedding event planning
- `garden_planner_persistent` - Garden planning with seasonal tracking

### Tier 3: Guarded Operations
Requires human approval for sensitive operations.

- `deep_safe_coder` - Code with write approval
- `deep_executor` - Shell execution with approval
- `legal_contract_reviewer_interrupt` - Contract analysis with write approval

### Tier 3.5: Docker Sandbox (Unguarded)
Fully automated code execution in isolated Docker containers.

- `deep_python_sandbox` - Execute Python in Docker (remote via SSH)

### Tier 4: Skills
Includes loadable skill packages.

- `deep_analyst` - Data analysis skills
- `deep_fullstack` - Full-stack development skills

### Tier 5: Multi-Agent Orchestration
Full subagent delegation capabilities.

- `deep_orchestrator` - General orchestration with coder, tester, reviewer, documenter
- `deep_code_reviewer` - Security review orchestration
- `deep_devops` - Deployment documentation
- `deep_architect` - System architecture design
- `film_production_orchestrator` - Film pre-production with script_analyst, casting_director, location_scout, line_producer

## Code Execution (Docker Sandbox)

The `include_execute` option enables code execution in an isolated Docker container:

```yaml
agent:
  kind: "deep"
  include_execute: true

  # Use Docker sandbox for isolated execution
  backend_kind: "docker"
  docker_config:
    image: "python:3.12-slim"
    work_dir: "/workspace"

  # Set to false for fully automated execution (no approval required)
  interrupt_on:
    execute: false
```

### Remote Docker Execution

To run code on a remote Docker host via SSH:

1. Set `DOCKER_HOST` in your `.env` file:
   ```
   DOCKER_HOST=ssh://hostname
   ```

2. Ensure SSH key authentication is configured for the remote host

3. The Docker daemon on the remote host will execute the code

### Docker Sandbox Features

- **Isolated execution**: Code runs in a fresh container
- **Package installation**: Can install packages with `pip install`
- **File persistence**: Files persist within the container session
- **Network access**: Containers have network access for web requests

### Example Room Configuration

```yaml
id: "deep_python_sandbox"
name: "Python Sandbox"

agent:
  kind: "deep"
  model_name: "gpt-oss:latest"
  system_prompt: "./prompt.txt"

  include_todo: true
  include_filesystem: true
  include_execute: true

  backend_kind: "docker"
  docker_config:
    image: "python:3.12-slim"
    work_dir: "/workspace"

  # Fully automated - no approval required
  interrupt_on:
    execute: false
    write_file: false
```

### Container Cleanup

Call `agent.cleanup()` when done to stop Docker containers:

```python
agent = create_deep_agent_from_config(agent_config)
try:
    result = await agent.run("Execute some code")
finally:
    agent.cleanup()  # Stops Docker container
```

## Domain-Specific Rooms

### Wedding Planner (`wedding_planner_todo_fs`)

Plans weddings with guest management, vendor coordination, and budget tracking.

**Expected outputs:**
- `guests/master_list.md` - Guest categories, RSVPs, dietary needs
- `vendors/comparison.md` - Venue, catering, photography comparisons
- `budget/tracker.md` - Line items with estimates
- `timeline/milestones.md` - 12-month planning timeline

### Garden Planner (`garden_planner_persistent`)

Plans vegetable gardens with seasonal tracking and persistent state.

**Expected outputs:**
- `garden/layout.md` - Bed layout with spacing
- `plants/selection.md` - Variety recommendations
- `calendar/planting_schedule.md` - Zone-based timing
- `care/maintenance_tasks.md` - Watering, fertilizing schedules
- `harvest/yield_tracker.md` - Yield estimates

### Legal Contract Reviewer (`legal_contract_reviewer_interrupt`)

Reviews contracts with risk identification. Requires approval before writing files.

**Expected outputs:**
- `analysis/clause_breakdown.md` - Section analysis
- `analysis/risk_assessment.md` - Risk ratings
- `recommendations/negotiation_points.md` - Redline suggestions

### Film Production Orchestrator (`film_production_orchestrator`)

Coordinates film pre-production using 4 specialized subagents.

**Subagents:**
- `script_analyst` - Screenplay breakdown
- `casting_director` - Character profiles with detailed backstories
- `location_scout` - Location requirements
- `line_producer` - Budget and scheduling

**Expected outputs:**
- `script/breakdown.md` - Scene-by-scene breakdown
- `casting/character_profiles.md` - Detailed character bios with backstories
- `locations/requirements.md` - Location matrix
- `production/budget.md` - Above/below-the-line costs
- `production/schedule.md` - Shooting schedule

## Running Tests

### Prerequisites

```bash
# Ensure OLLAMA_BASE_URL is set
export OLLAMA_BASE_URL=http://localhost:11434

# Or configure in .env
echo "OLLAMA_BASE_URL=http://localhost:11434" >> .env
```

### Run All Deep Agent Tests

```bash
uv run pytest tests/integration/test_deep_rooms.py -v --no-cov
```

### Run Specific Room Tests

```bash
# Test wedding planner
uv run pytest tests/integration/test_deep_rooms.py -k "wedding_planner" -v --no-cov

# Test non-interactive domain rooms
uv run pytest tests/integration/test_deep_rooms.py -k "wedding or garden or film" -v --no-cov

# Test all tier 5 orchestrators
uv run pytest tests/integration/test_deep_rooms.py -k "orchestrator or architect" -v --no-cov
```

### Test Configuration

Tests are configured in `tests/integration/test_deep_rooms.py` with:

- **Prompts** - Complex, multi-step task prompts
- **Expected files** - Files the agent should create
- **Validators** - Content validation functions

Example test configuration:

```python
"wedding_planner_todo_fs": {
    "tier": 2,
    "capabilities": ["todo", "filesystem"],
    "prompt": "Plan a wedding for 150 guests with a $50,000 budget...",
    "expected_files": [
        "guests/master_list.md",
        "vendors/comparison.md",
        "budget/tracker.md",
        "timeline/milestones.md",
    ],
    "file_validators": {
        "guests/master_list.md": [
            has_markdown_structure(),
            contains_all(["family", "friend"]),
        ],
    },
}
```

## File Structure

```
src/soliplex/deepagents/
├── __init__.py
├── agent.py          # SoliplexDeepAgent wrapper
├── config.py         # DeepAgentConfig dataclass
├── factory.py        # create_deep_agent_from_config()
└── _pydantic_deep_patch.py  # Patched create_deep_agent with nesting support

example/rooms/
├── deep_*/           # Technical deep agent rooms
├── wedding_planner_todo_fs/
├── garden_planner_persistent/
├── legal_contract_reviewer_interrupt/
└── film_production_orchestrator/

tests/integration/
└── test_deep_rooms.py  # Integration tests with validators
```

## Troubleshooting

### "OLLAMA_BASE_URL not set"

Subagents require the `OLLAMA_BASE_URL` environment variable. Set it in `.env` or export it:

```bash
export OLLAMA_BASE_URL=http://localhost:11434
```

### Tests fail with "async def functions not supported"

Ensure test functions have `@pytest.mark.asyncio`:

```python
@pytest.mark.asyncio
@pytest.mark.needs_llm
async def test_deep_room(room_id, installation):
    ...
```

### Agent returns `DeferredToolRequests`

This happens when `interrupt_on` is configured. The agent pauses for approval. For testing, either:
- Test output validators only (no expected files)
- Remove `interrupt_on` for the test room

### Subagent "model not found"

Ensure the model string is propagated correctly. The patch extracts provider info:
- Ollama models: `ollama:{model_name}`
- OpenAI models: `openai:{model_name}`
