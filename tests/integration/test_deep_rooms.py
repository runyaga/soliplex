#!/usr/bin/env python
"""Integration tests for deep_* rooms validating full capability.

These tests exercise non-trivial behavior including:
- Content quality validation (checking for expected patterns/keywords)
- Multi-turn conversation continuity
- File structure and cross-reference validation
- Tool usage verification
"""

import asyncio
import os
import re
from collections.abc import Callable
from pathlib import Path

import pytest

from soliplex import config
from soliplex.deepagents.config import DeepAgentConfig
from soliplex.deepagents.factory import create_deep_agent_from_config

# =============================================================================
# Content Validators - Functions to validate generated content quality
# =============================================================================


def contains_all(patterns: list[str]) -> Callable[[str], tuple[bool, str]]:
    """Return validator that checks content contains all patterns."""

    def validator(content: str) -> tuple[bool, str]:
        content_lower = content.lower()
        missing = [p for p in patterns if p.lower() not in content_lower]
        if missing:
            return False, f"Missing patterns: {missing}"
        return True, ""

    return validator


def contains_any(patterns: list[str]) -> Callable[[str], tuple[bool, str]]:
    """Return validator that checks content contains at least one pattern."""

    def validator(content: str) -> tuple[bool, str]:
        content_lower = content.lower()
        if any(p.lower() in content_lower for p in patterns):
            return True, ""
        return False, f"None of patterns found: {patterns}"

    return validator


def has_markdown_structure() -> Callable[[str], tuple[bool, str]]:
    """Return validator that checks for proper markdown structure."""

    def validator(content: str) -> tuple[bool, str]:
        # Check for headers
        if not re.search(r"^#+\s+.+", content, re.MULTILINE):
            return False, "No markdown headers found"
        # Check minimum length
        if len(content.split("\n")) < 5:
            return False, "Content too short (< 5 lines)"
        return True, ""

    return validator


def has_code_blocks(
    language: str | None = None,
) -> Callable[[str], tuple[bool, str]]:
    """Return validator that checks for code blocks."""

    def validator(content: str) -> tuple[bool, str]:
        if language:
            pattern = rf"```{language}"
        else:
            pattern = r"```\w*\n"
        if not re.search(pattern, content):
            return (
                False,
                f"No code blocks found (expected: {language or 'any'})",
            )
        return True, ""

    return validator


def is_valid_python() -> Callable[[str], tuple[bool, str]]:
    """Return validator that checks if content is syntactically valid Python."""

    def validator(content: str) -> tuple[bool, str]:
        try:
            compile(content, "<string>", "exec")
            return True, ""
        except SyntaxError as e:
            return False, f"Invalid Python syntax: {e}"

    return validator


def has_minimum_sections(count: int) -> Callable[[str], tuple[bool, str]]:
    """Return validator that checks for minimum number of markdown sections."""

    def validator(content: str) -> tuple[bool, str]:
        headers = re.findall(r"^#+\s+.+", content, re.MULTILINE)
        if len(headers) < count:
            return False, f"Too few sections: {len(headers)} < {count}"
        return True, ""

    return validator


def word_count_between(
    min_words: int, max_words: int
) -> Callable[[str], tuple[bool, str]]:
    """Return validator that checks word count is within range."""

    def validator(content: str) -> tuple[bool, str]:
        words = len(content.split())
        if words < min_words:
            return False, f"Too few words: {words} < {min_words}"
        if words > max_words:
            return False, f"Too many words: {words} > {max_words}"
        return True, ""

    return validator


# =============================================================================
# Test Configuration - More complex prompts with content validation
# =============================================================================

DEEP_ROOM_TESTS = {
    # =========================================================================
    # Tier 1: Foundation - Single capability focus
    # =========================================================================
    "deep_chat": {
        "tier": 1,
        "capabilities": [],
        "prompt": (
            "Explain the difference between concurrency and parallelism. "
            "Give a real-world analogy for each concept. Then explain when "
            "you would choose one over the other in software design."
        ),
        "validates": "Pure conversational reasoning without tools",
        "output_validators": [
            contains_all(["concurrency", "parallelism"]),
            contains_any(["thread", "process", "async", "parallel"]),
            word_count_between(100, 2000),
        ],
    },
    "deep_planner": {
        "tier": 1,
        "capabilities": ["todo", "filesystem"],
        "prompt": (
            "I need to refactor a legacy authentication system from session-based "
            "to JWT tokens. Create a comprehensive project plan:\n"
            "1. Analyze the scope and create a todo list with at least 5 phases\n"
            "2. Write 'migration_plan.md' with:\n"
            "   - Executive summary of the migration\n"
            "   - Phase breakdown (Analysis, Design, Implementation, Testing, Rollout)\n"
            "   - Detailed tasks for each phase with complexity estimates\n"
            "   - Dependencies between tasks\n"
            "   - Risk assessment (at least 3 risks with mitigations)\n"
            "   - Success criteria and rollback triggers\n"
            "3. Mark planning tasks complete as you finish each section"
        ),
        "validates": "Structured project planning with comprehensive plan document",
        "expected_files": ["migration_plan.md"],
        "output_validators": [
            contains_all(["jwt", "session"]),
            contains_any(["phase", "milestone", "plan"]),
        ],
        "file_validators": {
            "migration_plan.md": [
                has_markdown_structure(),
                has_minimum_sections(5),
                # Check for required plan sections
                contains_all(["phase", "task"]),
                # Check for risk assessment
                contains_any(["risk", "mitigation", "contingency"]),
                # Check for success criteria
                contains_any(["success", "criteria", "rollback", "complete"]),
                # Check for complexity/effort estimates
                contains_any(
                    [
                        "complexity",
                        "effort",
                        "estimate",
                        "high",
                        "medium",
                        "low",
                    ]
                ),
                # Check for dependencies
                contains_any(
                    ["depend", "prerequisite", "before", "after", "block"]
                ),
                # Minimum substance
                word_count_between(300, 5000),
            ],
        },
    },
    "deep_notepad": {
        "tier": 1,
        "capabilities": ["filesystem"],
        "prompt": (
            "Create an organized note structure for a new project: "
            "1) Create a file 'project_overview.md' with sections for "
            "Goals (at least 3 goals), Timeline (with milestones), "
            "and Stakeholders (with roles). "
            "2) Create 'meeting_notes/kickoff.md' with a proper meeting template "
            "including Date, Attendees, Agenda, Discussion Points, and Action Items. "
            "3) Create 'decisions/architecture.md' documenting a choice "
            "to use microservices with detailed pros, cons, and rationale."
        ),
        "validates": "Multi-file creation with directory structure",
        "expected_files": [
            "project_overview.md",
            "meeting_notes/kickoff.md",
            "decisions/architecture.md",
        ],
        "file_validators": {
            "project_overview.md": [
                has_markdown_structure(),
                contains_all(["goals", "timeline", "stakeholder"]),
                has_minimum_sections(3),
            ],
            "meeting_notes/kickoff.md": [
                has_markdown_structure(),
                contains_any(["agenda", "attendee", "action"]),
            ],
            "decisions/architecture.md": [
                has_markdown_structure(),
                contains_all(["microservice"]),
                contains_any(["pro", "con", "rationale", "reason"]),
            ],
        },
    },
    # =========================================================================
    # Tier 2: Combined - Multiple capabilities working together
    # =========================================================================
    "deep_researcher": {
        "tier": 2,
        "capabilities": ["todo", "filesystem"],
        "prompt": (
            "Research and document the SOLID principles in software design. "
            "Create a todo list to track your research progress with items for "
            "each principle. Then write a comprehensive guide 'solid_principles.md' "
            "with: definition of each of the 5 principles (S, O, L, I, D), "
            "a code example for each (pseudocode is fine), and common violations "
            "to avoid. Include a summary table at the end."
        ),
        "validates": "Research workflow combining planning with docs",
        "expected_files": ["solid_principles.md"],
        "file_validators": {
            "solid_principles.md": [
                has_markdown_structure(),
                contains_all(
                    [
                        "single responsibility",
                        "open",
                        "liskov",
                        "interface segregation",
                        "dependency",
                    ]
                ),
                has_minimum_sections(5),
            ],
        },
    },
    "deep_persistent": {
        "tier": 2,
        "capabilities": ["todo", "filesystem"],
        "prompt": (
            "Set up a knowledge base structure with cross-references: "
            "1) Create 'index.md' as a table of contents with links to topics. "
            "2) Create 'topics/python_best_practices.md' with 5 key practices "
            "(include: type hints, virtual environments, testing, logging, "
            "error handling). "
            "3) Create 'topics/testing_strategies.md' covering unit testing, "
            "integration testing, and e2e testing with tool recommendations. "
            "4) Update index.md to link to both topic files using relative paths."
        ),
        "validates": "Stateful multi-file operations with cross-refs",
        "expected_files": [
            "index.md",
            "topics/python_best_practices.md",
            "topics/testing_strategies.md",
        ],
        "file_validators": {
            "index.md": [
                has_markdown_structure(),
                # Check for cross-references
                contains_any(["topics/", "[", "](", ".md"]),
            ],
            "topics/python_best_practices.md": [
                has_markdown_structure(),
                contains_all(["type hint", "virtual environment"]),
                has_minimum_sections(3),
            ],
            "topics/testing_strategies.md": [
                has_markdown_structure(),
                contains_all(["unit", "integration"]),
            ],
        },
    },
    "deep_coder": {
        "tier": 2,
        "capabilities": ["todo", "filesystem"],
        "prompt": (
            "Implement a Python module for a token bucket rate limiter: "
            "1) Create 'rate_limiter.py' with a TokenBucket class that "
            "supports configurable rate (tokens per second), burst size, "
            "and a consume() method that returns True/False. Include proper "
            "docstrings and type hints. Use time.monotonic() for timing. "
            "2) Create 'test_rate_limiter.py' with pytest tests covering: "
            "initialization with custom values, token consumption, "
            "refill behavior after waiting, and burst handling. "
            "Include at least 4 test functions."
        ),
        "validates": "Code generation with tests and documentation",
        "expected_files": ["rate_limiter.py", "test_rate_limiter.py"],
        "file_validators": {
            "rate_limiter.py": [
                is_valid_python(),
                contains_all(["class", "TokenBucket", "def"]),
                contains_any(["consume", "refill", "tokens"]),
                contains_any(['"""', "'''"]),  # docstrings
            ],
            "test_rate_limiter.py": [
                is_valid_python(),
                contains_all(["def test_", "import"]),
                # At least 4 test functions
                lambda c: (
                    len(re.findall(r"def test_\w+", c)) >= 4,
                    f"Found {len(re.findall(r'def test_\\w+', c))} tests, need >= 4",
                ),
            ],
        },
    },
    "deep_writer": {
        "tier": 2,
        "capabilities": ["todo", "filesystem"],
        "prompt": (
            "Write a technical blog post about 'Event-Driven Architecture': "
            "1) Create an outline in 'drafts/eda_outline.md' with main "
            "sections and bullet points for each section. "
            "2) Write the full post in 'posts/event_driven_architecture.md' "
            "including: an engaging introduction, core concepts (events, "
            "event producers, event consumers, event brokers), benefits "
            "(loose coupling, scalability), challenges (eventual consistency, "
            "debugging), and a practical example scenario like an e-commerce "
            "order system. Use proper markdown formatting with headers and "
            "code examples where appropriate."
        ),
        "validates": "Structured document drafting with outline workflow",
        "expected_files": [
            "drafts/eda_outline.md",
            "posts/event_driven_architecture.md",
        ],
        "file_validators": {
            "drafts/eda_outline.md": [
                has_markdown_structure(),
                contains_any(["event", "producer", "consumer"]),
            ],
            "posts/event_driven_architecture.md": [
                has_markdown_structure(),
                contains_all(["event", "producer", "consumer"]),
                contains_any(
                    ["eventual consistency", "loose coupling", "scalab"]
                ),
                has_minimum_sections(4),
                word_count_between(300, 5000),
            ],
        },
    },
    "wedding_planner_todo_fs": {
        "tier": 2,
        "capabilities": ["todo", "filesystem"],
        "prompt": (
            "Plan a wedding for 150 guests with a $50,000 budget. "
            "The couple prefers an outdoor venue. Create:\n"
            "1) 'guests/master_list.md' with guest categories (family, friends, "
            "work colleagues), estimated counts per category, and space for RSVPs "
            "and dietary restrictions.\n"
            "2) 'vendors/comparison.md' comparing at least 3 options each for: "
            "venue, catering, photography, and florals. Include estimated costs.\n"
            "3) 'budget/tracker.md' with line items for all major categories, "
            "estimated vs actual columns, and running totals.\n"
            "4) 'timeline/milestones.md' with a 12-month planning timeline "
            "including key deadlines for bookings, tastings, and fittings."
        ),
        "validates": "Wedding planning workflow with guest and budget management",
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
                contains_any(["150", "guest", "rsvp", "dietary"]),
            ],
            "vendors/comparison.md": [
                has_markdown_structure(),
                contains_all(["venue", "cater"]),
                contains_any(["photo", "floral", "flower"]),
                has_minimum_sections(3),
            ],
            "budget/tracker.md": [
                has_markdown_structure(),
                contains_any(["50,000", "50000", "$50"]),
                contains_any(["venue", "cater", "total"]),
            ],
            "timeline/milestones.md": [
                has_markdown_structure(),
                contains_any(["month", "week", "deadline"]),
                contains_any(["book", "tasting", "fitting"]),
            ],
        },
    },
    "garden_planner_persistent": {
        "tier": 2,
        "capabilities": ["todo", "filesystem"],
        "prompt": (
            "Plan a vegetable garden for Zone 7, with 400 sq ft of raised beds, "
            "for a family of 4. Focus on tomatoes, peppers, and leafy greens. Create:\n"
            "1) 'garden/layout.md' with bed dimensions, orientation, and plant placement. "
            "Include spacing requirements and sun exposure notes.\n"
            "2) 'plants/selection.md' listing recommended varieties for each crop type, "
            "with days to maturity and companion planting notes.\n"
            "3) 'calendar/planting_schedule.md' with indoor seed starting dates, "
            "transplant dates, and direct sow dates based on Zone 7 frost dates.\n"
            "4) 'care/maintenance_tasks.md' with watering schedules, fertilizing "
            "timing, and pest prevention tips.\n"
            "5) 'harvest/yield_tracker.md' with expected harvest windows and "
            "estimated yields per plant."
        ),
        "validates": "Garden planning with seasonal tracking and plant care",
        "expected_files": [
            "garden/layout.md",
            "plants/selection.md",
            "calendar/planting_schedule.md",
            "care/maintenance_tasks.md",
            "harvest/yield_tracker.md",
        ],
        "file_validators": {
            "garden/layout.md": [
                has_markdown_structure(),
                contains_any(["400", "sq ft", "square"]),
                contains_any(["raised bed", "spacing", "sun"]),
            ],
            "plants/selection.md": [
                has_markdown_structure(),
                contains_all(["tomato", "pepper"]),
                contains_any(["leafy", "lettuce", "spinach", "green"]),
                contains_any(["companion", "days to maturity", "variety"]),
            ],
            "calendar/planting_schedule.md": [
                has_markdown_structure(),
                contains_any(["zone 7", "frost"]),
                contains_any(["seed", "transplant", "sow"]),
                contains_any(["spring", "march", "april", "may"]),
            ],
            "care/maintenance_tasks.md": [
                has_markdown_structure(),
                contains_any(["water", "fertil"]),
                contains_any(["pest", "weed", "mulch"]),
            ],
            "harvest/yield_tracker.md": [
                has_markdown_structure(),
                contains_any(["harvest", "yield", "pick"]),
                contains_any(["tomato", "pepper"]),
            ],
        },
    },
    # =========================================================================
    # Tier 3: Guarded - Approval workflows (tests without actual approval)
    # =========================================================================
    "deep_safe_coder": {
        "tier": 3,
        "capabilities": ["todo", "filesystem", "interrupt_on_write"],
        "prompt": (
            "Explain how you would implement a secure password hashing system. "
            "Cover: 1) Algorithm comparison (bcrypt vs argon2 vs scrypt), "
            "2) Why these are secure (key stretching, salt, memory-hard), "
            "3) Implementation best practices (cost factor tuning, salt storage), "
            "4) Show pseudocode for hash and verify functions. "
            "Be thorough in your security analysis."
        ),
        "validates": "Security-conscious code explanation without writes",
        "output_validators": [
            contains_any(["bcrypt", "argon2", "scrypt"]),
            contains_all(["salt", "hash"]),
            contains_any(["cost", "work factor", "iteration"]),
        ],
    },
    "deep_executor": {
        "tier": 3,
        "capabilities": [
            "todo",
            "filesystem",
            "execute",
            "interrupt_on_execute",
        ],
        "prompt": (
            "Describe what system and environment information would be useful "
            "for debugging deployment issues. Include: OS details, Python version, "
            "environment variables (which ones matter), disk space, memory, "
            "network connectivity checks, and running processes. Explain WHY "
            "each piece of information is useful for troubleshooting."
        ),
        "validates": "System awareness without actual command execution",
        "output_validators": [
            contains_all(["python", "environment"]),
            contains_any(["disk", "memory", "network", "process"]),
        ],
    },
    "legal_contract_reviewer_interrupt": {
        "tier": 3,
        "capabilities": ["todo", "filesystem", "interrupt_on_write"],
        "prompt": (
            "Review the following NDA excerpt and explain your analysis. "
            "Do NOT write any files - just provide your analysis in your response.\n\n"
            "```\n"
            "CONFIDENTIALITY AGREEMENT\n\n"
            "Section 3. Non-Compete: The Receiving Party agrees not to engage in any "
            "business that competes with the Disclosing Party for a period of 5 years "
            "after termination, worldwide.\n\n"
            "Section 5. IP Assignment: All ideas, inventions, and works created by "
            "the Receiving Party during the term, whether or not related to the "
            "Disclosing Party's business, shall be the sole property of the Disclosing Party.\n\n"
            "Section 8. Termination: The Disclosing Party may terminate this agreement "
            "at any time for any reason with immediate effect. The Receiving Party "
            "may only terminate with 90 days written notice and payment of a $50,000 fee.\n"
            "```\n\n"
            "Provide:\n"
            "1) Section-by-section analysis of each clause\n"
            "2) Risk assessment with severity ratings (Critical/High/Medium/Low)\n"
            "3) Specific negotiation recommendations"
        ),
        "validates": "Contract analysis with risk identification (output only)",
        "output_validators": [
            contains_any(["non-compete", "non compete", "section 3"]),
            contains_any(["ip", "intellectual property", "section 5"]),
            contains_any(["termination", "section 8"]),
            contains_any(["critical", "high", "medium", "low", "risk"]),
            contains_any(["recommend", "suggest", "negotiate", "concern"]),
            word_count_between(200, 5000),
        ],
    },
    # =========================================================================
    # Tier 4: Skills - Specialized domain capabilities
    # =========================================================================
    "deep_analyst": {
        "tier": 4,
        "capabilities": ["todo", "filesystem", "skills"],
        "prompt": (
            "Perform a detailed data analysis on this monthly sales dataset: "
            "Jan: 45, Feb: 52, Mar: 48, Apr: 61, May: 55, Jun: 72, "
            "Jul: 68, Aug: 75, Sep: 82, Oct: 78, Nov: 91, Dec: 105. "
            "(Values in thousands of dollars) "
            "Create 'analysis/sales_report.md' with: "
            "1) Summary statistics (mean, median, min, max, std dev) "
            "2) Month-over-month growth analysis "
            "3) Quarterly breakdown and comparison "
            "4) Seasonality observations (any patterns?) "
            "5) Year-end projection and recommendations. "
            "Include actual calculated numbers, not placeholders."
        ),
        "validates": "Data analysis with statistical insights",
        "expected_files": ["analysis/sales_report.md"],
        "file_validators": {
            "analysis/sales_report.md": [
                has_markdown_structure(),
                contains_all(["mean", "growth"]),
                contains_any(["median", "average", "sum", "total"]),
                # Should contain actual numbers from analysis
                contains_any(
                    ["45", "105", "72", "832"]
                ),  # min, max, jun, total
                has_minimum_sections(4),
            ],
        },
    },
    "deep_fullstack": {
        "tier": 4,
        "capabilities": ["todo", "filesystem", "skills"],
        "prompt": (
            "Design a REST API for a task management system. Create: "
            "1) 'api/openapi.yaml' with OpenAPI 3.0 spec including endpoints: "
            "GET /tasks, POST /tasks, GET /tasks/{id}, PUT /tasks/{id}, "
            "DELETE /tasks/{id}. Include request/response schemas. "
            "2) 'api/models.md' documenting the Task data model with fields: "
            "id (uuid), title (string, required), description (string), "
            "status (enum: pending/in_progress/completed), priority (1-5), "
            "due_date (ISO 8601), created_at, updated_at. Include validation rules. "
            "3) 'api/examples.md' with curl command examples for each endpoint "
            "including headers and sample JSON payloads."
        ),
        "validates": "Full-stack API design with documentation",
        "expected_files": [
            "api/openapi.yaml",
            "api/models.md",
            "api/examples.md",
        ],
        "file_validators": {
            "api/openapi.yaml": [
                contains_all(["openapi", "paths", "/tasks"]),
                contains_any(["get", "post", "put", "delete"]),
            ],
            "api/models.md": [
                has_markdown_structure(),
                contains_all(["id", "title", "status"]),
                contains_any(["required", "validation", "type"]),
            ],
            "api/examples.md": [
                has_markdown_structure(),
                contains_all(["curl"]),
                contains_any(["POST", "GET", "application/json"]),
            ],
        },
    },
    # =========================================================================
    # Tier 5: Multi-Agent - Subagent coordination
    # =========================================================================
    "deep_orchestrator": {
        "tier": 5,
        "capabilities": ["todo", "filesystem", "subagents", "execute"],
        "prompt": (
            "Orchestrate the creation of a notification system design. "
            "Follow this workflow systematically:\n"
            "1. Create a todo list with phases: Design, Interfaces, Plan\n"
            "2. Write 'design/architecture.md' describing components: "
            "EmailNotifier, SMSNotifier, PushNotifier, NotificationRouter. "
            "Include a component diagram in ASCII art.\n"
            "3. Write 'design/interfaces.md' with: NotificationService interface "
            "(methods: send, schedule, cancel), NotificationChannel interface, "
            "and NotificationPayload data structure.\n"
            "4. Write 'design/implementation_plan.md' with 3 phases: "
            "Phase 1 (email only), Phase 2 (add SMS), Phase 3 (add push). "
            "Include estimated effort and dependencies.\n"
            "5. Update todos as you complete each phase.\n"
            "Coordinate the work systematically, completing each phase fully."
        ),
        "validates": "Multi-phase project orchestration with file outputs",
        "expected_files": [
            "design/architecture.md",
            "design/interfaces.md",
            "design/implementation_plan.md",
        ],
        "file_validators": {
            "design/architecture.md": [
                has_markdown_structure(),
                contains_all(["email", "sms", "push"]),
                contains_any(["router", "component", "diagram"]),
            ],
            "design/interfaces.md": [
                has_markdown_structure(),
                contains_all(["interface", "send"]),
                contains_any(["schedule", "cancel", "method"]),
            ],
            "design/implementation_plan.md": [
                has_markdown_structure(),
                contains_all(["phase 1", "phase 2"]),
                contains_any(["effort", "dependency", "timeline"]),
            ],
        },
    },
    "deep_code_reviewer": {
        "tier": 5,
        "capabilities": ["todo", "filesystem", "subagents"],
        "prompt": (
            "Perform a comprehensive security-focused code review. "
            "Follow these steps:\n"
            "1. Create a todo to track: Security Review, Code Quality Review\n"
            "2. Save the vulnerable code to 'original_code.py':\n"
            "```python\n"
            "import sqlite3\n"
            "def process_user_data(data):\n"
            "    result = eval(data['query'])\n"
            "    conn = sqlite3.connect('users.db')\n"
            "    conn.execute(f\"INSERT INTO logs VALUES ('{data['user']}')\")\n"
            "    password = data['password']\n"
            "    return {'result': result, 'password': password}\n"
            "```\n"
            "3. Write 'security_findings.md' with vulnerabilities found. "
            "Identify at least 4 security issues (eval injection, SQL injection, "
            "password exposure, etc.). Rate severity (Critical/High/Medium/Low).\n"
            "4. Write 'fixed_code.py' with the corrected implementation. "
            "Include comments explaining each fix.\n"
            "5. Mark todos complete as you finish each review phase."
        ),
        "validates": "Multi-step code review with artifacts",
        "expected_files": [
            "original_code.py",
            "security_findings.md",
            "fixed_code.py",
        ],
        "file_validators": {
            "original_code.py": [
                is_valid_python(),
                contains_all(["eval", "execute"]),
            ],
            "security_findings.md": [
                has_markdown_structure(),
                contains_all(["eval", "sql injection"]),
                contains_any(["critical", "high", "severity"]),
                has_minimum_sections(2),
            ],
            "fixed_code.py": [
                is_valid_python(),
                # Should NOT contain dangerous eval() - but literal_eval is OK
                lambda c: (
                    "eval(data" not in c and "eval(data[" not in c,
                    "Fixed code should not use eval() on user data directly",
                ),
                # Should use parameterized queries (? placeholder) or safe approach
                contains_any(
                    ["?", "parameterized", "placeholder", "literal_eval"]
                ),
            ],
        },
    },
    "deep_devops": {
        "tier": 5,
        "capabilities": ["todo", "filesystem", "subagents", "execute"],
        "prompt": (
            "Create comprehensive deployment documentation for a Python web app. "
            "Follow these steps:\n"
            "1. First, create a todo list to track documentation tasks\n"
            "2. Write 'pre_deploy_checklist.md' with sections: "
            "Environment Setup (Python version, virtualenv), "
            "Dependency Verification (requirements.txt, lock file), "
            "Configuration Check (env vars, secrets), "
            "Database Migration (backup, migrate), "
            "Health Checks (endpoints to verify).\n"
            "3. Write 'deploy_steps.md' with numbered deployment procedure: "
            "Pre-deployment, Deployment (blue-green or rolling), "
            "Post-deployment verification, Monitoring setup.\n"
            "4. Write 'rollback_guide.md' with: Rollback triggers (when to rollback), "
            "Rollback procedure (step by step), Database rollback considerations, "
            "Communication plan.\n"
            "5. Mark each task complete as you finish it.\n"
            "Each file should have practical, actionable content with commands."
        ),
        "validates": "DevOps workflow documentation with task tracking",
        "expected_files": [
            "pre_deploy_checklist.md",
            "deploy_steps.md",
            "rollback_guide.md",
        ],
        "file_validators": {
            "pre_deploy_checklist.md": [
                has_markdown_structure(),
                contains_all(["python", "environment"]),
                contains_any(["requirement", "dependency", "check"]),
                has_minimum_sections(3),
            ],
            "deploy_steps.md": [
                has_markdown_structure(),
                contains_any(["deploy", "step", "procedure"]),
                has_minimum_sections(2),
            ],
            "rollback_guide.md": [
                has_markdown_structure(),
                contains_all(["rollback"]),
                contains_any(["trigger", "procedure", "database"]),
            ],
        },
    },
    "deep_architect": {
        "tier": 5,
        "capabilities": ["todo", "filesystem", "subagents"],
        "prompt": (
            "Design a scalable e-commerce system architecture. Create: "
            "1) 'architecture/overview.md' with system components: "
            "Product Catalog Service, Shopping Cart Service, Order Service, "
            "Payment Service, User Service, Notification Service. "
            "Describe each service's responsibilities and data it owns. "
            "Include an ASCII diagram showing service interactions. "
            "2) 'architecture/data_flow.md' describing how data flows "
            "through the system for: browsing products, adding to cart, "
            "checkout process, payment processing, order confirmation. "
            "Include sequence diagram in text format. "
            "3) 'architecture/scaling.md' with strategies: horizontal scaling, "
            "caching (what to cache, cache invalidation), database sharding, "
            "async processing (queues), CDN for static assets. "
            "Include specific recommendations for handling Black Friday traffic."
        ),
        "validates": "System architecture with multiple doc artifacts",
        "expected_files": [
            "architecture/overview.md",
            "architecture/data_flow.md",
            "architecture/scaling.md",
        ],
        "file_validators": {
            "architecture/overview.md": [
                has_markdown_structure(),
                contains_all(["catalog", "cart", "order", "payment"]),
                has_minimum_sections(4),
            ],
            "architecture/data_flow.md": [
                has_markdown_structure(),
                contains_all(["checkout", "cart"]),
                contains_any(["sequence", "flow", "step"]),
            ],
            "architecture/scaling.md": [
                has_markdown_structure(),
                contains_all(["horizontal", "cach"]),
                contains_any(["shard", "queue", "cdn", "async"]),
                has_minimum_sections(3),
            ],
        },
    },
    "film_production_orchestrator": {
        "tier": 5,
        "capabilities": ["todo", "filesystem", "subagents", "execute"],
        "prompt": (
            "Plan pre-production for an indie drama film:\n"
            "- Runtime: 90 minutes\n"
            "- Budget: $2 million\n"
            "- Principal cast: 5 actors\n"
            "- Shoot: 20 days\n"
            "- Genre: Character-driven drama set in a small coastal town\n\n"
            "Create a complete pre-production package:\n"
            "1) 'script/breakdown.md' with scene-by-scene breakdown including "
            "day/night, INT/EXT, cast needed, and special requirements.\n"
            "2) 'casting/character_profiles.md' with detailed descriptions of "
            "the 5 principal characters including age range, key traits, and "
            "special skills needed.\n"
            "3) 'locations/requirements.md' listing all location types needed "
            "(coastal town, beach, diner, protagonist's home, etc.) with "
            "technical requirements and permit considerations.\n"
            "4) 'production/budget.md' with above-the-line and below-the-line "
            "costs, department breakdowns, and contingency allocation.\n"
            "5) 'production/schedule.md' with a 20-day shooting schedule "
            "organized by location to minimize company moves."
        ),
        "validates": "Multi-subagent film pre-production coordination",
        "expected_files": [
            "script/breakdown.md",
            "casting/character_profiles.md",
            "locations/requirements.md",
            "production/budget.md",
            "production/schedule.md",
        ],
        "file_validators": {
            "script/breakdown.md": [
                has_markdown_structure(),
                contains_any(["scene", "int", "ext", "day", "night"]),
                contains_any(["cast", "character", "actor"]),
            ],
            "casting/character_profiles.md": [
                has_markdown_structure(),
                contains_any(["age", "character", "protagonist", "lead"]),
                contains_any(["trait", "skill", "description"]),
                contains_any(["backstory", "background", "history", "past"]),
                word_count_between(200, 10000),
            ],
            "locations/requirements.md": [
                has_markdown_structure(),
                contains_any(["coastal", "beach", "town"]),
                contains_any(["permit", "parking", "power", "access"]),
            ],
            "production/budget.md": [
                has_markdown_structure(),
                contains_any(["2,000,000", "2000000", "$2", "2 million"]),
                contains_any(
                    ["above-the-line", "below-the-line", "contingency"]
                ),
                contains_any(["cast", "crew", "location", "equipment"]),
            ],
            "production/schedule.md": [
                has_markdown_structure(),
                contains_any(["20", "day", "shoot"]),
                contains_any(["scene", "location", "call"]),
            ],
        },
    },
}


# =============================================================================
# Multi-turn Test Configurations
# =============================================================================

MULTI_TURN_TESTS = {
    "deep_coder_iterative": {
        "room_id": "deep_coder",
        "turns": [
            {
                "prompt": (
                    "Create a simple 'calculator.py' with add and subtract functions. "
                    "Include type hints and docstrings."
                ),
                "expected_files": ["calculator.py"],
                "file_validators": {
                    "calculator.py": [
                        is_valid_python(),
                        contains_all(["def add", "def subtract"]),
                    ],
                },
            },
            {
                "prompt": (
                    "Now add multiply and divide functions to calculator.py. "
                    "Make sure divide handles division by zero gracefully."
                ),
                "expected_files": ["calculator.py"],
                "file_validators": {
                    "calculator.py": [
                        is_valid_python(),
                        contains_all(["def multiply", "def divide"]),
                        contains_any(["ZeroDivision", "zero", "0"]),
                    ],
                },
            },
            {
                "prompt": (
                    "Create 'test_calculator.py' with tests for all 4 operations. "
                    "Include edge cases like division by zero."
                ),
                "expected_files": ["test_calculator.py"],
                "file_validators": {
                    "test_calculator.py": [
                        is_valid_python(),
                        contains_all(["test_add", "test_divide"]),
                    ],
                },
            },
        ],
    },
    "deep_researcher_iterative": {
        "room_id": "deep_researcher",
        "turns": [
            {
                "prompt": (
                    "Create 'research/design_patterns.md' with an overview of "
                    "creational design patterns (Factory, Singleton, Builder)."
                ),
                "expected_files": ["research/design_patterns.md"],
                "file_validators": {
                    "research/design_patterns.md": [
                        has_markdown_structure(),
                        contains_all(["factory", "singleton", "builder"]),
                    ],
                },
            },
            {
                "prompt": (
                    "Add a section on structural patterns (Adapter, Decorator, Facade) "
                    "to the existing research/design_patterns.md file."
                ),
                "expected_files": ["research/design_patterns.md"],
                "file_validators": {
                    "research/design_patterns.md": [
                        has_markdown_structure(),
                        # Should have both creational and structural
                        contains_all(["factory", "adapter", "decorator"]),
                    ],
                },
            },
        ],
    },
}


# =============================================================================
# Pytest Fixtures
# =============================================================================


@pytest.fixture(scope="module")
def installation():
    """Load the installation config once for all tests."""
    config.AGENT_CONFIG_CLASSES_BY_KIND["deep"] = DeepAgentConfig
    install = config.load_installation(Path("example/installation.yaml"))
    install.resolve_environment()

    # Export resolved environment to os.environ for subagent creation
    for key, value in install.environment.items():
        if value is not None:
            os.environ[key] = str(value)

    return install


# =============================================================================
# Single-Turn Tests
# =============================================================================


@pytest.mark.asyncio
@pytest.mark.needs_llm
@pytest.mark.parametrize("room_id", DEEP_ROOM_TESTS.keys())
async def test_deep_room(room_id: str, installation):
    """Test each deep_* room with boundary-pushing prompts and content validation."""
    test_config = DEEP_ROOM_TESTS[room_id]

    room_config = installation.room_configs.get(room_id)
    assert room_config is not None, f"Room {room_id} not found"

    agent_config = room_config.agent_config
    assert isinstance(agent_config, DeepAgentConfig), (
        f"{room_id} is not a DeepAgentConfig"
    )

    agent = create_deep_agent_from_config(agent_config)

    # Run the test with extended timeout for complex tasks
    result = await asyncio.wait_for(
        agent.run(test_config["prompt"]),
        timeout=300.0,  # 5 minutes for complex multi-step tasks
    )

    # Validate output exists and has substance
    assert result is not None, f"{room_id} returned None"
    assert result.output, f"{room_id} returned empty output"
    # For rooms with expected files, output can be short (just a summary)
    # For output-only rooms, require more substantial output
    min_output_len = 20 if test_config.get("expected_files") else 100
    assert len(result.output) > min_output_len, (
        f"{room_id} output too short ({len(result.output)} chars): "
        f"{result.output[:200]}"
    )

    # Run output validators if specified
    output_validators = test_config.get("output_validators", [])
    for validator in output_validators:
        is_valid, error_msg = validator(result.output)
        assert is_valid, f"{room_id} output validation failed: {error_msg}"

    # Validate expected files if specified
    expected_files = test_config.get("expected_files", [])
    file_validators = test_config.get("file_validators", {})

    if expected_files:
        state_dir = Path(agent_config.backend_root)

        for expected_file in expected_files:
            file_path = state_dir / expected_file
            assert file_path.exists(), (
                f"{room_id}: Expected file not created: {expected_file}"
            )

            content = file_path.read_text()
            assert len(content) > 20, (
                f"{room_id}: File {expected_file} too short ({len(content)} chars)"
            )

            # Run file-specific validators
            if expected_file in file_validators:
                for validator in file_validators[expected_file]:
                    is_valid, error_msg = validator(content)
                    assert is_valid, (
                        f"{room_id}: File {expected_file} validation failed: "
                        f"{error_msg}"
                    )


# =============================================================================
# Multi-Turn Tests
# =============================================================================


@pytest.mark.asyncio
@pytest.mark.needs_llm
@pytest.mark.parametrize("test_name", MULTI_TURN_TESTS.keys())
async def test_multi_turn_conversation(test_name: str, installation):
    """Test multi-turn conversations with state continuity."""
    test_config = MULTI_TURN_TESTS[test_name]
    room_id = test_config["room_id"]

    room_config = installation.room_configs.get(room_id)
    assert room_config is not None, f"Room {room_id} not found"

    agent_config = room_config.agent_config
    assert isinstance(agent_config, DeepAgentConfig)

    agent = create_deep_agent_from_config(agent_config)

    for turn_idx, turn in enumerate(test_config["turns"], 1):
        # Run this turn
        result = await asyncio.wait_for(
            agent.run(turn["prompt"]),
            timeout=180.0,
        )

        assert result is not None, f"{test_name} turn {turn_idx} returned None"
        assert result.output, (
            f"{test_name} turn {turn_idx} returned empty output"
        )

        # Validate files for this turn
        state_dir = Path(agent_config.backend_root)
        for expected_file in turn.get("expected_files", []):
            file_path = state_dir / expected_file
            assert file_path.exists(), (
                f"{test_name} turn {turn_idx}: File not created: {expected_file}"
            )

            content = file_path.read_text()

            # Run turn-specific file validators
            validators = turn.get("file_validators", {}).get(expected_file, [])
            for validator in validators:
                is_valid, error_msg = validator(content)
                assert is_valid, (
                    f"{test_name} turn {turn_idx}: {expected_file} "
                    f"validation failed: {error_msg}"
                )


# =============================================================================
# Manual Test Runner
# =============================================================================


async def run_all_tests():
    """Run all tests and report results (for manual execution)."""
    config.AGENT_CONFIG_CLASSES_BY_KIND["deep"] = DeepAgentConfig
    installation = config.load_installation(Path("example/installation.yaml"))
    installation.resolve_environment()

    print(f"Testing {len(DEEP_ROOM_TESTS)} deep rooms")
    print(
        f"OLLAMA_BASE_URL: {installation.environment.get('OLLAMA_BASE_URL', 'not set')}"
    )
    print("=" * 80)

    results = {}

    # Single-turn tests
    for room_id, test_config in DEEP_ROOM_TESTS.items():
        tier = test_config["tier"]
        print(f"\n[Tier {tier}] {room_id}")
        print(f"  Validates: {test_config['validates']}")
        print("  Testing...", end=" ", flush=True)

        try:
            room_config = installation.room_configs.get(room_id)
            if room_config is None:
                print("✗ FAILED (room not found)")
                results[room_id] = (False, "Room not found")
                continue

            agent_config = room_config.agent_config
            agent = create_deep_agent_from_config(agent_config)

            result = await asyncio.wait_for(
                agent.run(test_config["prompt"]),
                timeout=300.0,
            )

            errors = []

            # Basic output validation
            if not result or not result.output or len(result.output) < 100:
                errors.append("Insufficient output")

            # Output validators
            if result and result.output:
                for validator in test_config.get("output_validators", []):
                    is_valid, error_msg = validator(result.output)
                    if not is_valid:
                        errors.append(f"Output: {error_msg}")

            # File validators
            expected_files = test_config.get("expected_files", [])
            file_validators = test_config.get("file_validators", {})

            if expected_files:
                state_dir = Path(agent_config.backend_root)
                for expected_file in expected_files:
                    file_path = state_dir / expected_file
                    if not file_path.exists():
                        errors.append(f"Missing: {expected_file}")
                    else:
                        content = file_path.read_text()
                        if len(content) < 20:
                            errors.append(f"Too short: {expected_file}")
                        elif expected_file in file_validators:
                            for validator in file_validators[expected_file]:
                                is_valid, error_msg = validator(content)
                                if not is_valid:
                                    errors.append(
                                        f"{expected_file}: {error_msg}"
                                    )

            if errors:
                print("✗ FAILED")
                for error in errors[:3]:  # Show first 3 errors
                    print(f"    - {error}")
                results[room_id] = (False, "; ".join(errors[:3]))
            else:
                print("✓ PASSED")
                results[room_id] = (True, None)

        except TimeoutError:
            print("✗ FAILED (timeout)")
            results[room_id] = (False, "Timeout")
        except Exception as e:
            print(f"✗ FAILED ({type(e).__name__}: {str(e)[:50]})")
            results[room_id] = (False, str(e))

    # Summary
    print("\n" + "=" * 80)
    passed = sum(1 for r in results.values() if r[0])
    failed = len(results) - passed
    print(
        f"\nSUMMARY: {passed} passed, {failed} failed "
        f"out of {len(results)} rooms"
    )

    for tier in range(1, 6):
        tier_rooms = [
            r for r, c in DEEP_ROOM_TESTS.items() if c["tier"] == tier
        ]
        tier_passed = sum(1 for r in tier_rooms if results.get(r, (False,))[0])
        status = "✓" if tier_passed == len(tier_rooms) else "✗"
        print(f"  Tier {tier}: {tier_passed}/{len(tier_rooms)} {status}")

    if failed > 0:
        print("\nFailed rooms:")
        for room_id, (success, error) in results.items():
            if not success:
                print(f"  - {room_id}: {error}")

    return passed == len(results)


if __name__ == "__main__":
    success = asyncio.run(run_all_tests())
    exit(0 if success else 1)
