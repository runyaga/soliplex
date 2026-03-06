"""Spike test helpers for Monty code generation validation."""

from __future__ import annotations

import re


FORBIDDEN_PATTERNS = [
    (re.compile(r"^\s*import\s"), "import statement"),
    (re.compile(r"^\s*from\s+\S+\s+import\s"), "from-import statement"),
    (re.compile(r"^\s*class\s"), "class definition"),
    (re.compile(r"\basync\s+def\b"), "async def"),
    (re.compile(r"\bawait\s"), "await expression"),
    (re.compile(r"\bopen\s*\("), "open() call"),
    (re.compile(r"\bos\.\w"), "os module usage"),
    (re.compile(r"```"), "markdown fence"),
]


def validate_monty_code(code: str) -> list[str]:
    """Validate that generated code is valid Monty sandbox code.

    Returns a list of error strings. Empty list means valid.
    """
    errors = []

    if not code or not code.strip():
        return ["empty code"]

    for line in code.splitlines():
        for pattern, description in FORBIDDEN_PATTERNS:
            if pattern.search(line):
                errors.append(
                    f"forbidden: {description} on line: "
                    f"{line.strip()[:60]}"
                )

    return errors
