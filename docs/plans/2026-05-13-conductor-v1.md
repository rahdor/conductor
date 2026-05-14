# Conductor v1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Python CLI tool that intelligently routes LLM prompts to cost-optimal models via semantic triage, with built-in research capabilities to validate routing effectiveness.

**Architecture:** Async-first Python application using OpenRouter as unified LLM gateway. SQLite for persistence. Click for CLI. Triage via Sonnet, shadow sampling against Opus baseline for research data collection. Adaptive learning adjusts routing rules based on AutoJudge quality scores.

**Tech Stack:** Python 3.11+, Click, httpx (async), SQLite, Pydantic v2, PyYAML

---

## File Structure

```
conductor/
├── pyproject.toml                 # Project config, dependencies, entry point
├── conductor.yaml.example         # Example configuration file
├── src/
│   └── conductor/
│       ├── __init__.py            # Package init, version
│       ├── __main__.py            # Entry point for `python -m conductor`
│       ├── cli.py                 # Click CLI commands (ask, bench, stats, etc.)
│       ├── config.py              # Config loading, env var resolution, validation
│       ├── models.py              # Shared Pydantic models (RequestRecord, etc.)
│       ├── executor.py            # OpenRouter async client, retries, shadow sampling
│       ├── router.py              # Triage prompt, routing decision logic
│       ├── tracker.py             # SQLite operations, schema init
│       ├── learner.py             # Stats aggregation, rule generation
│       └── autojudge.py           # Quality scoring via LLM
└── tests/
    ├── conftest.py                # Shared fixtures (mock OpenRouter, temp DB)
    ├── test_config.py             # Config loading tests
    ├── test_models.py             # Pydantic model tests
    ├── test_executor.py           # Executor tests with mocked API
    ├── test_router.py             # Router tests with mocked executor
    ├── test_tracker.py            # SQLite operations tests
    ├── test_learner.py            # Learning system tests
    ├── test_autojudge.py          # AutoJudge tests
    └── test_cli.py                # CLI integration tests
```

---

## Task 1: Project Setup

**Files:**
- Create: `pyproject.toml`
- Create: `src/conductor/__init__.py`
- Create: `src/conductor/__main__.py`
- Create: `tests/conftest.py`

- [ ] **Step 1: Create pyproject.toml**

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "conductor"
version = "0.1.0"
description = "Intelligent LLM request routing with empirical validation"
readme = "README.md"
requires-python = ">=3.11"
dependencies = [
    "click>=8.1.0",
    "httpx>=0.27.0",
    "pydantic>=2.0.0",
    "pyyaml>=6.0.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0.0",
    "pytest-asyncio>=0.23.0",
    "respx>=0.21.0",
]

[project.scripts]
conductor = "conductor.cli:main"

[tool.hatch.build.targets.wheel]
packages = ["src/conductor"]

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]
```

- [ ] **Step 2: Create package init**

```python
# src/conductor/__init__.py
"""Conductor: Intelligent LLM request routing with empirical validation."""

__version__ = "0.1.0"
```

- [ ] **Step 3: Create __main__.py entry point**

```python
# src/conductor/__main__.py
"""Allow running as `python -m conductor`."""

from conductor.cli import main

if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Create test conftest with fixtures**

```python
# tests/conftest.py
"""Shared test fixtures."""

import os
import tempfile
from pathlib import Path
from typing import Generator

import pytest


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def temp_db_path(temp_dir: Path) -> Path:
    """Return path for temporary test database."""
    return temp_dir / "test_conductor.db"


@pytest.fixture
def temp_config_path(temp_dir: Path) -> Path:
    """Return path for temporary config file."""
    return temp_dir / "conductor.yaml"


@pytest.fixture(autouse=True)
def mock_env_vars(monkeypatch: pytest.MonkeyPatch) -> None:
    """Set test environment variables."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-api-key-12345")
```

- [ ] **Step 5: Create directory structure**

Run:
```bash
mkdir -p src/conductor tests
touch src/conductor/__init__.py src/conductor/__main__.py tests/conftest.py
```

- [ ] **Step 6: Install in development mode**

Run:
```bash
pip install -e ".[dev]"
```

Expected: Installation succeeds, `conductor` command available (will error since cli.py doesn't exist yet)

- [ ] **Step 7: Verify pytest runs**

Run:
```bash
pytest --collect-only
```

Expected: "no tests ran" (no test files yet), but no errors

- [ ] **Step 8: Commit**

```bash
git add pyproject.toml src/ tests/
git commit -m "feat: initial project setup with dependencies"
```

---

## Task 2: Pydantic Models

**Files:**
- Create: `src/conductor/models.py`
- Create: `tests/test_models.py`

- [ ] **Step 1: Write failing test for TaskType enum**

```python
# tests/test_models.py
"""Tests for Pydantic models."""

from conductor.models import TaskType


def test_task_type_values():
    """TaskType enum has expected values."""
    assert TaskType.CODING == "coding"
    assert TaskType.REASONING == "reasoning"
    assert TaskType.CREATIVE == "creative"
    assert TaskType.SIMPLE_QA == "simple_qa"
    assert TaskType.GENERAL == "general"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_models.py -v`

Expected: FAIL with "ModuleNotFoundError: No module named 'conductor.models'"

- [ ] **Step 3: Create models.py with TaskType**

```python
# src/conductor/models.py
"""Shared Pydantic models for Conductor."""

from enum import Enum


class TaskType(str, Enum):
    """Classification of prompt task types."""

    CODING = "coding"
    REASONING = "reasoning"
    CREATIVE = "creative"
    SIMPLE_QA = "simple_qa"
    GENERAL = "general"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_models.py::test_task_type_values -v`

Expected: PASS

- [ ] **Step 5: Write failing test for ModelTier enum**

```python
# tests/test_models.py (append)


def test_model_tier_values():
    """ModelTier enum has expected values."""
    from conductor.models import ModelTier

    assert ModelTier.FAST == "fast"
    assert ModelTier.MID == "mid"
    assert ModelTier.BEST == "best"
```

- [ ] **Step 6: Run test to verify it fails**

Run: `pytest tests/test_models.py::test_model_tier_values -v`

Expected: FAIL with "cannot import name 'ModelTier'"

- [ ] **Step 7: Add ModelTier to models.py**

```python
# src/conductor/models.py (append after TaskType)


class ModelTier(str, Enum):
    """Model capability tiers for routing."""

    FAST = "fast"
    MID = "mid"
    BEST = "best"
```

- [ ] **Step 8: Run test to verify it passes**

Run: `pytest tests/test_models.py::test_model_tier_values -v`

Expected: PASS

- [ ] **Step 9: Write failing test for TokenUsage model**

```python
# tests/test_models.py (append)


def test_token_usage_model():
    """TokenUsage model validates and calculates total."""
    from conductor.models import TokenUsage

    usage = TokenUsage(input_tokens=100, output_tokens=50)
    assert usage.input_tokens == 100
    assert usage.output_tokens == 50
    assert usage.total_tokens == 150
```

- [ ] **Step 10: Run test to verify it fails**

Run: `pytest tests/test_models.py::test_token_usage_model -v`

Expected: FAIL with "cannot import name 'TokenUsage'"

- [ ] **Step 11: Add TokenUsage model**

```python
# src/conductor/models.py (append)

from pydantic import BaseModel, computed_field


class TokenUsage(BaseModel):
    """Token usage for an API call."""

    input_tokens: int
    output_tokens: int

    @computed_field
    @property
    def total_tokens(self) -> int:
        """Total tokens used."""
        return self.input_tokens + self.output_tokens
```

- [ ] **Step 12: Run test to verify it passes**

Run: `pytest tests/test_models.py::test_token_usage_model -v`

Expected: PASS

- [ ] **Step 13: Write failing test for RoutingDecision model**

```python
# tests/test_models.py (append)


def test_routing_decision_model():
    """RoutingDecision model with all fields."""
    from conductor.models import RoutingDecision, TaskType, TokenUsage

    decision = RoutingDecision(
        complexity=3,
        task_type=TaskType.CODING,
        model="anthropic/claude-sonnet-4",
        reasoning="Code generation task with moderate complexity",
        triage_tokens=TokenUsage(input_tokens=150, output_tokens=45),
        triage_cost=0.0003,
        triage_latency_ms=234,
    )
    assert decision.complexity == 3
    assert decision.task_type == TaskType.CODING
    assert decision.model == "anthropic/claude-sonnet-4"


def test_routing_decision_complexity_validation():
    """RoutingDecision rejects invalid complexity."""
    import pytest
    from pydantic import ValidationError
    from conductor.models import RoutingDecision, TaskType, TokenUsage

    with pytest.raises(ValidationError):
        RoutingDecision(
            complexity=6,  # Invalid: must be 1-5
            task_type=TaskType.CODING,
            model="anthropic/claude-sonnet-4",
            reasoning="Test",
            triage_tokens=TokenUsage(input_tokens=10, output_tokens=10),
            triage_cost=0.0001,
            triage_latency_ms=100,
        )
```

- [ ] **Step 14: Run tests to verify they fail**

Run: `pytest tests/test_models.py::test_routing_decision_model tests/test_models.py::test_routing_decision_complexity_validation -v`

Expected: FAIL with "cannot import name 'RoutingDecision'"

- [ ] **Step 15: Add RoutingDecision model**

```python
# src/conductor/models.py (append)

from pydantic import Field


class RoutingDecision(BaseModel):
    """Result of routing triage."""

    complexity: int = Field(ge=1, le=5)
    task_type: TaskType
    model: str
    reasoning: str
    triage_tokens: TokenUsage
    triage_cost: float
    triage_latency_ms: int
```

- [ ] **Step 16: Run tests to verify they pass**

Run: `pytest tests/test_models.py -v`

Expected: All PASS

- [ ] **Step 17: Write failing test for ExecutionResult model**

```python
# tests/test_models.py (append)


def test_execution_result_model():
    """ExecutionResult model with successful response."""
    from conductor.models import ExecutionResult

    result = ExecutionResult(
        content="def hello(): print('world')",
        model="anthropic/claude-3.5-haiku",
        input_tokens=50,
        output_tokens=20,
        cost=0.0001,
        latency_ms=412,
    )
    assert result.content == "def hello(): print('world')"
    assert result.error is None


def test_execution_result_with_error():
    """ExecutionResult can capture errors."""
    from conductor.models import ExecutionResult

    result = ExecutionResult(
        content="",
        model="anthropic/claude-3.5-haiku",
        input_tokens=50,
        output_tokens=0,
        cost=0.0,
        latency_ms=5000,
        error="Timeout after 5000ms",
    )
    assert result.error == "Timeout after 5000ms"
```

- [ ] **Step 18: Run tests to verify they fail**

Run: `pytest tests/test_models.py::test_execution_result_model tests/test_models.py::test_execution_result_with_error -v`

Expected: FAIL with "cannot import name 'ExecutionResult'"

- [ ] **Step 19: Add ExecutionResult model**

```python
# src/conductor/models.py (append)


class ExecutionResult(BaseModel):
    """Result of an LLM execution."""

    content: str
    model: str
    input_tokens: int
    output_tokens: int
    cost: float
    latency_ms: int
    error: str | None = None
```

- [ ] **Step 20: Run all model tests**

Run: `pytest tests/test_models.py -v`

Expected: All PASS

- [ ] **Step 21: Write failing test for RequestRecord model**

```python
# tests/test_models.py (append)

from datetime import datetime


def test_request_record_model():
    """RequestRecord captures full request lifecycle."""
    from conductor.models import RequestRecord, TaskType, TokenUsage

    record = RequestRecord(
        id="abc123",
        timestamp=datetime.fromisoformat("2026-05-13T10:30:00"),
        prompt_hash="sha256abc",
        prompt_text="Write hello world in Python",
        complexity=1,
        task_type=TaskType.CODING,
        routed_model="anthropic/claude-3.5-haiku",
        routing_reasoning="Simple coding task",
        triage_input_tokens=100,
        triage_output_tokens=40,
        triage_cost=0.0002,
        triage_latency_ms=200,
        executed_model="anthropic/claude-3.5-haiku",
        response_text="print('hello world')",
        input_tokens=50,
        output_tokens=10,
        cost=0.0001,
        latency_ms=300,
    )
    assert record.id == "abc123"
    assert record.auto_judge_score is None
    assert record.baseline_model is None
```

- [ ] **Step 22: Run test to verify it fails**

Run: `pytest tests/test_models.py::test_request_record_model -v`

Expected: FAIL with "cannot import name 'RequestRecord'"

- [ ] **Step 23: Add RequestRecord model**

```python
# src/conductor/models.py (append)

from datetime import datetime


class RequestRecord(BaseModel):
    """Complete record of a conductor request."""

    id: str
    timestamp: datetime
    prompt_hash: str
    prompt_text: str

    # Routing decision
    complexity: int = Field(ge=1, le=5)
    task_type: TaskType
    routed_model: str
    routing_reasoning: str

    # Triage overhead
    triage_input_tokens: int
    triage_output_tokens: int
    triage_cost: float
    triage_latency_ms: int

    # Primary execution
    executed_model: str
    response_text: str | None = None
    input_tokens: int
    output_tokens: int
    cost: float
    latency_ms: int
    error: str | None = None

    # Quality metrics
    auto_judge_score: float | None = Field(default=None, ge=1, le=10)
    human_rating: str | None = Field(default=None, pattern="^(up|down)$")

    # Baseline comparison (shadow samples only)
    baseline_model: str | None = None
    baseline_response: str | None = None
    baseline_input_tokens: int | None = None
    baseline_output_tokens: int | None = None
    baseline_cost: float | None = None
    baseline_latency_ms: int | None = None
    baseline_auto_judge_score: float | None = Field(default=None, ge=1, le=10)
```

- [ ] **Step 24: Run all model tests**

Run: `pytest tests/test_models.py -v`

Expected: All PASS

- [ ] **Step 25: Commit**

```bash
git add src/conductor/models.py tests/test_models.py
git commit -m "feat: add Pydantic models for routing, execution, and records"
```

---

## Task 3: Configuration System

**Files:**
- Create: `src/conductor/config.py`
- Create: `tests/test_config.py`
- Create: `conductor.yaml.example`

- [ ] **Step 1: Write failing test for ModelConfig**

```python
# tests/test_config.py
"""Tests for configuration loading."""

from conductor.config import ModelConfig, ModelTier


def test_model_config_basic():
    """ModelConfig with required fields."""
    config = ModelConfig(
        tier=ModelTier.MID,
        strengths=["coding", "reasoning"],
    )
    assert config.tier == ModelTier.MID
    assert "coding" in config.strengths
    assert config.cost_per_1k_input is None  # fetched from API later
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_config.py::test_model_config_basic -v`

Expected: FAIL with "ModuleNotFoundError: No module named 'conductor.config'"

- [ ] **Step 3: Create config.py with ModelConfig**

```python
# src/conductor/config.py
"""Configuration loading and validation."""

from pydantic import BaseModel

from conductor.models import ModelTier


class ModelConfig(BaseModel):
    """Configuration for a single model in the pool."""

    tier: ModelTier
    strengths: list[str]
    cost_per_1k_input: float | None = None
    cost_per_1k_output: float | None = None
    context_length: int | None = None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_config.py::test_model_config_basic -v`

Expected: PASS

- [ ] **Step 5: Write failing test for ConductorConfig**

```python
# tests/test_config.py (append)

from conductor.config import ConductorConfig


def test_conductor_config_minimal():
    """ConductorConfig with minimal required fields."""
    config = ConductorConfig(
        openrouter_api_key="sk-test-123",
        models={
            "anthropic/claude-3.5-haiku": ModelConfig(
                tier=ModelTier.FAST,
                strengths=["simple-qa"],
            ),
        },
    )
    assert config.openrouter_api_key == "sk-test-123"
    assert config.router_model == "anthropic/claude-sonnet-4"  # default
    assert config.shadow_sample_rate == 0.1  # default


def test_conductor_config_custom_values():
    """ConductorConfig with custom values."""
    config = ConductorConfig(
        openrouter_api_key="sk-test-123",
        router_model="openai/gpt-4o",
        baseline_model="anthropic/claude-opus-4",
        shadow_sample_rate=0.2,
        models={
            "anthropic/claude-3.5-haiku": ModelConfig(
                tier=ModelTier.FAST,
                strengths=["simple-qa"],
            ),
        },
    )
    assert config.router_model == "openai/gpt-4o"
    assert config.shadow_sample_rate == 0.2
```

- [ ] **Step 6: Run tests to verify they fail**

Run: `pytest tests/test_config.py -v`

Expected: FAIL with "cannot import name 'ConductorConfig'"

- [ ] **Step 7: Add ConductorConfig model**

```python
# src/conductor/config.py (append)

from pathlib import Path
from pydantic import Field


class LearningConfig(BaseModel):
    """Configuration for adaptive learning."""

    enabled: bool = True
    min_sample_size: int = 20
    quality_block_threshold: float = 5.0
    quality_prefer_threshold: float = 8.5
    rule_expiry_days: int = 30


class ConductorConfig(BaseModel):
    """Root configuration for Conductor."""

    openrouter_api_key: str
    router_model: str = "anthropic/claude-sonnet-4"
    baseline_model: str = "anthropic/claude-opus-4"
    shadow_sample_rate: float = Field(default=0.1, ge=0.0, le=1.0)
    models: dict[str, ModelConfig]
    timeout_seconds: int = 120
    max_retries: int = 3
    database_path: Path = Path("~/.config/conductor/conductor.db")
    learning: LearningConfig = Field(default_factory=LearningConfig)
```

- [ ] **Step 8: Run tests to verify they pass**

Run: `pytest tests/test_config.py -v`

Expected: All PASS

- [ ] **Step 9: Write failing test for env var resolution**

```python
# tests/test_config.py (append)

import os


def test_resolve_env_vars():
    """resolve_env_vars substitutes ${VAR} patterns."""
    from conductor.config import resolve_env_vars

    os.environ["TEST_KEY"] = "secret123"
    result = resolve_env_vars("${TEST_KEY}")
    assert result == "secret123"


def test_resolve_env_vars_missing():
    """resolve_env_vars raises on missing env var."""
    import pytest
    from conductor.config import resolve_env_vars

    with pytest.raises(ValueError, match="Environment variable"):
        resolve_env_vars("${NONEXISTENT_VAR_12345}")


def test_resolve_env_vars_literal():
    """resolve_env_vars returns literal strings unchanged."""
    from conductor.config import resolve_env_vars

    result = resolve_env_vars("literal-value")
    assert result == "literal-value"
```

- [ ] **Step 10: Run tests to verify they fail**

Run: `pytest tests/test_config.py::test_resolve_env_vars tests/test_config.py::test_resolve_env_vars_missing tests/test_config.py::test_resolve_env_vars_literal -v`

Expected: FAIL with "cannot import name 'resolve_env_vars'"

- [ ] **Step 11: Add resolve_env_vars function**

```python
# src/conductor/config.py (add at top after imports)

import os
import re


def resolve_env_vars(value: str) -> str:
    """Resolve ${VAR} patterns to environment variable values."""
    pattern = r"\$\{([^}]+)\}"
    match = re.match(pattern, value)
    if match:
        var_name = match.group(1)
        env_value = os.environ.get(var_name)
        if env_value is None:
            raise ValueError(f"Environment variable {var_name} not set")
        return env_value
    return value
```

- [ ] **Step 12: Run tests to verify they pass**

Run: `pytest tests/test_config.py -v`

Expected: All PASS

- [ ] **Step 13: Write failing test for load_config**

```python
# tests/test_config.py (append)

from pathlib import Path


def test_load_config_from_file(temp_config_path: Path):
    """load_config reads YAML and resolves env vars."""
    from conductor.config import load_config

    config_content = """
openrouter_api_key: ${OPENROUTER_API_KEY}
router_model: anthropic/claude-sonnet-4
models:
  anthropic/claude-3.5-haiku:
    tier: fast
    strengths:
      - simple-qa
      - fast-tasks
"""
    temp_config_path.write_text(config_content)

    config = load_config(temp_config_path)
    assert config.openrouter_api_key == "test-api-key-12345"  # from fixture
    assert "anthropic/claude-3.5-haiku" in config.models


def test_load_config_file_not_found():
    """load_config raises on missing file."""
    import pytest
    from conductor.config import load_config

    with pytest.raises(FileNotFoundError):
        load_config(Path("/nonexistent/config.yaml"))
```

- [ ] **Step 14: Run tests to verify they fail**

Run: `pytest tests/test_config.py::test_load_config_from_file tests/test_config.py::test_load_config_file_not_found -v`

Expected: FAIL with "cannot import name 'load_config'"

- [ ] **Step 15: Add load_config function**

```python
# src/conductor/config.py (append)

import yaml


def load_config(path: Path) -> ConductorConfig:
    """Load configuration from YAML file."""
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path) as f:
        raw_config = yaml.safe_load(f)

    # Resolve environment variables in API key
    if "openrouter_api_key" in raw_config:
        raw_config["openrouter_api_key"] = resolve_env_vars(
            raw_config["openrouter_api_key"]
        )

    return ConductorConfig(**raw_config)
```

- [ ] **Step 16: Run all config tests**

Run: `pytest tests/test_config.py -v`

Expected: All PASS

- [ ] **Step 17: Create conductor.yaml.example**

```yaml
# conductor.yaml.example
# Copy to ~/.config/conductor/conductor.yaml or ./conductor.yaml

# OpenRouter API key (get one at https://openrouter.ai/keys)
# Use environment variable reference or literal value
openrouter_api_key: ${OPENROUTER_API_KEY}

# Model used for routing triage (fast, smart enough)
router_model: anthropic/claude-sonnet-4

# Baseline model for research comparison (the "always-best" option)
baseline_model: anthropic/claude-opus-4

# Percentage of requests to also run against baseline (0.0 to 1.0)
shadow_sample_rate: 0.1

# Model pool with tiers and strengths
# Costs are fetched from OpenRouter API automatically
models:
  anthropic/claude-opus-4:
    tier: best
    strengths:
      - reasoning
      - coding
      - creative

  anthropic/claude-sonnet-4:
    tier: mid
    strengths:
      - coding
      - reasoning
      - general

  anthropic/claude-3.5-haiku:
    tier: fast
    strengths:
      - simple-qa
      - fast-tasks

  openai/gpt-4o:
    tier: mid
    strengths:
      - general
      - creative

# Execution settings
timeout_seconds: 120
max_retries: 3

# Learning settings
learning:
  enabled: true
  min_sample_size: 20
  quality_block_threshold: 5.0
  quality_prefer_threshold: 8.5
  rule_expiry_days: 30
```

- [ ] **Step 18: Run all tests**

Run: `pytest tests/test_config.py -v`

Expected: All PASS

- [ ] **Step 19: Commit**

```bash
git add src/conductor/config.py tests/test_config.py conductor.yaml.example
git commit -m "feat: add configuration system with env var resolution"
```

---

## Task 4: SQLite Tracker

**Files:**
- Create: `src/conductor/tracker.py`
- Create: `tests/test_tracker.py`

- [ ] **Step 1: Write failing test for schema initialization**

```python
# tests/test_tracker.py
"""Tests for SQLite tracker."""

import sqlite3
from pathlib import Path


def test_tracker_creates_schema(temp_db_path: Path):
    """Tracker initializes database schema."""
    from conductor.tracker import Tracker

    tracker = Tracker(temp_db_path)
    tracker.initialize()

    # Verify tables exist
    conn = sqlite3.connect(temp_db_path)
    cursor = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    )
    tables = [row[0] for row in cursor.fetchall()]
    conn.close()

    assert "requests" in tables
    assert "model_stats" in tables
    assert "routing_rules" in tables
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_tracker.py::test_tracker_creates_schema -v`

Expected: FAIL with "ModuleNotFoundError: No module named 'conductor.tracker'"

- [ ] **Step 3: Create tracker.py with schema initialization**

```python
# src/conductor/tracker.py
"""SQLite persistence for request tracking."""

import sqlite3
from pathlib import Path


SCHEMA = """
CREATE TABLE IF NOT EXISTS requests (
    id TEXT PRIMARY KEY,
    timestamp TEXT NOT NULL,
    prompt_hash TEXT NOT NULL,
    prompt_text TEXT NOT NULL,
    complexity INTEGER NOT NULL CHECK (complexity BETWEEN 1 AND 5),
    task_type TEXT NOT NULL,
    routed_model TEXT NOT NULL,
    routing_reasoning TEXT,
    triage_input_tokens INTEGER NOT NULL,
    triage_output_tokens INTEGER NOT NULL,
    triage_cost REAL NOT NULL,
    triage_latency_ms INTEGER NOT NULL,
    executed_model TEXT NOT NULL,
    response_text TEXT,
    input_tokens INTEGER NOT NULL,
    output_tokens INTEGER NOT NULL,
    cost REAL NOT NULL,
    latency_ms INTEGER NOT NULL,
    error TEXT,
    auto_judge_score REAL CHECK (auto_judge_score BETWEEN 1 AND 10),
    human_rating TEXT CHECK (human_rating IN ('up', 'down')),
    baseline_model TEXT,
    baseline_response TEXT,
    baseline_input_tokens INTEGER,
    baseline_output_tokens INTEGER,
    baseline_cost REAL,
    baseline_latency_ms INTEGER,
    baseline_auto_judge_score REAL CHECK (baseline_auto_judge_score BETWEEN 1 AND 10)
);

CREATE INDEX IF NOT EXISTS idx_requests_timestamp ON requests(timestamp);
CREATE INDEX IF NOT EXISTS idx_requests_task_type ON requests(task_type);
CREATE INDEX IF NOT EXISTS idx_requests_model ON requests(executed_model);

CREATE TABLE IF NOT EXISTS model_stats (
    model TEXT NOT NULL,
    task_type TEXT NOT NULL,
    total_requests INTEGER NOT NULL DEFAULT 0,
    total_quality_sum REAL NOT NULL DEFAULT 0,
    avg_latency_ms REAL,
    avg_cost REAL,
    last_updated TEXT NOT NULL,
    PRIMARY KEY (model, task_type)
);

CREATE TABLE IF NOT EXISTS routing_rules (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    rule_type TEXT NOT NULL CHECK (rule_type IN ('block', 'prefer', 'min_tier')),
    condition_json TEXT NOT NULL,
    action_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    expires_at TEXT,
    reason TEXT NOT NULL,
    is_active INTEGER NOT NULL DEFAULT 1
);

CREATE INDEX IF NOT EXISTS idx_rules_active ON routing_rules(is_active);
"""


class Tracker:
    """SQLite-based request tracker."""

    def __init__(self, db_path: Path):
        self.db_path = db_path

    def initialize(self) -> None:
        """Create database schema if not exists."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        conn.executescript(SCHEMA)
        conn.commit()
        conn.close()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_tracker.py::test_tracker_creates_schema -v`

Expected: PASS

- [ ] **Step 5: Write failing test for log_request**

```python
# tests/test_tracker.py (append)

from datetime import datetime
from conductor.models import TaskType


def test_tracker_log_request(temp_db_path: Path):
    """Tracker logs request records."""
    from conductor.tracker import Tracker
    from conductor.models import RequestRecord

    tracker = Tracker(temp_db_path)
    tracker.initialize()

    record = RequestRecord(
        id="test-123",
        timestamp=datetime.fromisoformat("2026-05-13T10:30:00"),
        prompt_hash="abc123",
        prompt_text="Write hello world",
        complexity=1,
        task_type=TaskType.CODING,
        routed_model="anthropic/claude-3.5-haiku",
        routing_reasoning="Simple task",
        triage_input_tokens=100,
        triage_output_tokens=40,
        triage_cost=0.0002,
        triage_latency_ms=200,
        executed_model="anthropic/claude-3.5-haiku",
        response_text="print('hello world')",
        input_tokens=50,
        output_tokens=10,
        cost=0.0001,
        latency_ms=300,
    )

    tracker.log_request(record)

    # Verify it was stored
    retrieved = tracker.get_request("test-123")
    assert retrieved is not None
    assert retrieved.prompt_text == "Write hello world"
    assert retrieved.complexity == 1
```

- [ ] **Step 6: Run test to verify it fails**

Run: `pytest tests/test_tracker.py::test_tracker_log_request -v`

Expected: FAIL with "AttributeError: 'Tracker' object has no attribute 'log_request'"

- [ ] **Step 7: Add log_request and get_request methods**

```python
# src/conductor/tracker.py (add import at top)
from conductor.models import RequestRecord

# src/conductor/tracker.py (add to Tracker class)

    def _get_connection(self) -> sqlite3.Connection:
        """Get database connection."""
        return sqlite3.connect(self.db_path)

    def log_request(self, record: RequestRecord) -> None:
        """Log a request record to the database."""
        conn = self._get_connection()
        conn.execute(
            """
            INSERT INTO requests (
                id, timestamp, prompt_hash, prompt_text,
                complexity, task_type, routed_model, routing_reasoning,
                triage_input_tokens, triage_output_tokens, triage_cost, triage_latency_ms,
                executed_model, response_text, input_tokens, output_tokens,
                cost, latency_ms, error,
                auto_judge_score, human_rating,
                baseline_model, baseline_response, baseline_input_tokens,
                baseline_output_tokens, baseline_cost, baseline_latency_ms,
                baseline_auto_judge_score
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.id,
                record.timestamp.isoformat(),
                record.prompt_hash,
                record.prompt_text,
                record.complexity,
                record.task_type.value,
                record.routed_model,
                record.routing_reasoning,
                record.triage_input_tokens,
                record.triage_output_tokens,
                record.triage_cost,
                record.triage_latency_ms,
                record.executed_model,
                record.response_text,
                record.input_tokens,
                record.output_tokens,
                record.cost,
                record.latency_ms,
                record.error,
                record.auto_judge_score,
                record.human_rating,
                record.baseline_model,
                record.baseline_response,
                record.baseline_input_tokens,
                record.baseline_output_tokens,
                record.baseline_cost,
                record.baseline_latency_ms,
                record.baseline_auto_judge_score,
            ),
        )
        conn.commit()
        conn.close()

    def get_request(self, request_id: str) -> RequestRecord | None:
        """Retrieve a request record by ID."""
        conn = self._get_connection()
        conn.row_factory = sqlite3.Row
        cursor = conn.execute("SELECT * FROM requests WHERE id = ?", (request_id,))
        row = cursor.fetchone()
        conn.close()

        if row is None:
            return None

        return RequestRecord(
            id=row["id"],
            timestamp=datetime.fromisoformat(row["timestamp"]),
            prompt_hash=row["prompt_hash"],
            prompt_text=row["prompt_text"],
            complexity=row["complexity"],
            task_type=TaskType(row["task_type"]),
            routed_model=row["routed_model"],
            routing_reasoning=row["routing_reasoning"],
            triage_input_tokens=row["triage_input_tokens"],
            triage_output_tokens=row["triage_output_tokens"],
            triage_cost=row["triage_cost"],
            triage_latency_ms=row["triage_latency_ms"],
            executed_model=row["executed_model"],
            response_text=row["response_text"],
            input_tokens=row["input_tokens"],
            output_tokens=row["output_tokens"],
            cost=row["cost"],
            latency_ms=row["latency_ms"],
            error=row["error"],
            auto_judge_score=row["auto_judge_score"],
            human_rating=row["human_rating"],
            baseline_model=row["baseline_model"],
            baseline_response=row["baseline_response"],
            baseline_input_tokens=row["baseline_input_tokens"],
            baseline_output_tokens=row["baseline_output_tokens"],
            baseline_cost=row["baseline_cost"],
            baseline_latency_ms=row["baseline_latency_ms"],
            baseline_auto_judge_score=row["baseline_auto_judge_score"],
        )
```

- [ ] **Step 8: Add missing import to tracker.py**

```python
# src/conductor/tracker.py (add at top)
from datetime import datetime
from conductor.models import RequestRecord, TaskType
```

- [ ] **Step 9: Run test to verify it passes**

Run: `pytest tests/test_tracker.py::test_tracker_log_request -v`

Expected: PASS

- [ ] **Step 10: Write failing test for update_rating**

```python
# tests/test_tracker.py (append)


def test_tracker_update_rating(temp_db_path: Path):
    """Tracker updates human rating."""
    from conductor.tracker import Tracker
    from conductor.models import RequestRecord

    tracker = Tracker(temp_db_path)
    tracker.initialize()

    record = RequestRecord(
        id="rate-test",
        timestamp=datetime.fromisoformat("2026-05-13T10:30:00"),
        prompt_hash="abc123",
        prompt_text="Test prompt",
        complexity=1,
        task_type=TaskType.GENERAL,
        routed_model="anthropic/claude-3.5-haiku",
        routing_reasoning="Test",
        triage_input_tokens=10,
        triage_output_tokens=10,
        triage_cost=0.0001,
        triage_latency_ms=100,
        executed_model="anthropic/claude-3.5-haiku",
        response_text="Response",
        input_tokens=10,
        output_tokens=10,
        cost=0.0001,
        latency_ms=100,
    )
    tracker.log_request(record)

    tracker.update_rating("rate-test", "up")

    retrieved = tracker.get_request("rate-test")
    assert retrieved.human_rating == "up"
```

- [ ] **Step 11: Run test to verify it fails**

Run: `pytest tests/test_tracker.py::test_tracker_update_rating -v`

Expected: FAIL with "AttributeError: 'Tracker' object has no attribute 'update_rating'"

- [ ] **Step 12: Add update_rating method**

```python
# src/conductor/tracker.py (add to Tracker class)

    def update_rating(self, request_id: str, rating: str) -> None:
        """Update the human rating for a request."""
        if rating not in ("up", "down"):
            raise ValueError("Rating must be 'up' or 'down'")
        conn = self._get_connection()
        conn.execute(
            "UPDATE requests SET human_rating = ? WHERE id = ?",
            (rating, request_id),
        )
        conn.commit()
        conn.close()
```

- [ ] **Step 13: Run test to verify it passes**

Run: `pytest tests/test_tracker.py::test_tracker_update_rating -v`

Expected: PASS

- [ ] **Step 14: Write failing test for update_auto_judge_score**

```python
# tests/test_tracker.py (append)


def test_tracker_update_auto_judge_score(temp_db_path: Path):
    """Tracker updates auto judge score."""
    from conductor.tracker import Tracker
    from conductor.models import RequestRecord

    tracker = Tracker(temp_db_path)
    tracker.initialize()

    record = RequestRecord(
        id="judge-test",
        timestamp=datetime.fromisoformat("2026-05-13T10:30:00"),
        prompt_hash="abc123",
        prompt_text="Test prompt",
        complexity=1,
        task_type=TaskType.GENERAL,
        routed_model="anthropic/claude-3.5-haiku",
        routing_reasoning="Test",
        triage_input_tokens=10,
        triage_output_tokens=10,
        triage_cost=0.0001,
        triage_latency_ms=100,
        executed_model="anthropic/claude-3.5-haiku",
        response_text="Response",
        input_tokens=10,
        output_tokens=10,
        cost=0.0001,
        latency_ms=100,
    )
    tracker.log_request(record)

    tracker.update_auto_judge_score("judge-test", 8.5)

    retrieved = tracker.get_request("judge-test")
    assert retrieved.auto_judge_score == 8.5
```

- [ ] **Step 15: Run test to verify it fails**

Run: `pytest tests/test_tracker.py::test_tracker_update_auto_judge_score -v`

Expected: FAIL with "AttributeError: 'Tracker' object has no attribute 'update_auto_judge_score'"

- [ ] **Step 16: Add update_auto_judge_score method**

```python
# src/conductor/tracker.py (add to Tracker class)

    def update_auto_judge_score(
        self, request_id: str, score: float, is_baseline: bool = False
    ) -> None:
        """Update the auto judge score for a request."""
        column = "baseline_auto_judge_score" if is_baseline else "auto_judge_score"
        conn = self._get_connection()
        conn.execute(
            f"UPDATE requests SET {column} = ? WHERE id = ?",
            (score, request_id),
        )
        conn.commit()
        conn.close()
```

- [ ] **Step 17: Run test to verify it passes**

Run: `pytest tests/test_tracker.py::test_tracker_update_auto_judge_score -v`

Expected: PASS

- [ ] **Step 18: Write failing test for get_stats**

```python
# tests/test_tracker.py (append)


def test_tracker_get_stats(temp_db_path: Path):
    """Tracker returns usage statistics."""
    from conductor.tracker import Tracker
    from conductor.models import RequestRecord

    tracker = Tracker(temp_db_path)
    tracker.initialize()

    # Add some test records
    for i in range(3):
        record = RequestRecord(
            id=f"stats-test-{i}",
            timestamp=datetime.fromisoformat("2026-05-13T10:30:00"),
            prompt_hash=f"hash{i}",
            prompt_text=f"Prompt {i}",
            complexity=2,
            task_type=TaskType.CODING,
            routed_model="anthropic/claude-3.5-haiku",
            routing_reasoning="Test",
            triage_input_tokens=100,
            triage_output_tokens=40,
            triage_cost=0.0002,
            triage_latency_ms=200,
            executed_model="anthropic/claude-3.5-haiku",
            response_text="Response",
            input_tokens=50,
            output_tokens=20,
            cost=0.001,
            latency_ms=300,
            auto_judge_score=7.0 + i,
        )
        tracker.log_request(record)

    stats = tracker.get_stats()
    assert stats["total_requests"] == 3
    assert stats["total_cost"] > 0
    assert "anthropic/claude-3.5-haiku" in stats["by_model"]
```

- [ ] **Step 19: Run test to verify it fails**

Run: `pytest tests/test_tracker.py::test_tracker_get_stats -v`

Expected: FAIL with "AttributeError: 'Tracker' object has no attribute 'get_stats'"

- [ ] **Step 20: Add get_stats method**

```python
# src/conductor/tracker.py (add to Tracker class)

    def get_stats(self, days: int = 30) -> dict:
        """Get usage statistics for the last N days."""
        conn = self._get_connection()
        conn.row_factory = sqlite3.Row

        # Total requests and costs
        cursor = conn.execute(
            """
            SELECT
                COUNT(*) as total_requests,
                SUM(cost) as total_execution_cost,
                SUM(triage_cost) as total_triage_cost,
                AVG(latency_ms) as avg_latency
            FROM requests
            WHERE timestamp >= datetime('now', ?)
            """,
            (f"-{days} days",),
        )
        totals = cursor.fetchone()

        # By model breakdown
        cursor = conn.execute(
            """
            SELECT
                executed_model,
                COUNT(*) as request_count,
                SUM(cost) as total_cost,
                AVG(auto_judge_score) as avg_quality
            FROM requests
            WHERE timestamp >= datetime('now', ?)
            GROUP BY executed_model
            """,
            (f"-{days} days",),
        )
        by_model = {
            row["executed_model"]: {
                "request_count": row["request_count"],
                "total_cost": row["total_cost"] or 0,
                "avg_quality": row["avg_quality"],
            }
            for row in cursor.fetchall()
        }

        conn.close()

        return {
            "total_requests": totals["total_requests"],
            "total_cost": (totals["total_execution_cost"] or 0)
            + (totals["total_triage_cost"] or 0),
            "total_execution_cost": totals["total_execution_cost"] or 0,
            "total_triage_cost": totals["total_triage_cost"] or 0,
            "avg_latency_ms": totals["avg_latency"],
            "by_model": by_model,
        }
```

- [ ] **Step 21: Run test to verify it passes**

Run: `pytest tests/test_tracker.py::test_tracker_get_stats -v`

Expected: PASS

- [ ] **Step 22: Run all tracker tests**

Run: `pytest tests/test_tracker.py -v`

Expected: All PASS

- [ ] **Step 23: Commit**

```bash
git add src/conductor/tracker.py tests/test_tracker.py
git commit -m "feat: add SQLite tracker for request persistence"
```

---

## Task 5: OpenRouter Executor

**Files:**
- Create: `src/conductor/executor.py`
- Create: `tests/test_executor.py`

- [ ] **Step 1: Write failing test for basic execution**

```python
# tests/test_executor.py
"""Tests for OpenRouter executor."""

import pytest
import httpx
import respx
from conductor.executor import Executor


@pytest.fixture
def executor():
    """Create executor with test API key."""
    return Executor(api_key="test-key-123")


@respx.mock
@pytest.mark.asyncio
async def test_executor_execute_success(executor: Executor):
    """Executor makes successful API call."""
    respx.post("https://openrouter.ai/api/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "id": "gen-123",
                "model": "anthropic/claude-3.5-haiku",
                "choices": [{"message": {"content": "Hello, world!"}}],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "total_tokens": 15,
                },
            },
        )
    )

    result = await executor.execute(
        model="anthropic/claude-3.5-haiku",
        prompt="Say hello",
    )

    assert result.content == "Hello, world!"
    assert result.model == "anthropic/claude-3.5-haiku"
    assert result.input_tokens == 10
    assert result.output_tokens == 5
    assert result.error is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_executor.py::test_executor_execute_success -v`

Expected: FAIL with "ModuleNotFoundError: No module named 'conductor.executor'"

- [ ] **Step 3: Create executor.py with basic execution**

```python
# src/conductor/executor.py
"""OpenRouter API executor."""

import time
import httpx

from conductor.models import ExecutionResult


class Executor:
    """Async executor for OpenRouter API calls."""

    BASE_URL = "https://openrouter.ai/api/v1"

    def __init__(
        self,
        api_key: str,
        timeout: int = 120,
        max_retries: int = 3,
    ):
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries

    async def execute(
        self,
        model: str,
        prompt: str,
        max_tokens: int = 4096,
        temperature: float = 0.7,
    ) -> ExecutionResult:
        """Execute a prompt against a model."""
        start_time = time.perf_counter()

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                f"{self.BASE_URL}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://github.com/conductor",
                    "X-Title": "Conductor",
                },
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                },
            )

        latency_ms = int((time.perf_counter() - start_time) * 1000)

        if response.status_code != 200:
            return ExecutionResult(
                content="",
                model=model,
                input_tokens=0,
                output_tokens=0,
                cost=0.0,
                latency_ms=latency_ms,
                error=f"API error: {response.status_code} - {response.text}",
            )

        data = response.json()
        usage = data.get("usage", {})

        return ExecutionResult(
            content=data["choices"][0]["message"]["content"],
            model=data.get("model", model),
            input_tokens=usage.get("prompt_tokens", 0),
            output_tokens=usage.get("completion_tokens", 0),
            cost=self._calculate_cost(model, usage),
            latency_ms=latency_ms,
        )

    def _calculate_cost(self, model: str, usage: dict) -> float:
        """Calculate cost based on usage. Placeholder until we fetch prices."""
        # TODO: Fetch actual prices from OpenRouter /models endpoint
        # For now, use approximate values
        input_tokens = usage.get("prompt_tokens", 0)
        output_tokens = usage.get("completion_tokens", 0)
        # Rough estimate: $0.001 per 1K tokens average
        return (input_tokens + output_tokens) * 0.000001
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_executor.py::test_executor_execute_success -v`

Expected: PASS

- [ ] **Step 5: Write failing test for API error handling**

```python
# tests/test_executor.py (append)


@respx.mock
@pytest.mark.asyncio
async def test_executor_handles_api_error(executor: Executor):
    """Executor captures API errors in result."""
    respx.post("https://openrouter.ai/api/v1/chat/completions").mock(
        return_value=httpx.Response(429, text="Rate limited")
    )

    result = await executor.execute(
        model="anthropic/claude-3.5-haiku",
        prompt="Test",
    )

    assert result.error is not None
    assert "429" in result.error
    assert result.content == ""
```

- [ ] **Step 6: Run test to verify it passes**

Run: `pytest tests/test_executor.py::test_executor_handles_api_error -v`

Expected: PASS (error handling already implemented)

- [ ] **Step 7: Write failing test for retry logic**

```python
# tests/test_executor.py (append)


@respx.mock
@pytest.mark.asyncio
async def test_executor_retries_on_failure(executor: Executor):
    """Executor retries failed requests."""
    route = respx.post("https://openrouter.ai/api/v1/chat/completions")
    route.side_effect = [
        httpx.Response(500, text="Server error"),
        httpx.Response(500, text="Server error"),
        httpx.Response(
            200,
            json={
                "id": "gen-123",
                "model": "anthropic/claude-3.5-haiku",
                "choices": [{"message": {"content": "Success after retry"}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            },
        ),
    ]

    result = await executor.execute(
        model="anthropic/claude-3.5-haiku",
        prompt="Test retry",
    )

    assert result.content == "Success after retry"
    assert result.error is None
    assert route.call_count == 3
```

- [ ] **Step 8: Run test to verify it fails**

Run: `pytest tests/test_executor.py::test_executor_retries_on_failure -v`

Expected: FAIL (no retry logic yet)

- [ ] **Step 9: Add retry logic to executor**

```python
# src/conductor/executor.py (replace execute method)

    async def execute(
        self,
        model: str,
        prompt: str,
        max_tokens: int = 4096,
        temperature: float = 0.7,
    ) -> ExecutionResult:
        """Execute a prompt against a model with retries."""
        start_time = time.perf_counter()
        last_error = None

        for attempt in range(self.max_retries):
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    response = await client.post(
                        f"{self.BASE_URL}/chat/completions",
                        headers={
                            "Authorization": f"Bearer {self.api_key}",
                            "Content-Type": "application/json",
                            "HTTP-Referer": "https://github.com/conductor",
                            "X-Title": "Conductor",
                        },
                        json={
                            "model": model,
                            "messages": [{"role": "user", "content": prompt}],
                            "max_tokens": max_tokens,
                            "temperature": temperature,
                        },
                    )

                if response.status_code == 200:
                    latency_ms = int((time.perf_counter() - start_time) * 1000)
                    data = response.json()
                    usage = data.get("usage", {})

                    return ExecutionResult(
                        content=data["choices"][0]["message"]["content"],
                        model=data.get("model", model),
                        input_tokens=usage.get("prompt_tokens", 0),
                        output_tokens=usage.get("completion_tokens", 0),
                        cost=self._calculate_cost(model, usage),
                        latency_ms=latency_ms,
                    )

                last_error = f"API error: {response.status_code} - {response.text}"

                # Don't retry on 4xx client errors (except 429)
                if 400 <= response.status_code < 500 and response.status_code != 429:
                    break

            except httpx.TimeoutException:
                last_error = f"Timeout after {self.timeout}s"
            except httpx.RequestError as e:
                last_error = f"Request error: {e}"

            # Exponential backoff
            if attempt < self.max_retries - 1:
                import asyncio
                await asyncio.sleep(2 ** attempt)

        latency_ms = int((time.perf_counter() - start_time) * 1000)
        return ExecutionResult(
            content="",
            model=model,
            input_tokens=0,
            output_tokens=0,
            cost=0.0,
            latency_ms=latency_ms,
            error=last_error,
        )
```

- [ ] **Step 10: Run test to verify it passes**

Run: `pytest tests/test_executor.py::test_executor_retries_on_failure -v`

Expected: PASS

- [ ] **Step 11: Write failing test for shadow execution**

```python
# tests/test_executor.py (append)

import random


@respx.mock
@pytest.mark.asyncio
async def test_executor_shadow_execution(executor: Executor, monkeypatch):
    """Executor runs shadow request when sampling triggers."""
    # Force shadow sampling to trigger
    monkeypatch.setattr(random, "random", lambda: 0.05)

    respx.post("https://openrouter.ai/api/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "id": "gen-123",
                "model": "test-model",
                "choices": [{"message": {"content": "Response"}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            },
        )
    )

    primary, shadow = await executor.execute_with_shadow(
        model="anthropic/claude-3.5-haiku",
        prompt="Test",
        shadow_model="anthropic/claude-opus-4",
        shadow_rate=0.1,
    )

    assert primary.content == "Response"
    assert shadow is not None
    assert shadow.content == "Response"


@respx.mock
@pytest.mark.asyncio
async def test_executor_shadow_not_triggered(executor: Executor, monkeypatch):
    """Executor skips shadow when sampling doesn't trigger."""
    # Force shadow sampling to NOT trigger
    monkeypatch.setattr(random, "random", lambda: 0.5)

    respx.post("https://openrouter.ai/api/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "id": "gen-123",
                "model": "test-model",
                "choices": [{"message": {"content": "Response"}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            },
        )
    )

    primary, shadow = await executor.execute_with_shadow(
        model="anthropic/claude-3.5-haiku",
        prompt="Test",
        shadow_model="anthropic/claude-opus-4",
        shadow_rate=0.1,
    )

    assert primary.content == "Response"
    assert shadow is None
```

- [ ] **Step 12: Run tests to verify they fail**

Run: `pytest tests/test_executor.py::test_executor_shadow_execution tests/test_executor.py::test_executor_shadow_not_triggered -v`

Expected: FAIL with "AttributeError: 'Executor' object has no attribute 'execute_with_shadow'"

- [ ] **Step 13: Add execute_with_shadow method**

```python
# src/conductor/executor.py (add import at top)
import asyncio
import random

# src/conductor/executor.py (add method to Executor class)

    async def execute_with_shadow(
        self,
        model: str,
        prompt: str,
        shadow_model: str,
        shadow_rate: float = 0.1,
        max_tokens: int = 4096,
    ) -> tuple[ExecutionResult, ExecutionResult | None]:
        """Execute with optional shadow request to baseline model."""
        should_shadow = random.random() < shadow_rate

        if should_shadow:
            primary_task = self.execute(model, prompt, max_tokens)
            shadow_task = self.execute(shadow_model, prompt, max_tokens)
            primary, shadow = await asyncio.gather(primary_task, shadow_task)
            return primary, shadow
        else:
            primary = await self.execute(model, prompt, max_tokens)
            return primary, None
```

- [ ] **Step 14: Run tests to verify they pass**

Run: `pytest tests/test_executor.py -v`

Expected: All PASS

- [ ] **Step 15: Commit**

```bash
git add src/conductor/executor.py tests/test_executor.py
git commit -m "feat: add OpenRouter executor with retries and shadow sampling"
```

---

## Task 6: Router with Triage

**Files:**
- Create: `src/conductor/router.py`
- Create: `tests/test_router.py`

- [ ] **Step 1: Write failing test for triage prompt building**

```python
# tests/test_router.py
"""Tests for routing logic."""

import pytest
from conductor.router import Router
from conductor.config import ModelConfig, ConductorConfig
from conductor.models import ModelTier


@pytest.fixture
def config() -> ConductorConfig:
    """Create test configuration."""
    return ConductorConfig(
        openrouter_api_key="test-key",
        models={
            "anthropic/claude-opus-4": ModelConfig(
                tier=ModelTier.BEST,
                strengths=["reasoning", "coding"],
            ),
            "anthropic/claude-3.5-haiku": ModelConfig(
                tier=ModelTier.FAST,
                strengths=["simple-qa"],
            ),
        },
    )


def test_router_builds_triage_prompt(config: ConductorConfig):
    """Router builds triage prompt with model pool."""
    router = Router(config)
    prompt = router._build_triage_prompt("Write hello world in Python")

    assert "Write hello world in Python" in prompt
    assert "anthropic/claude-opus-4" in prompt
    assert "anthropic/claude-3.5-haiku" in prompt
    assert "tier: best" in prompt.lower() or "best" in prompt.lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_router.py::test_router_builds_triage_prompt -v`

Expected: FAIL with "ModuleNotFoundError: No module named 'conductor.router'"

- [ ] **Step 3: Create router.py with triage prompt builder**

```python
# src/conductor/router.py
"""Routing logic with LLM triage."""

from conductor.config import ConductorConfig


TRIAGE_TEMPLATE = """You are a routing assistant. Analyze the user's prompt and select the optimal model.

## Complexity Scale
1: Direct factual question, single-step task
2: Simple task with clear instructions
3: Multi-step reasoning, moderate ambiguity
4: Complex analysis, nuanced judgment required
5: Expert-level, novel problem-solving, extended output

## Task Types
- coding: Writing, debugging, or reviewing code
- reasoning: Logic, mathematics, analysis, planning
- creative: Writing, brainstorming, open-ended generation
- simple_qa: Factual lookup, definitions, brief answers
- general: Does not fit above categories

## Available Models
{model_pool}

{constraints}

## User Prompt
\"\"\"
{user_prompt}
\"\"\"

Return ONLY this JSON:
{{"complexity": <1-5>, "task_type": "<type>", "model": "<model_id>", "reasoning": "<one sentence>"}}"""


class Router:
    """Routes prompts to optimal models via LLM triage."""

    def __init__(self, config: ConductorConfig):
        self.config = config

    def _build_model_pool(self) -> str:
        """Build model pool description for triage prompt."""
        lines = []
        for model_id, model_config in self.config.models.items():
            strengths = ", ".join(model_config.strengths)
            lines.append(f"- {model_id} (tier: {model_config.tier.value}, strengths: {strengths})")
        return "\n".join(lines)

    def _build_triage_prompt(self, user_prompt: str, constraints: str = "") -> str:
        """Build the full triage prompt."""
        model_pool = self._build_model_pool()
        constraints_section = f"## Routing Constraints\n{constraints}" if constraints else ""
        return TRIAGE_TEMPLATE.format(
            model_pool=model_pool,
            constraints=constraints_section,
            user_prompt=user_prompt,
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_router.py::test_router_builds_triage_prompt -v`

Expected: PASS

- [ ] **Step 5: Write failing test for route method**

```python
# tests/test_router.py (append)

import httpx
import respx
from unittest.mock import AsyncMock, patch


@respx.mock
@pytest.mark.asyncio
async def test_router_route_returns_decision(config: ConductorConfig):
    """Router returns routing decision from triage."""
    respx.post("https://openrouter.ai/api/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "id": "gen-123",
                "model": "anthropic/claude-sonnet-4",
                "choices": [
                    {
                        "message": {
                            "content": '{"complexity": 2, "task_type": "coding", "model": "anthropic/claude-3.5-haiku", "reasoning": "Simple coding task"}'
                        }
                    }
                ],
                "usage": {"prompt_tokens": 200, "completion_tokens": 50},
            },
        )
    )

    router = Router(config)
    decision = await router.route("Write hello world in Python")

    assert decision.complexity == 2
    assert decision.task_type.value == "coding"
    assert decision.model == "anthropic/claude-3.5-haiku"
    assert decision.triage_tokens.input_tokens == 200
```

- [ ] **Step 6: Run test to verify it fails**

Run: `pytest tests/test_router.py::test_router_route_returns_decision -v`

Expected: FAIL with "AttributeError: 'Router' object has no attribute 'route'"

- [ ] **Step 7: Add route method**

```python
# src/conductor/router.py (add imports at top)
import json
import time
from conductor.executor import Executor
from conductor.models import RoutingDecision, TaskType, TokenUsage

# src/conductor/router.py (add method to Router class)

    async def route(self, prompt: str) -> RoutingDecision:
        """Route a prompt to the optimal model via triage."""
        triage_prompt = self._build_triage_prompt(prompt)

        executor = Executor(api_key=self.config.openrouter_api_key)

        start_time = time.perf_counter()
        result = await executor.execute(
            model=self.config.router_model,
            prompt=triage_prompt,
            max_tokens=100,
            temperature=0,
        )
        latency_ms = int((time.perf_counter() - start_time) * 1000)

        if result.error:
            # Fallback to mid-tier model on triage failure
            return RoutingDecision(
                complexity=3,
                task_type=TaskType.GENERAL,
                model=self._get_fallback_model(),
                reasoning=f"Triage failed: {result.error}",
                triage_tokens=TokenUsage(
                    input_tokens=result.input_tokens,
                    output_tokens=result.output_tokens,
                ),
                triage_cost=result.cost,
                triage_latency_ms=latency_ms,
            )

        # Parse JSON response
        try:
            data = json.loads(result.content)
        except json.JSONDecodeError:
            return RoutingDecision(
                complexity=3,
                task_type=TaskType.GENERAL,
                model=self._get_fallback_model(),
                reasoning="Failed to parse triage response",
                triage_tokens=TokenUsage(
                    input_tokens=result.input_tokens,
                    output_tokens=result.output_tokens,
                ),
                triage_cost=result.cost,
                triage_latency_ms=latency_ms,
            )

        return RoutingDecision(
            complexity=data["complexity"],
            task_type=TaskType(data["task_type"]),
            model=data["model"],
            reasoning=data["reasoning"],
            triage_tokens=TokenUsage(
                input_tokens=result.input_tokens,
                output_tokens=result.output_tokens,
            ),
            triage_cost=result.cost,
            triage_latency_ms=latency_ms,
        )

    def _get_fallback_model(self) -> str:
        """Get a fallback mid-tier model."""
        from conductor.models import ModelTier
        for model_id, model_config in self.config.models.items():
            if model_config.tier == ModelTier.MID:
                return model_id
        # If no mid-tier, return first available
        return next(iter(self.config.models.keys()))
```

- [ ] **Step 8: Run test to verify it passes**

Run: `pytest tests/test_router.py::test_router_route_returns_decision -v`

Expected: PASS

- [ ] **Step 9: Write failing test for triage failure fallback**

```python
# tests/test_router.py (append)


@respx.mock
@pytest.mark.asyncio
async def test_router_fallback_on_triage_error(config: ConductorConfig):
    """Router falls back on triage error."""
    respx.post("https://openrouter.ai/api/v1/chat/completions").mock(
        return_value=httpx.Response(500, text="Server error")
    )

    router = Router(config)
    decision = await router.route("Test prompt")

    assert decision.complexity == 3
    assert decision.task_type == TaskType.GENERAL
    assert "failed" in decision.reasoning.lower() or "error" in decision.reasoning.lower()
```

- [ ] **Step 10: Run test to verify it passes**

Run: `pytest tests/test_router.py::test_router_fallback_on_triage_error -v`

Expected: PASS (fallback already implemented)

- [ ] **Step 11: Add import for TaskType in test file**

```python
# tests/test_router.py (add import at top)
from conductor.models import TaskType
```

- [ ] **Step 12: Run all router tests**

Run: `pytest tests/test_router.py -v`

Expected: All PASS

- [ ] **Step 13: Commit**

```bash
git add src/conductor/router.py tests/test_router.py
git commit -m "feat: add router with LLM triage for model selection"
```

---

## Task 7: AutoJudge

**Files:**
- Create: `src/conductor/autojudge.py`
- Create: `tests/test_autojudge.py`

- [ ] **Step 1: Write failing test for AutoJudge**

```python
# tests/test_autojudge.py
"""Tests for AutoJudge quality scoring."""

import pytest
import httpx
import respx
from conductor.autojudge import AutoJudge


@pytest.fixture
def autojudge():
    """Create AutoJudge instance."""
    return AutoJudge(api_key="test-key", model="anthropic/claude-sonnet-4")


@respx.mock
@pytest.mark.asyncio
async def test_autojudge_scores_response(autojudge: AutoJudge):
    """AutoJudge returns quality score."""
    respx.post("https://openrouter.ai/api/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "id": "gen-123",
                "model": "anthropic/claude-sonnet-4",
                "choices": [
                    {"message": {"content": '{"score": 8.5, "reasoning": "Good response"}'}}
                ],
                "usage": {"prompt_tokens": 100, "completion_tokens": 20},
            },
        )
    )

    score = await autojudge.score(
        prompt="Write hello world",
        response="print('hello world')",
    )

    assert score == 8.5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_autojudge.py::test_autojudge_scores_response -v`

Expected: FAIL with "ModuleNotFoundError: No module named 'conductor.autojudge'"

- [ ] **Step 3: Create autojudge.py**

```python
# src/conductor/autojudge.py
"""AutoJudge for automated quality scoring."""

import json
from conductor.executor import Executor


JUDGE_TEMPLATE = """Rate the quality of this AI response on a scale of 1-10.

Consider:
- Accuracy and correctness
- Completeness (addresses all aspects of the prompt)
- Clarity and coherence
- Appropriate level of detail

## Prompt
\"\"\"
{prompt}
\"\"\"

## Response
\"\"\"
{response}
\"\"\"

Return ONLY this JSON: {{"score": <1-10>, "reasoning": "<brief explanation>"}}"""


class AutoJudge:
    """Automated quality scoring using LLM."""

    def __init__(self, api_key: str, model: str = "anthropic/claude-sonnet-4"):
        self.api_key = api_key
        self.model = model

    async def score(self, prompt: str, response: str) -> float:
        """Score a response on a 1-10 scale."""
        judge_prompt = JUDGE_TEMPLATE.format(prompt=prompt, response=response)

        executor = Executor(api_key=self.api_key)
        result = await executor.execute(
            model=self.model,
            prompt=judge_prompt,
            max_tokens=100,
            temperature=0,
        )

        if result.error:
            return 5.0  # Neutral score on failure

        try:
            data = json.loads(result.content)
            score = float(data["score"])
            return max(1.0, min(10.0, score))  # Clamp to 1-10
        except (json.JSONDecodeError, KeyError, ValueError):
            return 5.0  # Neutral score on parse failure
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_autojudge.py::test_autojudge_scores_response -v`

Expected: PASS

- [ ] **Step 5: Write failing test for error handling**

```python
# tests/test_autojudge.py (append)


@respx.mock
@pytest.mark.asyncio
async def test_autojudge_returns_neutral_on_error(autojudge: AutoJudge):
    """AutoJudge returns neutral score on API error."""
    respx.post("https://openrouter.ai/api/v1/chat/completions").mock(
        return_value=httpx.Response(500, text="Server error")
    )

    score = await autojudge.score(
        prompt="Test",
        response="Test response",
    )

    assert score == 5.0


@respx.mock
@pytest.mark.asyncio
async def test_autojudge_clamps_score(autojudge: AutoJudge):
    """AutoJudge clamps scores to 1-10 range."""
    respx.post("https://openrouter.ai/api/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "id": "gen-123",
                "model": "anthropic/claude-sonnet-4",
                "choices": [
                    {"message": {"content": '{"score": 15, "reasoning": "Amazing"}'}}
                ],
                "usage": {"prompt_tokens": 100, "completion_tokens": 20},
            },
        )
    )

    score = await autojudge.score(prompt="Test", response="Test")

    assert score == 10.0  # Clamped from 15
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `pytest tests/test_autojudge.py -v`

Expected: All PASS

- [ ] **Step 7: Commit**

```bash
git add src/conductor/autojudge.py tests/test_autojudge.py
git commit -m "feat: add AutoJudge for automated quality scoring"
```

---

## Task 8: Learner

**Files:**
- Create: `src/conductor/learner.py`
- Create: `tests/test_learner.py`

- [ ] **Step 1: Write failing test for stats update**

```python
# tests/test_learner.py
"""Tests for adaptive learning system."""

import pytest
from pathlib import Path
from datetime import datetime
from conductor.learner import Learner
from conductor.tracker import Tracker
from conductor.config import ConductorConfig, ModelConfig, LearningConfig
from conductor.models import ModelTier, TaskType, RequestRecord


@pytest.fixture
def config() -> ConductorConfig:
    """Create test configuration."""
    return ConductorConfig(
        openrouter_api_key="test-key",
        models={
            "anthropic/claude-3.5-haiku": ModelConfig(
                tier=ModelTier.FAST,
                strengths=["simple-qa"],
            ),
        },
        learning=LearningConfig(
            enabled=True,
            min_sample_size=5,  # Lower for testing
            quality_block_threshold=5.0,
        ),
    )


@pytest.fixture
def tracker(temp_db_path: Path) -> Tracker:
    """Create and initialize tracker."""
    t = Tracker(temp_db_path)
    t.initialize()
    return t


def test_learner_updates_stats(config: ConductorConfig, tracker: Tracker):
    """Learner updates model stats after request."""
    learner = Learner(config, tracker)

    record = RequestRecord(
        id="test-1",
        timestamp=datetime.now(),
        prompt_hash="abc",
        prompt_text="Test",
        complexity=1,
        task_type=TaskType.CODING,
        routed_model="anthropic/claude-3.5-haiku",
        routing_reasoning="Test",
        triage_input_tokens=10,
        triage_output_tokens=10,
        triage_cost=0.0001,
        triage_latency_ms=100,
        executed_model="anthropic/claude-3.5-haiku",
        response_text="Response",
        input_tokens=10,
        output_tokens=10,
        cost=0.0001,
        latency_ms=100,
        auto_judge_score=8.0,
    )

    learner.learn(record)

    stats = tracker.get_model_stats("anthropic/claude-3.5-haiku", "coding")
    assert stats is not None
    assert stats["total_requests"] == 1
    assert stats["avg_quality"] == 8.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_learner.py::test_learner_updates_stats -v`

Expected: FAIL with "ModuleNotFoundError: No module named 'conductor.learner'"

- [ ] **Step 3: Create learner.py with stats update**

```python
# src/conductor/learner.py
"""Adaptive learning system for routing optimization."""

from datetime import datetime
from conductor.config import ConductorConfig
from conductor.tracker import Tracker
from conductor.models import RequestRecord


class Learner:
    """Learns from request outcomes to improve routing."""

    def __init__(self, config: ConductorConfig, tracker: Tracker):
        self.config = config
        self.tracker = tracker

    def learn(self, record: RequestRecord) -> None:
        """Process a completed request and update stats."""
        if not self.config.learning.enabled:
            return

        if record.auto_judge_score is None:
            return

        self.tracker.update_model_stats(
            model=record.executed_model,
            task_type=record.task_type.value,
            quality=record.auto_judge_score,
            latency_ms=record.latency_ms,
            cost=record.cost,
        )

        self._evaluate_rules(record.executed_model, record.task_type.value)

    def _evaluate_rules(self, model: str, task_type: str) -> None:
        """Check if routing rules should be created or updated."""
        stats = self.tracker.get_model_stats(model, task_type)
        if stats is None:
            return

        if stats["total_requests"] < self.config.learning.min_sample_size:
            return

        avg_quality = stats["avg_quality"]
        if avg_quality is None:
            return

        # Check if model should be blocked for this task type
        if avg_quality < self.config.learning.quality_block_threshold:
            self._create_block_rule(model, task_type, avg_quality, stats["total_requests"])

    def _create_block_rule(
        self, model: str, task_type: str, avg_quality: float, sample_size: int
    ) -> None:
        """Create a rule to block a model for a task type."""
        import json

        # Check if rule already exists
        existing_rules = self.tracker.get_active_rules()
        for rule in existing_rules:
            if rule["rule_type"] == "block":
                condition = json.loads(rule["condition_json"])
                action = json.loads(rule["action_json"])
                if condition.get("task_type") == task_type and action.get("model") == model:
                    return  # Rule already exists

        self.tracker.add_rule(
            rule_type="block",
            condition={"task_type": task_type},
            action={"model": model},
            reason=f"Average quality {avg_quality:.1f}/10 over {sample_size} samples",
        )
```

- [ ] **Step 4: Add missing methods to tracker.py**

```python
# src/conductor/tracker.py (add methods to Tracker class)

    def update_model_stats(
        self,
        model: str,
        task_type: str,
        quality: float,
        latency_ms: int,
        cost: float,
    ) -> None:
        """Update aggregated stats for a model/task_type pair."""
        conn = self._get_connection()
        now = datetime.now().isoformat()

        # Upsert stats
        conn.execute(
            """
            INSERT INTO model_stats (model, task_type, total_requests, total_quality_sum, avg_latency_ms, avg_cost, last_updated)
            VALUES (?, ?, 1, ?, ?, ?, ?)
            ON CONFLICT(model, task_type) DO UPDATE SET
                total_requests = total_requests + 1,
                total_quality_sum = total_quality_sum + excluded.total_quality_sum,
                avg_latency_ms = (avg_latency_ms * total_requests + excluded.avg_latency_ms) / (total_requests + 1),
                avg_cost = (avg_cost * total_requests + excluded.avg_cost) / (total_requests + 1),
                last_updated = excluded.last_updated
            """,
            (model, task_type, quality, latency_ms, cost, now),
        )
        conn.commit()
        conn.close()

    def get_model_stats(self, model: str, task_type: str) -> dict | None:
        """Get stats for a model/task_type pair."""
        conn = self._get_connection()
        conn.row_factory = sqlite3.Row
        cursor = conn.execute(
            """
            SELECT
                total_requests,
                total_quality_sum,
                CASE WHEN total_requests > 0 THEN total_quality_sum / total_requests ELSE NULL END as avg_quality,
                avg_latency_ms,
                avg_cost
            FROM model_stats
            WHERE model = ? AND task_type = ?
            """,
            (model, task_type),
        )
        row = cursor.fetchone()
        conn.close()

        if row is None:
            return None

        return {
            "total_requests": row["total_requests"],
            "avg_quality": row["avg_quality"],
            "avg_latency_ms": row["avg_latency_ms"],
            "avg_cost": row["avg_cost"],
        }

    def get_active_rules(self) -> list[dict]:
        """Get all active routing rules."""
        conn = self._get_connection()
        conn.row_factory = sqlite3.Row
        cursor = conn.execute(
            "SELECT * FROM routing_rules WHERE is_active = 1"
        )
        rules = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return rules

    def add_rule(
        self,
        rule_type: str,
        condition: dict,
        action: dict,
        reason: str,
    ) -> None:
        """Add a new routing rule."""
        import json
        from datetime import timedelta

        conn = self._get_connection()
        now = datetime.now()
        expires_at = now + timedelta(days=30)

        conn.execute(
            """
            INSERT INTO routing_rules (rule_type, condition_json, action_json, created_at, expires_at, reason, is_active)
            VALUES (?, ?, ?, ?, ?, ?, 1)
            """,
            (
                rule_type,
                json.dumps(condition),
                json.dumps(action),
                now.isoformat(),
                expires_at.isoformat(),
                reason,
            ),
        )
        conn.commit()
        conn.close()
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_learner.py::test_learner_updates_stats -v`

Expected: PASS

- [ ] **Step 6: Write failing test for rule creation**

```python
# tests/test_learner.py (append)


def test_learner_creates_block_rule(config: ConductorConfig, tracker: Tracker):
    """Learner creates block rule for poor performing model."""
    learner = Learner(config, tracker)

    # Add enough poor-quality samples to trigger rule
    for i in range(6):
        record = RequestRecord(
            id=f"poor-{i}",
            timestamp=datetime.now(),
            prompt_hash=f"hash{i}",
            prompt_text="Test",
            complexity=2,
            task_type=TaskType.CODING,
            routed_model="anthropic/claude-3.5-haiku",
            routing_reasoning="Test",
            triage_input_tokens=10,
            triage_output_tokens=10,
            triage_cost=0.0001,
            triage_latency_ms=100,
            executed_model="anthropic/claude-3.5-haiku",
            response_text="Bad response",
            input_tokens=10,
            output_tokens=10,
            cost=0.0001,
            latency_ms=100,
            auto_judge_score=3.0,  # Poor quality
        )
        learner.learn(record)

    rules = tracker.get_active_rules()
    block_rules = [r for r in rules if r["rule_type"] == "block"]
    assert len(block_rules) == 1
    assert "claude-3.5-haiku" in block_rules[0]["action_json"]
```

- [ ] **Step 7: Run test to verify it passes**

Run: `pytest tests/test_learner.py::test_learner_creates_block_rule -v`

Expected: PASS

- [ ] **Step 8: Run all learner tests**

Run: `pytest tests/test_learner.py -v`

Expected: All PASS

- [ ] **Step 9: Commit**

```bash
git add src/conductor/learner.py src/conductor/tracker.py tests/test_learner.py
git commit -m "feat: add adaptive learning system for routing optimization"
```

---

## Task 9: CLI Commands

**Files:**
- Create: `src/conductor/cli.py`
- Create: `tests/test_cli.py`

- [ ] **Step 1: Write failing test for CLI entry point**

```python
# tests/test_cli.py
"""Tests for CLI commands."""

import pytest
from click.testing import CliRunner
from conductor.cli import main


@pytest.fixture
def runner():
    """Create CLI test runner."""
    return CliRunner()


def test_cli_help(runner: CliRunner):
    """CLI shows help text."""
    result = runner.invoke(main, ["--help"])
    assert result.exit_code == 0
    assert "conductor" in result.output.lower() or "routing" in result.output.lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_cli.py::test_cli_help -v`

Expected: FAIL with "ModuleNotFoundError: No module named 'conductor.cli'"

- [ ] **Step 3: Create cli.py with basic structure**

```python
# src/conductor/cli.py
"""CLI interface for Conductor."""

import click


@click.group()
@click.version_option(version="0.1.0")
def main():
    """Conductor: Intelligent LLM request routing."""
    pass


@main.command()
@click.argument("prompt")
@click.option("--model", "-m", help="Override routing, use specific model")
@click.option("--no-shadow", is_flag=True, help="Disable shadow sampling")
@click.option("--json", "output_json", is_flag=True, help="Output as JSON")
@click.option("--quiet", "-q", is_flag=True, help="Only output response")
def ask(prompt: str, model: str | None, no_shadow: bool, output_json: bool, quiet: bool):
    """Route and execute a prompt."""
    import asyncio
    asyncio.run(_ask(prompt, model, no_shadow, output_json, quiet))


async def _ask(
    prompt: str,
    model: str | None,
    no_shadow: bool,
    output_json: bool,
    quiet: bool,
):
    """Async implementation of ask command."""
    from conductor.config import load_config, ConductorConfig
    from conductor.router import Router
    from conductor.executor import Executor
    from conductor.tracker import Tracker
    from conductor.autojudge import AutoJudge
    from conductor.learner import Learner
    from conductor.models import RequestRecord, TaskType
    from pathlib import Path
    from datetime import datetime
    import hashlib
    import uuid
    import json

    # Load config
    config_paths = [
        Path("conductor.yaml"),
        Path.home() / ".config" / "conductor" / "conductor.yaml",
    ]
    config = None
    for path in config_paths:
        if path.exists():
            config = load_config(path)
            break

    if config is None:
        click.echo("Error: No conductor.yaml found", err=True)
        raise SystemExit(1)

    # Initialize components
    tracker = Tracker(Path(config.database_path).expanduser())
    tracker.initialize()

    executor = Executor(api_key=config.openrouter_api_key)
    autojudge = AutoJudge(api_key=config.openrouter_api_key, model=config.router_model)
    learner = Learner(config, tracker)

    # Route or use override
    if model:
        from conductor.models import RoutingDecision, TokenUsage
        decision = RoutingDecision(
            complexity=0,
            task_type=TaskType.GENERAL,
            model=model,
            reasoning="Manual override",
            triage_tokens=TokenUsage(input_tokens=0, output_tokens=0),
            triage_cost=0.0,
            triage_latency_ms=0,
        )
    else:
        router = Router(config)
        decision = await router.route(prompt)

    if not quiet:
        click.echo(f"Routing: complexity={decision.complexity}, task_type={decision.task_type.value} -> {decision.model}")

    # Execute
    shadow_rate = 0.0 if no_shadow else config.shadow_sample_rate
    primary, baseline = await executor.execute_with_shadow(
        model=decision.model,
        prompt=prompt,
        shadow_model=config.baseline_model,
        shadow_rate=shadow_rate,
    )

    # Score with AutoJudge
    primary_score = await autojudge.score(prompt, primary.content)
    baseline_score = None
    if baseline:
        baseline_score = await autojudge.score(prompt, baseline.content)

    # Create record
    record = RequestRecord(
        id=str(uuid.uuid4())[:8],
        timestamp=datetime.now(),
        prompt_hash=hashlib.sha256(prompt.encode()).hexdigest()[:16],
        prompt_text=prompt,
        complexity=decision.complexity,
        task_type=decision.task_type,
        routed_model=decision.model,
        routing_reasoning=decision.reasoning,
        triage_input_tokens=decision.triage_tokens.input_tokens,
        triage_output_tokens=decision.triage_tokens.output_tokens,
        triage_cost=decision.triage_cost,
        triage_latency_ms=decision.triage_latency_ms,
        executed_model=primary.model,
        response_text=primary.content,
        input_tokens=primary.input_tokens,
        output_tokens=primary.output_tokens,
        cost=primary.cost,
        latency_ms=primary.latency_ms,
        error=primary.error,
        auto_judge_score=primary_score,
        baseline_model=config.baseline_model if baseline else None,
        baseline_response=baseline.content if baseline else None,
        baseline_input_tokens=baseline.input_tokens if baseline else None,
        baseline_output_tokens=baseline.output_tokens if baseline else None,
        baseline_cost=baseline.cost if baseline else None,
        baseline_latency_ms=baseline.latency_ms if baseline else None,
        baseline_auto_judge_score=baseline_score,
    )

    # Log and learn
    tracker.log_request(record)
    learner.learn(record)

    # Output
    if output_json:
        click.echo(json.dumps(record.model_dump(), default=str, indent=2))
    else:
        total_cost = decision.triage_cost + primary.cost
        if not quiet:
            click.echo(f"Cost: ${total_cost:.4f} (triage: ${decision.triage_cost:.4f} + execution: ${primary.cost:.4f})")
            click.echo(f"Latency: {primary.latency_ms}ms")
            click.echo()
        click.echo(primary.content)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_cli.py::test_cli_help -v`

Expected: PASS

- [ ] **Step 5: Write failing test for models command**

```python
# tests/test_cli.py (append)


def test_cli_models_command(runner: CliRunner, temp_config_path, monkeypatch):
    """CLI models command lists configured models."""
    config_content = """
openrouter_api_key: test-key
models:
  anthropic/claude-3.5-haiku:
    tier: fast
    strengths:
      - simple-qa
"""
    temp_config_path.write_text(config_content)
    monkeypatch.chdir(temp_config_path.parent)

    result = runner.invoke(main, ["models"])

    assert result.exit_code == 0
    assert "claude-3.5-haiku" in result.output
```

- [ ] **Step 6: Run test to verify it fails**

Run: `pytest tests/test_cli.py::test_cli_models_command -v`

Expected: FAIL with "Error: No such command 'models'"

- [ ] **Step 7: Add models command**

```python
# src/conductor/cli.py (add command)

@main.command()
def models():
    """List configured models."""
    from conductor.config import load_config
    from pathlib import Path

    config_paths = [
        Path("conductor.yaml"),
        Path.home() / ".config" / "conductor" / "conductor.yaml",
    ]
    config = None
    for path in config_paths:
        if path.exists():
            config = load_config(path)
            break

    if config is None:
        click.echo("Error: No conductor.yaml found", err=True)
        raise SystemExit(1)

    click.echo("\nCONFIGURED MODELS")
    click.echo("-" * 60)
    click.echo(f"{'Model':<35} {'Tier':<6} Strengths")
    click.echo("-" * 60)

    for model_id, model_config in config.models.items():
        strengths = ", ".join(model_config.strengths)
        click.echo(f"{model_id:<35} {model_config.tier.value:<6} {strengths}")

    click.echo()
    click.echo(f"Router Model: {config.router_model}")
    click.echo(f"Baseline Model: {config.baseline_model}")
    click.echo(f"Shadow Sample Rate: {config.shadow_sample_rate * 100:.0f}%")
```

- [ ] **Step 8: Run test to verify it passes**

Run: `pytest tests/test_cli.py::test_cli_models_command -v`

Expected: PASS

- [ ] **Step 9: Add stats command**

```python
# src/conductor/cli.py (add command)

@main.command()
@click.option("--days", "-d", default=30, help="Number of days to include")
@click.option("--rules", is_flag=True, help="Show active routing rules")
def stats(days: int, rules: bool):
    """Show usage statistics."""
    from conductor.config import load_config
    from conductor.tracker import Tracker
    from pathlib import Path

    config_paths = [
        Path("conductor.yaml"),
        Path.home() / ".config" / "conductor" / "conductor.yaml",
    ]
    config = None
    for path in config_paths:
        if path.exists():
            config = load_config(path)
            break

    if config is None:
        click.echo("Error: No conductor.yaml found", err=True)
        raise SystemExit(1)

    tracker = Tracker(Path(config.database_path).expanduser())
    tracker.initialize()

    data = tracker.get_stats(days=days)

    click.echo("\nCONDUCTOR STATS")
    click.echo("=" * 60)
    click.echo(f"Period: Last {days} days")
    click.echo(f"Total Requests: {data['total_requests']}")
    click.echo()
    click.echo("COST ANALYSIS")
    click.echo("-" * 60)
    click.echo(f"  Execution Cost:  ${data['total_execution_cost']:.4f}")
    click.echo(f"  Triage Overhead: ${data['total_triage_cost']:.4f}")
    click.echo(f"  Total Cost:      ${data['total_cost']:.4f}")
    click.echo()

    if data["by_model"]:
        click.echo("MODEL DISTRIBUTION")
        click.echo("-" * 60)
        for model, model_data in data["by_model"].items():
            quality = model_data["avg_quality"]
            quality_str = f"{quality:.1f}" if quality else "N/A"
            click.echo(
                f"  {model:<30} {model_data['request_count']:>5} reqs  ${model_data['total_cost']:.4f}  Q:{quality_str}"
            )

    if rules:
        click.echo()
        click.echo("ACTIVE ROUTING RULES")
        click.echo("-" * 60)
        active_rules = tracker.get_active_rules()
        if active_rules:
            for rule in active_rules:
                click.echo(f"  [{rule['rule_type']}] {rule['reason']}")
        else:
            click.echo("  No active rules")
```

- [ ] **Step 10: Add rate command**

```python
# src/conductor/cli.py (add command)

@main.command()
@click.argument("request_id")
@click.argument("rating", type=click.Choice(["up", "down"]))
def rate(request_id: str, rating: str):
    """Rate a past response (up/down)."""
    from conductor.config import load_config
    from conductor.tracker import Tracker
    from pathlib import Path

    config_paths = [
        Path("conductor.yaml"),
        Path.home() / ".config" / "conductor" / "conductor.yaml",
    ]
    config = None
    for path in config_paths:
        if path.exists():
            config = load_config(path)
            break

    if config is None:
        click.echo("Error: No conductor.yaml found", err=True)
        raise SystemExit(1)

    tracker = Tracker(Path(config.database_path).expanduser())
    tracker.initialize()

    record = tracker.get_request(request_id)
    if record is None:
        click.echo(f"Error: Request {request_id} not found", err=True)
        raise SystemExit(1)

    tracker.update_rating(request_id, rating)
    click.echo(f"Recorded: thumbs {rating} for request {request_id}")
```

- [ ] **Step 11: Add bench command**

```python
# src/conductor/cli.py (add command)

@main.command()
@click.argument("prompt")
@click.option("--suite", type=click.Path(exists=True), help="Benchmark suite JSONL file")
def bench(prompt: str | None, suite: str | None):
    """Run benchmark: routed vs baseline."""
    import asyncio
    asyncio.run(_bench(prompt, suite))


async def _bench(prompt: str | None, suite: str | None):
    """Async implementation of bench command."""
    from conductor.config import load_config
    from conductor.router import Router
    from conductor.executor import Executor
    from conductor.autojudge import AutoJudge
    from pathlib import Path
    import json

    config_paths = [
        Path("conductor.yaml"),
        Path.home() / ".config" / "conductor" / "conductor.yaml",
    ]
    config = None
    for path in config_paths:
        if path.exists():
            config = load_config(path)
            break

    if config is None:
        click.echo("Error: No conductor.yaml found", err=True)
        raise SystemExit(1)

    prompts = []
    if suite:
        with open(suite) as f:
            for line in f:
                data = json.loads(line)
                prompts.append(data["prompt"])
    elif prompt:
        prompts = [prompt]
    else:
        click.echo("Error: Provide a prompt or --suite", err=True)
        raise SystemExit(1)

    router = Router(config)
    executor = Executor(api_key=config.openrouter_api_key)
    autojudge = AutoJudge(api_key=config.openrouter_api_key)

    for p in prompts:
        click.echo(f"\nBenchmarking: {p[:50]}...")

        # Route
        decision = await router.route(p)
        click.echo(f"  Routed to: {decision.model} (complexity={decision.complexity})")

        # Execute both
        routed_result = await executor.execute(decision.model, p)
        baseline_result = await executor.execute(config.baseline_model, p)

        # Score both
        routed_score = await autojudge.score(p, routed_result.content)
        baseline_score = await autojudge.score(p, baseline_result.content)

        # Display results
        click.echo()
        click.echo(f"  {'Metric':<12} {'Routed':<15} {'Baseline':<15}")
        click.echo(f"  {'-'*42}")
        click.echo(f"  {'Quality':<12} {routed_score:<15.1f} {baseline_score:<15.1f}")
        click.echo(f"  {'Cost':<12} ${routed_result.cost:<14.4f} ${baseline_result.cost:<14.4f}")
        click.echo(f"  {'Latency':<12} {routed_result.latency_ms:<15}ms {baseline_result.latency_ms:<15}ms")

        savings = baseline_result.cost - routed_result.cost - decision.triage_cost
        quality_delta = baseline_score - routed_score
        click.echo()
        click.echo(f"  Savings: ${savings:.4f} | Quality delta: {quality_delta:+.1f}")
```

- [ ] **Step 12: Add compare command**

```python
# src/conductor/cli.py (add command)

@main.command()
@click.option("--export", "export_format", type=click.Choice(["csv", "json"]), help="Export dataset")
def compare(export_format: str | None):
    """Compare routed vs always-baseline strategy."""
    from conductor.config import load_config
    from conductor.tracker import Tracker
    from pathlib import Path
    import json
    import csv
    import sys

    config_paths = [
        Path("conductor.yaml"),
        Path.home() / ".config" / "conductor" / "conductor.yaml",
    ]
    config = None
    for path in config_paths:
        if path.exists():
            config = load_config(path)
            break

    if config is None:
        click.echo("Error: No conductor.yaml found", err=True)
        raise SystemExit(1)

    tracker = Tracker(Path(config.database_path).expanduser())
    tracker.initialize()

    # Get requests with baseline data
    conn = tracker._get_connection()
    conn.row_factory = __import__("sqlite3").Row
    cursor = conn.execute(
        """
        SELECT * FROM requests
        WHERE baseline_model IS NOT NULL
        ORDER BY timestamp DESC
        """
    )
    rows = cursor.fetchall()
    conn.close()

    if not rows:
        click.echo("No shadow-sampled data available. Run more requests or use 'conductor bench'.")
        raise SystemExit(0)

    if export_format == "csv":
        writer = csv.writer(sys.stdout)
        writer.writerow([
            "request_id", "timestamp", "task_type", "complexity",
            "routed_model", "routed_cost", "routed_quality",
            "baseline_model", "baseline_cost", "baseline_quality",
            "cost_savings", "quality_delta"
        ])
        for row in rows:
            savings = (row["baseline_cost"] or 0) - row["cost"] - row["triage_cost"]
            quality_delta = (row["baseline_auto_judge_score"] or 0) - (row["auto_judge_score"] or 0)
            writer.writerow([
                row["id"], row["timestamp"], row["task_type"], row["complexity"],
                row["executed_model"], row["cost"], row["auto_judge_score"],
                row["baseline_model"], row["baseline_cost"], row["baseline_auto_judge_score"],
                savings, quality_delta
            ])
        return

    if export_format == "json":
        data = []
        for row in rows:
            savings = (row["baseline_cost"] or 0) - row["cost"] - row["triage_cost"]
            quality_delta = (row["baseline_auto_judge_score"] or 0) - (row["auto_judge_score"] or 0)
            data.append({
                "request_id": row["id"],
                "timestamp": row["timestamp"],
                "task_type": row["task_type"],
                "complexity": row["complexity"],
                "routed_model": row["executed_model"],
                "routed_cost": row["cost"],
                "routed_quality": row["auto_judge_score"],
                "baseline_model": row["baseline_model"],
                "baseline_cost": row["baseline_cost"],
                "baseline_quality": row["baseline_auto_judge_score"],
                "cost_savings": savings,
                "quality_delta": quality_delta,
            })
        click.echo(json.dumps(data, indent=2))
        return

    # Summary view
    total_routed_cost = sum(r["cost"] + r["triage_cost"] for r in rows)
    total_baseline_cost = sum(r["baseline_cost"] or 0 for r in rows)
    avg_routed_quality = sum(r["auto_judge_score"] or 0 for r in rows) / len(rows)
    avg_baseline_quality = sum(r["baseline_auto_judge_score"] or 0 for r in rows) / len(rows)

    click.echo("\nROUTING EFFICIENCY ANALYSIS")
    click.echo("=" * 60)
    click.echo(f"Based on {len(rows)} shadow-sampled requests")
    click.echo()
    click.echo(f"{'Strategy':<20} {'Cost':<12} {'Avg Quality':<12}")
    click.echo("-" * 60)
    click.echo(f"{'Conductor routed':<20} ${total_routed_cost:<11.4f} {avg_routed_quality:<12.1f}")
    click.echo(f"{'Always baseline':<20} ${total_baseline_cost:<11.4f} {avg_baseline_quality:<12.1f}")
    click.echo("-" * 60)

    savings_pct = ((total_baseline_cost - total_routed_cost) / total_baseline_cost * 100) if total_baseline_cost > 0 else 0
    quality_delta = avg_baseline_quality - avg_routed_quality
    click.echo(f"\nSavings: ${total_baseline_cost - total_routed_cost:.4f} ({savings_pct:.1f}%)")
    click.echo(f"Quality tradeoff: {quality_delta:+.1f}")
```

- [ ] **Step 13: Run all CLI tests**

Run: `pytest tests/test_cli.py -v`

Expected: All PASS

- [ ] **Step 14: Test CLI manually**

Run: `conductor --help`

Expected: Shows help with all commands listed

- [ ] **Step 15: Commit**

```bash
git add src/conductor/cli.py tests/test_cli.py
git commit -m "feat: add CLI commands (ask, bench, stats, compare, models, rate)"
```

---

## Task 10: Integration Test and Polish

**Files:**
- Create: `tests/test_integration.py`
- Modify: `src/conductor/__init__.py`

- [ ] **Step 1: Write integration test**

```python
# tests/test_integration.py
"""Integration tests for full conductor flow."""

import pytest
import httpx
import respx
from pathlib import Path
from click.testing import CliRunner
from conductor.cli import main


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def config_file(temp_dir: Path) -> Path:
    """Create test config file."""
    config = temp_dir / "conductor.yaml"
    config.write_text("""
openrouter_api_key: ${OPENROUTER_API_KEY}
router_model: anthropic/claude-sonnet-4
baseline_model: anthropic/claude-opus-4
shadow_sample_rate: 0.0
database_path: {db_path}
models:
  anthropic/claude-3.5-haiku:
    tier: fast
    strengths:
      - simple-qa
  anthropic/claude-sonnet-4:
    tier: mid
    strengths:
      - coding
      - reasoning
""".format(db_path=temp_dir / "test.db"))
    return config


@respx.mock
@pytest.mark.asyncio
async def test_full_ask_flow(runner: CliRunner, config_file: Path, monkeypatch):
    """Full ask flow: route -> execute -> score -> log."""
    monkeypatch.chdir(config_file.parent)

    # Mock triage call
    respx.post("https://openrouter.ai/api/v1/chat/completions").mock(
        side_effect=[
            # Triage response
            httpx.Response(
                200,
                json={
                    "model": "anthropic/claude-sonnet-4",
                    "choices": [{"message": {"content": '{"complexity": 1, "task_type": "simple_qa", "model": "anthropic/claude-3.5-haiku", "reasoning": "Simple question"}'}}],
                    "usage": {"prompt_tokens": 100, "completion_tokens": 30},
                },
            ),
            # Execution response
            httpx.Response(
                200,
                json={
                    "model": "anthropic/claude-3.5-haiku",
                    "choices": [{"message": {"content": "Hello, world!"}}],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 5},
                },
            ),
            # AutoJudge response
            httpx.Response(
                200,
                json={
                    "model": "anthropic/claude-sonnet-4",
                    "choices": [{"message": {"content": '{"score": 8.5, "reasoning": "Good"}'}}],
                    "usage": {"prompt_tokens": 50, "completion_tokens": 20},
                },
            ),
        ]
    )

    result = runner.invoke(main, ["ask", "Say hello"])

    assert result.exit_code == 0
    assert "Hello, world!" in result.output
    assert "haiku" in result.output.lower()
```

- [ ] **Step 2: Run integration test**

Run: `pytest tests/test_integration.py -v`

Expected: PASS

- [ ] **Step 3: Update package exports**

```python
# src/conductor/__init__.py
"""Conductor: Intelligent LLM request routing with empirical validation."""

__version__ = "0.1.0"

from conductor.config import ConductorConfig, load_config
from conductor.executor import Executor
from conductor.router import Router
from conductor.tracker import Tracker
from conductor.learner import Learner
from conductor.autojudge import AutoJudge
from conductor.models import (
    TaskType,
    ModelTier,
    TokenUsage,
    RoutingDecision,
    ExecutionResult,
    RequestRecord,
)

__all__ = [
    "ConductorConfig",
    "load_config",
    "Executor",
    "Router",
    "Tracker",
    "Learner",
    "AutoJudge",
    "TaskType",
    "ModelTier",
    "TokenUsage",
    "RoutingDecision",
    "ExecutionResult",
    "RequestRecord",
]
```

- [ ] **Step 4: Run full test suite**

Run: `pytest tests/ -v`

Expected: All tests PASS

- [ ] **Step 5: Test CLI end-to-end (requires API key)**

Run:
```bash
export OPENROUTER_API_KEY="your-key-here"
cp conductor.yaml.example conductor.yaml
conductor models
conductor ask "What is 2+2?"
conductor stats
```

Expected: Commands execute successfully

- [ ] **Step 6: Final commit**

```bash
git add src/conductor/__init__.py tests/test_integration.py
git commit -m "feat: add integration tests and finalize package exports"
```

- [ ] **Step 7: Create final summary commit**

```bash
git log --oneline
```

Expected: Clean commit history showing incremental progress

---

## Summary

This plan implements Conductor v1 in 10 tasks:

1. **Project Setup** - pyproject.toml, directory structure
2. **Pydantic Models** - TaskType, ModelTier, RoutingDecision, ExecutionResult, RequestRecord
3. **Configuration** - YAML loading, env var resolution, validation
4. **SQLite Tracker** - Schema, logging, stats queries
5. **OpenRouter Executor** - Async client, retries, shadow sampling
6. **Router** - Triage prompt, model selection
7. **AutoJudge** - Quality scoring
8. **Learner** - Stats aggregation, rule generation
9. **CLI Commands** - ask, bench, stats, compare, models, rate
10. **Integration** - End-to-end test, polish

Each task follows TDD: write failing test, implement, verify, commit.
