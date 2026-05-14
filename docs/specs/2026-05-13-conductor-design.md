# Conductor: Intelligent LLM Request Routing with Empirical Validation

**Version:** 1.0 Draft
**Date:** May 13, 2026
**Status:** Design Specification

---

## Abstract

Conductor is a Python CLI tool and research framework for intelligent routing of LLM prompts to cost-optimal models. The system employs a lightweight frontier model as a triage layer to assess prompt complexity and task characteristics, routing requests to the most appropriate model from a configured pool. Unlike static routing rules, Conductor continuously learns from execution outcomes, automatically adjusting routing decisions based on measured quality metrics.

The dual nature of Conductor (as both a practical tool and research instrument) enables empirical investigation of a central question in LLM operations: **Does intelligent routing reduce costs while preserving output quality compared to always using the most capable model?**

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [System Architecture](#2-system-architecture)
3. [Component Specifications](#3-component-specifications)
4. [Data Model](#4-data-model)
5. [Routing Algorithm](#5-routing-algorithm)
6. [Adaptive Learning System](#6-adaptive-learning-system)
7. [Research Methodology](#7-research-methodology)
8. [Command-Line Interface](#8-command-line-interface)
9. [Configuration](#9-configuration)
10. [Future Work](#10-future-work)
11. [Appendices](#11-appendices)

---

## 1. Introduction

### 1.1 Problem Statement

The proliferation of large language models presents practitioners with a cost-quality tradeoff. Frontier models (Claude Opus, GPT-4, etc.) deliver superior performance but at 10-50x the cost of smaller models. Conversely, lightweight models handle simple tasks adequately but fail on complex reasoning.

Current approaches fall into two categories:
- **Static routing:** Hardcoded rules based on keyword matching or prompt length
- **Always-best:** Send everything to the frontier model, accepting high costs

Neither approach is optimal. Static routing lacks semantic understanding; always-best wastes resources on trivial tasks.

### 1.2 Proposed Solution

Conductor introduces **semantic triage routing**: using a mid-tier LLM to analyze each prompt's complexity and task type, then routing to the optimal model. The triage call is constrained to <100 output tokens, keeping overhead minimal.

Key innovation: Conductor treats routing as an empirical question. Every request generates data comparing the routed outcome against a baseline (the always-best strategy). Over time, the system:

1. Validates whether routing delivers cost savings
2. Measures the quality tradeoff
3. Automatically adjusts routing rules based on observed performance

### 1.3 Design Principles

| Principle | Implementation |
|-----------|----------------|
| **Empirical over theoretical** | Every routing decision is measured, not assumed correct |
| **Transparent learning** | All rule adjustments are logged with reasoning |
| **Minimal overhead** | Triage calls capped at 100 tokens; async execution |
| **Provider agnostic** | OpenRouter abstraction enables any model combination |
| **Research-first** | Data collection is a primary feature, not an afterthought |

---

## 2. System Architecture

### 2.1 High-Level Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                              CLI Layer                                  │
│         ask │ bench │ stats │ compare │ models │ rate                   │
└──────┬──────┴───┬───┴───┬───┴────┬────┴───┬────┴───┬────────────────────┘
       │          │       │        │        │        │
       ▼          ▼       │        │        │        │
┌──────────┐ ┌──────────┐ │        │        │        │
│  Router  │ │ Executor │◄┼────────┼────────┘        │
│          │ │          │ │        │                 │
│ ┌──────┐ │ │ OpenRouter│ │        │                 │
│ │Triage│ │ │   Client │ │        │                 │
│ └──────┘ │ └──────────┘ │        │                 │
└────┬─────┘      │       │        │                 │
     │            │       │        │                 │
     ▼            ▼       ▼        ▼                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         Tracker (SQLite)                                │
│                                                                         │
│  ┌─────────────┐  ┌─────────────┐  ┌───────────────┐  ┌──────────────┐ │
│  │  requests   │  │ model_stats │  │ routing_rules │  │    config    │ │
│  └─────────────┘  └─────────────┘  └───────────────┘  └──────────────┘ │
└───────────────────────────────┬─────────────────────────────────────────┘
                                │
                                ▼
                        ┌──────────────┐
                        │   Learner    │
                        │              │
                        │ Statistics   │
                        │ Analysis     │
                        │ Rule Genesis │
                        └──────────────┘
```

### 2.2 Data Flow

**Standard Request (`conductor ask`):**

```
User Prompt
     │
     ▼
┌─────────┐    RoutingDecision    ┌──────────┐
│ Router  │ ───────────────────▶  │ Executor │
│         │  {model, complexity,  │          │
│ triage()│   task_type, reason}  │execute() │
└─────────┘                       └────┬─────┘
                                       │
                    ┌──────────────────┴──────────────────┐
                    │                                     │
                    ▼                                     ▼
            Primary Request                    Shadow Request (10%)
            (routed model)                     (baseline model)
                    │                                     │
                    └──────────────┬──────────────────────┘
                                   │
                                   ▼
                            ┌─────────────┐
                            │  Tracker    │
                            │  (persist)  │
                            └──────┬──────┘
                                   │
                                   ▼
                            ┌─────────────┐
                            │  Learner    │
                            │  (analyze)  │
                            └─────────────┘
```

### 2.3 Concurrency Model

Conductor employs Python's `asyncio` with `httpx.AsyncClient` for all network operations:

- **Triage → Execute:** Sequential (triage must complete before execution)
- **Primary + Shadow:** Concurrent (baseline doesn't block response delivery)
- **Batch Benchmark:** Concurrent across requests, respecting rate limits

---

## 3. Component Specifications

### 3.1 Router

**Responsibility:** Analyze incoming prompts and determine optimal model routing.

**Interface:**
```python
class Router:
    async def route(self, prompt: str) -> RoutingDecision
```

**RoutingDecision Schema:**
```python
@dataclass
class RoutingDecision:
    complexity: int          # 1-5 scale
    task_type: TaskType      # enum: coding, reasoning, creative, simple_qa, general
    model: str               # selected model identifier
    reasoning: str           # one-sentence explanation
    triage_tokens: TokenUsage
    triage_cost: float
    triage_latency_ms: int
```

**Triage Prompt Template:**
```
You are a routing assistant. Analyze the user's prompt and select the optimal model.

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
{{model_pool}}

## Active Routing Constraints
{{learned_rules}}

## User Prompt
"""
{{user_prompt}}
"""

Return ONLY this JSON:
{"complexity": <1-5>, "task_type": "<type>", "model": "<model_id>", "reasoning": "<one sentence>"}
```

**Constraints:**
- Maximum output tokens: 100
- Temperature: 0 (deterministic)
- Default router model: `anthropic/claude-sonnet-4`

### 3.2 Executor

**Responsibility:** Execute LLM requests via OpenRouter API.

**Interface:**
```python
class Executor:
    async def execute(
        self,
        model: str,
        prompt: str,
        max_tokens: int = 4096
    ) -> ExecutionResult

    async def execute_with_shadow(
        self,
        model: str,
        prompt: str,
        shadow_model: str,
        shadow_rate: float = 0.1
    ) -> tuple[ExecutionResult, ExecutionResult | None]
```

**ExecutionResult Schema:**
```python
@dataclass
class ExecutionResult:
    content: str
    model: str               # confirmed model from response
    input_tokens: int
    output_tokens: int
    cost: float
    latency_ms: int
    error: str | None = None
```

**OpenRouter Integration:**
- Endpoint: `https://openrouter.ai/api/v1/chat/completions`
- Authentication: `Authorization: Bearer $OPENROUTER_API_KEY`
- Required headers: `HTTP-Referer`, `X-Title` for attribution
- Pricing fetched dynamically from `/api/v1/models`

**Error Handling:**
- Exponential backoff: 3 attempts, 1s → 2s → 4s delays
- Timeout: 120 seconds (configurable)
- Graceful degradation: errors logged, not raised

### 3.3 Tracker

**Responsibility:** Persist all request data for analysis and research.

**Interface:**
```python
class Tracker:
    def log_request(self, request: RequestRecord) -> str  # returns request_id
    def get_request(self, request_id: str) -> RequestRecord
    def get_model_stats(self, model: str, task_type: str) -> ModelStats
    def update_model_stats(self, model: str, task_type: str, quality: float)
    def get_active_rules(self) -> list[RoutingRule]
    def add_rule(self, rule: RoutingRule)
    def export_research_data(self, format: str) -> str  # csv or json
```

**Storage:** SQLite database at `~/.config/conductor/conductor.db`

### 3.4 Learner

**Responsibility:** Analyze execution outcomes and adjust routing rules.

**Interface:**
```python
class Learner:
    async def learn(self, request: RequestRecord)
    def evaluate_rules(self) -> list[RuleChange]
    def get_performance_report(self) -> PerformanceReport
```

**Trigger Conditions:** See [Section 6](#6-adaptive-learning-system).

### 3.5 AutoJudge

**Responsibility:** Score output quality without human intervention.

**Interface:**
```python
class AutoJudge:
    async def score(self, prompt: str, response: str) -> float  # 1-10
```

**Scoring Prompt:**
```
Rate the quality of this AI response on a scale of 1-10.

Consider:
- Accuracy and correctness
- Completeness (addresses all aspects of the prompt)
- Clarity and coherence
- Appropriate level of detail

Prompt: """{{prompt}}"""

Response: """{{response}}"""

Return ONLY a JSON object: {"score": <1-10>, "reasoning": "<brief explanation>"}
```

**Execution:** Uses the router model (Sonnet) with temperature 0.

**Invocation:** AutoJudge runs automatically after every execution, scoring both primary and baseline (if shadow sampled) responses. Scores are stored immediately; no batching.

---

## 4. Data Model

### 4.1 SQLite Schema

```sql
-- Primary request log
CREATE TABLE requests (
    id TEXT PRIMARY KEY,
    timestamp TEXT NOT NULL,
    prompt_hash TEXT NOT NULL,
    prompt_text TEXT NOT NULL,

    -- Routing decision
    complexity INTEGER NOT NULL CHECK (complexity BETWEEN 1 AND 5),
    task_type TEXT NOT NULL,
    routed_model TEXT NOT NULL,
    routing_reasoning TEXT,

    -- Triage overhead
    triage_input_tokens INTEGER NOT NULL,
    triage_output_tokens INTEGER NOT NULL,
    triage_cost REAL NOT NULL,
    triage_latency_ms INTEGER NOT NULL,

    -- Primary execution
    executed_model TEXT NOT NULL,
    response_text TEXT,
    input_tokens INTEGER NOT NULL,
    output_tokens INTEGER NOT NULL,
    cost REAL NOT NULL,
    latency_ms INTEGER NOT NULL,
    error TEXT,

    -- Quality metrics
    auto_judge_score REAL CHECK (auto_judge_score BETWEEN 1 AND 10),
    human_rating TEXT CHECK (human_rating IN ('up', 'down')),

    -- Baseline comparison (shadow samples only)
    baseline_model TEXT,
    baseline_response TEXT,
    baseline_input_tokens INTEGER,
    baseline_output_tokens INTEGER,
    baseline_cost REAL,
    baseline_latency_ms INTEGER,
    baseline_auto_judge_score REAL,

    -- Indexes
    UNIQUE(prompt_hash, executed_model)
);

CREATE INDEX idx_requests_timestamp ON requests(timestamp);
CREATE INDEX idx_requests_task_type ON requests(task_type);
CREATE INDEX idx_requests_model ON requests(executed_model);

-- Aggregated statistics for learning
CREATE TABLE model_stats (
    model TEXT NOT NULL,
    task_type TEXT NOT NULL,
    total_requests INTEGER NOT NULL DEFAULT 0,
    total_quality_sum REAL NOT NULL DEFAULT 0,
    avg_quality REAL GENERATED ALWAYS AS (
        CASE WHEN total_requests > 0
        THEN total_quality_sum / total_requests
        ELSE NULL END
    ) STORED,
    avg_latency_ms REAL,
    avg_cost REAL,
    last_updated TEXT NOT NULL,
    PRIMARY KEY (model, task_type)
);

-- Learned routing rules
CREATE TABLE routing_rules (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    rule_type TEXT NOT NULL CHECK (rule_type IN ('block', 'prefer', 'min_tier')),
    condition_json TEXT NOT NULL,
    action_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    expires_at TEXT,
    reason TEXT NOT NULL,
    is_active INTEGER NOT NULL DEFAULT 1
);

CREATE INDEX idx_rules_active ON routing_rules(is_active);
```

### 4.2 Pydantic Models

```python
class TaskType(str, Enum):
    CODING = "coding"
    REASONING = "reasoning"
    CREATIVE = "creative"
    SIMPLE_QA = "simple_qa"
    GENERAL = "general"

class ModelTier(str, Enum):
    FAST = "fast"
    MID = "mid"
    BEST = "best"

class ModelConfig(BaseModel):
    provider: str
    model_id: str
    strengths: list[str]
    tier: ModelTier
    # Populated from OpenRouter API:
    cost_per_1k_input: float | None = None
    cost_per_1k_output: float | None = None
    context_length: int | None = None

class RequestRecord(BaseModel):
    id: str
    timestamp: datetime
    prompt_hash: str
    prompt_text: str
    complexity: int
    task_type: TaskType
    routed_model: str
    routing_reasoning: str
    # ... full schema as per SQLite table
```

---

## 5. Routing Algorithm

### 5.1 Default Routing Logic

The triage model receives the complexity scale and model pool, then selects based on:

```
Complexity 1-2  →  tier: fast   (Haiku, GPT-4o-mini)
Complexity 3-4  →  tier: mid    (Sonnet, GPT-4o)
Complexity 5    →  tier: best   (Opus)
```

**Task-Type Affinity:**
- `coding` → Prefer Claude models
- `creative` → Prefer GPT models or Claude
- `reasoning` → Prefer Claude Opus/Sonnet
- `simple_qa` → Prefer fastest available

### 5.2 Rule-Based Overrides

Learned rules inject constraints into the triage prompt:

```python
def build_constraints(rules: list[RoutingRule]) -> str:
    lines = []
    for rule in rules:
        if rule.rule_type == "block":
            lines.append(f"- Do not select {rule.action['model']} for {rule.condition['task_type']} tasks")
        elif rule.rule_type == "prefer":
            lines.append(f"- Prefer {rule.action['model']} for {rule.condition['task_type']} tasks")
        elif rule.rule_type == "min_tier":
            lines.append(f"- For {rule.condition['task_type']} tasks, minimum tier is {rule.action['tier']}")
    return "\n".join(lines)
```

### 5.3 Override Behavior

When user specifies `--model`:
- Triage is skipped entirely
- Request logged with `routed_model = None`, `executed_model = <override>`
- Still contributes to research data (useful for manual A/B testing)

---

## 6. Adaptive Learning System

### 6.1 Learning Loop

```
Request Completed
       │
       ▼
┌─────────────────┐
│ Update Stats    │  model_stats[model, task_type] += quality_score
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Evaluate Rules  │  Check if any threshold crossed
└────────┬────────┘
         │
    ┌────┴────┐
    │ Change? │
    └────┬────┘
    yes  │  no
    ▼    ▼
┌───────┐ (done)
│ Apply │
│ Rule  │
└───────┘
```

### 6.2 Rule Triggers

| Condition | Threshold | Action |
|-----------|-----------|--------|
| Model underperforms on task type | avg_quality < 5.0, n ≥ 20 | Block model for task type |
| Model excels on task type | avg_quality > 8.5, n ≥ 20 | Allow in lower complexity tiers |
| Baseline consistently superior | quality_gap > 1.5, n ≥ 30 | Increase minimum tier for task type |
| Routing overhead exceeds savings | triage_cost > savings, n ≥ 50 | Flag for review |

### 6.3 Rule Lifecycle

- **Creation:** Automatic when threshold crossed
- **Expiration:** 30 days after creation (forces re-evaluation)
- **Override:** User can manually disable via `conductor rules --disable <id>`
- **Logging:** All changes recorded with timestamp and reasoning

### 6.4 Statistical Safeguards

- **Minimum sample size:** No rules created with n < 20
- **Confidence intervals:** Future versions may incorporate Bayesian updating
- **Decay factor:** Older samples weighted less (recency bias)

---

## 7. Research Methodology

### 7.1 Core Research Question

> Does intelligent semantic routing reduce LLM operational costs while maintaining acceptable output quality, compared to a strategy of always using the most capable model?

### 7.2 Experimental Design

**Control Group:** Always-Opus baseline (captured via shadow sampling)

**Treatment Group:** Conductor-routed requests

**Metrics:**
| Metric | Definition |
|--------|------------|
| Cost Savings | `(baseline_cost - routed_cost) / baseline_cost` |
| Quality Delta | `baseline_auto_judge - routed_auto_judge` |
| Routing Overhead | `triage_cost / total_cost` |
| Net Savings | `cost_savings - routing_overhead` |

### 7.3 Data Collection

**Shadow Sampling:**
- 10% of requests (configurable) also execute against baseline model
- Both responses scored by AutoJudge
- Enables direct comparison on identical prompts

**Deliberate Benchmarking:**
```bash
conductor bench --suite evaluation_prompts.jsonl
```

Suite format:
```json
{"prompt": "...", "category": "coding-easy", "expected_complexity": 2}
{"prompt": "...", "category": "reasoning-hard", "expected_complexity": 5}
```

### 7.4 Analysis Outputs

**`conductor compare` generates:**

1. **Summary Statistics**
   - Total cost: routed vs baseline
   - Average quality: routed vs baseline
   - Net savings percentage

2. **Breakdown by Task Type**
   - Which categories benefit most from routing?
   - Where does routing fail?

3. **Routing Accuracy**
   - Complexity prediction vs AutoJudge scores
   - Model selection vs optimal hindsight selection

**Export:** `conductor compare --export csv` for external analysis.

### 7.5 Validity Considerations

**Threats to validity:**
- AutoJudge bias (same model family scoring outputs)
- Shadow sampling may miss edge cases
- OpenRouter latency variance

**Mitigations:**
- Human ratings as ground truth validation
- Sufficient sample sizes before conclusions
- Latency measured end-to-end, reported separately

---

## 8. Command-Line Interface

### 8.1 Command Reference

#### `conductor ask <prompt>`

Execute a routed request.

```
$ conductor ask "Implement a binary search in Python"

Routing: complexity=2, task_type=coding → anthropic/claude-3.5-haiku
Cost: $0.0012 (triage: $0.0003 + execution: $0.0009)
Latency: 847ms

def binary_search(arr: list, target: int) -> int:
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1
```

**Flags:**
| Flag | Description |
|------|-------------|
| `--model <model>` | Skip routing, use specified model |
| `--no-shadow` | Disable shadow sampling for this request |
| `--json` | Output full result as JSON |
| `--quiet` | Suppress routing info, output only response |

#### `conductor bench <prompt>`

Run comparative benchmark: routed vs baseline.

```
$ conductor bench "Explain the CAP theorem in distributed systems"

Running routed path...  → anthropic/claude-3.5-haiku (complexity=2)
Running baseline...     → anthropic/claude-opus-4

Results:
┌─────────────┬────────────────┬─────────────────┐
│             │ Routed (haiku) │ Baseline (opus) │
├─────────────┼────────────────┼─────────────────┤
│ Quality     │ 7.2/10         │ 9.1/10          │
│ Cost        │ $0.0008        │ $0.0340         │
│ Latency     │ 412ms          │ 2,847ms         │
└─────────────┴────────────────┴─────────────────┘

Routing saved $0.033 (97%) with -1.9 quality delta
```

**Flags:**
| Flag | Description |
|------|-------------|
| `--suite <file>` | Batch benchmark from JSONL file |
| `--parallel <n>` | Concurrent requests (default: 3) |

#### `conductor stats`

Display usage statistics.

```
$ conductor stats

═══════════════════════════════════════════════════════
                    CONDUCTOR STATS
═══════════════════════════════════════════════════════

Period: 2026-04-13 → 2026-05-13 (30 days)
Total Requests: 1,247

COST ANALYSIS
─────────────────────────────────────────────────────────
  Execution Cost:     $13.58
  Triage Overhead:    $1.24 (8.4%)
  Total Cost:         $14.82

MODEL DISTRIBUTION
─────────────────────────────────────────────────────────
  anthropic/claude-3.5-haiku  │ 623 reqs │ $2.14  │ ★ 6.8
  anthropic/claude-sonnet-4   │ 401 reqs │ $8.92  │ ★ 8.1
  anthropic/claude-opus-4     │ 223 reqs │ $3.52  │ ★ 8.9

TASK TYPE BREAKDOWN
─────────────────────────────────────────────────────────
  coding      │ 412 reqs │ 33%
  reasoning   │ 298 reqs │ 24%
  simple_qa   │ 287 reqs │ 23%
  creative    │ 156 reqs │ 13%
  general     │  94 reqs │  7%
```

**Flags:**
| Flag | Description |
|------|-------------|
| `--rules` | Show active learned routing rules |
| `--learning` | Show recent rule changes |
| `--days <n>` | Limit to last n days (default: 30) |

#### `conductor compare`

Research output: routed vs always-best comparison.

```
$ conductor compare

═══════════════════════════════════════════════════════
              ROUTING EFFICIENCY ANALYSIS
═══════════════════════════════════════════════════════

Based on 847 shadow-sampled requests

STRATEGY COMPARISON
─────────────────────────────────────────────────────────
                    │ Conductor │ Always-Opus │ Delta
────────────────────┼───────────┼─────────────┼─────────
Total Cost          │ $14.82    │ $89.20      │ -83.4%
Avg Quality         │ 7.4/10    │ 8.7/10      │ -1.3
Avg Latency         │ 923ms     │ 2,341ms     │ -60.6%
────────────────────┴───────────┴─────────────┴─────────

VERDICT: Routing saved $74.38 with 1.3 quality tradeoff

BREAKDOWN BY TASK TYPE
─────────────────────────────────────────────────────────
Task Type   │ Savings │ Quality Δ │ Verdict
────────────┼─────────┼───────────┼────────────────────
simple_qa   │ 94%     │ -0.4      │ Routing optimal
coding      │ 71%     │ -1.8      │ Routing beneficial
creative    │ 68%     │ -2.1      │ Consider higher tier
reasoning   │ 45%     │ -1.1      │ Routing beneficial
```

**Flags:**
| Flag | Description |
|------|-------------|
| `--export csv` | Export full dataset to CSV |
| `--export json` | Export full dataset to JSON |

#### `conductor models`

List configured model pool.

```
$ conductor models

═══════════════════════════════════════════════════════
                  CONFIGURED MODELS
═══════════════════════════════════════════════════════

Model                       │ Tier │ Strengths           │ $/1K in  │ $/1K out
────────────────────────────┼──────┼─────────────────────┼──────────┼─────────
anthropic/claude-opus-4     │ best │ reasoning, coding   │ $0.0150  │ $0.0750
anthropic/claude-sonnet-4   │ mid  │ coding, reasoning   │ $0.0030  │ $0.0150
anthropic/claude-3.5-haiku  │ fast │ simple-qa, fast     │ $0.0008  │ $0.0040
openai/gpt-4o               │ mid  │ general, creative   │ $0.0050  │ $0.0150

Router Model: anthropic/claude-sonnet-4
Baseline Model: anthropic/claude-opus-4
Shadow Sample Rate: 10%
```

#### `conductor rate <request_id> <up|down>`

Provide human feedback on a past result.

```
$ conductor rate 7f3a2b1c up
Recorded: thumbs up for request 7f3a2b1c
```

---

## 9. Configuration

### 9.1 Configuration File

**Location:** `~/.config/conductor/conductor.yaml` (or `./conductor.yaml`)

```yaml
# ═══════════════════════════════════════════════════════
#                 CONDUCTOR CONFIGURATION
# ═══════════════════════════════════════════════════════

# OpenRouter API (single key for all models)
openrouter_api_key: ${OPENROUTER_API_KEY}

# Routing configuration
router_model: anthropic/claude-sonnet-4
baseline_model: anthropic/claude-opus-4
shadow_sample_rate: 0.1  # 10% of requests get baseline comparison

# Model pool
# Costs are fetched from OpenRouter API; define tiers and strengths here
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

# Paths
database_path: ~/.config/conductor/conductor.db
```

### 9.2 API Key Resolution

**Priority order:**
1. Environment variable: `OPENROUTER_API_KEY`
2. Config file value (supports `${VAR}` syntax)
3. Encrypted keystore (future)

**Security:**
- Never commit API keys to version control
- Config file should be in `.gitignore`
- CLI warns if keys detected in git-tracked files

### 9.3 Environment Variables

| Variable | Description |
|----------|-------------|
| `OPENROUTER_API_KEY` | OpenRouter API key (required) |
| `CONDUCTOR_CONFIG` | Override config file path |
| `CONDUCTOR_DB` | Override database path |
| `CONDUCTOR_DEBUG` | Enable debug logging |

---

## 10. Future Work

### 10.1 Version 2 Roadmap

#### Trained Routing Classifier

**Motivation:** Reduce triage overhead from ~8% to <1%.

**Approach:**
1. Collect labeled dataset from v1 usage (prompt → optimal_model)
2. Train lightweight classifier (embedding + logistic regression, or small fine-tuned model)
3. Replace LLM triage with local inference

**Prerequisites:**
- Minimum 10,000 labeled examples
- Validated AutoJudge accuracy against human ratings

#### Direct Provider APIs

**Motivation:** Eliminate OpenRouter markup (~10-20%).

**Implementation:**
```yaml
providers:
  anthropic:
    api_key: ${ANTHROPIC_API_KEY}
    direct: true
  openai:
    api_key: ${OPENAI_API_KEY}
    direct: true
  openrouter:
    api_key: ${OPENROUTER_API_KEY}
    fallback: true  # Use for models without direct access
```

**Complexity:** Requires provider-specific API handling, unified error mapping.

### 10.2 Extended Features

| Feature | Description | Priority |
|---------|-------------|----------|
| Streaming | Stream responses with `--stream` flag | High |
| Conversation context | Multi-turn routing (consider history) | Medium |
| Cost budgets | `--max-cost $0.05` constraint | Medium |
| A/B testing | Compare Conductor vs OpenRouter Pareto | Low |
| Web UI | Dashboard for stats and research analysis | Low |
| Team sharing | Shared database for team-wide learning | Low |

### 10.3 Research Extensions

- **Cross-model AutoJudge:** Use different model family for judging to reduce bias
- **Human-in-the-loop calibration:** Periodic human rating to validate AutoJudge
- **Prompt embedding analysis:** Cluster similar prompts to identify routing patterns
- **Temporal analysis:** Track routing efficiency over time as models update

---

## 11. Appendices

### Appendix A: Technology Stack

| Component | Technology | Rationale |
|-----------|------------|-----------|
| Language | Python 3.11+ | Ecosystem, async support |
| CLI | Click | Clean interface, composable |
| HTTP | httpx | Async support, modern API |
| Database | SQLite | Zero config, portable |
| Validation | Pydantic | Type safety, serialization |
| Config | PyYAML | Human-readable, standard |

### Appendix B: OpenRouter API Reference

**Endpoint:** `https://openrouter.ai/api/v1/chat/completions`

**Request:**
```json
{
  "model": "anthropic/claude-sonnet-4",
  "messages": [
    {"role": "user", "content": "..."}
  ],
  "max_tokens": 4096,
  "temperature": 0
}
```

**Response:**
```json
{
  "id": "gen-...",
  "model": "anthropic/claude-sonnet-4",
  "choices": [
    {"message": {"content": "..."}}
  ],
  "usage": {
    "prompt_tokens": 150,
    "completion_tokens": 500,
    "total_tokens": 650
  }
}
```

**Models Endpoint:** `GET /api/v1/models` returns pricing and capabilities.

### Appendix C: Example Research Dataset Export

```csv
request_id,timestamp,prompt_hash,task_type,complexity,routed_model,routed_cost,routed_quality,baseline_model,baseline_cost,baseline_quality,cost_savings,quality_delta
7f3a2b1c,2026-05-13T10:30:00Z,a1b2c3...,coding,2,claude-3.5-haiku,0.0012,7.2,claude-opus-4,0.0340,9.1,0.97,-1.9
8e4d3c2b,2026-05-13T10:31:00Z,d4e5f6...,reasoning,4,claude-sonnet-4,0.0089,8.4,claude-opus-4,0.0280,8.9,0.68,-0.5
...
```

---

## Document History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2026-05-13 | - | Initial specification |

---

*This document serves as the authoritative specification for Conductor v1. Implementation should follow this design; deviations require spec updates.*
