#!/usr/bin/env python3
"""Add mandatory 7-field headers to the 13 root-level governance documents in .agents/."""

from pathlib import Path

HEADERS = {
    "AGENT_INTERACTION_LOOP.md": """- **Motivation/Background**: Defines the micro-level step-by-step collaboration loop between the Human Engineer and the AI Agent.
- **Purpose**: Ensure deterministic, test-driven, contract-first pair programming without hallucinations or regressions.
- **Overview Pipeline**: Task Intake -> Contract Formulation -> Failing Test -> Minimal Vectorized Code -> Smoke Verification -> Atomic Commit.
- **Detailed Plan**: §1 The 7-Step Micro Pair Loop; §2 Failure Recovery Protocol; §3 DoD Checklist; §4 Communication Best Practices.
- **References**: `README.md`, `AI_FIRST_LIFECYCLE.md`, `rules/TEST_BEFORE_CODE_RULE.md`.
- **Created**: 2026-11-02T07:30:00+07:00
- **Last Updated**: 2026-11-11T23:25:00+07:00""",

    "AI_FIRST_LIFECYCLE.md": """- **Motivation/Background**: Complex CV/Deepfake systems require an end-to-end disciplined lifecycle from problem framing to deployment.
- **Purpose**: Specify the 9 macro engineering phases governing project evolution.
- **Overview Pipeline**: Phase 1 Problem Definition -> Phase 2 Dataset Factory -> Phase 3 Baseline Model -> Phase 4 Deep Architecture -> Phase 5 Hard Mining -> Phase 6 Active Learning -> Phase 7 OOD Benchmark -> Phase 8 Production Packaging -> Phase 9 Defense Presentation.
- **Detailed Plan**: §1 Macro Lifecycle Map; §2 Phase-by-Phase Deliverables & Gates; §3 Milestone Checkpoints; §4 Governance Rules.
- **References**: `README.md`, `WORKFLOW_OVERVIEW.md`, `PROGRESS_TRACKING.md`.
- **Created**: 2026-11-02T07:30:00+07:00
- **Last Updated**: 2026-11-11T23:25:00+07:00""",

    "AI_REFERENCE.md": """- **Motivation/Background**: AI agents require an unambiguous, consolidated reference manual for domain invariants and technical formulas.
- **Purpose**: Provide the authoritative domain dictionary, mathematical formulas, and tensor contracts for Computer Vision and Deepfake Defense.
- **Overview Pipeline**: Domain Invariants Ingestion -> Tensor Contracts -> Metrics Formulation -> Architectural Reference.
- **Detailed Plan**: §1 5 CV Domain Invariants; §2 Mathematical Formulas (AUC, EER, Contrastive Loss); §3 Architecture Specifications (DINOv2, SD 1.5); §4 Common Gotchas.
- **References**: `../GEMINI.md`, `rules/ZERO_MOCK_DATA_RULE.md`, `rules/COLOR_SPACE_RULE.md`.
- **Created**: 2026-11-02T07:30:00+07:00
- **Last Updated**: 2026-11-11T23:25:00+07:00""",

    "BEHAVIORAL_LAYER.md": """- **Motivation/Background**: Autonomous agents must maintain rigorous engineering discipline, zero-assumption mindset, and fail-fast behavior.
- **Purpose**: Define behavioral invariants, persona traits, and operational guardrails for AI coding assistants.
- **Overview Pipeline**: Persona Alignment -> Operational Axioms -> Conflict Resolution -> Output Structuring.
- **Detailed Plan**: §1 Cognitive Persona; §2 Zero-Guesswork Protocol; §3 Fail-Fast Invariant; §4 Quality Assurance Standards.
- **References**: `README.md`, `AGENT_INTERACTION_LOOP.md`, `../GEMINI.md`.
- **Created**: 2026-11-02T07:30:00+07:00
- **Last Updated**: 2026-11-11T23:25:00+07:00""",

    "CODEBASE_AUDIT.md": """- **Motivation/Background**: Continuous verification of architectural boundaries and code hygiene prevents technical debt accumulation.
- **Purpose**: Guide comprehensive codebase inspection, boundary auditing, and structural refactoring.
- **Overview Pipeline**: Static AST Analysis -> Rule Verification -> Clean Architecture Auditing -> Automated Remediation.
- **Detailed Plan**: §1 Audit Dimensions; §2 Architectural Separation Check; §3 Zero-Hardcode & Zero-Mock Verification; §4 Reporting Format.
- **References**: `rules/ZERO_HARDCODE_RULE.md`, `rules/ZERO_MOCK_DATA_RULE.md`, `PIPELINE_SEPARATION.md`.
- **Created**: 2026-11-02T07:30:00+07:00
- **Last Updated**: 2026-11-11T23:25:00+07:00""",

    "CONTEXT_MANAGEMENT.md": """- **Motivation/Background**: Long-horizon coding tasks suffer from context drift and window exhaustion if memory is unmanaged.
- **Purpose**: Establish protocols for context compaction, memory offloading to documentation, and Second Brain navigation.
- **Overview Pipeline**: Working Memory Ingestion -> Periodic Compaction -> Artifact Offloading -> Epistemic Retrieval.
- **Detailed Plan**: §1 Context Hierarchy; §2 Markdown Memory Anchoring; §3 Subagent Handoff Patterns; §4 State Restoration.
- **References**: `README.md`, `PROGRESS_TRACKING.md`, `../docs/README.md`.
- **Created**: 2026-11-02T07:30:00+07:00
- **Last Updated**: 2026-11-11T23:25:00+07:00""",

    "PIPELINE_SEPARATION.md": """- **Motivation/Background**: Monolithic code inside Jupyter Notebooks causes silent state corruption, non-reproducibility, and testing failures.
- **Purpose**: Enforce the absolute architectural separation between modular library logic (`src/`), executable pipelines (`src/pipelines/`, `scripts/`), and visualization notebooks (`notebooks/`).
- **Overview Pipeline**: Logic Modularization (`src/`) -> Script Pipeline Execution -> Visual Inspection in Notebooks (`notebooks/`).
- **Detailed Plan**: §1 Script vs Notebook Boundary; §2 Prohibited Patterns; §3 Notebook Client Paradigm; §4 Enforcement Mechanisms.
- **References**: `../GEMINI.md`, `rules/NOTEBOOK_INSPECTION_RULE.md`, `rules/SCRIPT_TRAINING_RULE.md`.
- **Created**: 2026-11-02T07:30:00+07:00
- **Last Updated**: 2026-11-11T23:25:00+07:00""",

    "PROGRESS_TRACKING.md": """- **Motivation/Background**: Transparent, verifiable task progress is required to maintain momentum across extended development sessions.
- **Purpose**: Define standards for task breakdown, progress ledgers, milestone tracking, and phase sign-offs.
- **Overview Pipeline**: Roadmap Ingestion -> Task Ledger Logging -> Smoke Verification -> Milestone Sign-off.
- **Detailed Plan**: §1 Task Tracking Taxonomy; §2 Progress Ledger Formatting; §3 Definition of Done; §4 Rollback Protocol.
- **References**: `AI_FIRST_LIFECYCLE.md`, `../docs/progress/TASK_PROGRESS.md`, `../docs/progress/PROJECT_ROADMAP.md`.
- **Created**: 2026-11-02T07:30:00+07:00
- **Last Updated**: 2026-11-11T23:25:00+07:00""",

    "README.md": """- **Motivation/Background**: Establishes the cognitive operating system, architectural constraints, and lifecycle workflows for AI agents operating in the repository.
- **Purpose**: Serve as the root orientation and Second Brain guide for autonomous and paired AI agents.
- **Overview Pipeline**: Cognitive Protocol Ingestion -> Context Anchoring -> Task Routing -> Contract Output -> Verification.
- **Detailed Plan**: §1 Agent Cognitive Protocol; §2 Second Brain Directory Architecture; §3 9-Phase Macro Engineering Lifecycle; §4 7-Step Micro Pair Programming Loop; §5 Immutable Rules & Quality Gates.
- **References**: `../GEMINI.md`, `rules/`, `templates/`.
- **Created**: 2026-11-02T07:30:00+07:00
- **Last Updated**: 2026-11-11T23:25:00+07:00""",

    "TASK_DESCRIPTION.md": """- **Motivation/Background**: Vague task definitions lead to scope creep, mismatched assumptions, and broken tests.
- **Purpose**: Specify the standard schema for task descriptions, input/output contracts, and acceptance criteria.
- **Overview Pipeline**: Scope Identification -> Tensor Contract Definition -> Granular Plan -> DoD Checklist.
- **Detailed Plan**: §1 Task Description Schema; §2 Contract-First Requirements; §3 Acceptance Verification; §4 Examples.
- **References**: `AGENT_INTERACTION_LOOP.md`, `rules/CONTRACT_FIRST_RULE.md`.
- **Created**: 2026-11-02T07:30:00+07:00
- **Last Updated**: 2026-11-11T23:25:00+07:00""",

    "TESTING_STRATEGY.md": """- **Motivation/Background**: Deep learning codebases without fast, deterministic testing degrade rapidly through silent regression.
- **Purpose**: Define the testing pyramid, smoke test gate (< 5s), numerical tolerance thresholds, and mock isolation.
- **Overview Pipeline**: Unit Tests -> Smoke Tests (< 5s) -> Integration Pipelines -> Checkpoint Verification.
- **Detailed Plan**: §1 Testing Pyramid; §2 Fast Smoke Test Gate; §3 Tensor Shape & Numerical Assertions; §4 CI/CD Makefile Targets.
- **References**: `../GEMINI.md`, `rules/TEST_BEFORE_CODE_RULE.md`, `rules/ZERO_MOCK_DATA_RULE.md`.
- **Created**: 2026-11-02T07:30:00+07:00
- **Last Updated**: 2026-11-11T23:25:00+07:00""",

    "TROUBLESHOOTING.md": """- **Motivation/Background**: Machine learning development involves subtle numerical instabilities, CUDA errors, and shape mismatches.
- **Purpose**: Provide a fast-lookup diagnostic manual for common failure modes and deterministic remedies.
- **Overview Pipeline**: Symptom Recognition -> Root Cause Isolation -> Minimal Test Reproduction -> Verified Remedy.
- **Detailed Plan**: §1 CUDA & Memory Out-of-Memory; §2 Shape & Color Space Mismatches; §3 Training Loss NaN/Divergence; §4 Git Sandbox Issues.
- **References**: `AI_REFERENCE.md`, `rules/COLOR_SPACE_RULE.md`.
- **Created**: 2026-11-02T07:30:00+07:00
- **Last Updated**: 2026-11-11T23:25:00+07:00""",

    "WORKFLOW_OVERVIEW.md": """- **Motivation/Background**: High-level overview of the entire repository workflow, connecting human developers, agents, tools, and documentation.
- **Purpose**: Provide a macro view of the Second Brain ecosystem and engineering operational mechanics.
- **Overview Pipeline**: Repository Orientation -> Second Brain Navigation -> Tool Execution -> Milestone Verification.
- **Detailed Plan**: §1 Ecosystem Map; §2 Core Workflows; §3 Quality Assurance Gates; §4 Governance Maintenance.
- **References**: `README.md`, `AI_FIRST_LIFECYCLE.md`, `../GEMINI.md`.
- **Created**: 2026-11-02T07:30:00+07:00
- **Last Updated**: 2026-11-11T23:25:00+07:00""",
}


def add_headers():
    agents_dir = Path(".agents")
    for filename, header in HEADERS.items():
        file_path = agents_dir / filename
        if not file_path.exists():
            print(f"Skipping missing: {file_path}")
            continue

        content = file_path.read_text(encoding="utf-8")
        if "- **Motivation/Background**:" in content:
            print(f"Already has header: {filename}")
            continue

        lines = content.split("\n")
        title_idx = 0
        for i, line in enumerate(lines):
            if line.startswith("# "):
                title_idx = i
                break

        new_content = (
            "\n".join(lines[: title_idx + 1])
            + "\n\n"
            + header
            + "\n\n---\n"
            + "\n".join(lines[title_idx + 1 :])
        )

        file_path.write_text(new_content, encoding="utf-8")
        print(f"Successfully added header to: {filename}")


if __name__ == "__main__":
    add_headers()
