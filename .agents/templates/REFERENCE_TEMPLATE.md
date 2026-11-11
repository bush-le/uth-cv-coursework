# [Reference Title] — Reference Guide Template

- **Motivation/Background**: Reusable knowledge, tool configurations, and architectural guides must follow a uniform, structured format across the repository.
- **Purpose**: Serve as a canonical template for creating reference guides in the project.
- **Overview Pipeline**: Ingestion of reference context -> Structuring core concepts -> Documenting implementation guidelines -> Reference verification.
- **Detailed Plan**: §1 Context & Purpose; §2 Core Concepts & Architectural Rules; §3 Technical Specifications & Code Patterns; §4 Operational Recipes & Edge Cases.
- **References**: `../rules/MD_CONVENTION.md`, `CREATE_FOLDER_STRUCTURE_TEMPLATE.md`.
- **Created**: 2026-09-06T21:05:00+07:00
- **Last Updated**: 2026-11-11T23:25:00+07:00

---

## 1. Context & Purpose

Document reusable architectural guides, external API quirks, tool configurations, dataset formats, or operational recipes here. When instantiated, save as `docs/references/<NAME>.md`.

- **Primary Purpose:** [Why does this reference exist and what problem or misunderstanding does it prevent?]
- **Target Audience:** [AI agents, human practitioners, or both]
- **Relevant Project Scope:** [Specific phases, models, or global project lifecycle]

---

## 2. Core Concepts & Architectural Rules

[Detail the core facts, architectural assumptions, invariants, and guidelines.]

---

## 3. Practical Implementation & Code Snippets

```python
# Provide concrete, production-grade snippets or CLI commands
```

---

## 4. Known Pitfalls & Edge Cases

| Anti-Pattern / Pitfall | Expected Failure | Correct Handling |
| :--- | :--- | :--- |
| [e.g., Unpinned dependency] | Silent build failure | Explicit pinning in requirements/base.txt |
| [e.g., Implicit CUDA assumptions] | Test crash on CPU CI | Conditional device detection (`get_device()`) |

---

## 5. Related Documentation

- Master References Index: [docs/references/README.md](../../docs/references/README.md)
- Constitutional Rules: [.agents/rules/CREATE_FOLDER_STRUCTURE_TEMPLATE.md](../rules/CREATE_FOLDER_STRUCTURE_TEMPLATE.md)
