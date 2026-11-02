# Project Documentation Index (`/docs`)

- **Motivation/Background**: Maintain a single discoverable portal for all evolving research memory, phase specifications, progress tracking, technical reports, and operational guides.
- **Purpose**: Map out the documentation structure and establish the relationship between `/docs` and the immutable governance in `/.agents`.
- **Overview Pipeline**: Project lifecycle documentation flows from Phase Specs (`phases/`) -> Live Task Tracking (`progress/`) -> Technical Milestone Reports (`reports/` / `coursework/`).
- **Detailed Plan**: §1 Distinction Between Agents and Docs; §2 Directory Map; §3 Documentation Standards; §4 Report Requirements (Dual MD + HTML).
- **References**: `.agents/rules/MD_CONVENTION.md`, `.agents/rules/FOLDER_STRUCTURE.md`, `.agents/AI_REFERENCE.md`.
- **Created**: 2026-11-02T07:22:00+07:00
- **Last Updated**: 2026-11-02T07:22:00+07:00

---

## 1. Distinction Between `/.agents` and `/docs`

- **`/.agents` (The Immutable Constitution)**: Universal behavior rules, quality gates, coding conventions, and document templates. Never accumulates evolving project outputs.
- **`/docs` (The Evolving Memory)**: Active project plans, roadmap, phase specifications, progress tracking, experiment logs, and academic/coursework reports.

---

## 2. Directory Map

```text
docs/
├── README.md                 # Master documentation index (this file)
├── PURPOSE.md                # Project mission statement, problem definition & success criteria
├── OVERVIEW.md               # Living roadmap, milestones & deliverables tracker
├── coursework/               # Coursework topic specifications, rubric & submission artifacts
│   └── .gitkeep
├── phases/                   # Technical phase specifications (01_DATA_PREP.md, 02_BASELINE.md, etc.)
│   └── .gitkeep
├── progress/                 # Live Kanban trackers (01_DATA_PREP_STATUS.md, etc.)
│   └── .gitkeep
├── experiments/              # Empirical test logs & 5W1H experiment reports
│   └── .gitkeep
├── reports/                  # Comprehensive technical reports (Markdown + Interactive HTML)
│   └── .gitkeep
├── shared/                   # Shared SOPs, developer onboarding manuals & handoffs
│   └── .gitkeep
└── assets/                   # Architecture diagrams, benchmark figures & visual plots
    └── .gitkeep
```

---

## 3. Documentation Standards

1. **7-Field Header**: Every `.md` document in `docs/` must begin with the standard 7-field header block.
2. **Naming Convention**: File names in `docs/` must be ALL CAPS (`UPPER_SNAKE_CASE.md` or `UPPERCASE.md`).
3. **Dual MD + HTML Report Policy**: Final technical reports must provide both Markdown (`REPORT.md`) for Git diffing and HTML (`REPORT.html`) for interactive visual inspection of galleries, charts, and metrics.
