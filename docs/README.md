# Project Documentation Index (`/docs`)

- **Motivation/Background**: Maintain a single discoverable portal for all evolving research memory, phase specifications, progress tracking, technical reports, and presentation materials for the refactored UTH Computer Vision Coursework on Stable Diffusion Synthetic Deepfake Generation, `dinoViT` Baseline Benchmarking, and Hard-Sample Mining.
- **Purpose**: Map out the documentation structure and establish the relationship between `/docs` and the immutable governance in `/.agents`.
- **Overview Pipeline**: Project lifecycle documentation flows from Purpose & Overview (`PURPOSE.md`, `OVERVIEW.md`) -> 3 Phase Specs (`phases/`) -> Controlled Experiment Protocol (`experiments/`) -> Technical Reports & Slide Deck (`reports/`).
- **Detailed Plan**: §1 Distinction Between Agents and Docs; §2 Directory Map; §3 Documentation Standards; §4 Report Requirements (Dual MD + HTML).
- **References**: `.agents/rules/MD_CONVENTION.md`, `.agents/rules/FOLDER_STRUCTURE.md`, `.agents/AI_REFERENCE.md`, `docs/PURPOSE.md`.
- **Created**: 2026-11-02T07:22:00+07:00
- **Last Updated**: 2026-11-08T17:45:00+07:00

---

## 1. Distinction Between `/.agents` and `/docs`

- **`/.agents` (The Immutable Constitution)**: Universal behavior rules, quality gates, coding conventions, and document templates. Never accumulates evolving project outputs.
- **`/docs` (The Evolving Memory)**: Active project specifications, roadmap, 3-phase technical specs (Phase 1 SD Generation, Phase 2 `dinoViT` Hard-Sample Mining, Phase 3 Fine-Tuning & Proof of Utility), experiment protocols, and presentation slide decks.

---

## 2. Directory Map

```text
docs/
├── README.md                 # Master documentation index (this file)
├── PURPOSE.md                # Project mission statement, closed-loop dinoViT + Hard Sample problem definition & KPIs
├── OVERVIEW.md               # Living roadmap, milestones & deliverables tracker
├── coursework/               # Literature review (GAN vs Diffusion + DINO ViT) & data schema specification
│   ├── LITERATURE_REVIEW.md
│   └── TAXONOMY_AND_DATA_SPEC.md
├── phases/                   # Technical phase specifications
│   ├── PHASE_01_DIFFUSION_SYNTHESIS_SPEC.md
│   ├── PHASE_02_DINOVIT_HARD_SAMPLE_MINING_SPEC.md
│   └── PHASE_03_AUDIT_RISK_AND_QUALITY_SPEC.md
├── progress/                 # Macro roadmap & live milestone trackers
│   └── PROJECT_ROADMAP.md
├── experiments/              # Controlled ablation & baseline-vs-finetuned experiment protocol
│   └── EXPERIMENT_PROTOCOL.md
├── reports/                  # Comprehensive technical reports & 5-minute presentation slide deck
│   ├── PART_01_PROBLEM_DEFINITION.md
│   ├── PART_02_SYSTEM_ARCHITECTURE.md
│   ├── SYSTEM_DEFINE_AND_DESIGN_REPORT.md
│   ├── PRESENTATION_5MIN_DEFINE_AND_DESIGN.md
│   ├── SLIDE_PRESENTATION_STRUCTURE_5MIN.md
│   ├── PRESENTATION_DEFENSE_QA_CHEAT_SHEET.md
│   └── presentation_5min_define_and_design.html
├── shared/                   # Shared SOPs & developer handoffs
│   └── .gitkeep
└── assets/                   # Architecture diagrams, benchmark figures & visual plots
    └── .gitkeep
```

---

## 3. Documentation Standards

1. **7-Field Header**: Every `.md` document in `docs/` must begin with the standard 7-field header block.
2. **Naming Convention**: File names in `docs/` must be ALL CAPS (`UPPER_SNAKE_CASE.md` or `UPPERCASE.md`), except interactive HTML decks (`presentation_5min_define_and_design.html`).
3. **Zero Absolute Paths**: All links across documentation must use relative paths.
