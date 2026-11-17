# Task 2 Brief: Notebook Construction Engine & Capture Runner

## Target
Implement `scripts/build_unified_coursework_suite.py` and unit test `tests/test_unified_suite_builder.py`.

## Core Requirements
1. **Engine Architecture (`scripts/build_unified_coursework_suite.py`)**:
   - Standalone script using standard library (`json`, `io`, `base64`, `sys`, `pathlib`) and installed dependencies (`IPython`, `matplotlib`, `torch`).
   - Uses `IPython.core.interactiveshell.InteractiveShell.instance()` to execute code cells in order within the repository root context.
   - Captures stdout (`io.StringIO` redirect / sys.stdout capture) and matplotlib figure canvas to PNG base64 strings.
   - Closes all matplotlib figures with `plt.close("all")` after capture.
   - Configures matplotlib figure size and DPI (120-130) to keep base64 payloads under 2.5MB per notebook.
   - Provides clean helper functions:
     - `create_notebook(title: str, subtitle: str, part_num: int, cells: list) -> dict`
     - `make_header_cell(title: str, subtitle: str, part_num: int, time_str: str) -> dict`
     - `make_roadmap_cell(current_part: int) -> dict`
     - `make_markdown_cell(source: str) -> dict`
     - `make_code_cell(source: str, shell: InteractiveShell, execute: bool = True) -> dict`
     - `save_notebook(nb_dict: dict, path: Path)`
   - Blueprint compliance:
     - Cell 0: Header with Topic, Group, Part info, Date, Objective.
     - Cell 1: 4-column Roadmap Markdown table linking the 4 notebooks with relative paths. Active notebook row highlighted.
     - Cells 2+: Code & narrative cells.
2. **Testing (`tests/test_unified_suite_builder.py`)**:
   - Marked `@pytest.mark.smoke`.
   - Tests `InteractiveShell` execution capture (verifies stdout and figure display_data outputs).
   - Tests `create_notebook` output schema compliance (`nbformat == 4`, `cells`, `metadata`).
   - Tests that `plt.close("all")` is called and no dangling figures remain.
   - Tests roadmap table generation with correct relative paths.

## Execution Rules
- Follow TDD: write failing test `tests/test_unified_suite_builder.py`, confirm RED, implement `scripts/build_unified_coursework_suite.py`, confirm GREEN.
- Run `make test-smoke`.
- Zero absolute paths, zero mock data.
