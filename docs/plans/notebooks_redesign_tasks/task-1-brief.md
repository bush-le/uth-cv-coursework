### Task 1: Archive Legacy Notebooks & Prepare Architecture Blueprint

**Files:**
- Create: `notebooks/coursework/legacy_v1/README.md`
- Move: `notebooks/coursework/00_master_deepfake_defense_coursework.ipynb` -> `notebooks/coursework/legacy_v1/`
- Move: `notebooks/coursework/01_eda_faces_demographics.ipynb` -> `notebooks/coursework/legacy_v1/`
- Move: `notebooks/coursework/02_sd_generation_and_curation.ipynb` -> `notebooks/coursework/legacy_v1/`
- Move: `notebooks/coursework/03_hard_samples_and_benchmark_comparison.ipynb` -> `notebooks/coursework/legacy_v1/`
- Move: `notebooks/coursework/04_active_learning_production_visualizer.ipynb` -> `notebooks/coursework/legacy_v1/`
- Move: `notebooks/coursework/05_synthetic_face_6k_realimage_benchmark.ipynb` -> `notebooks/coursework/legacy_v1/`
- Move: `notebooks/coursework/06_specialization_vs_generalization_synthesis.ipynb` -> `notebooks/coursework/legacy_v1/`
- Move: `notebooks/coursework/07_universal_deepfake_benchmark.ipynb` -> `notebooks/coursework/legacy_v1/`
- Move: `notebooks/coursework/08_champion_m3_dual_engine_quad_vault_benchmark.ipynb` -> `notebooks/coursework/legacy_v1/`
- Move: `notebooks/coursework/generator_realistic_vision_kaggle.ipynb` -> `notebooks/coursework/legacy_v1/`
- Test: `tests/test_notebook_migration.py`

**Interfaces:**
- Consumes: Existing files in `notebooks/coursework/`
- Produces: Safe legacy storage `notebooks/coursework/legacy_v1/` with 10 archived notebook files and clear documentation.

- [ ] **Step 1: Write the failing test for legacy migration**

```python
# tests/test_notebook_migration.py
from pathlib import Path

def test_legacy_v1_directory_structure():
    legacy_dir = Path("notebooks/coursework/legacy_v1")
    assert legacy_dir.is_dir(), "legacy_v1 directory must exist"
    readme_path = legacy_dir / "README.md"
    assert readme_path.is_file(), "legacy_v1/README.md must exist"
    
    expected_files = [
        "00_master_deepfake_defense_coursework.ipynb",
        "01_eda_faces_demographics.ipynb",
        "02_sd_generation_and_curation.ipynb",
        "03_hard_samples_and_benchmark_comparison.ipynb",
        "04_active_learning_production_visualizer.ipynb",
        "05_synthetic_face_6k_realimage_benchmark.ipynb",
        "06_specialization_vs_generalization_synthesis.ipynb",
        "07_universal_deepfake_benchmark.ipynb",
        "08_champion_m3_dual_engine_quad_vault_benchmark.ipynb",
        "generator_realistic_vision_kaggle.ipynb",
    ]
    for fname in expected_files:
        assert (legacy_dir / fname).is_file(), f"Missing archived file: {fname}"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_notebook_migration.py -v`
Expected: FAIL with "legacy_v1 directory must exist"

- [ ] **Step 3: Move notebooks and create legacy README.md**

Move all 10 legacy `.ipynb` files to `notebooks/coursework/legacy_v1/` and write `notebooks/coursework/legacy_v1/README.md` explaining the legacy version history and the transition to the unified 4-notebook suite.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_notebook_migration.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add notebooks/coursework/ tests/test_notebook_migration.py
git commit -m "chore(notebooks): archive legacy coursework notebooks to legacy_v1"
```

---

