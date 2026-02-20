# VIS Spinoff Workfile

Last updated: 2026-02-20

## Purpose
Shared progress log for the `vis` standalone spinoff so multiple agents can continue in small, safe batches.

## Current status
- Overall plan source: `vis/docs/VIS_SPINOFF_PLAN.md`
- Batch policy: small, incremental changes only (avoid full migration in one pass)

## Batch 1 (completed)
### Goal
Localize direct `concept_graphs` utility imports used by `vis_with_viser.py`.

### Changes made
1. Added local core package:
   - `vis/core/__init__.py`
   - `vis/core/seed.py`
   - `vis/core/color.py`
   - `vis/core/floor.py`
2. Rewired imports in `vis/vis_with_viser.py`:
   - `segment_floor_points` now from `vis.core.floor`
   - `set_seed` now from `vis.core.seed`
   - `similarities_to_rgb` now from `vis.core.color`

### Notes
- `vis/core/color.py` includes a safe degenerate-case handling for constant similarity vectors (`max == min`) and empty input.
- `sys.path.insert(...)` in `vis_with_viser.py` is intentionally untouched for now (planned in later batch).
- Sparse segmentation seam (`SparseSegmentMasker`) not yet implemented.

## Batch 2 (completed)
### Goal
Introduce `SparseSegmentMasker` and move segment splitting out of `ConceptGraphData` inline logic.

### Changes made
1. Added `vis/core/segmentation.py`:
   - `SparseSegmentMasker.split_objects(...)` implemented.
2. Updated `vis/cg_dataclass.py`:
   - replaced inline `_split_segments(...)` logic with `SparseSegmentMasker` seam.
3. Preserved fallback behavior contract:
   - no sparse cloud -> `[]`
   - missing/empty annotations -> `[sparse_pcd]`
   - empty/invalid/all-empty groups -> fallback `[sparse_pcd]`

### Notes
- Masker skips malformed group entries and non-list/empty `segments` values.
- `ConceptGraphData` external data contract is unchanged (`pcd_o3d`, `segments_anno`, etc.).

## Batch 3 (completed)
### Goal
Remove runtime path mutation and ensure module-based startup for standalone `vis` execution.

### Changes made
1. Updated `vis/vis_with_viser.py`:
   - removed `sys.path.insert(...)` startup path hack.
2. Startup/import validation:
   - `uv run python -c "import vis.vis_with_viser; print('ok')"` -> `ok`.
3. Dependency validation:
   - no `concept_graphs` references found in `vis/**/*.py` (only planning/work docs mention it).

### Notes
- `uv` warns that `requires-python ==3.10` is an exact minor version and suggests `==3.10.*`; this does not block runtime.

## Batch 4 (completed)
### Goal
Copy required runtime config dependencies into a spinoff-local config tree and verify Hydra composition without parent `conf/` coupling.

### Changes made
1. Added local config tree under `vis/conf/`:
   - `vis/conf/visualizer.yaml`
   - `vis/conf/paths/local.yaml`
   - `vis/conf/agent/agent.yaml`
   - `vis/conf/agent/llm/openai_gpt4.yaml`
2. Updated Hydra entrypoint in `vis/vis_with_viser.py`:
   - `config_path` switched from `../conf` -> `conf`.
3. Trimmed local defaults to minimal runtime dependencies:
   - removed `ft_extraction` config coupling from visualizer defaults.

### Validation
- `uv run python -m vis.vis_with_viser --cfg job --resolve` succeeds and resolves config from local `vis/conf` tree.
- `vis_with_viser.py` diagnostics report no errors.

### Notes
- `uv` still warns about `requires-python ==3.10` exact-minor pin; non-blocking for runtime.

## Batch 5 (completed)
### Goal
Eliminate remaining root-module dependency (`agent.py` / `chat_history.py`) so `vis` startup is self-contained.

### Changes made
1. Added local chat modules:
   - `vis/chat_agent.py` (localized `ChatAgent` implementation)
   - `vis/chat_history.py` (localized `ChatManager` / history)
2. Updated `vis/vis_with_viser.py` imports:
   - `ChatAgent` now from `vis.chat_agent`
   - `ChatManager` now from `vis.chat_history`
3. Updated Python version pin in `pyproject.toml`:
   - `requires-python` changed from `==3.10` to `==3.10.*` (silences uv warning)

### Validation
- `uv run vis/vis_with_viser.py --cfg job --resolve` succeeds (no `ModuleNotFoundError: agent`).
- `uv run python -m vis.vis_with_viser --cfg job --resolve` succeeds.
- No `from agent import` or `from chat_history import` references remain in `vis/**/*.py`.

## Next batch (proposed)
### Batch 6 goal
Add targeted tests for segmentation/floor/color edge cases to lock in standalone behavior.

### Batch 6 tasks
1. Add tests for `SparseSegmentMasker` fallback and malformed-segment handling.
2. Add tests for `similarities_to_rgb` degenerate/empty inputs.
3. Add tests for `segment_floor_points` empty input and shape validation.
4. Run tests via `uv` and record results.

### Validation for Batch 6
- `uv run pytest` on added/updated test modules passes.

## Open items after Batch 6
- None currently required for standalone startup; proceed with targeted tests.
