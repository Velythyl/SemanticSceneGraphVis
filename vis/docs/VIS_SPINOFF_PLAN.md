# VIS Spinoff Plan (Remove `concept_graphs` dependency)

## Goal

Make `vis/` runnable as its own package without importing from `concept_graphs/`.

## Scope

This plan focuses on current direct couplings in `vis/vis_with_viser.py`:

- `from concept_graphs.floor_segment import segment_floor_points`
- `from concept_graphs.utils import set_seed`
- `from concept_graphs.viz.utils import similarities_to_rgb`

And on preserving map loading behavior from `vis/cg_dataclass.py`, where segmented objects are reconstructed from sparse point cloud indices.

It also includes copying the required runtime config files into the `vis` spinoff so it can run without the parent repo config tree.

## Non-goals

- No redesign of rendering/UI behavior.
- No map-format changes.
- No broad refactor outside what is required for decoupling.

---

## Proposed target boundary

Create a local `vis/core/` package and move all required logic behind small, stable interfaces:

- `vis/core/seed.py` → local `set_seed()`
- `vis/core/color.py` → local `similarities_to_rgb()`
- `vis/core/floor.py` → local `segment_floor_points()`
- `vis/core/segmentation.py` → **thin sparse segment masker/extractor class**

`vis` should only depend on:

- standard library
- already-used third-party libs (`numpy`, `open3d`, `matplotlib`, etc.)
- local `vis.*` modules

---

## Thin class to mask sparse/non-dense cloud

### Why this class

Right now object extraction is embedded in `ConceptGraphData._split_segments()`. Pulling this into an explicit class gives `vis` a clean seam for future external integrations while preserving current behavior.

### Suggested class

`vis/core/segmentation.py`

```python
class SparseSegmentMasker:
    def split_objects(
        self,
        sparse_pcd: o3d.geometry.PointCloud | None,
        segments_anno: dict[str, Any] | None,
    ) -> list[o3d.geometry.PointCloud]:
        ...
```

### Behavior contract (match current behavior)

1. If sparse cloud is missing → return `[]`.
2. If annotations missing/empty → return `[sparse_pcd]`.
3. Read `segGroups[*].segments` as point indices.
4. For each group: `select_by_index(indices)`.
5. Drop empty objects.
6. If all objects empty/invalid → fallback `[sparse_pcd]`.
7. If the only thing available is the dense cloud, display that

### Optional extension (not required for first pass)

Add `return_indices: bool = False` to expose masks/index arrays for tooling, while defaulting to point-cloud output for compatibility.

---

## Migration steps

### Phase 1 — Localize imported utilities

1. Add local modules under `vis/core/`:
   - `seed.py` (copy/adapt `set_seed`)
   - `color.py` (copy/adapt `similarities_to_rgb`; handle constant-similarity edge case safely)
   - `floor.py` (copy/adapt `segment_floor_points`)
2. Update `vis/vis_with_viser.py` imports to use `vis.core.*`.
3. Keep function names/signatures stable to minimize churn.

### Phase 2 — Introduce sparse masker seam

1. Implement `SparseSegmentMasker` in `vis/core/segmentation.py`.
2. Update `vis/cg_dataclass.py` to call this class instead of inline `_split_segments` logic.
3. Keep `ConceptGraphData` data contract unchanged (`pcd_o3d`, `segments_anno`, etc.).

### Phase 3 — Remove packaging/path coupling

1. Remove runtime `sys.path.insert(...)` hack from `vis/vis_with_viser.py`.
2. Ensure all imports are package-correct for standalone execution.
3. Add/adjust package entrypoint (e.g., `python -m vis.vis_with_viser`) so direct script assumptions are unnecessary.

### Phase 3.5 — Copy required config into spinoff

1. Identify `vis` runtime config dependencies currently resolved from `conf/` (e.g., visualizer + any `agent/` + `llm/` settings used by `vis_with_viser.py`).
2. Copy those files/directories into the spinoff package (for example `vis/conf/`), preserving relative structure expected by Hydra.
3. Update config search path/defaults so the standalone package loads local config first.
4. Verify `vis` startup works in a clean environment with only spinoff-local config present.

### Phase 4 — Validate standalone behavior

1. Open a known map directory and verify:
   - segmented objects render
   - floor segmentation toggle still works
   - similarity coloring still works
2. Verify behavior when artifacts are missing:
   - no `segments_anno.json`
   - no `dense_point_cloud.pcd`
   - empty segmentation groups

---

## Acceptance criteria

- No import in `vis/**` references `concept_graphs`.
- `vis` map loading still reconstructs segmented objects from sparse cloud + segment indices.
- Floor segmentation output parity is maintained on existing maps.
- `vis` can be run without `concept_graphs` installed/importable.
- `vis` can be run with only spinoff-local config files (no dependency on parent `conf/` paths).

---

## Risks and mitigations

- **Risk:** subtle drift in floor segmentation behavior after copy.
  - **Mitigation:** keep algorithm identical first; optimize later.
- **Risk:** color mapping differs for degenerate similarity vectors.
  - **Mitigation:** explicitly handle `max == min` case and document expected output.
- **Risk:** import/runtime break when removing `sys.path` mutation.
  - **Mitigation:** enforce module-based execution and absolute package imports.
- **Risk:** missing config fragments after spinoff causes Hydra composition failures.
  - **Mitigation:** enumerate required config groups up front and add a clean-env startup check.

---

## Suggested implementation order (small PRs)

1. PR1: Add `vis/core/seed.py`, `vis/core/color.py`, `vis/core/floor.py` + import rewires.
2. PR2: Add `SparseSegmentMasker` and wire `cg_dataclass`.
3. PR3: Remove `sys.path.insert(...)`, tighten standalone run path/docs.
4. PR4: Copy required config tree into spinoff and validate Hydra config resolution.
5. PR5: Add small tests around sparse masking and floor/color edge cases.
