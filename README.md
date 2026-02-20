# vis

Standalone visualization package split out from the main `cg` workspace.

This folder is structured as if it were its own repository, so the Python package is intentionally nested:

- package root: `vis/`
- module path example: `vis/vis_with_viser.py`
- full workspace path example: `vis/vis/vis_with_viser.py`

## Setup

From this folder (`cg/vis`):

```bash
uv venv
uv pip install -e .
```

Or with pip:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Run

```bash
python -m vis.vis_with_viser
```

To inspect resolved Hydra config:

```bash
python -m vis.vis_with_viser --cfg job --resolve
```

## Included assets

- Runtime configs: `vis/conf/`
- Internal docs: `vis/docs/`
- Core utility modules: `vis/core/`

## Notes

- The nested `vis/vis` path is temporary by design while this package is still inside the parent monorepo.
- Once moved to its own repository, this becomes the standard `repo_root/vis/...` layout.
