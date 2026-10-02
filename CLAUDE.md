# CLAUDE.md

Guidance for Claude Code when working in this repo.

## Project Overview

`emergent-matter-sdm-physics` is the **physics layer** of the SDM stack -- the fourth
layer in the org's per-CEM contract (decided 2026-07-24):

1. **core** (`emergent-matter-sdm-core`) -- SDF tree, params, DOF/joint declarations
   (the part's configuration space). Does NOT own how the part moves.
2. **materials** (`emergent-matter-sdm-materials`) -- constitutive properties.
3. **physics** (this repo) -- BCs, loads, couplings, solvers, time. Maps
   (configuration space + materials + BCs) → trajectories, deformation fields,
   metrics (compliance, stress, contact).
4. **view** (`emergent-matter-sdm-view`) -- renders any configuration; playback of
   trajectories; authoring physics inputs in-viewport; field visualization. Does NOT
   own animation semantics or solve anything.

This is a shared substrate repo mirroring sdm-core/sdm-view. It currently ships the
`sdm-traj` trajectory contract (producer and reader, stdlib only), the Beam
Constraint Model solvers behind the optional `solvers` extra (jax), and the hexafoil
sweep producer that turns a hexafoil `.sdm` into a solved `sdm-traj` stream. Any
additional solver backend is orchestrated behind the same seam, never merged into
the base install.

## Build & Run

```bash
uv sync                    # base install: trajectory contract, dev tooling
uv sync --extra solvers    # adds jax for the BCM solvers and the sweep producer
uv run pytest              # round-trip / forward-compat / version-gate tests
uv run ruff check .        # lint (shared org ruleset, target py311)
uv run mypy src            # strict typecheck; clean with or without jax installed
uv run --extra solvers python -m sdm_physics.producers.hexafoil_sweep part.sdm
```

## Architecture

```
src/sdm_physics/
├── trajectory.py   -- sdm-traj v1 writer (producer) + reader. Pure Python, stdlib
│                     only. TrajectoryWriter (streaming, flush-per-line, context
│                     manager, eos-on-close), write_trajectory() one-shot,
│                     read_header(), iter_samples() (generator over any open
│                     line-iterable), read_trajectory().
├── solvers/
│   ├── bcm.py      -- Beam Constraint Model primitive: clamped-clamped
│   │                 Euler-Bernoulli beam with Awtar's third-order axial
│   │                 shortening. Pure JAX, differentiable end to end.
│   └── hexafoil_bcm.py -- the hexafoil rotary-flexure reduction on top of bcm.py:
│                     HexafoilGeometry (the .sdm seam), analyze_rotation(),
│                     linear_rotational_stiffness() closed-form oracle.
└── producers/
    └── hexafoil_sweep.py -- CLI producer: solves a twist sweep at every sample and
                      writes an sdm-traj stream with a provenance source block.
docs/
└── trajectory-format.md -- the sdm-traj v1 spec (normative). The consumer
                      (sdm-view's playback path) parses the same format; there is
                      exactly one player, and authored .sdm animations compile to
                      this format too.
tests/
└── test_trajectory.py -- plain Python, no bpy/jax in the collection path.
```

## Technical Context

- **File-based handoff is v1 by design** -- JSONL configuration streams
  (`.traj.jsonl`) for trajectories, grid files for deformation fields. The framing
  (line-delimited records, key-presence discrimination, flush-per-line, mid-stream
  re-header) is designed so the identical messages stream over a socket in v2 (live
  viewer) with no format change.
- **Record discrimination is by key presence**, not line position: `"format"` →
  header, `"t"` → sample, `"eos": true` → end-of-stream; anything else is ignored
  (forward compatibility). Unknown keys everywhere are ignored; `version` gates
  breaking changes only, with the exact rejection message
  `ValueError("Unsupported sdm-traj version {found!r} (expected {expected!r})")`.
  A missing header or missing `version` is malformed -- never defaulted.
- **The grid-file encoding is a sibling contract** -- this format only carries field
  references (`path` + `kind`), so trajectory framing and grid encoding version
  independently. In the viewer, a displacement field is a query-point warp applied
  above the baked-grid texture fetch: playback never re-bakes or re-meshes.
- **Trajectories live outside the `.sdm`** -- embedding them would change the `.sdm`
  hash and invalidate the emission cache; `sdm_sha256` is the advisory staleness
  link. Date-stamp solved-trajectory filenames per org convention.
- **Solver deps stay behind the `solvers` extra** -- sdm-view loads the trajectory
  reader inside Blender's Python, so the base install must remain stdlib-only. The
  package root exports trajectory symbols only; solver modules are imported by
  explicit submodule path. Keep that boundary.
- **The solvers run in float64** -- `sdm_physics/solvers/__init__.py` enables jax's
  x64 mode at import (process-wide by jax design; see the comment there). Without
  it jax silently downcasts the solver's float64 coercions to float32;
  tests/test_hexafoil_bcm.py pins against that regression.

## Code standard

Follow STYLE.md at the repo root for all code, comments, tests, and docs.
Pull request descriptions follow `.github/PULL_REQUEST_TEMPLATE.md`.

## Naming Conventions

Typed prefixes on primitive scalars, per STYLE.md: `d_` float, `n_` int, `b_` bool,
`s_` str. JSON wire keys are fixed by the spec and unprefixed. Verb-first API naming
(`write_trajectory`, `read_header`, `iter_samples`).
