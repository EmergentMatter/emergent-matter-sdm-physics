<a id="readme-top"></a>

# emergent-matter-sdm-physics

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-033388.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-0055FF.svg)](https://www.python.org/downloads/)
[![Built with JAX](https://img.shields.io/badge/built%20with-JAX-orange.svg)](https://docs.jax.dev/)
[![uv](https://img.shields.io/badge/packaged%20with-uv-DE5FE9.svg)](https://docs.astral.sh/uv/)

The **physics layer** of the Software Defined Matter stack: boundary conditions, loads,
couplings, solvers, and time. It maps (configuration space + materials + BCs) to
trajectories, deformation fields, and metrics such as compliance and stress.

## The four-layer contract per CEM

1. **core** (`emergent-matter-sdm-core`): SDF tree, params, DOF/joint declarations (the part's configuration space). Does not own how the part moves.
2. **materials** (`emergent-matter-sdm-materials`): constitutive properties.
3. **physics** (this repo): BCs, loads, couplings, solvers, time. Produces trajectories, fields, and metrics.
4. **view** (`emergent-matter-sdm-view`): renders any configuration, plays back trajectories, authors physics inputs. Solves nothing.

## What ships today

- **The `sdm-traj` trajectory contract** (`docs/trajectory-format.md`): the producer
  and reader for the file-based handoff between physics and sdm-view playback. Pure
  Python, stdlib only; the base install has no third-party dependencies.
- **Beam Constraint Model solvers** (`sdm_physics.solvers`, behind the optional
  `solvers` extra): a clamped-clamped Euler-Bernoulli beam element with Awtar's
  third-order axial-shortening term (`solvers.bcm`), and the hexafoil rotary-flexure
  reduction built on it (`solvers.hexafoil_bcm`). Closed form, JAX-differentiable
  end to end. Importing the solver subpackage enables jax's float64 (x64) mode for
  the process; the solvers' precision contract is float64.
- **The hexafoil sweep producer** (`sdm_physics.producers.hexafoil_sweep`): a CLI
  that solves a twist sweep for a hexafoil `.sdm` and writes it as an `sdm-traj`
  stream, reporting torque, rotational stiffness, and peak stresses per sample.

Solver backends stay optional on purpose: sdm-view loads the trajectory reader inside
Blender's Python, and the base install must not put a heavyweight jax wheel behind a
JSONL parser. The package root exports trajectory symbols only; solvers are reached by
explicit submodule import.

## Install

From the Software Defined Matter package index (not PyPI):

```bash
uv pip install "emergent-matter-sdm-physics[solvers]" --index https://get.softwaredefinedmatter.com/simple
```

In a uv project, declare the index once and pin the package to it, so its name never resolves from public PyPI:

```toml
[[tool.uv.index]]
name = "em"
url = "https://get.softwaredefinedmatter.com/simple"
explicit = true

[tool.uv.sources]
emergent-matter-sdm-physics = { index = "em" }
```

From a clone:

```bash
uv sync                  # trajectory contract only, stdlib
uv sync --extra solvers  # adds jax for the BCM solvers
```

Or with pip, from the repo root:

```bash
pip install ".[solvers]"
```

## Quickstart

Write a trajectory (base install, stdlib only):

```python
from sdm_physics import Channel, Sample, TrajectoryHeader, TrajectoryWriter

header = TrajectoryHeader(
    s_part="finger",
    channels=(Channel(s_name="finger__curl", s_kind="pose", s_unit="rad", d_neutral=0.0),),
    d_rate_hz=120.0,
)
with TrajectoryWriter("finger_tendon_2026-07-24T1802.traj.jsonl", header) as w:
    w.write_sample(Sample(d_t=0.0, q={"finger__curl": 0.0}))
    w.write_sample(Sample(d_t=0.008333, q={"finger__curl": 0.0021}))
```

Analyze a hexafoil flexure (needs the `solvers` extra):

```python
from sdm_physics.solvers.hexafoil_bcm import HexafoilGeometry, analyze_rotation

geom = HexafoilGeometry(
    d_cylinder_length=10.0,
    d_blade_thickness=0.8,
    d_cylinder_inner_radius=12.0,
    d_inner_radius=3.0,
    n_stages=2,
    n_foil_count=6,
)
result = analyze_rotation(geom, d_E=1850.0, d_delta_theta=0.1)
print(float(result.d_rotational_stiffness))  # N*mm per rad
```

Solve a sweep for a part straight from its `.sdm`:

```bash
uv run --extra solvers python -m sdm_physics.producers.hexafoil_sweep part.sdm -o out.traj.jsonl
```


## Development

```bash
uv sync
uv run pytest
uv run ruff check .
uv run mypy src
```

See CONTRIBUTING.md for the changelog-note workflow and STYLE.md for code
conventions.

## Documentation

| Page | What it covers |
|---|---|
| [`docs/trajectory-format.md`](docs/trajectory-format.md) | The `sdm-traj` v1 spec (normative): the cross-repo contract between this layer's producers and sdm-view's playback path |
| [STYLE.md](STYLE.md) / [CONTRIBUTING.md](CONTRIBUTING.md) | House style and how a change ships |

Those pages document this library. For the wider SDM ecosystem and the
other repositories in it, see
[`emergent-matter-sdm`](https://github.com/EmergentMatter/emergent-matter-sdm).

## Built with

| | |
|---|---|
| [JAX](https://docs.jax.dev/) | Autodiff for the BCM solvers, behind the optional `solvers` extra |
| [uv](https://docs.astral.sh/uv/) | Packaging and the locked dev environment |

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for how a change ships: the
changeset a pull request needs, what counts as major, minor, or patch,
and how a release is cut. [STYLE.md](STYLE.md) is the house style for
code, tests, and docs, and it wins over habit.

By participating you agree to the
[Code of Conduct](CODE_OF_CONDUCT.md).

## Support

Questions and usage help go to
[Discussions](https://github.com/EmergentMatter/emergent-matter-sdm/discussions);
bugs and feature requests go to
[Issues](https://github.com/EmergentMatter/emergent-matter-sdm-physics/issues).
See [SUPPORT.md](SUPPORT.md) for what is and is not supported.

For security reports, do not open a public issue. Follow
[SECURITY.md](SECURITY.md).

## License

Apache-2.0. See [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE).

## Acknowledgments

- The Beam Constraint Model's third-order axial-shortening term
  (`solvers.bcm`) follows Awtar, Slocum, and Sevincer's 2007
  "Characteristics of Beam-Based Flexure Modules".

<p align="right">(<a href="#readme-top">back to top</a>)</p>
