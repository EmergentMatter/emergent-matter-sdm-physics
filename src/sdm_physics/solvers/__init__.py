"""Solver backends for the SDM physics layer.

Optional by design: the base install is stdlib-only, and everything under
this subpackage needs the `solvers` extra (jax). Import solver modules by
explicit submodule path; the package root deliberately does not re-export
them.
"""

from __future__ import annotations

try:
    import jax
except ImportError as e:
    raise ImportError(
        "sdm_physics.solvers needs jax. Install the 'solvers' extra: "
        "`uv sync --extra solvers` or `pip install 'emergent-matter-sdm-physics[solvers]'`."
    ) from e

# The BCM solvers are written for float64: analyze_rotation coerces its
# rotation to float64 explicitly, and the quantity the model exists for
# (the geometric-stiffening tail on top of a large linear term) is the
# kind of small-correction arithmetic float32 quietly erodes. jax defaults
# to float32 and SILENTLY downcasts float64 requests unless x64 is
# enabled, so without this line the dtype at that asarray site was a lie.
# The flag is process-wide by jax design; setting it here scopes the
# decision to the moment a consumer explicitly opts into the solver
# backend, before any solver math traces. A consumer that needs float32
# jax elsewhere in the same process should not import this subpackage.
jax.config.update("jax_enable_x64", True)
