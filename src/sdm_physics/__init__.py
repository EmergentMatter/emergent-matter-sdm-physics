"""emergent-matter-sdm-physics: the SDM physics layer.

Scope: the sdm-traj trajectory contract (producer + reader), covering both
per-part channel streams and assembly motion (per-instance poses, spec §5a),
plus the optional BCM solvers. The package root stays importable without jax.
"""

from sdm_physics.trajectory import (
    TRAJ_EXTENSION,
    TRAJ_FORMAT,
    TRAJ_VERSION,
    Channel,
    FieldRef,
    Sample,
    TrajectoryHeader,
    TrajectoryWriter,
    iter_samples,
    read_header,
    read_trajectory,
    write_trajectory,
)

__version__ = "1.1.0"

__all__ = [
    "TRAJ_EXTENSION",
    "TRAJ_FORMAT",
    "TRAJ_VERSION",
    "Channel",
    "FieldRef",
    "Sample",
    "TrajectoryHeader",
    "TrajectoryWriter",
    "iter_samples",
    "read_header",
    "read_trajectory",
    "write_trajectory",
    "__version__",
]
