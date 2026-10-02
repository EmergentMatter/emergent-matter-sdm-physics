"""emergent-matter-sdm-physics: the SDM physics layer.

v1 scope: the sdm-traj trajectory contract (producer + reader). Solver
backends (jax-fem, magnetostatics) are onboarded later; this package stays
importable without jax.
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

__version__ = "0.0.0"

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
