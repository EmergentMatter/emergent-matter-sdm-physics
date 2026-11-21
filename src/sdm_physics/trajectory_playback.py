"""Reference evaluator for sdm-traj playback (spec §5 and §5a).

Every consumer of the format keeps its own player (the engine, sdm-ui, the
fabrication navigator, sdm-view). This module states the playback rules once,
as code, so those players can be checked against it and against the expected
frames under ``tests/data/expected/``. It is not a player: it evaluates one
stream segment at one time and returns what a player should show.

Pure Python, stdlib only, like the reader and writer.
"""

from __future__ import annotations

import math
from bisect import bisect_right
from collections.abc import Sequence
from dataclasses import dataclass

from sdm_physics.trajectory_types import Sample, TrajectoryHeader

__all__ = ["Frame", "Playback", "slerp_shortest", "stream_time"]

# Below this angle between two quaternions slerp's sin(theta) denominator loses
# precision; normalised linear interpolation is exact enough there.
_D_SLERP_LINEAR_DOT = 0.9995


@dataclass(frozen=True)
class Frame:
    """What a player shows at one moment.

    ``q`` holds every declared ``pose`` channel. ``fields`` maps a field
    ``kind`` to the held reference's path. ``poses`` holds every instance the
    stream moves; ``None`` means the instance sits where its ``.sdm`` puts it.
    """

    q: dict[str, float]
    fields: dict[str, str]
    poses: dict[str, tuple[float, ...] | None]


def stream_time(s_loop: str, d_t_first: float, d_t_last: float, d_elapsed_s: float) -> float:
    """Map playback time onto the stream's own ``t`` axis (spec §5 loop rule).

    Args:
        s_loop: The header's ``loop``. Unknown values play as ``"once"``.
        d_t_first: ``t`` of the first sample, the playback origin.
        d_t_last: ``t`` of the last sample.
        d_elapsed_s: Seconds since playback started. Negative values clamp to 0.

    Returns:
        The stream ``t`` to evaluate. ``"pingpong"`` mirrors the second half
        of its period, so the way back evaluates the same stream at the
        mirrored ``t`` and holds follow stream order.
    """
    d_span = d_t_last - d_t_first
    if d_span <= 0.0:
        return d_t_first
    d_e = max(d_elapsed_s, 0.0)
    if s_loop == "cycle":
        return d_t_first + math.fmod(d_e, d_span)
    if s_loop == "pingpong":
        d_u = math.fmod(d_e, 2.0 * d_span)
        return d_t_first + (d_u if d_u <= d_span else 2.0 * d_span - d_u)
    return d_t_first + min(d_e, d_span)


def slerp_shortest(
    qa: Sequence[float], qb: Sequence[float], d_frac: float
) -> tuple[float, float, float, float]:
    """Interpolate unit quaternions ``(w, x, y, z)`` on the shortest arc.

    ``q`` and ``-q`` are the same rotation. Without flipping ``qb`` when the
    dot product is negative, a sign change between samples spins the long way
    round (spec §5a).
    """
    d_dot = sum(a * b for a, b in zip(qa, qb, strict=True))
    if d_dot < 0.0:
        qb = [-b for b in qb]
        d_dot = -d_dot
    if d_dot > _D_SLERP_LINEAR_DOT:
        q = [a + d_frac * (b - a) for a, b in zip(qa, qb, strict=True)]
    else:
        d_theta = math.acos(min(d_dot, 1.0))
        d_sin = math.sin(d_theta)
        d_wa = math.sin((1.0 - d_frac) * d_theta) / d_sin
        d_wb = math.sin(d_frac * d_theta) / d_sin
        q = [d_wa * a + d_wb * b for a, b in zip(qa, qb, strict=True)]
    d_norm = math.sqrt(sum(c * c for c in q))
    return (q[0] / d_norm, q[1] / d_norm, q[2] / d_norm, q[3] / d_norm)


class Playback:
    """One stream segment (a header and the samples after it), evaluable at any time.

    Build it from ``read_trajectory``'s result, which is the last segment of
    the file. Hold state is precomputed per sample (the held-value timeline of
    spec §5), so evaluation is a binary search plus one interpolation.
    """

    def __init__(self, header: TrajectoryHeader, samples: Sequence[Sample]) -> None:
        self.header = header
        self.times = [s.d_t for s in samples]
        # Channels of unknown kind are ignored (spec §3).
        neutrals = {ch.s_name: ch.d_neutral for ch in header.channels if ch.s_kind == "pose"}
        paths = list(header.poses)
        for s in samples:
            paths += [p for p in (s.poses or {}) if p not in paths]

        held_q = dict(neutrals)
        held_fields: dict[str, str] = {}
        held_poses: dict[str, tuple[float, ...] | None] = dict.fromkeys(paths)
        self._q: list[dict[str, float]] = []
        self._fields: list[dict[str, str]] = []
        self._poses: list[dict[str, tuple[float, ...] | None]] = []
        for s in samples:
            # Undeclared q keys are ignored (spec §4).
            held_q.update({k: v for k, v in (s.q or {}).items() if k in neutrals})
            held_fields.update({ref.s_kind: ref.s_path for ref in s.fields})
            held_poses.update(s.poses or {})
            self._q.append(dict(held_q))
            self._fields.append(dict(held_fields))
            self._poses.append(dict(held_poses))
        self._neutrals = neutrals
        self._paths = paths

    def at(self, d_elapsed_s: float) -> Frame:
        """The frame ``d_elapsed_s`` seconds after playback started, loop applied."""
        if not self.times:
            return Frame(q=dict(self._neutrals), fields={}, poses=dict.fromkeys(self._paths))
        d_t = stream_time(self.header.s_loop, self.times[0], self.times[-1], d_elapsed_s)
        return self.at_stream_time(d_t)

    def at_stream_time(self, d_t: float) -> Frame:
        """The frame at stream time ``d_t``, clamped to the sampled range, no loop."""
        if not self.times:
            return Frame(q=dict(self._neutrals), fields={}, poses=dict.fromkeys(self._paths))
        n_last = len(self.times) - 1
        i = min(max(bisect_right(self.times, d_t) - 1, 0), n_last)
        if i == n_last:
            j, d_frac = i, 0.0
        else:
            j = i + 1
            d_frac = (d_t - self.times[i]) / (self.times[j] - self.times[i])
            d_frac = min(max(d_frac, 0.0), 1.0)

        qa, qb = self._q[i], self._q[j]
        q = {k: qa[k] + d_frac * (qb[k] - qa[k]) for k in qa}

        poses: dict[str, tuple[float, ...] | None] = {}
        for s_path in self._paths:
            pa, pb = self._poses[i][s_path], self._poses[j][s_path]
            if pa is None or pb is None:
                # Not placed yet at sample i: the instance stays at its .sdm
                # placement and snaps to its first pose when t reaches it,
                # with no interpolation from rest (spec §5a).
                poses[s_path] = pa
                continue
            pos = tuple(a + d_frac * (b - a) for a, b in zip(pa[:3], pb[:3], strict=True))
            poses[s_path] = (*pos, *slerp_shortest(pa[3:], pb[3:], d_frac))

        return Frame(q=q, fields=dict(self._fields[i]), poses=poses)
