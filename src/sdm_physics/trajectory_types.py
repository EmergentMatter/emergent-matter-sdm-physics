"""sdm-traj v1 wire-format dataclasses and constants.

Split out of ``trajectory.py`` (the writer/reader) to keep both files small.
Everything here is re-exported from ``sdm_physics.trajectory``. Pure Python,
stdlib only.

Naming follows the org convention: ``d_`` float, ``n_`` int, ``b_`` bool,
``s_`` str. JSON wire keys are fixed by the spec and unprefixed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

TRAJ_FORMAT = "sdm-traj"
TRAJ_VERSION = 1
TRAJ_EXTENSION = ".traj.jsonl"

# Floats are absolute values, so precision loss does not accumulate; the spec
# says producers SHOULD round to ~6 decimal places.
_N_FLOAT_DECIMALS = 6

# A pose quaternion must be unit length to this tolerance (spec §5a). Rounding
# to 6 decimals leaves norms off by ~3e-7, far inside it.
_D_QUAT_NORM_TOL = 1e-3
_N_POSE_LEN = 7


def check_pose(s_path: str, pose: Any) -> tuple[float, ...]:
    """Validate one pose ``[x, y, z, qw, qx, qy, qz]`` and return it as floats.

    Raises ValueError for a wrong length, a non-numeric or non-finite element, or a
    quaternion whose norm is off by 1e-3 or more. The quaternion is renormalised, so
    the returned pose is unit length to float precision (spec §5a).
    """
    if not isinstance(pose, (list, tuple)) or len(pose) != _N_POSE_LEN:
        raise ValueError(f"Malformed pose for {s_path!r}: expected 7 numbers [x,y,z,qw,qx,qy,qz]")
    vals: list[float] = []
    for v in pose:
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
            raise ValueError(f"Malformed pose for {s_path!r}: non-finite or non-numeric {v!r}")
        vals.append(float(v))
    d_norm = math.sqrt(sum(v * v for v in vals[3:]))
    if abs(d_norm - 1.0) >= _D_QUAT_NORM_TOL:
        raise ValueError(f"Malformed pose for {s_path!r}: quaternion norm {d_norm:.6g} is not 1")
    return (*vals[:3], *(v / d_norm for v in vals[3:]))


# ──────────────────────────────────────────────────────────────────────────────
# Dataclasses (wire-format mirrors)
# ──────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Channel:
    """A declared configuration channel (spec §3)."""

    s_name: str  # .sdm manifest param name, e.g. "finger__curl"
    s_kind: str = "pose"  # v1 defines "pose" only
    d_neutral: float = 0.0  # value restored on Stop / Rest Pose (required on wire)
    s_unit: str = ""  # informational ("rad", "mm", "")

    def to_obj(self) -> dict[str, Any]:
        obj: dict[str, Any] = {
            "name": self.s_name,
            "kind": self.s_kind,
            "neutral": self.d_neutral,
        }
        if self.s_unit:
            obj["unit"] = self.s_unit
        return obj

    @classmethod
    def from_obj(cls, obj: dict[str, Any]) -> Channel:
        # Unknown channel keys are ignored (spec §6).
        return cls(
            s_name=str(obj["name"]),
            s_kind=str(obj["kind"]),
            d_neutral=float(obj["neutral"]),
            s_unit=str(obj.get("unit", "")),
        )


@dataclass(frozen=True)
class FieldRef:
    """A field reference on a sample (spec §4.1)."""

    s_path: str  # relative to the trajectory file's directory, forward slashes
    s_kind: str = "displacement"  # v1 defines "displacement" only

    def to_obj(self) -> dict[str, Any]:
        return {"path": self.s_path, "kind": self.s_kind}

    @classmethod
    def from_obj(cls, obj: dict[str, Any]) -> FieldRef:
        # Unknown field-ref keys are ignored (spec §6).
        return cls(s_path=str(obj["path"]), s_kind=str(obj["kind"]))


@dataclass(frozen=True)
class Sample:
    """One sample record (spec §4)."""

    d_t: float  # seconds, strictly increasing within a stream
    q: dict[str, float] | None = None  # sparse: subset of declared channels
    fields: tuple[FieldRef, ...] = ()
    poses: dict[str, tuple[float, ...]] | None = (
        None  # assembly motion (spec §5a): path -> x y z qw qx qy qz
    )

    def to_obj(self) -> dict[str, Any]:
        obj: dict[str, Any] = {"t": round(float(self.d_t), _N_FLOAT_DECIMALS)}
        if self.q is not None:
            obj["q"] = {
                s_key: round(float(d_val), _N_FLOAT_DECIMALS) for s_key, d_val in self.q.items()
            }
        if self.fields:
            obj["fields"] = [ref.to_obj() for ref in self.fields]
        if self.poses is not None:
            obj["poses"] = {
                s_path: [round(d_v, _N_FLOAT_DECIMALS) for d_v in check_pose(s_path, pose)]
                for s_path, pose in self.poses.items()
            }
        return obj

    @classmethod
    def from_obj(cls, obj: dict[str, Any]) -> Sample:
        # Unknown sample keys are ignored (spec §4/§6).
        q_raw = obj.get("q")
        q = (
            {str(s_key): float(d_val) for s_key, d_val in q_raw.items()}
            if isinstance(q_raw, dict)
            else None
        )
        fields_raw = obj.get("fields")
        fields: tuple[FieldRef, ...] = ()
        if isinstance(fields_raw, list):
            fields = tuple(
                FieldRef.from_obj(ref)
                for ref in fields_raw
                if isinstance(ref, dict) and "path" in ref and "kind" in ref
            )
        poses_raw = obj.get("poses")
        poses = (
            {str(s_path): check_pose(str(s_path), pose) for s_path, pose in poses_raw.items()}
            if isinstance(poses_raw, dict)
            else None
        )
        d_t = float(obj["t"])
        if not math.isfinite(d_t):
            raise ValueError(f"Malformed sample: non-finite t {d_t!r}")
        return cls(d_t=d_t, q=q, fields=fields, poses=poses)


@dataclass(frozen=True)
class TrajectoryHeader:
    """The header record (spec §2)."""

    s_part: str
    channels: tuple[Channel, ...] = ()
    s_name: str | None = None  # display name; consumer defaults to file stem
    s_sdm: str | None = None  # path to the source .sdm, relative, fwd slashes
    s_sdm_sha256: str | None = None  # advisory staleness check
    s_loop: str = "once"  # "once" (hold last) | "cycle" (wrap) | "pingpong" (forward, back)
    d_rate_hz: float | None = None  # pacing hint only; per-sample t authoritative
    d_duration_s: float | None = None  # hint; never the authoritative period
    source: dict[str, Any] | None = None  # provenance, free-form
    poses: tuple[str, ...] = ()  # assembly motion: instance paths moved (spec §5a)

    def to_obj(self) -> dict[str, Any]:
        obj: dict[str, Any] = {
            "format": TRAJ_FORMAT,
            "version": TRAJ_VERSION,
            "part": self.s_part,
            "channels": [ch.to_obj() for ch in self.channels],
        }
        if self.s_name is not None:
            obj["name"] = self.s_name
        if self.s_sdm is not None:
            obj["sdm"] = self.s_sdm
        if self.s_sdm_sha256 is not None:
            obj["sdm_sha256"] = self.s_sdm_sha256
        if self.s_loop != "once":
            obj["loop"] = self.s_loop
        if self.d_rate_hz is not None:
            obj["rate_hz"] = self.d_rate_hz
        if self.d_duration_s is not None:
            obj["duration_s"] = self.d_duration_s
        if self.source is not None:
            obj["source"] = self.source
        if self.poses:
            obj["poses"] = list(self.poses)
        return obj

    @classmethod
    def from_obj(cls, obj: dict[str, Any]) -> TrajectoryHeader:
        """Validate + parse a header object. Raises ValueError on malformed
        headers and on unsupported versions (spec §2/§6)."""
        s_format = obj.get("format")
        if s_format != TRAJ_FORMAT:
            raise ValueError(
                f"Not an sdm-traj header: format={s_format!r} (expected {TRAJ_FORMAT!r})"
            )
        if "version" not in obj:
            # Malformed: never version-defaulted (spec §6).
            raise ValueError("Malformed sdm-traj header: missing 'version' key")
        n_version = obj["version"]
        if n_version != TRAJ_VERSION:
            raise ValueError(
                f"Unsupported sdm-traj version {n_version!r} (expected {TRAJ_VERSION!r})"
            )
        if "part" not in obj:
            raise ValueError("Malformed sdm-traj header: missing 'part'")
        # channels is optional (an assembly-motion stream may declare none).
        channels_raw = obj.get("channels", [])
        if not isinstance(channels_raw, list):
            raise ValueError("Malformed sdm-traj header: 'channels' must be a list")
        poses_raw = obj.get("poses", [])
        if not isinstance(poses_raw, list):
            raise ValueError("Malformed sdm-traj header: 'poses' must be a list")
        channels = tuple(
            Channel.from_obj(ch)
            for ch in channels_raw
            if isinstance(ch, dict) and "name" in ch and "kind" in ch and "neutral" in ch
        )
        loop = obj.get("loop", "once")
        source = obj.get("source")
        rate_hz = obj.get("rate_hz")
        duration_s = obj.get("duration_s")
        return cls(
            s_part=str(obj["part"]),
            channels=channels,
            s_name=obj.get("name"),
            s_sdm=obj.get("sdm"),
            s_sdm_sha256=obj.get("sdm_sha256"),
            s_loop=str(loop),
            d_rate_hz=float(rate_hz) if rate_hz is not None else None,
            d_duration_s=float(duration_s) if duration_s is not None else None,
            source=source if isinstance(source, dict) else None,
            poses=tuple(p for p in poses_raw if isinstance(p, str)),
        )
