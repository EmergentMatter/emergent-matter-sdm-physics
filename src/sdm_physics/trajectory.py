"""sdm-traj v1 writer (producer) and reader.

Implements the SDM trajectory / configuration stream contract between a
trajectory producer (this repo's physics layer, or sdm-view's authored-animation
compiler) and the consumer (sdm-view's trajectory playback path). The full spec
lives in ``docs/trajectory-format.md``; the reader here is the same format both
sides parse.

Pure Python, stdlib only (no jax, no bpy). Streaming-first: the writer flushes
after every line so ``tail -f`` / socket consumers see samples as the solver
integrates; ``iter_samples`` is a generator over any open line-iterable and
never loads a whole file.

Naming follows the org convention: ``d_`` float, ``n_`` int, ``b_`` bool,
``s_`` str. JSON wire keys are fixed by the spec and unprefixed.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any

TRAJ_FORMAT = "sdm-traj"
TRAJ_VERSION = 1
TRAJ_EXTENSION = ".traj.jsonl"

# Floats are absolute values, so precision loss does not accumulate; the spec
# says producers SHOULD round to ~6 decimal places.
_N_FLOAT_DECIMALS = 6


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

    def to_obj(self) -> dict[str, Any]:
        obj: dict[str, Any] = {"t": round(float(self.d_t), _N_FLOAT_DECIMALS)}
        if self.q is not None:
            obj["q"] = {
                s_key: round(float(d_val), _N_FLOAT_DECIMALS) for s_key, d_val in self.q.items()
            }
        if self.fields:
            obj["fields"] = [ref.to_obj() for ref in self.fields]
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
        return cls(d_t=float(obj["t"]), q=q, fields=fields)


@dataclass(frozen=True)
class TrajectoryHeader:
    """The header record (spec §2)."""

    s_part: str
    channels: tuple[Channel, ...] = ()
    s_name: str | None = None  # display name; consumer defaults to file stem
    s_sdm: str | None = None  # path to the source .sdm, relative, fwd slashes
    s_sdm_sha256: str | None = None  # advisory staleness check
    s_loop: str = "once"  # "once" (hold last) | "cycle" (wrap)
    d_rate_hz: float | None = None  # pacing hint only; per-sample t authoritative
    d_duration_s: float | None = None  # hint; never the authoritative period
    source: dict[str, Any] | None = None  # provenance, free-form

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
        channels_raw = obj.get("channels")
        if not isinstance(channels_raw, list):
            raise ValueError("Malformed sdm-traj header: missing 'channels' list")
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
        )


# ──────────────────────────────────────────────────────────────────────────────
# Record discrimination (by key presence, not line position; spec §1)
# ──────────────────────────────────────────────────────────────────────────────


def is_header_obj(obj: dict[str, Any]) -> bool:
    return "format" in obj


def is_sample_obj(obj: dict[str, Any]) -> bool:
    return "t" in obj


def is_eos_obj(obj: dict[str, Any]) -> bool:
    return obj.get("eos") is True


# ──────────────────────────────────────────────────────────────────────────────
# Writer / producer API
# ──────────────────────────────────────────────────────────────────────────────


class TrajectoryWriter:
    """Streaming sdm-traj producer.

    Opens the stream, writes the header immediately, appends samples as the
    solver integrates, and writes the end-of-stream record on close. Flushes
    after every line (spec §1) so live consumers can tail the file.

    Usage::

        with TrajectoryWriter(path, header) as w:
            for step in solve():
                w.write_sample(Sample(d_t=step.t, q={"finger__curl": step.curl}))
        # eos written and file closed on clean exit
    """

    def __init__(self, path: str | Path, header: TrajectoryHeader) -> None:
        self.path = Path(path)
        self._f: IO[str] | None = self.path.open("w", encoding="utf-8", newline="\n")
        self.b_eos_written = False
        self.n_samples_written = 0
        self.write_header(header)

    # -- record writers --------------------------------------------------

    def _write_line(self, obj: dict[str, Any]) -> None:
        if self._f is None:
            raise ValueError(f"TrajectoryWriter for {self.path} is closed")
        self._f.write(json.dumps(obj, separators=(",", ":")) + "\n")
        self._f.flush()

    def write_header(self, header: TrajectoryHeader) -> None:
        """Write a header record. Line 1 of a file is always a header; a
        producer MAY also write a new header mid-stream (solver restart),
        which replaces the previous one and resets consumer hold state."""
        self._write_line(header.to_obj())
        self.header = header

    def write_sample(self, sample: Sample) -> None:
        self._write_line(sample.to_obj())
        self.n_samples_written += 1

    def write_eos(self, **info: Any) -> None:
        """Write the end-of-stream record, e.g. ``write_eos(reason="converged")``.
        Informational keys are carried through; readers ignore them."""
        self._write_line({"eos": True, **info})
        self.b_eos_written = True

    # -- lifecycle --------------------------------------------------------

    def close(self, b_write_eos: bool = True) -> None:
        """Close the stream. Writes ``{"eos": true}`` first unless one was
        already written or ``b_write_eos=False``."""
        if self._f is None:
            return
        if b_write_eos and not self.b_eos_written:
            self.write_eos()
        self._f.close()
        self._f = None

    def __enter__(self) -> TrajectoryWriter:
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        # On an exception, do not stamp a clean eos: a truncated stream reads
        # as "solver stalled", which is the truth.
        self.close(b_write_eos=exc_type is None)


def write_trajectory(
    path: str | Path,
    header: TrajectoryHeader,
    samples: Iterable[Sample],
) -> Path:
    """Convenience one-shot producer: header + samples + eos. Returns the path."""
    with TrajectoryWriter(path, header) as writer:
        for sample in samples:
            writer.write_sample(sample)
    return Path(path)


# ──────────────────────────────────────────────────────────────────────────────
# Reader (the same format both sides parse)
# ──────────────────────────────────────────────────────────────────────────────


def _iter_objs(lines: Iterable[str]) -> Iterator[dict[str, Any]]:
    """Parse a line-iterable into JSON objects, skipping blank lines
    defensively (spec §1). Non-object lines are ignored (unknown records)."""
    for s_line in lines:
        s_line = s_line.strip()
        if not s_line:
            continue
        obj = json.loads(s_line)
        if isinstance(obj, dict):
            yield obj


def read_header(path: str | Path) -> TrajectoryHeader:
    """Read + validate the header of a trajectory file. The first record MUST
    be a header (spec §1); raises ValueError otherwise, and on version
    mismatch (spec §6)."""
    with Path(path).open("r", encoding="utf-8") as f:
        for obj in _iter_objs(f):
            if not is_header_obj(obj):
                raise ValueError(f"Malformed sdm-traj stream {path}: first record is not a header")
            return TrajectoryHeader.from_obj(obj)
    raise ValueError(f"Malformed sdm-traj stream {path}: empty stream (no header)")


def iter_samples(lines: Iterable[str]) -> Iterator[Sample]:
    """Generator over any open line-iterable (file handle today, socket
    line-stream in v2) yielding Sample records.

    - Blank lines are skipped.
    - Header records encountered mid-stream are validated (version gate) but
      not yielded: hold-state reset is the consumer's job.
    - Unknown records (neither header, sample, nor eos) are ignored.
    - ``{"eos": true}`` marks solve completion, NOT end-of-parse: a producer
      may re-header and keep writing after it (solver restart / reconnect,
      spec §1/§4.2), so iteration continues to end of input: parity with
      sdm-view's reader, which parses the whole stream.
    """
    for obj in _iter_objs(lines):
        if is_header_obj(obj):
            TrajectoryHeader.from_obj(obj)  # version gate, even mid-stream
        elif is_sample_obj(obj):
            yield Sample.from_obj(obj)
        # else: ignore eos / unknown records (forward compat, spec §1)


def read_trajectory(path: str | Path) -> tuple[TrajectoryHeader, list[Sample]]:
    """Read a whole trajectory file: (header, samples). The first record MUST
    be a header; raises ValueError otherwise. A mid-stream re-header (solver
    restart) replaces the previous header and resets all hold state (spec
    §1/§5.2). Prior samples belong to the old header and are dropped, matching
    sdm-view's ``read_trajectory``. ``eos`` marks solve completion, not
    end-of-parse, so reading continues (a restart stream may re-header after
    it, spec §4.2). For streaming consumption use ``read_header`` +
    ``iter_samples`` instead."""
    header: TrajectoryHeader | None = None
    samples: list[Sample] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for obj in _iter_objs(f):
            if header is None:
                if not is_header_obj(obj):
                    raise ValueError(
                        f"Malformed sdm-traj stream {path}: first record is not a header"
                    )
                header = TrajectoryHeader.from_obj(obj)
            elif is_header_obj(obj):
                # Mid-stream re-header: replaces the header and resets hold
                # state. Samples accumulated so far are the old solve's.
                header = TrajectoryHeader.from_obj(obj)
                samples = []
            elif is_sample_obj(obj):
                samples.append(Sample.from_obj(obj))
            # else: ignore eos / unknown records
    if header is None:
        raise ValueError(f"Malformed sdm-traj stream {path}: empty stream (no header)")
    return header, samples
