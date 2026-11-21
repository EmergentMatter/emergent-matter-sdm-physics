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
from pathlib import Path
from typing import IO, Any

from sdm_physics.trajectory_types import (
    TRAJ_EXTENSION,
    TRAJ_FORMAT,
    TRAJ_VERSION,
    Channel,
    FieldRef,
    Sample,
    TrajectoryHeader,
)

__all__ = [
    "TRAJ_EXTENSION",
    "TRAJ_FORMAT",
    "TRAJ_VERSION",
    "Channel",
    "FieldRef",
    "Sample",
    "TrajectoryHeader",
    "TrajectoryWriter",
    "is_eos_obj",
    "is_header_obj",
    "is_sample_obj",
    "iter_samples",
    "read_header",
    "read_trajectory",
    "write_trajectory",
]

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
        self._f.write(json.dumps(obj, separators=(",", ":"), allow_nan=False) + "\n")
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


def _check_increasing(d_last_t: float | None, d_t: float) -> None:
    """Spec §4: ``t`` is strictly increasing within a stream (per header)."""
    if d_last_t is not None and d_t <= d_last_t:
        raise ValueError(f"Malformed sdm-traj stream: t={d_t!r} does not follow t={d_last_t!r}")


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
    - Non-increasing ``t`` raises ValueError (spec §4).
    - Unknown records (neither header, sample, nor eos) are ignored.
    - ``{"eos": true}`` marks solve completion, NOT end-of-parse: a producer
      may re-header and keep writing after it (solver restart / reconnect,
      spec §1/§4.2), so iteration continues to end of input: parity with
      sdm-view's reader, which parses the whole stream.
    """
    d_last_t: float | None = None
    for obj in _iter_objs(lines):
        if is_header_obj(obj):
            TrajectoryHeader.from_obj(obj)  # version gate, even mid-stream
            d_last_t = None  # a re-header starts a new solve
        elif is_sample_obj(obj):
            sample = Sample.from_obj(obj)
            _check_increasing(d_last_t, sample.d_t)
            d_last_t = sample.d_t
            yield sample
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
                sample = Sample.from_obj(obj)
                _check_increasing(samples[-1].d_t if samples else None, sample.d_t)
                samples.append(sample)
            # else: ignore eos / unknown records
    if header is None:
        raise ValueError(f"Malformed sdm-traj stream {path}: empty stream (no header)")
    return header, samples
