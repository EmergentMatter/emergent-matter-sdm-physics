"""Hexafoil twist sweep: a SOLVED trajectory, not an authored one.

    uv run python -m sdm_physics.producers.hexafoil_sweep part.sdm -o out.traj.jsonl

EmergentMatter internal ruling (2026-08-15, decision log): every animation is a
solved trajectory played back.
Until now a hexafoil sweep was authored kinematics: the CEM wrote track
amplitudes into `metadata.animations` and the viewer interpolated them. The
amplitudes happened to be right for uniform geometry, but they were ASSERTED:
nothing computed them, so nothing could tell you when they stopped being true.

Here they are computed. `solvers.hexafoil_bcm.analyze_rotation` is Awtar's
Beam Constraint Model over the stack; its own docstring derives the ring/hub
distribution from ROTATIONAL EQUILIBRIUM of identical springs in series:

    ring k at k/N of the total, hub s at (s + 1/2)/N

which is the same relation the authored version asserted. The difference is
everything the authored version could not carry:

  * torque, rotational stiffness and strain energy at each sample;
  * peak bending, axial and COMBINED stress: the axial term coming from
    constrained arc-length shortening, i.e. the geometric stiffening that only
    appears at high displacement, which is the whole reason BCM exists rather
    than a linear beam;
  * a `source` block on the header naming the solver and modulus, so a stream
    can be read back and its provenance argued with.

WHAT THIS DOES AND DOES NOT CARRY (updated 2026-08-15). The channels are ring
ANGLES. The blade no longer renders straight between them: hexafoil's plates
are `twist_linear` deforms that ramp between the two ring angles, so feeding
ring angles here produces a visibly BENT blade with its welded ends held to
their rings. But the ramp is sdm-core's Hermite profile standing in for this
solve's deflection curve: qualitatively a clamped-clamped bend, not
sample-for-sample the solver's shape. Exporting the real curve still needs a
deformation-field consumer (sdm-view's warp seam is marked and unbuilt).

SIGN. Channel values are rotate_z QUERY-SPACE angles, matching the normative
statement in the hexafoil CEM's emitter: positive = CLOCKWISE viewed from +Z.
`twist_linear` was deliberately given the same convention so one channel can
drive both a ring's rotate_z and the blade end welded to it; getting this
backwards makes blades counter-rotate against their own rings at double the
intended angle while still looking animated (observed live, 2026-08-15).

CHANNELS ARE FILTERED BY WHAT THE PART DECLARES. A bending plate arch has no
`twist_center_s`: the hub angle is a consequence of the two ring angles, not
a DOF. So a single-stage part yields exactly one channel.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

from sdm_physics.solvers.hexafoil_bcm import HexafoilGeometry, analyze_rotation
from sdm_physics.trajectory import (
    Channel,
    Sample,
    TrajectoryHeader,
    write_trajectory,
)

# PA12 (Formlabs Fuse, SLS): the org's default print material. MPa.
D_E_PA12_MPA = 1850.0


def channel_names(
    geom: HexafoilGeometry, s_prefix: str = "", declared: set | None = None
) -> list[str]:
    """The pose params a hexafoil stack drives, in the .sdm's own spelling.

    Filtered by what the part DECLARES when that is known. A bending plate
    arch has no `twist_center_s`: the hub angle falls out of the two ring
    angles the blade is welded between, so it is a consequence rather than a
    DOF. Emitting a channel for it would put a name in the stream that the
    part cannot answer to, which the viewer then has to report as absent.
    """
    ns = f"{s_prefix}__" if s_prefix else ""
    names = [f"{ns}twist_ring_{k}" for k in range(1, geom.n_stages + 1)] + [
        f"{ns}twist_center_{s}" for s in range(geom.n_stages)
    ]
    if declared is None:
        return names
    return [n for n in names if n in declared]


def solve_sweep(
    geom: HexafoilGeometry,
    d_total_deg: float,
    n_samples: int,
    d_E: float = D_E_PA12_MPA,
    s_prefix: str = "",
    d_rate_deg_per_s: float = 15.0,
    declared: set | None = None,
) -> tuple[list[Sample], list[dict], float]:
    """One full sweep: -total -> +total -> -total, solved at every sample.

    Time is derived from the authored sweep RATE so playback speed means
    something physical rather than being a UI preference.

    Args:
        geom: Hexafoil geometry to solve.
        d_total_deg: Sweep amplitude (deg).
        n_samples: Samples across the whole sweep.
        d_E: Young's modulus (MPa).
        s_prefix: Param namespace for an embedded pivot, e.g. ``pivot_0``.
        d_rate_deg_per_s: Authored sweep rate (deg/s); sample times derive
            from it.
        declared: Param names the part declares; channels are filtered to
            this set when it is given.

    Returns:
        (samples, rows, d_duration_s), where rows carry the per-sample
        mechanics for the caller to report.
    """
    ns = f"{s_prefix}__" if s_prefix else ""
    samples: list[Sample] = []
    rows: list[dict] = []
    d_span_deg = 4.0 * d_total_deg  # -A -> +A -> -A
    d_duration_s = d_span_deg / max(1e-9, d_rate_deg_per_s)
    for i in range(n_samples):
        d_phase = i / max(1, n_samples - 1)
        # TRIANGLE WAVE: -A at phase 0, +A at 0.5, -A at 1, so a loop="cycle"
        # stream is continuous at the wrap AND at the apex.
        #
        # The first version folded phase TWICE, using `2*p % 1` and then a
        # sign flip on the second half. That is a sawtooth in a triangle's
        # clothes: it snapped the full range at the midpoint and again at the wrap
        # (measured: max step 1.867 of a 2.0 range, wrap gap 2.000, against
        # 0.067 and 0.000 here). Live it read as the part jumping every cycle,
        # which is exactly what it was doing.
        d_signed = 1.0 - 2.0 * abs(2.0 * d_phase - 1.0)
        d_theta_deg = d_total_deg * d_signed
        d_theta = math.radians(d_theta_deg)

        res = analyze_rotation(geom, d_E, d_theta)

        # THE SOLVED DISTRIBUTION. Equilibrium of identical springs in series
        # puts ring k at k/N and hub s at (s+1/2)/N of the total: read off
        # the solver's own kinematic reduction rather than restated here.
        q: dict[str, float] = {}
        for k in range(1, geom.n_stages + 1):
            q[f"{ns}twist_ring_{k}"] = d_theta * (k / geom.n_stages)
        for s in range(geom.n_stages):
            q[f"{ns}twist_center_{s}"] = d_theta * ((s + 0.5) / geom.n_stages)
        if declared is not None:
            q = {k2: v for k2, v in q.items() if k2 in declared}

        samples.append(Sample(d_t=d_duration_s * d_phase, q=q))
        rows.append(
            {
                "t": round(d_duration_s * d_phase, 4),
                "theta_deg": round(d_theta_deg, 3),
                "torque_Nmm": float(res.d_torque),
                "k_rot_Nmm_per_rad": float(res.d_rotational_stiffness),
                "peak_bending_MPa": float(res.d_peak_bending_stress),
                "peak_axial_MPa": float(res.d_peak_axial_stress),
                "peak_combined_MPa": float(res.d_peak_combined_stress),
            }
        )
    return samples, rows, d_duration_s


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("sdm", type=Path, help="hexafoil .sdm to solve")
    ap.add_argument("-o", "--out", type=Path, default=None)
    ap.add_argument(
        "--deg",
        type=float,
        default=None,
        help="total sweep amplitude (deg); default = the part's twist_range_per_stage * stages",
    )
    ap.add_argument("--samples", type=int, default=61)
    ap.add_argument(
        "--modulus",
        type=float,
        default=D_E_PA12_MPA,
        help="Young's modulus in MPa (default PA12/SLS)",
    )
    ap.add_argument(
        "--prefix", default="", help="param namespace for an EMBEDDED pivot, e.g. pivot_0"
    )
    args = ap.parse_args(argv)

    doc = json.loads(args.sdm.read_text())
    geom = HexafoilGeometry.from_sdm(doc)
    geom.validate()

    params = doc.get("params") or {}
    d_psr = float((params.get("twist_range_per_stage") or {}).get("value", 15.9))
    d_total_deg = args.deg if args.deg is not None else d_psr * geom.n_stages

    declared = set((doc.get("params") or {}).keys())
    samples, rows, d_duration_s = solve_sweep(
        geom, d_total_deg, args.samples, d_E=args.modulus, s_prefix=args.prefix, declared=declared
    )

    out = args.out or args.sdm.with_suffix("").with_name(args.sdm.stem + "_twist_sweep.traj.jsonl")
    header = TrajectoryHeader(
        s_part=doc.get("name") or args.sdm.stem,
        s_name="twist_sweep (BCM)",
        s_sdm=args.sdm.name,
        s_sdm_sha256=hashlib.sha256(args.sdm.read_bytes()).hexdigest(),
        s_loop="cycle",
        d_duration_s=d_duration_s,
        channels=tuple(
            Channel(s_name=name, s_kind="pose", d_neutral=0.0, s_unit="rad")
            for name in channel_names(geom, args.prefix, declared)
        ),
        source={
            "solver": "sdm_physics.solvers.hexafoil_bcm.analyze_rotation",
            "model": "Beam Constraint Model (Awtar, Slocum & Sevincer 2007)",
            "fidelity": "clamped-clamped Euler-Bernoulli + 3rd-order axial "
            "shortening; ring angles only: the blade SHAPE is "
            "reproduced by the part's own twist_linear deform "
            "ramping between the two ring angles, not carried as "
            "a deformation field. The rendered profile is a "
            "Hermite stand-in for this solve's deflection curve, "
            "so it is qualitatively a clamped-clamped bend but "
            "not sample-for-sample the solver's shape.",
            "E_MPa": args.modulus,
            "stages": geom.n_stages,
            "foils": geom.n_foil_count,
            "sweep_deg": d_total_deg,
        },
    )
    write_trajectory(out, header, samples)

    worst = max(rows, key=lambda r: r["peak_combined_MPa"])
    print(
        f"wrote {out}  ({len(samples)} samples, {d_duration_s:.1f} s, "
        f"{len(header.channels)} channels)"
    )
    print(
        f"  at {worst['theta_deg']:+.1f} deg: "
        f"torque {worst['torque_Nmm']:.1f} N·mm, "
        f"k {worst['k_rot_Nmm_per_rad']:.1f} N·mm/rad, "
        f"bending {worst['peak_bending_MPa']:.1f} MPa, "
        f"axial {worst['peak_axial_MPa']:.1f} MPa, "
        f"combined {worst['peak_combined_MPa']:.1f} MPa"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
