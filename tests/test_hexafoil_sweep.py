"""Pins the hexafoil sweep producer's happy path: a solved sweep emits the
channels the part declares, follows the triangle-wave shape, and reports
finite per-sample mechanics.

Drives the BCM solver, so the module skips loudly without the solvers
extra.
"""

from __future__ import annotations

import math

import pytest

pytest.importorskip(
    "jax", reason="producer tests drive the BCM solver; install the 'solvers' extra"
)

from sdm_physics.producers.hexafoil_sweep import channel_names, solve_sweep
from sdm_physics.solvers.hexafoil_bcm import HexafoilGeometry

D_AMPLITUDE_DEG = 5.0
N_SAMPLES = 5  # phase hits 0, 1/4, 1/2, 3/4, 1: both apexes and the wrap


def _make_geometry() -> HexafoilGeometry:
    return HexafoilGeometry(
        d_cylinder_length=10.0,
        d_blade_thickness=0.8,
        d_cylinder_inner_radius=12.0,
        d_inner_radius=3.0,
        n_stages=2,
        n_foil_count=6,
    )


def test_sweep_emits_ring_and_center_channels_when_all_declared() -> None:
    geom = _make_geometry()
    names = channel_names(geom)
    assert names == [
        "twist_ring_1",
        "twist_ring_2",
        "twist_center_0",
        "twist_center_1",
    ]


def test_channels_are_filtered_to_what_the_part_declares() -> None:
    geom = _make_geometry()
    declared = {"twist_ring_1", "twist_ring_2"}
    assert channel_names(geom, declared=declared) == ["twist_ring_1", "twist_ring_2"]
    samples, _, _ = solve_sweep(geom, D_AMPLITUDE_DEG, N_SAMPLES, declared=declared)
    for sample in samples:
        assert set(sample.q) == declared


def test_sweep_is_a_triangle_wave_continuous_at_wrap_and_apex() -> None:
    geom = _make_geometry()
    samples, rows, d_duration_s = solve_sweep(geom, D_AMPLITUDE_DEG, N_SAMPLES)
    assert len(samples) == N_SAMPLES
    assert d_duration_s > 0.0
    d_amplitude_rad = math.radians(D_AMPLITUDE_DEG)
    # The top ring carries the full imposed rotation. The triangle runs
    # -A at phase 0, +A at phase 1/2, back to -A at phase 1, so a cycle
    # loop is continuous at both the wrap and the apex.
    d_top_ring = [s.q["twist_ring_2"] for s in samples]
    assert d_top_ring[0] == pytest.approx(-d_amplitude_rad)
    assert d_top_ring[N_SAMPLES // 2] == pytest.approx(d_amplitude_rad)
    assert d_top_ring[-1] == pytest.approx(-d_amplitude_rad)
    assert rows[0]["theta_deg"] == pytest.approx(-D_AMPLITUDE_DEG)


def test_sweep_rows_report_finite_mechanics() -> None:
    geom = _make_geometry()
    _, rows, _ = solve_sweep(geom, D_AMPLITUDE_DEG, N_SAMPLES)
    for row in rows:
        for key in (
            "torque_Nmm",
            "k_rot_Nmm_per_rad",
            "peak_bending_MPa",
            "peak_axial_MPa",
            "peak_combined_MPa",
        ):
            assert math.isfinite(row[key]), f"{key} is not finite in {row}"
