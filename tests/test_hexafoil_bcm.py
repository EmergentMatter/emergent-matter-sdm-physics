"""Pins the hexafoil BCM solver to its closed-form small-angle oracle.

The invariants this file pins: at small rotation the autodiff stiffness
from analyze_rotation equals linear_rotational_stiffness's closed form,
torque is linear in rotation, total stiffness scales inversely with the
stage count, the zero-inner-radius geometry degenerates to the cantilever
result, and the solver actually runs in float64 (the solvers package
enables jax x64 at import; a float32 downcast regression fails here).

Needs jax, so the whole module skips loudly without the solvers extra.
"""

from __future__ import annotations

import pytest

jax = pytest.importorskip(
    "jax", reason="solver tests need jax; install the 'solvers' extra (uv sync --extra solvers)"
)

import jax.numpy as jnp

from sdm_physics.solvers.hexafoil_bcm import (
    HexafoilGeometry,
    analyze_rotation,
    linear_rotational_stiffness,
)

# PA12 print material, the org default. MPa.
D_E_MPA = 1850.0

# The BCM strain energy is exactly quadratic in the imposed rotation
# (geometric-stiffness feedback into K is deliberately deferred), so the
# autodiff stiffness equals the closed form up to pure float64 roundoff.
# 1e-9 relative leaves generous margin over that roundoff while sitting
# far below float32 resolution (about 1e-7 relative), so a downcast
# regression cannot pass.
REL_TOL_ORACLE = 1e-9

# Torque linearity compares two solves at different angles, so roundoff
# enters twice; same reasoning, same margin.
REL_TOL_LINEARITY = 1e-9

# Small imposed rotation (rad). Small enough to be the linear regime by
# any definition, large enough to stay far from underflow.
D_THETA_SMALL = 1e-4


def _make_geometry(**overrides: float | int) -> HexafoilGeometry:
    kwargs: dict = {
        "d_cylinder_length": 10.0,
        "d_blade_thickness": 0.8,
        "d_cylinder_inner_radius": 12.0,
        "d_inner_radius": 3.0,
        "n_stages": 2,
        "n_foil_count": 6,
    }
    kwargs.update(overrides)
    return HexafoilGeometry(**kwargs)


def test_autodiff_stiffness_matches_closed_form_oracle_at_small_angle() -> None:
    geom = _make_geometry()
    res = analyze_rotation(geom, D_E_MPA, D_THETA_SMALL)
    d_k_oracle = linear_rotational_stiffness(geom, D_E_MPA)
    assert float(res.d_rotational_stiffness) == pytest.approx(d_k_oracle, rel=REL_TOL_ORACLE)


def test_torque_is_linear_in_rotation_at_small_angles() -> None:
    geom = _make_geometry()
    d_torque_one = float(analyze_rotation(geom, D_E_MPA, D_THETA_SMALL).d_torque)
    d_torque_two = float(analyze_rotation(geom, D_E_MPA, 2.0 * D_THETA_SMALL).d_torque)
    assert d_torque_two == pytest.approx(2.0 * d_torque_one, rel=REL_TOL_LINEARITY)


def test_torque_equals_stiffness_times_rotation_at_small_angles() -> None:
    geom = _make_geometry()
    res = analyze_rotation(geom, D_E_MPA, D_THETA_SMALL)
    d_k_oracle = linear_rotational_stiffness(geom, D_E_MPA)
    assert float(res.d_torque) == pytest.approx(d_k_oracle * D_THETA_SMALL, rel=REL_TOL_LINEARITY)


def test_total_stiffness_scales_inversely_with_stage_count() -> None:
    d_k_single = linear_rotational_stiffness(_make_geometry(n_stages=1), D_E_MPA)
    for n_stages in (2, 3, 4):
        geom = _make_geometry(n_stages=n_stages)
        d_k_closed = linear_rotational_stiffness(geom, D_E_MPA)
        assert d_k_closed == pytest.approx(d_k_single / n_stages, rel=REL_TOL_ORACLE)
        d_k_bcm = float(analyze_rotation(geom, D_E_MPA, D_THETA_SMALL).d_rotational_stiffness)
        assert d_k_bcm == pytest.approx(d_k_single / n_stages, rel=REL_TOL_ORACLE)


def test_zero_inner_radius_degenerates_to_cantilever_blades() -> None:
    # r_0 = 0 collapses the per-stage closed form to 2 N E I / L, the
    # familiar result for 2N cantilever blades terminating on the axis
    # (see linear_rotational_stiffness's docstring).
    geom = _make_geometry(d_inner_radius=0.0, n_stages=1)
    d_inertia = geom.d_cylinder_length * geom.d_blade_thickness**3 / 12.0
    d_length = geom.d_blade_radial_length
    d_k_expected = 2.0 * geom.n_foil_count * D_E_MPA * d_inertia / d_length
    assert linear_rotational_stiffness(geom, D_E_MPA) == pytest.approx(
        d_k_expected, rel=REL_TOL_ORACLE
    )
    d_k_bcm = float(analyze_rotation(geom, D_E_MPA, D_THETA_SMALL).d_rotational_stiffness)
    assert d_k_bcm == pytest.approx(d_k_expected, rel=REL_TOL_ORACLE)


def test_solver_runs_in_float64_not_downcast_to_float32() -> None:
    # Regression gate: with jax x64 off, the float64 request at
    # analyze_rotation's asarray site is SILENTLY downcast to float32 and
    # every result field follows. This fails if that downcast returns.
    res = analyze_rotation(_make_geometry(), D_E_MPA, D_THETA_SMALL)
    assert res.d_delta_theta.dtype == jnp.float64
    assert res.d_torque.dtype == jnp.float64
    assert res.d_rotational_stiffness.dtype == jnp.float64
