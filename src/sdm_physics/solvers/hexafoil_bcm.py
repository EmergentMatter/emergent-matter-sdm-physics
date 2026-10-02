"""Beam Constraint Model analysis for the hexafoil rotary flexure.

Kinematic model:
  - Cylinder 0 (lower) is grounded; cylinder n_stages (top) rotates by
    d_delta_theta about the z-axis.
  - Each cylinder carries n_foil_count blades on its inner wall. Adjacent
    cylinders share a stage: their blades interleave and fuse at the
    centre of the stage into a hub.
  - Each blade is a clamped-clamped Euler-Bernoulli beam in pure
    tangential bending. Axial slant (d_axial_slant) is neglected; the
    beam axis is treated as radial with length = d_blade_radial_length
    and cross-section d_cylinder_length x d_blade_thickness.
  - Beam local frame: node 1 = hub side (inner), node 2 = cylinder side
    (outer); local +x points radially OUTWARD so local +z aligns with the
    global rotation axis and rigid z-rotations of the whole assembly are
    the zero-energy mode of the element.
  - End DOFs for one blade (weak-axis bending = tangential):
        v_hub  = theta_hub * r_inner         phi_hub = theta_hub
        v_cyl  = theta_cyl * r_cyl_inner     phi_cyl = theta_cyl

Multi-stage reduction:
  - For UNIFORM geometry (every cylinder has the same OD), every stage
    is an identical spring in series. Rotational equilibrium places each
    interior cylinder at k/N · Δθ_total and each hub at (s + 1/2)/N · Δθ.
    Every stage therefore sees an effective rotation Δθ_stage = Δθ / N;
    the series composition gives
        K_total = K_per_stage / N      (stages in series)
        T_total = T_stage              (same torque through the chain)
    and every blade experiences the SAME deformation magnitude, so peak
    stresses are the per-stage values.
  - For tapered geometry (a_outer_radii set) stages have unequal stiffness
    and this reduction no longer holds. Not yet implemented; raise.

BCM layer (on top of linear Euler-Bernoulli):
  - Kinematic axial shortening of each bent blade is computed; by n-fold
    symmetry the hubs cannot translate radially, so each blade is held at
    fixed end-to-end length and develops axial tension.
  - Tension adds to bending fiber stress at worst-case fibers, and to
    leading order in Δθ softens / stiffens the transverse stiffness. We
    expose the induced tension and combined stress; geometric-stiffness
    feedback into K_θ is deferred to a later pass.

All math is closed-form and JAX-differentiable.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from sdm_physics.solvers.bcm import analyze_beam


@dataclass(frozen=True)
class HexafoilGeometry:
    """The geometry this solver reads, decoupled from any CEM's Parameters.

    Ported from the internal prototype (2026-08-15) with the BCM itself. The
    prototype's solver read that CEM's own parameters class, which would have
    made the PHYSICS layer depend on one CEM repo. That is backwards: physics
    reads `.sdm` files, so the seam belongs at the .sdm, not at a Python
    class. Every field below is either a declared hexafoil param or derived
    from two of them.
    """

    d_cylinder_length: float
    d_blade_thickness: float
    d_cylinder_inner_radius: float
    d_inner_radius: float
    n_stages: int
    n_foil_count: int
    a_outer_radii: tuple | None = None

    @property
    def d_blade_radial_length(self) -> float:
        """Hub to cylinder wall: the beam's span."""
        return self.d_cylinder_inner_radius - self.d_inner_radius

    def validate(self) -> None:
        if self.d_cylinder_inner_radius <= 0:
            raise ValueError("d_cylinder_inner_radius must be > 0")
        if self.d_blade_radial_length < 1.0:
            raise ValueError(
                f"d_blade_radial_length ({self.d_blade_radial_length:.2f} mm) "
                "is below the 1 mm floor the beam model is meaningful at"
            )
        if self.n_stages < 1 or self.n_foil_count < 1:
            raise ValueError("n_stages and n_foil_count must be >= 1")

    @classmethod
    def from_sdm(cls, doc: dict) -> HexafoilGeometry:
        """Read a hexafoil `.sdm`'s param block. THE seam between layers."""
        p = doc.get("params") or {}

        def val(name: str) -> float:
            spec = p.get(name)
            if spec is None:
                raise KeyError(f"{name} is not a param of this .sdm")
            if isinstance(spec, dict):
                raw = spec.get("value")
                if raw is None:
                    raise KeyError(f"{name} has a param block with no value in this .sdm")
                return float(raw)
            return float(spec)

        return cls(
            d_cylinder_length=val("cylinder_length"),
            d_blade_thickness=val("blade_thickness"),
            d_cylinder_inner_radius=val("cylinder_inner_radius"),
            d_inner_radius=val("inner_radius"),
            n_stages=int(val("stages")),
            n_foil_count=int(val("foil_count")),
        )


@dataclass(frozen=True)
class HexafoilBCMResult:
    """Aggregated BCM analysis of a hexafoil under rotation.

    Fields are JAX scalars so gradients propagate end-to-end.
    """

    d_delta_theta: jnp.ndarray  # total imposed relative rotation (rad)
    d_delta_theta_per_stage: jnp.ndarray  # Δθ / n_stages (rad)
    d_torque: jnp.ndarray  # restoring torque at the input end (N·mm)
    d_rotational_stiffness: jnp.ndarray  # dT/dθ_total (N·mm / rad)
    d_total_strain_energy: jnp.ndarray  # summed over all blades (N·mm)
    d_peak_bending_stress: jnp.ndarray  # max across blades (MPa)
    d_peak_axial_stress: jnp.ndarray  # from constrained arc-length shortening (MPa)
    d_peak_combined_stress: jnp.ndarray  # bending + axial on worst-case fiber (MPa)
    d_blade_axial_shortening: jnp.ndarray  # per-blade kinematic Δu_kin (mm)
    d_hub_rotation: jnp.ndarray  # first hub's rotation (rad); mostly informational


def _per_blade_end_dofs(
    d_theta_cyl: jnp.ndarray,
    d_theta_hub: jnp.ndarray,
    d_r_cyl_inner: float,
    d_r_inner: float,
) -> jnp.ndarray:
    """Build the 4-DOF vector for a single blade.

    Node 1 = hub side (inner), node 2 = cylinder side (outer). The beam's
    local +x therefore points radially OUTWARD so that local +z aligns
    with the global +z rotation axis; this keeps rigid z-rotations of the
    whole assembly as the zero-energy mode of the element.

    Returns [v_hub, phi_hub, v_cyl, phi_cyl].
    """
    return jnp.array(
        [
            d_theta_hub * d_r_inner,
            d_theta_hub,
            d_theta_cyl * d_r_cyl_inner,
            d_theta_cyl,
        ]
    )


def _analyze_one_stage(
    d_E: float,
    d_b: float,
    d_h: float,
    d_L: float,
    d_r_cyl_inner: float,
    d_r_inner: float,
    n_foil_count: int,
    d_delta_theta_stage: jnp.ndarray,
) -> tuple[jnp.ndarray, ...]:
    """Analyse a single stage (two cylinders + one hub) at its own Δθ_stage.

    Returns:
        (per-stage strain energy, per-stage torque, per-stage K_θ,
        peak_bending, peak_axial, peak_combined, peak_shortening, theta_hub).
    """
    d_theta_hub = 0.5 * d_delta_theta_stage

    d_dofs_lower = _per_blade_end_dofs(
        d_theta_cyl=jnp.asarray(0.0),
        d_theta_hub=d_theta_hub,
        d_r_cyl_inner=d_r_cyl_inner,
        d_r_inner=d_r_inner,
    )
    d_dofs_upper = _per_blade_end_dofs(
        d_theta_cyl=d_delta_theta_stage,
        d_theta_hub=d_theta_hub,
        d_r_cyl_inner=d_r_cyl_inner,
        d_r_inner=d_r_inner,
    )

    res_lo = analyze_beam(d_E, d_b, d_h, d_L, d_dofs_lower, b_axially_constrained=True)
    res_hi = analyze_beam(d_E, d_b, d_h, d_L, d_dofs_upper, b_axially_constrained=True)

    d_U_stage = n_foil_count * (res_lo.strain_energy + res_hi.strain_energy)

    def _energy_stage(dth_stage: jnp.ndarray) -> jnp.ndarray:
        hub = 0.5 * dth_stage
        d0 = _per_blade_end_dofs(jnp.asarray(0.0), hub, d_r_cyl_inner, d_r_inner)
        d1 = _per_blade_end_dofs(dth_stage, hub, d_r_cyl_inner, d_r_inner)
        r0 = analyze_beam(d_E, d_b, d_h, d_L, d0, b_axially_constrained=True)
        r1 = analyze_beam(d_E, d_b, d_h, d_L, d1, b_axially_constrained=True)
        return n_foil_count * (r0.strain_energy + r1.strain_energy)

    d_T_stage = jax.grad(_energy_stage)(d_delta_theta_stage)
    d_K_stage = jax.grad(jax.grad(_energy_stage))(d_delta_theta_stage)

    d_peak_bending = jnp.maximum(res_lo.bending_stress, res_hi.bending_stress)
    d_peak_axial = jnp.maximum(res_lo.axial_stress, res_hi.axial_stress)
    d_peak_combined = jnp.maximum(res_lo.combined_stress, res_hi.combined_stress)
    d_peak_shortening = jnp.minimum(res_lo.axial_shortening, res_hi.axial_shortening)

    return (
        d_U_stage,
        d_T_stage,
        d_K_stage,
        d_peak_bending,
        d_peak_axial,
        d_peak_combined,
        d_peak_shortening,
        d_theta_hub,
    )


def analyze_rotation(
    params: HexafoilGeometry,
    d_E: float,
    d_delta_theta: float,
) -> HexafoilBCMResult:
    """Analyse a hexafoil (any number of stages) under imposed relative rotation.

    Args:
        params: Hexafoil geometry. Tapered cylinders (a_outer_radii set) are
            not yet supported for multi-stage; a ValueError is raised.
        d_E: Young's modulus (MPa).
        d_delta_theta: Total relative rotation of the top cylinder w.r.t.
            the grounded bottom cylinder (rad).

    Returns:
        HexafoilBCMResult.
    """
    params.validate()
    if params.n_stages > 1 and params.a_outer_radii is not None:
        raise NotImplementedError(
            "BCM multi-stage analysis currently assumes uniform cylinder "
            "geometry. Tapered (a_outer_radii) multi-stage needs a full "
            "interior-DOF solve, deferred."
        )

    d_b = params.d_cylinder_length
    d_h = params.d_blade_thickness
    d_L = params.d_blade_radial_length
    d_r_cyl_inner = params.d_cylinder_inner_radius
    d_r_inner = params.d_inner_radius
    n_stages = params.n_stages

    # float64 is real here, not aspirational: the solvers package __init__
    # enables jax's x64 mode at import, so this coercion is honored. With
    # x64 off, jax silently downcasts this to float32, which is exactly the
    # regression tests/test_hexafoil_bcm.py pins against.
    d_theta_total = jnp.asarray(d_delta_theta, dtype=jnp.float64)
    d_delta_theta_stage = d_theta_total / n_stages

    (
        d_U_stage,
        d_T_stage,
        d_K_stage,
        d_peak_bending,
        d_peak_axial,
        d_peak_combined,
        d_peak_shortening,
        d_theta_hub,
    ) = _analyze_one_stage(
        d_E,
        d_b,
        d_h,
        d_L,
        d_r_cyl_inner,
        d_r_inner,
        params.n_foil_count,
        d_delta_theta_stage,
    )

    # Series composition: torque is transmitted unchanged through each
    # stage; strain energy adds; rotational stiffness at the INPUT end
    # equals per-stage stiffness divided by n_stages.
    d_total_U = n_stages * d_U_stage
    d_torque = d_T_stage
    d_K_theta = d_K_stage / n_stages

    return HexafoilBCMResult(
        d_delta_theta=d_theta_total,
        d_delta_theta_per_stage=d_delta_theta_stage,
        d_torque=d_torque,
        d_rotational_stiffness=d_K_theta,
        d_total_strain_energy=d_total_U,
        d_peak_bending_stress=d_peak_bending,
        d_peak_axial_stress=d_peak_axial,
        d_peak_combined_stress=d_peak_combined,
        d_blade_axial_shortening=d_peak_shortening,
        d_hub_rotation=d_theta_hub,
    )


def linear_rotational_stiffness(
    params: HexafoilGeometry,
    d_E: float,
) -> float:
    """Closed-form linear rotational stiffness K_θ = dT/dθ at θ=0.

    Useful as a sanity check, a cheap objective for topology sweeps,
    and a unit-test oracle. Per-stage derivation: assemble all 2N blades,
    take the hub's internal rotation DOF to its energy-minimising value,
    and differentiate the resulting quadratic form in Δθ_stage twice:

        K_stage = 2 * N * (E * I / L^3) * (3 r_0² + 3 L r_0 + L²)

    where N = n_foil_count, L = d_blade_radial_length, r_0 = d_inner_radius.
    For n_stages > 1 the stages act as identical torsional springs in
    series, so the stiffness at the INPUT end of the stack is

        K_total = K_stage / n_stages.

    Note that it is the INNER radius (where each blade meets the hub) that
    sets the leverage of hub-rotation onto blade deformation; the outer
    (cylinder) radius enters only through L = r_cyl_inner - r_0. For
    r_0 = 0 this collapses to K_stage = 2 N E I / L -- the familiar result
    for 2N cantilever blades terminating on the axis.

    Args:
        params: Hexafoil geometry.
        d_E: Young's modulus (MPa).

    Returns:
        K_θ at the input end of the stack (N·mm per rad).
    """
    d_I = params.d_cylinder_length * params.d_blade_thickness**3 / 12.0
    d_L = params.d_blade_radial_length
    d_r0 = params.d_inner_radius
    d_shape = 3.0 * d_r0**2 + 3.0 * d_L * d_r0 + d_L**2
    d_K_stage = 2.0 * params.n_foil_count * d_E * d_I * d_shape / d_L**3
    return d_K_stage / params.n_stages
