"""Beam Constraint Model primitive for slender flexure blades.

A clamped-clamped Euler-Bernoulli beam element with the third-order
kinematic axial-shortening term from Awtar's Beam Constraint Model
(Awtar, Slocum, Sevincer 2007, "Characteristics of Beam-Based Flexure
Modules"). Captures the two effects that matter for a rotary flexure
at non-trivial rotation:

  1. Transverse bending stiffness (linear Euler-Bernoulli K matrix).
  2. Kinematic "arc-length shortening" of the bent blade, which --
     when the ends are axially constrained by symmetry -- induces an
     axial tension that geometrically stiffens the transverse stiffness
     and adds axial stress on top of bending stress.

Both live in closed form. Pure JAX, differentiable end-to-end. Higher-
fidelity nonlinearity (post-buckling, blade-on-blade contact) belongs in
a geometrically-exact Cosserat rod model, not here.

Sign convention -- 2D bending in the x-y plane:
  - Beam axis is x, from root (x=0) to tip (x=L).
  - Transverse deflection v is in the y direction.
  - End DOFs, root-first, tip-last:  d = [v_0, theta_0, v_L, theta_L].
  - Rotations theta are dv/dx at the respective ends.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp


def beam_stiffness_matrix(d_E: float, d_I: float, d_L: float) -> jnp.ndarray:
    """4x4 Euler-Bernoulli stiffness matrix for 2D bending.

    Maps end DOFs [v_0, theta_0, v_L, theta_L] to end reactions
    [V_0, M_0, V_L, M_L] (shear and moment at each end).

    Args:
        d_E: Young's modulus (MPa ≡ N/mm²).
        d_I: Weak-axis second moment of area (mm⁴).
        d_L: Beam length (mm).

    Returns:
        (4, 4) stiffness matrix with units N/mm, N, N/mm, N consistent
        with end DOFs in mm and rad.
    """
    d_EI_over_L3 = d_E * d_I / (d_L**3)
    d_L2 = d_L * d_L
    return d_EI_over_L3 * jnp.array(
        [
            [12.0, 6.0 * d_L, -12.0, 6.0 * d_L],
            [6.0 * d_L, 4.0 * d_L2, -6.0 * d_L, 2.0 * d_L2],
            [-12.0, -6.0 * d_L, 12.0, -6.0 * d_L],
            [6.0 * d_L, 2.0 * d_L2, -6.0 * d_L, 4.0 * d_L2],
        ]
    )


def _kinematic_coupling_matrix(d_L: float) -> jnp.ndarray:
    """4x4 coupling matrix C such that integral of (dv/dx)^2 dx = d^T C d.

    Derived from Hermite cubic shape functions for the clamped-clamped
    beam. Used to compute the BCM axial kinematic shortening via

        Δu_kin = -(1/2) d^T C d   (shortening of beam along its axis)

    Tabulated integrals of N_i'(ξ) N_j'(ξ) over [0, 1] give, after
    dimensional rescaling, the block below.
    """
    return jnp.array(
        [
            [6.0 / (5.0 * d_L), 1.0 / 10.0, -6.0 / (5.0 * d_L), 1.0 / 10.0],
            [1.0 / 10.0, 2.0 * d_L / 15.0, -1.0 / 10.0, -d_L / 30.0],
            [-6.0 / (5.0 * d_L), -1.0 / 10.0, 6.0 / (5.0 * d_L), -1.0 / 10.0],
            [1.0 / 10.0, -d_L / 30.0, -1.0 / 10.0, 2.0 * d_L / 15.0],
        ]
    )


@dataclass(frozen=True)
class BeamResult:
    """Result of a single-beam BCM analysis.

    All fields are JAX scalars or arrays so gradients propagate.
    """

    strain_energy: jnp.ndarray  # bending strain energy (N·mm)
    end_reactions: jnp.ndarray  # (4,) = [V_0, M_0, V_L, M_L]
    peak_moment: jnp.ndarray  # max |M| along the beam (N·mm)
    bending_stress: jnp.ndarray  # peak bending stress (MPa)
    axial_shortening: jnp.ndarray  # Δu_kin along beam axis (mm, negative = shorter)
    axial_strain: jnp.ndarray  # induced axial strain if ends axially fixed
    axial_stress: jnp.ndarray  # induced axial stress (MPa), tensile > 0
    combined_stress: jnp.ndarray  # bending + axial at root fiber (MPa)


def analyze_beam(
    d_E: float,
    d_b: float,
    d_h: float,
    d_L: float,
    d_end_dofs: jnp.ndarray,
    b_axially_constrained: bool = True,
) -> BeamResult:
    """Analyze a single clamped-clamped slender beam under prescribed end DOFs.

    Args:
        d_E: Young's modulus (MPa).
        d_b: Cross-section dimension perpendicular to bending plane, i.e.
             the strong-axis width (mm). For a hexafoil blade: d_cylinder_length.
        d_h: Cross-section thickness in the bending plane, i.e. the weak-axis
             thickness (mm). For a hexafoil blade: d_blade_thickness.
        d_L: Beam length (mm).
        d_end_dofs: (4,) = [v_0, theta_0, v_L, theta_L].
        b_axially_constrained: If True, the beam ends are held at fixed axial
            separation (as in the hexafoil by symmetry). The kinematic
            shortening then induces axial tension. If False, the beam is free
            to shorten and axial stress is zero.

    Returns:
        BeamResult with strain energy, reactions, stresses.

    Notes:
        - Peak bending moment is at one of the ends for a clamped-clamped
          beam with linear transverse loads (it is a cubic). We return
          max(|M_0|, |M_L|).
        - Combined stress uses simple superposition at the root fiber;
          for a precise combined yield check, the caller should apply a
          material-appropriate failure criterion.
    """
    d_I = d_b * d_h**3 / 12.0
    d_c = d_h / 2.0  # distance to extreme fiber (bending)

    K = beam_stiffness_matrix(d_E, d_I, d_L)
    C = _kinematic_coupling_matrix(d_L)

    d_reactions = K @ d_end_dofs
    d_strain_energy = 0.5 * d_end_dofs @ d_reactions

    d_M0 = d_reactions[1]
    d_ML = d_reactions[3]
    d_peak_moment = jnp.maximum(jnp.abs(d_M0), jnp.abs(d_ML))
    d_bending_stress = d_peak_moment * d_c / d_I  # = M c / I

    # Kinematic axial shortening of the beam along its own axis (Awtar BCM):
    #   Δu_kin = -(1/2) ∫ (dv/dx)^2 dx = -(1/2) d^T C d
    d_int_slope_sq = d_end_dofs @ (C @ d_end_dofs)
    d_axial_shortening = -0.5 * d_int_slope_sq

    if b_axially_constrained:
        # Enforcing ends at fixed separation requires elastic elongation
        # equal to -Δu_kin; the resulting axial strain and stress are tensile
        # when the beam has tried to shorten.
        d_axial_strain = -d_axial_shortening / d_L
        d_axial_stress = d_E * d_axial_strain
    else:
        d_axial_strain = jnp.zeros_like(d_axial_shortening)
        d_axial_stress = jnp.zeros_like(d_axial_shortening)

    # Combined stress at the extreme bending fiber of the most loaded end.
    # Axial tension adds to the tensile bending fiber; this is the worst case.
    d_combined_stress = d_bending_stress + jnp.maximum(d_axial_stress, 0.0)

    return BeamResult(
        strain_energy=d_strain_energy,
        end_reactions=d_reactions,
        peak_moment=d_peak_moment,
        bending_stress=d_bending_stress,
        axial_shortening=d_axial_shortening,
        axial_strain=d_axial_strain,
        axial_stress=d_axial_stress,
        combined_stress=d_combined_stress,
    )
