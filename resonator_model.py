"""
resonator_model.py
==================
Wavelength-selective resonant filter model.

Physical context
----------------
The resonant element is any narrowband wavelength filter coupled to the FBG
output: a Fabry–Pérot etalon, a micro-ring resonator, a fibre loop mirror, or
a coupled-resonator optical waveguide (CROW).  Its transmission (or reflection)
as a function of wavelength is well approximated by a Lorentzian:

    H_r(λ) = 1 / [1 + (2(λ − λ_r) / Γ_r)²]

where λ_r is the resonance wavelength and Γ_r its FWHM linewidth.

The resonator acts as a spectral gate: it transmits the portion of the FBG
reflection that falls within Γ_r of λ_r.  The nonlinear feedback (modelled in
bistable_solver.py) is applied to the light that passes through this filter.
When λ_B ≈ λ_r, the combined response S = R_fbg · H_r is near its peak, and
the system is close to its switching threshold.

Design note
-----------
Choosing λ_r slightly offset from λ_B0 places the quiescent operating point
on the steepest slope of S(λ_B), which maximises dI_out/dλ_B and therefore
the sensitivity of the bistable interrogator.
"""

import numpy as np


def resonator_response(
    lambda_vals: np.ndarray,
    lambda_r: float,
    Gamma_r: float,
) -> np.ndarray:
    """
    Compute the Lorentzian resonator power transmission.

    Parameters
    ----------
    lambda_vals : np.ndarray
        Wavelength axis [nm], shape (N,).
    lambda_r : float
        Resonance centre wavelength [nm].
    Gamma_r : float
        Resonator linewidth (FWHM) [nm].  Smaller Γ_r → sharper filter →
        larger dS/dλ on the edge, potentially higher sensitivity.

    Returns
    -------
    H_r : np.ndarray
        Transmission spectrum (dimensionless, 0–1), shape (N,).

    Notes
    -----
    The filter satisfies H_r(λ_r ± Γ_r/2) = 0.5 by construction.
    """
    detuning = 2.0 * (lambda_vals - lambda_r) / Gamma_r  # normalised offset
    H_r = 1.0 / (1.0 + detuning ** 2)
    return H_r


def resonator_response_at(
    lambda_probe: float,
    lambda_r: float,
    Gamma_r: float,
) -> float:
    """
    Evaluate the resonator transmission at a single wavelength.

    Parameters
    ----------
    lambda_probe : float
        Wavelength at which to evaluate [nm].
    lambda_r : float
        Resonance centre wavelength [nm].
    Gamma_r : float
        Resonator linewidth [nm].

    Returns
    -------
    float
        Scalar transmission value.
    """
    return float(resonator_response(
        np.array([lambda_probe]), lambda_r, Gamma_r
    )[0])


# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------

def _run_checks(lambda_r: float = 1550.05, Gamma_r: float = 0.15):
    """
    Verify the resonator response is centred at λ_r and has the correct FWHM.
    """
    lam = np.linspace(lambda_r - 1.0, lambda_r + 1.0, 10_000)
    H = resonator_response(lam, lambda_r, Gamma_r)

    # Peak should be at λ_r
    peak_lam = lam[np.argmax(H)]
    assert abs(peak_lam - lambda_r) < 1e-3, (
        f"Peak at {peak_lam:.4f} nm, expected {lambda_r:.4f} nm"
    )
    print(f"[resonator_model] Peak wavelength: {peak_lam:.4f} nm  ✓")

    # Peak value should be 1.0
    assert abs(H.max() - 1.0) < 1e-6, f"Peak value {H.max():.6f}, expected 1.0"
    print(f"[resonator_model] Peak transmission: {H.max():.6f}  ✓")

    # FWHM check
    half_max_idx = np.where(H >= 0.5)[0]
    measured_fwhm = lam[half_max_idx[-1]] - lam[half_max_idx[0]]
    assert abs(measured_fwhm - Gamma_r) < 0.01, (
        f"FWHM {measured_fwhm:.4f} nm, expected {Gamma_r:.4f} nm"
    )
    print(f"[resonator_model] Measured FWHM: {measured_fwhm:.4f} nm  "
          f"(target {Gamma_r:.4f} nm)  ✓")


if __name__ == "__main__":
    _run_checks()
