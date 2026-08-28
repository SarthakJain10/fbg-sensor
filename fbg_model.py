"""
fbg_model.py
============
Fiber Bragg Grating (FBG) reflection spectrum model.

Assumption / simplification
----------------------------
We use a Lorentzian approximation for the FBG reflectivity:

    R_fbg(λ) = R0 / [1 + (2(λ − λ_B) / Γ_fbg)²]

This is equivalent to the power spectrum of an exponentially decaying impulse
response and closely matches the envelope of a uniform grating's reflectivity
for moderate index-contrast gratings (κL ≈ 2–4).  For strongly over-coupled
gratings the sidelobes are suppressed and the Lorentzian is a good engineering
approximation.  If higher fidelity is needed the function signature is
compatible with a drop-in replacement using coupled-mode theory (e.g. via
scipy transfer-matrix integration).

Physical meaning
----------------
An FBG is a periodic perturbation of the fibre refractive index. It reflects
light whose wavelength satisfies the Bragg condition λ_B = 2 n_eff Λ, where
Λ is the grating period and n_eff is the effective refractive index.  Strain
or temperature shifts n_eff and/or Λ, moving λ_B and therefore the reflection
peak.  Conventional interrogation tracks this peak; the bistable scheme
instead senses the power change on the slope of the combined response.
"""

import numpy as np


def fbg_reflection(
    lambda_vals: np.ndarray,
    lambda_B: float,
    Gamma_fbg: float,
    R0: float = 1.0,
) -> np.ndarray:
    """
    Compute the Lorentzian FBG reflection spectrum.

    Parameters
    ----------
    lambda_vals : np.ndarray
        Wavelength axis [nm], shape (N,).
    lambda_B : float
        Bragg wavelength [nm].  This is the parameter that shifts with
        strain or temperature.
    Gamma_fbg : float
        Full-width at half-maximum (FWHM) linewidth of the FBG reflection
        peak [nm].
    R0 : float, optional
        Peak reflectivity (dimensionless, 0 < R0 ≤ 1).  Default is 1.0
        (lossless grating with 100 % peak reflectivity).

    Returns
    -------
    R_fbg : np.ndarray
        Reflection spectrum, shape (N,), values in [0, R0].

    Notes
    -----
    The Lorentzian FWHM satisfies: R_fbg(λ_B ± Γ_fbg/2) = R0/2.
    """
    # Normalised detuning: dimensionless frequency offset from Bragg peak.
    # Factor of 2 in the numerator gives the FWHM = Gamma_fbg convention.
    detuning = 2.0 * (lambda_vals - lambda_B) / Gamma_fbg  # shape (N,)

    R_fbg = R0 / (1.0 + detuning ** 2)
    return R_fbg


def fbg_reflection_at(
    lambda_B_probe: float,
    lambda_B: float,
    Gamma_fbg: float,
    R0: float = 1.0,
) -> float:
    """
    Evaluate R_fbg at a single probe wavelength.

    Convenience wrapper used by the bistable solver when λ_B (the Bragg
    wavelength being swept) is the argument and we probe at a fixed wavelength.

    Parameters
    ----------
    lambda_B_probe : float
        Probe wavelength at which to evaluate reflectivity [nm].
    lambda_B : float
        Current Bragg wavelength [nm].
    Gamma_fbg : float
        FBG linewidth [nm].
    R0 : float, optional
        Peak reflectivity. Default 1.0.

    Returns
    -------
    float
        Scalar reflectivity value.
    """
    return float(fbg_reflection(
        np.array([lambda_B_probe]), lambda_B, Gamma_fbg, R0
    )[0])


# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------

def _run_checks(lambda_B: float = 1550.0, Gamma_fbg: float = 0.2):
    """
    Verify the spectrum is centred at λ_B and has the correct FWHM.
    Prints pass/fail messages.
    """
    import numpy as np

    lam = np.linspace(lambda_B - 1.0, lambda_B + 1.0, 10_000)
    R = fbg_reflection(lam, lambda_B, Gamma_fbg, R0=1.0)

    # Peak should be at λ_B
    peak_lam = lam[np.argmax(R)]
    assert abs(peak_lam - lambda_B) < 1e-3, (
        f"Peak at {peak_lam:.4f} nm, expected {lambda_B:.4f} nm"
    )
    print(f"[fbg_model] Peak wavelength: {peak_lam:.4f} nm  ✓")

    # Peak value should be R0 = 1.0
    assert abs(R.max() - 1.0) < 1e-6, f"Peak value {R.max():.6f}, expected 1.0"
    print(f"[fbg_model] Peak reflectivity: {R.max():.6f}  ✓")

    # FWHM check: find wavelengths where R = 0.5
    half_max_idx = np.where(R >= 0.5)[0]
    measured_fwhm = lam[half_max_idx[-1]] - lam[half_max_idx[0]]
    assert abs(measured_fwhm - Gamma_fbg) < 0.01, (
        f"FWHM {measured_fwhm:.4f} nm, expected {Gamma_fbg:.4f} nm"
    )
    print(f"[fbg_model] Measured FWHM: {measured_fwhm:.4f} nm  "
          f"(target {Gamma_fbg:.4f} nm)  ✓")


if __name__ == "__main__":
    _run_checks()
