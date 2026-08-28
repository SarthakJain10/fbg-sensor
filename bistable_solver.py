"""
bistable_solver.py
==================
Nonlinear steady-state solver for the dispersive (Kerr) bistable resonator.

Physical model
--------------
We use the standard single-mode dispersive bistability model (Ikeda / nonlinear
Fabry-Pérot resonator with Kerr nonlinearity):

    I_out · [1 + (Δ − α·I_out)²] = I_in          … (*)

where
    I_out   – intra-cavity / output intensity [a.u.]
    I_in    – effective input driving field intensity = I0 · S(λ_B) [a.u.]
    Δ       – normalised cavity detuning (in units of half-linewidth γ = 1)
    α       – nonlinear (Kerr) coefficient [a.u.]⁻¹

Rearranging into a cubic in I_out:

    α²·I_out³ − 2Δα·I_out² + (1+Δ²)·I_out − I_in = 0      … (**)

This cubic has THREE real positive roots (S-shaped transfer curve) whenever
the cavity detuning satisfies |Δ| > √3, and the input I_in falls within the
bistable window [I_fold_low, I_fold_high].

The two fold (saddle-node) bifurcation points are found analytically by
setting d(I_in)/d(I_out) = 0:

    I_out_fold = [2Δα ± √(4Δ²α² − 12α²(1+Δ²))] / (6α²)
               = [2Δ ± √(4Δ² − 12(1+Δ²))] / (6α)

Physical meaning
----------------
- The lower stable branch: low-power state.
- The upper stable branch: high-power state.
- The middle branch: unstable (not physically accessible).
- Hysteresis: up-sweep follows lower branch until I_in = I_fold_high (jumps up);
              down-sweep follows upper branch until I_in = I_fold_low (jumps down).

As λ_B is swept, S(λ_B) = R_fbg(λ_probe | λ_B) · H_r(λ_probe) changes, and
I_in = I0 · S changes accordingly.  When I_in crosses a fold point, the output
switches abruptly, enabling fine-resolution sensing.

Bistability condition
---------------------
    Δ > √3 ≈ 1.732       (detuning criterion)
    I_fold_high > I_in > I_fold_low   (input in bistable window)

These are checked in `bistability_window()`.
"""

import numpy as np
from scipy.optimize import brentq


# ---------------------------------------------------------------------------
# Analytic fold-bifurcation points
# ---------------------------------------------------------------------------

def bistability_window(delta: float, alpha: float, I0: float = 1.0) -> dict:
    """
    Compute the bistable window (fold bifurcation points) analytically.

    Parameters
    ----------
    delta : float
        Normalised cavity detuning.  Must satisfy |delta| > sqrt(3).
    alpha : float
        Kerr nonlinear coefficient.
    I0 : float
        Input scale factor (for display; does not affect the window shape
        since I_in = I0·S and S ∈ [0,1]).

    Returns
    -------
    dict with keys:
        'bistable'       : bool   – True if |delta| > sqrt(3)
        'I_out_fold_low' : float  – I_out at lower fold (up-switching)
        'I_out_fold_high': float  – I_out at upper fold (down-switching)
        'I_in_fold_low'  : float  – I_in at lower fold  (= S_fold_high * I0)
        'I_in_fold_high' : float  – I_in at upper fold  (= S_fold_low * I0)
        'S_fold_low'     : float  – S value triggering down-switch
        'S_fold_high'    : float  – S value triggering up-switch
    """
    bistable = abs(delta) > np.sqrt(3.0)
    if not bistable:
        return {
            "bistable": False,
            "I_out_fold_low": None, "I_out_fold_high": None,
            "I_in_fold_low": None,  "I_in_fold_high": None,
            "S_fold_low": None,     "S_fold_high": None,
        }

    # dF/dI_out = 3α²I² − 4Δα·I + (1+Δ²) = 0
    A = 3.0 * alpha ** 2
    B = -4.0 * delta * alpha
    C = 1.0 + delta ** 2
    disc = B ** 2 - 4.0 * A * C  # = 16Δ²α² − 12α²(1+Δ²) = 4α²(4Δ²−3(1+Δ²))

    sqrt_disc = np.sqrt(max(disc, 0.0))
    I_fold1 = (-B - sqrt_disc) / (2.0 * A)  # lower fold (I_out_fold_low)
    I_fold2 = (-B + sqrt_disc) / (2.0 * A)  # upper fold (I_out_fold_high)

    # Corresponding I_in values (from the cubic at these I_out values)
    I_in_fold1 = _cubic_eval(I_fold1, delta, alpha)  # I_in at lower fold
    I_in_fold2 = _cubic_eval(I_fold2, delta, alpha)  # I_in at upper fold

    # S_fold = I_in_fold / I0 (normalised combined response at fold)
    return {
        "bistable":        True,
        "I_out_fold_low":  I_fold1,
        "I_out_fold_high": I_fold2,
        "I_in_fold_low":   I_in_fold1,
        "I_in_fold_high":  I_in_fold2,
        "S_fold_low":      I_in_fold1 / I0,
        "S_fold_high":     I_in_fold2 / I0,
    }


def _cubic_eval(I_out: float, delta: float, alpha: float) -> float:
    """Evaluate the I_in(I_out) relation: I_out·[1+(Δ−α·I_out)²]."""
    return I_out * (1.0 + (delta - alpha * I_out) ** 2)


# ---------------------------------------------------------------------------
# Residual function
# ---------------------------------------------------------------------------

def bistable_equation(
    I_out: float,
    I_in: float,
    delta: float,
    alpha: float,
) -> float:
    """
    Residual of the dispersive bistable equation.

    f(I_out) = α²·I_out³ − 2Δα·I_out² + (1+Δ²)·I_out − I_in

    Parameters
    ----------
    I_out : float
        Trial output intensity.
    I_in : float
        Effective input drive (= I0 · S).
    delta : float
        Normalised cavity detuning.
    alpha : float
        Kerr nonlinear coefficient.

    Returns
    -------
    float
        Residual (zero at steady state).
    """
    return (alpha ** 2 * I_out ** 3
            - 2.0 * delta * alpha * I_out ** 2
            + (1.0 + delta ** 2) * I_out
            - I_in)


# ---------------------------------------------------------------------------
# All real roots of the cubic
# ---------------------------------------------------------------------------

def find_all_roots(
    I_in: float,
    delta: float,
    alpha: float,
    I_max: float = 50.0,
    N_scan: int = 50_000,
) -> list:
    """
    Find all real, non-negative roots of the dispersive bistable cubic.

    Uses a dense scan + brentq to robustly locate all sign changes.

    Parameters
    ----------
    I_in : float
        Input drive intensity.
    delta : float
        Normalised detuning.
    alpha : float
        Kerr coefficient.
    I_max : float
        Upper bound of root search.
    N_scan : int
        Number of scan points (higher → fewer missed roots near fold).

    Returns
    -------
    list of float
        All real non-negative roots, sorted ascending.
    """
    x = np.linspace(1e-12, I_max, N_scan)
    f = bistable_equation(x, I_in, delta, alpha)
    sign_changes = np.where(np.diff(np.sign(f)))[0]

    roots = []
    for idx in sign_changes:
        try:
            r = brentq(
                bistable_equation,
                x[idx], x[idx + 1],
                args=(I_in, delta, alpha),
                xtol=1e-12, rtol=1e-12,
            )
            if r >= 0.0:
                roots.append(float(r))
        except ValueError:
            pass

    return sorted(roots)


# ---------------------------------------------------------------------------
# Vectorised sweep solver (hysteresis tracking)
# ---------------------------------------------------------------------------

def solve_bistable_sweep(
    I0: float,
    S_array: np.ndarray,
    delta: float,
    alpha: float,
    direction: str = "up",
) -> np.ndarray:
    """
    Solve the bistable equation for an array of S values, tracking hysteresis.

    The sweep direction determines which stable branch is followed:
    - 'up'   : start on lower branch; jump to upper at upper fold.
    - 'down' : start on upper branch; jump to lower at lower fold.

    Parameters
    ----------
    I0 : float
        Input power scale [a.u.].
    S_array : np.ndarray
        Combined spectral response values, shape (N,), in traversal order
        (ascending for 'up', descending for 'down').
    delta : float
        Normalised cavity detuning.
    alpha : float
        Kerr nonlinear coefficient.
    direction : str
        'up' or 'down'.

    Returns
    -------
    I_out : np.ndarray
        Steady-state output intensities, shape (N,).
    """
    N = len(S_array)
    I_out_arr = np.zeros(N)

    # Analytic fold bifurcation info for branch selection
    window = bistability_window(delta, alpha, I0)

    # Initial guess: start on the appropriate branch
    if direction == "up":
        # Begin below the lower fold (on lower branch)
        I_guess = 0.01
    else:
        # Begin above the upper fold (on upper branch)
        # Use large-I_in linear estimate
        I_in_start = I0 * float(S_array[0])
        I_guess = I_in_start / (1.0 + delta ** 2) * 3.0  # rough upper branch estimate

    for i, S in enumerate(S_array):
        I_in = I0 * float(S)

        # Find all roots for this I_in
        roots = find_all_roots(I_in, delta, alpha)

        if len(roots) == 0:
            I_out_arr[i] = 0.0
            continue

        if len(roots) == 1:
            # Monostable: only one choice
            I_out_arr[i] = roots[0]
            I_guess = roots[0]
            continue

        if len(roots) == 2:
            # Edge of bistable window: pick closest to previous state
            idx = int(np.argmin([abs(r - I_guess) for r in roots]))
            I_out_arr[i] = roots[idx]
            I_guess = roots[idx]
            continue

        # Three roots: lower (0), middle (1, unstable), upper (2)
        low_root  = roots[0]
        high_root = roots[-1]  # roots[2]

        # Branch tracking: stay on the current branch
        if direction == "up":
            # Prefer lower root until it disappears
            chosen = low_root if abs(I_guess - low_root) <= abs(I_guess - high_root) else high_root
        else:
            # Prefer upper root until it disappears
            chosen = high_root if abs(I_guess - high_root) <= abs(I_guess - low_root) else low_root

        I_out_arr[i] = chosen
        I_guess = chosen

    return I_out_arr


# ---------------------------------------------------------------------------
# Single-point solver (convenience)
# ---------------------------------------------------------------------------

def solve_bistable(
    I_in: float,
    delta: float,
    alpha: float,
    I_guess: float = None,
) -> float:
    """
    Find a single steady-state solution near I_guess.

    Parameters
    ----------
    I_in : float
        Input drive (= I0 · S).
    delta : float
        Normalised cavity detuning.
    alpha : float
        Kerr nonlinear coefficient.
    I_guess : float or None
        Initial guess for branch selection.

    Returns
    -------
    float
        Output intensity.
    """
    roots = find_all_roots(I_in, delta, alpha)
    if not roots:
        return 0.0
    if I_guess is None:
        return roots[0]
    idx = int(np.argmin([abs(r - I_guess) for r in roots]))
    return roots[idx]


# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------

def _run_checks():
    """
    Verify:
    - Bistability condition is detected correctly.
    - Three roots exist in the bistable window.
    - Up/down sweeps produce hysteresis.
    - Residuals are zero at all solutions.
    """
    delta = 3.0
    alpha = 1.0
    I0    = 1.0

    window = bistability_window(delta, alpha, I0)
    print(f"[bistable_solver] Bistable: {window['bistable']}")
    print(f"  Lower fold: I_out={window['I_out_fold_low']:.4f}, "
          f"S_fold={window['S_fold_high']:.4f}")
    print(f"  Upper fold: I_out={window['I_out_fold_high']:.4f}, "
          f"S_fold={window['S_fold_low']:.4f}")
    assert window["bistable"], "delta=3 should give bistability!"

    # Three roots in the middle of the bistable window
    I_in_mid = (window["I_in_fold_low"] + window["I_in_fold_high"]) / 2.0
    roots = find_all_roots(I_in_mid, delta, alpha)
    print(f"  Roots at mid-window (I_in={I_in_mid:.3f}): {[f'{r:.4f}' for r in roots]}")
    assert len(roots) == 3, f"Expected 3 roots, got {len(roots)}"
    print("[bistable_solver] 3 roots confirmed in bistable window  ✓")

    # Residual check
    for r in roots:
        res = abs(bistable_equation(r, I_in_mid, delta, alpha))
        assert res < 1e-8, f"Large residual {res:.2e} at root {r:.4f}"
    print("[bistable_solver] All residuals < 1e-8  ✓")

    # Hysteresis sweep: S from 0 to 1 and back
    # Scale I0 so that I_in = I0*S covers the bistable window
    I_in_max_needed = window["I_in_fold_high"] * 1.2
    I0_scaled = I_in_max_needed  # so that S=1 gives I_in = I0_scaled

    S_up   = np.linspace(0.0, 1.0, 2000)
    S_down = S_up[::-1].copy()

    I_up   = solve_bistable_sweep(I0_scaled, S_up,   delta, alpha, "up")
    I_down = solve_bistable_sweep(I0_scaled, S_down, delta, alpha, "down")

    # Align down-sweep for comparison
    I_down_asc = I_down[::-1]
    gap = np.abs(I_up - I_down_asc).max()
    print(f"[bistable_solver] Max hysteresis gap: {gap:.4f} a.u.")
    assert gap > 0.1, f"Hysteresis gap too small: {gap:.4f}"
    print("[bistable_solver] Hysteresis detected  ✓")


if __name__ == "__main__":
    _run_checks()
