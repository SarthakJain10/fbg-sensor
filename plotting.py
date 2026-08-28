"""
plotting.py
===========
Publication-quality figure generation for the FBG bistable interrogation study.

All functions accept data arrays and parameter dictionaries, produce a
matplotlib figure, and optionally save it to the output directory.

Style notes
-----------
- Line widths and font sizes are chosen for A4 / two-column journal figures.
- Wavelength axes are in nm; intensity axes are normalised (a.u.).
- Colour scheme: blue for FBG, orange for resonator, green for combined,
  purple for bistable, red for noise/peak-tracking.
- Grid lines are light grey for readability.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")          # non-interactive backend (safe in all envs)
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D


# ---------------------------------------------------------------------------
# Global style settings
# ---------------------------------------------------------------------------

STYLE = {
    "figure.dpi":         150,
    "figure.figsize":     (6.5, 4.0),
    "axes.labelsize":     11,
    "axes.titlesize":     12,
    "xtick.labelsize":    9,
    "ytick.labelsize":    9,
    "legend.fontsize":    9,
    "lines.linewidth":    1.8,
    "axes.grid":          True,
    "grid.alpha":         0.3,
    "grid.linestyle":     "--",
}

plt.rcParams.update(STYLE)

COLORS = {
    "fbg":        "#2166ac",   # blue
    "resonator":  "#d6604d",   # red-orange
    "combined":   "#1a9850",   # green
    "bistable":   "#762a83",   # purple
    "up":         "#4393c3",   # light blue (up-sweep)
    "down":       "#d73027",   # red (down-sweep)
    "noise":      "#f4a582",   # pale orange (noisy signal)
    "pt":         "#b2182b",   # dark red (peak tracking)
}


def _save(fig, output_dir: str, filename: str):
    """Save figure and close it."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# 1. FBG reflection spectrum
# ---------------------------------------------------------------------------

def plot_fbg_spectrum(
    lambda_vals: np.ndarray,
    R_fbg: np.ndarray,
    lambda_B: float,
    Gamma_fbg: float,
    output_dir: str = "output",
    filename: str = "fbg_spectrum.png",
):
    """
    Plot the FBG reflection spectrum R_fbg(λ).

    Parameters
    ----------
    lambda_vals : np.ndarray  Wavelength axis [nm].
    R_fbg       : np.ndarray  Reflection spectrum.
    lambda_B    : float       Bragg wavelength [nm] (shown as dashed line).
    Gamma_fbg   : float       FBG linewidth [nm] (shown as annotation).
    output_dir  : str         Directory to save the figure.
    filename    : str         Output filename.
    """
    fig, ax = plt.subplots()
    ax.plot(lambda_vals, R_fbg, color=COLORS["fbg"], label=r"$R_\mathrm{FBG}(\lambda)$")
    ax.axvline(lambda_B, color="k", ls="--", lw=1.0, alpha=0.6,
               label=fr"$\lambda_B = {lambda_B:.2f}$ nm")

    # Annotate FWHM
    ax.axhline(0.5, color="grey", ls=":", lw=0.8, alpha=0.7)
    ax.annotate(
        fr"$\Gamma_\mathrm{{FBG}}={Gamma_fbg*1e3:.0f}$ pm",
        xy=(lambda_B + Gamma_fbg / 2 * 1.2, 0.5),
        fontsize=8, color="grey",
    )

    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Reflectivity (a.u.)")
    ax.set_title("FBG Reflection Spectrum (Lorentzian approximation)")
    ax.set_ylim(-0.05, 1.10)
    ax.legend(loc="upper right")
    _save(fig, output_dir, filename)


# ---------------------------------------------------------------------------
# 2. Resonator transmission spectrum
# ---------------------------------------------------------------------------

def plot_resonator_response(
    lambda_vals: np.ndarray,
    H_r: np.ndarray,
    lambda_r: float,
    Gamma_r: float,
    output_dir: str = "output",
    filename: str = "resonator_response.png",
):
    """
    Plot the resonator Lorentzian transmission H_r(λ).

    Parameters
    ----------
    lambda_vals : np.ndarray  Wavelength axis [nm].
    H_r         : np.ndarray  Transmission spectrum.
    lambda_r    : float       Resonance wavelength [nm].
    Gamma_r     : float       Resonator linewidth [nm].
    output_dir  : str         Output directory.
    filename    : str         Output filename.
    """
    fig, ax = plt.subplots()
    ax.plot(lambda_vals, H_r, color=COLORS["resonator"],
            label=r"$H_r(\lambda)$")
    ax.axvline(lambda_r, color="k", ls="--", lw=1.0, alpha=0.6,
               label=fr"$\lambda_r = {lambda_r:.3f}$ nm")
    ax.axhline(0.5, color="grey", ls=":", lw=0.8, alpha=0.7)
    ax.annotate(
        fr"$\Gamma_r={Gamma_r*1e3:.0f}$ pm",
        xy=(lambda_r + Gamma_r / 2 * 1.2, 0.5),
        fontsize=8, color="grey",
    )

    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Transmission (a.u.)")
    ax.set_title("Resonator Transmission Spectrum")
    ax.set_ylim(-0.05, 1.10)
    ax.legend(loc="upper right")
    _save(fig, output_dir, filename)


# ---------------------------------------------------------------------------
# 3. Combined spectral response S(λ)
# ---------------------------------------------------------------------------

def plot_combined_response(
    lambda_vals: np.ndarray,
    R_fbg: np.ndarray,
    H_r: np.ndarray,
    S: np.ndarray,
    lambda_B: float,
    lambda_r: float,
    output_dir: str = "output",
    filename: str = "combined_response.png",
):
    """
    Overlay FBG, resonator, and combined spectral responses.

    Parameters
    ----------
    lambda_vals : np.ndarray  Wavelength axis [nm].
    R_fbg       : np.ndarray  FBG reflection.
    H_r         : np.ndarray  Resonator transmission.
    S           : np.ndarray  Combined response S = R_fbg * H_r.
    lambda_B    : float       Bragg wavelength [nm].
    lambda_r    : float       Resonance wavelength [nm].
    output_dir  : str         Output directory.
    filename    : str         Output filename.
    """
    fig, ax = plt.subplots()
    ax.plot(lambda_vals, R_fbg, color=COLORS["fbg"], alpha=0.7,
            ls="--", label=r"$R_\mathrm{FBG}$")
    ax.plot(lambda_vals, H_r, color=COLORS["resonator"], alpha=0.7,
            ls=":", label=r"$H_r$")
    ax.plot(lambda_vals, S, color=COLORS["combined"], lw=2.2,
            label=r"$S = R_\mathrm{FBG} \cdot H_r$")
    ax.axvline(lambda_B, color=COLORS["fbg"], ls="--", lw=0.8, alpha=0.5,
               label=fr"$\lambda_B={lambda_B:.2f}$ nm")
    ax.axvline(lambda_r, color=COLORS["resonator"], ls=":", lw=0.8, alpha=0.5,
               label=fr"$\lambda_r={lambda_r:.3f}$ nm")

    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Spectral response (a.u.)")
    ax.set_title("Combined FBG–Resonator Spectral Response")
    ax.set_ylim(-0.05, 1.10)
    ax.legend(loc="upper right", ncol=2, fontsize=8)
    _save(fig, output_dir, filename)


# ---------------------------------------------------------------------------
# 4. Bistable I_out(λ) curve
# ---------------------------------------------------------------------------

def plot_bistable_curve(
    lambda_vals: np.ndarray,
    S: np.ndarray,
    I_out: np.ndarray,
    output_dir: str = "output",
    filename: str = "bistable_curve.png",
):
    """
    Plot the bistable output intensity I_out vs. the combined response S.

    This shows the two stable branches as a function of S (input drive),
    illustrating the S-shaped bistable transfer function.

    Parameters
    ----------
    lambda_vals : np.ndarray  Wavelength axis [nm] (for x-label context).
    S           : np.ndarray  Combined spectral response values.
    I_out       : np.ndarray  Bistable output intensities.
    output_dir  : str         Output directory.
    filename    : str         Output filename.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Left panel: I_out vs wavelength
    ax = axes[0]
    ax.plot(lambda_vals, I_out, color=COLORS["bistable"],
            label=r"$I_\mathrm{out}(\lambda)$")
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel(r"$I_\mathrm{out}$ (a.u.)")
    ax.set_title("Bistable Output vs Wavelength")
    ax.legend()

    # Right panel: I_out vs S
    ax2 = axes[1]
    ax2.plot(S, I_out, color=COLORS["bistable"],
             label=r"$I_\mathrm{out}(S)$")
    ax2.set_xlabel("Combined response $S$ (a.u.)")
    ax2.set_ylabel(r"$I_\mathrm{out}$ (a.u.)")
    ax2.set_title("Bistable Transfer Function")
    ax2.legend()

    fig.suptitle("Bistable Optical Response", fontsize=12, y=1.01)
    fig.tight_layout()
    _save(fig, output_dir, filename)


# ---------------------------------------------------------------------------
# 5. Hysteresis loop
# ---------------------------------------------------------------------------

def plot_hysteresis_loop(
    lambda_B_up: np.ndarray,
    I_out_up: np.ndarray,
    lambda_B_down: np.ndarray,
    I_out_down: np.ndarray,
    lambda_switch_up: float = None,
    lambda_switch_down: float = None,
    hysteresis_width: float = None,
    output_dir: str = "output",
    filename: str = "hysteresis_loop.png",
):
    """
    Plot the bistable hysteresis loop (up and down λ_B sweeps).

    Parameters
    ----------
    lambda_B_up       : np.ndarray  Up-sweep λ_B values [nm].
    I_out_up          : np.ndarray  I_out on up-sweep.
    lambda_B_down     : np.ndarray  Down-sweep λ_B values [nm] (descending).
    I_out_down        : np.ndarray  I_out on down-sweep.
    lambda_switch_up  : float       Switching point on up-sweep [nm] (optional).
    lambda_switch_down: float       Switching point on down-sweep [nm] (optional).
    hysteresis_width  : float       Width of hysteresis window [nm] (optional).
    output_dir        : str         Output directory.
    filename          : str         Output filename.
    """
    fig, ax = plt.subplots(figsize=(7, 4.5))

    ax.plot(lambda_B_up, I_out_up, color=COLORS["up"], lw=2.0,
            label="Up-sweep →", zorder=3)
    ax.plot(lambda_B_down, I_out_down, color=COLORS["down"], lw=2.0,
            ls="--", label="Down-sweep ←", zorder=3)

    # Mark switching points
    if lambda_switch_up is not None:
        ax.axvline(lambda_switch_up, color=COLORS["up"], ls=":", lw=1.2,
                   alpha=0.8, label=fr"$\lambda_{{sw}}^{{↑}}={lambda_switch_up:.4f}$ nm")
    if lambda_switch_down is not None:
        ax.axvline(lambda_switch_down, color=COLORS["down"], ls=":", lw=1.2,
                   alpha=0.8, label=fr"$\lambda_{{sw}}^{{↓}}={lambda_switch_down:.4f}$ nm")

    # Annotate hysteresis width
    if hysteresis_width is not None and lambda_switch_up is not None \
            and lambda_switch_down is not None:
        y_ann = ax.get_ylim()[1] * 0.92 if ax.get_ylim()[1] > 0 else 0.9
        ax.annotate(
            "",
            xy=(lambda_switch_up, y_ann),
            xytext=(lambda_switch_down, y_ann),
            arrowprops=dict(arrowstyle="<->", color="k", lw=1.2),
        )
        mid = (lambda_switch_up + lambda_switch_down) / 2
        ax.text(
            mid, y_ann * 1.02,
            fr"$\Delta\lambda_{{hyst}}={hysteresis_width*1e3:.0f}$ pm",
            ha="center", fontsize=8,
        )

    ax.set_xlabel(r"Bragg wavelength $\lambda_B$ (nm)")
    ax.set_ylabel(r"$I_\mathrm{out}$ (a.u.)")
    ax.set_title("Bistable Hysteresis Loop")
    ax.legend(loc="upper left", fontsize=8)
    _save(fig, output_dir, filename)


# ---------------------------------------------------------------------------
# 6. Resolution comparison
# ---------------------------------------------------------------------------

def plot_resolution_comparison(
    delta_lam_bistable: float,
    delta_lam_peak: float,
    improvement: float,
    output_dir: str = "output",
    filename: str = "resolution_comparison.png",
):
    """
    Bar chart comparing Δλ_min for bistable vs. peak-tracking methods.

    Parameters
    ----------
    delta_lam_bistable : float  Bistable resolution [nm].
    delta_lam_peak     : float  Peak-tracking resolution [nm].
    improvement        : float  Improvement factor.
    output_dir         : str    Output directory.
    filename           : str    Output filename.
    """
    fig, ax = plt.subplots(figsize=(5, 4.5))

    methods = ["Peak\nTracking", "Bistable\nInterrogation"]
    values  = [delta_lam_peak * 1e3, delta_lam_bistable * 1e3]  # convert to pm
    colors  = [COLORS["pt"], COLORS["bistable"]]

    bars = ax.bar(methods, values, color=colors, width=0.5, zorder=3)

    # Annotate bar values
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() * 1.03,
            f"{val:.2f} pm",
            ha="center", va="bottom", fontsize=9,
        )

    # Improvement annotation
    ax.annotate(
        f"Improvement: {improvement:.1f}×",
        xy=(0.5, 0.88), xycoords="axes fraction",
        ha="center", fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="gold", alpha=0.9),
    )

    ax.set_ylabel(r"$\Delta\lambda_\mathrm{min}$ (pm)")
    ax.set_title("Minimum Resolvable Wavelength Shift")
    ax.set_ylim(0, max(values) * 1.35)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
    _save(fig, output_dir, filename)


# ---------------------------------------------------------------------------
# 7. Noise comparison on sweeps
# ---------------------------------------------------------------------------

def plot_noisy_sweeps(
    lambda_B_up: np.ndarray,
    I_out_up_clean: np.ndarray,
    I_out_up_noisy: np.ndarray,
    lambda_B_down: np.ndarray,
    I_out_down_clean: np.ndarray,
    I_out_down_noisy: np.ndarray,
    output_dir: str = "output",
    filename: str = "noisy_sweeps.png",
):
    """
    Overlay clean and noisy versions of the bistable sweeps.

    Parameters
    ----------
    lambda_B_up/down     : np.ndarray  Sweep arrays [nm].
    I_out_up/down_clean  : np.ndarray  Noise-free output.
    I_out_up/down_noisy  : np.ndarray  Noisy output.
    output_dir           : str
    filename             : str
    """
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)

    for ax, lam, I_clean, I_noisy, direction, color in [
        (axes[0], lambda_B_up,   I_out_up_clean,   I_out_up_noisy,   "Up",   COLORS["up"]),
        (axes[1], lambda_B_down, I_out_down_clean, I_out_down_noisy, "Down", COLORS["down"]),
    ]:
        ax.plot(lam, I_noisy, color=COLORS["noise"], lw=0.8, alpha=0.7,
                label="Noisy signal", zorder=2)
        ax.plot(lam, I_clean, color=color, lw=2.0,
                label="Clean signal", zorder=3)
        ax.set_xlabel(r"Bragg wavelength $\lambda_B$ (nm)")
        ax.set_title(f"{direction}-sweep")
        ax.legend(fontsize=8)

    axes[0].set_ylabel(r"$I_\mathrm{out}$ (a.u.)")
    fig.suptitle("Bistable Output: Clean vs. Noisy", fontsize=12)
    fig.tight_layout()
    _save(fig, output_dir, filename)


# ---------------------------------------------------------------------------
# 8. Derivative / slope plot
# ---------------------------------------------------------------------------

def plot_slope(
    lambda_B_array: np.ndarray,
    dI_dlam: np.ndarray,
    lambda_switch: float,
    max_slope: float,
    output_dir: str = "output",
    filename: str = "slope_profile.png",
):
    """
    Plot the derivative dI_out/dλ_B profile.

    Parameters
    ----------
    lambda_B_array : np.ndarray  λ_B sweep array [nm].
    dI_dlam        : np.ndarray  Derivative array [a.u./nm].
    lambda_switch  : float       λ_B at maximum slope [nm].
    max_slope      : float       Maximum |dI/dλ| [a.u./nm].
    output_dir     : str
    filename       : str
    """
    fig, ax = plt.subplots()
    ax.plot(lambda_B_array, np.abs(dI_dlam), color=COLORS["bistable"],
            label=r"$|dI_\mathrm{out}/d\lambda_B|$")
    ax.axvline(lambda_switch, color="k", ls="--", lw=1.0, alpha=0.7,
               label=fr"$\lambda_{{sw}}={lambda_switch:.4f}$ nm")
    ax.axhline(max_slope, color="grey", ls=":", lw=0.8, alpha=0.7,
               label=fr"Max slope = {max_slope:.3f} a.u./nm")

    ax.set_xlabel(r"Bragg wavelength $\lambda_B$ (nm)")
    ax.set_ylabel(r"$|dI_\mathrm{out}/d\lambda_B|$ (a.u./nm)")
    ax.set_title("Bistable Transfer-Function Slope Profile")
    ax.legend(fontsize=8)
    _save(fig, output_dir, filename)


# ---------------------------------------------------------------------------
# 9. Peak-tracking noise distribution
# ---------------------------------------------------------------------------

def plot_peak_tracking_distribution(
    lambda_B_estimates: np.ndarray,
    lambda_B_true: float,
    std_estimate: float,
    output_dir: str = "output",
    filename: str = "peak_tracking_distribution.png",
):
    """
    Histogram of λ_B estimates from Monte-Carlo peak tracking.

    Parameters
    ----------
    lambda_B_estimates : np.ndarray  Estimated λ_B values [nm].
    lambda_B_true      : float       True Bragg wavelength [nm].
    std_estimate       : float       Standard deviation of estimates [nm].
    output_dir         : str
    filename           : str
    """
    fig, ax = plt.subplots(figsize=(6, 4))

    ax.hist(
        (lambda_B_estimates - lambda_B_true) * 1e3,  # show in pm
        bins=60, color=COLORS["pt"], alpha=0.75, edgecolor="white",
        label="Estimated $\\lambda_B$ offsets",
    )
    ax.axvline(0, color="k", ls="--", lw=1.2, label="True $\\lambda_B$")
    ax.axvline(-std_estimate * 1e3, color="grey", ls=":", lw=1.0)
    ax.axvline(+std_estimate * 1e3, color="grey", ls=":",  lw=1.0,
               label=fr"$\pm\sigma={std_estimate*1e3:.2f}$ pm")

    ax.set_xlabel(r"$\hat{\lambda}_B - \lambda_B$ (pm)")
    ax.set_ylabel("Counts")
    ax.set_title("Peak-Tracking Noise Distribution (Monte-Carlo)")
    ax.legend(fontsize=8)
    _save(fig, output_dir, filename)
