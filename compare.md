This is a genuinely close paper — it's worth being precise about where we overlap and where we diverge, because the differences are real and defensible.

---

## What Tritschler et al. (2408.06247) do

They work with chip-integrated photonic microring resonators for phase sensing, and propose exploiting the nonlinear self-phase-modulation (SPM) effect to increase sensitivity by a multiplicative gain factor that appears when the operational point is chosen just at the crossover from the mono- to the bistable regime — demonstrating a gain factor of 22 on a silicon-nitride chip.

The core idea is that the SPM effect tilts the resonance curve, and the largest slope — and thus the largest nonlinear gain factor — appears when the operational point is chosen just at the crossover from the monostable to the bistable regime.

Critically, their sensing scheme works best *slightly below* P_max and breaks down in the bistable regime itself.

Their sensitivity formula is:

The sensitivity is boosted by a multiplicative gain factor G = P_max/(P_max − P_p), and if P_p is close to P_max, a large nonlinear gain can be achieved.

Their application domain is optical phase sensors such as ring-gyroscopes (Sagnac effect) and optical temperature sensors, where the optical phase and thus the resonance condition of the ring is changed, affecting output transmission power.

---

## Where our work is different

The differences are architectural, physical, and application-level:

### 1. Sensor modality: FBG vs. microring resonator
Their sensor *is* the microring — the resonator and the sensing element are the same device. In our architecture, **the FBG is the sensor** and the resonator is a separate interrogation element. The FBG encodes strain/temperature as a wavelength shift λ_B, which is a fundamentally different measurand (spectral shift) than a phase shift inside a cavity. The two problems have different signal models, noise sources, and resolution limits.

### 2. Bistable regime vs. edge of bistability
This is the sharpest physical distinction. Their maximum gain occurs at the crossover from monostable to bistable, and they explicitly avoid operating in the bistable regime itself, because the hysteresis there is a liability not an asset.

We do the opposite: **we operate inside the bistable window and exploit the switching discontinuity** (the fold bifurcation) as the detection event. We use hysteresis deliberately — the up/down sweep asymmetry and the abrupt contrast change are our signal. Our Δλ_min = σ_I / max|dI/dλ_B| is dominated by the near-infinite slope at the switching point, which only exists deep in the bistable regime.

### 3. What is being discriminated
- Tritschler et al.: discriminate **small phase shifts** (continuous, linear-regime sensitivity enhancement near the monostable-bistable boundary).
- Our work: discriminate **small Bragg wavelength shifts** by detecting whether a switching event occurs or not — a **threshold/binary detection** scheme, not a continuous slope measurement.

### 4. Interrogation architecture
Their system is a single integrated device with a tunable pump laser swept over the resonance. Our system is an **interrogator** — a separate readout instrument attached to a passive FBG fiber sensor. The resonator is not the sensing element; it is a signal-processing stage. This is a different engineering problem (remotely interrogating distributed fiber sensors vs. reading out a chip sensor locally).

### 5. The comparison baseline
Their baseline is the linear resonator sensitivity S₀, and they show S/S₀ rising with input power following the theoretical model. Our baseline is **conventional FBG peak-tracking interrogation** — a completely different reference point. We compare wavelength resolution (pm), not dimensionless sensitivity ratios.

### 6. Thermal SPM vs. pure Kerr
In their experiment, the total gain factor is g_tot ≈ 112 Hz with the optical contribution g_opt ≈ 2 Hz, showing that self-heating is the dominant process over the optical SPM effect. Our model uses a pure Kerr (dispersive) nonlinearity without thermal dynamics, which is appropriate for a numerically-demonstrated interrogation architecture study rather than a specific material platform characterization.

---

## Summary table

| Dimension | Tritschler et al. | Our work |
|---|---|---|
| Sensing element | Microring (phase sensor) | FBG (wavelength-shift sensor) |
| Nonlinear element | Same microring | Separate resonator (interrogator) |
| Operating regime | Edge of bistability (monostable side) | Inside bistable window |
| Hysteresis | Avoided / liability | Exploited / signal |
| Measurand | Phase shift | Bragg wavelength shift (pm) |
| Comparison baseline | Linear resonator S₀ | Peak-tracking interrogation |
| Nonlinearity origin | Thermal SPM + optical Kerr | Pure dispersive Kerr (model) |
| Implementation | Chip experiment (Si₃N₄) | Numerical simulation |

---

## Recommended positioning

In a paper, you would cite Tritschler et al. as prior art showing that **nonlinear resonator physics enhances sensing sensitivity**, then distinguish your work on three axes: (1) you apply this to FBG fiber sensors rather than integrated phase sensors; (2) you exploit the bistable switching discontinuity rather than the pre-bistable slope enhancement; and (3) you demonstrate a new interrogation architecture concept rather than a new sensor material/geometry. That framing is clean and honest — the physics inspiration is shared, but the problem, architecture, and operating regime are distinct.