# Extinction coefficients: notes for model_issues_resolved.md item 5

All coefficients are extinction per unit Bayestar19 reddening E, defined by
E(g-r)_PS1 = 0.901 E (Green et al. 2019, Sec. 2.1). Tables per E(B-V)_SFD
(SF11, Yuan13) are already on this scale: SF11 gives
E(g-r)_PS1 = 3.172 - 2.271 = 0.901 exactly.

## Sources

| Short name | Paper | What it gives | How the absolute level (gray) is set |
|---|---|---|---|
| Green19 | Green et al. 2019, arXiv:1905.02734, Table 1 + Sec. 2.1 | PS1 grizy, 2MASS JHKs (what Bayestar uses) | Schlafly16 vector + assumed A_H/A_K = 1.55 (Indebetouw+2005) |
| Schlafly16 | Schlafly et al. 2016, arXiv:1602.03928, Table 2 | Mean reddening vector R0 and its main variation dR/dx, g..W2 | Not measurable: "any additive offset ... (a gray component) is not measurable by our technique" (Sec. 3.1.2) |
| SF11 | Schlafly & Finkbeiner 2011, arXiv:1012.4804, Table 6 | F99 R_V=3.1 coefficients for many systems (PS1, UKIRT JHK) | From the F99 curve shape. Only colors were tested against data (g-r, r-i, i-z to 3%) |
| Yuan13 | Yuan, Liu & Xiang 2013, arXiv:1301.1427, Table 2 | Empirical; SDSS, 2MASS, WISE | Assumed R(Ks) = 0.306 |
| WC19 | Wang & Chen 2019, arXiv:1904.04575, Tables 2-3 | Red clump stars; PS1, 2MASS, WISE **and Gaia G, BP, RP** | Measured: A_BP/A_RP from red-clump stars with Gaia parallaxes (Sec. 3.4) |
| Babusiaux18 | Gaia Collaboration, Babusiaux et al. 2018, A&A 616, A10, arXiv:1804.09378, Table 1 / Eq. 1 | k = A_m/A_0 for G, BP, RP vs (BP-RP)_0 and A_0; DR2 passbands, 3500-10000 K | FM07-based; superseded by dustapprox here |
| dustapprox | Fouesneau et al., `dustapprox` 0.2.0 (PyPI), precomputed polynomial models | k = A_X/A_0 vs Teff, A_0, R_0 for Gaia DR3, 2MASS, WISE, SDSS (no PS1); 3500-50000 K; F99 or G23 law | Per A_0 of the model law; only its Teff dependence is used |

The repo's W1/W2 values (0.179, 0.132) come from
`scripts/derive_extinction_coefficients.py`, which fits the Green19
scale/offset to the Schlafly16 vector.

Where the papers compare themselves with each other:
- Schlafly16 Sec. 5.2 and its comparison table: Yuan+13, Indebetouw+05,
  Davenport+14. They argue F99/CCM fit poorly.
- Schlafly16 Appendix: the gray fix is "problematic if there is significant
  variation in the infrared extinction curve". Curve unreliable outside
  5000-45000 A.
- WC19 Sec. 5.2-5.3: comparison of color excess ratios and curves with
  earlier work.

## Comparison on the Bayestar scale

WC19 is rescaled from A/E(BP-RP) by 0.901/(A_g - A_r).

| band | Green19 | SF11 | Yuan13 | WC19 | SF11-G | Y13-G | WC19-G |
|---|---|---|---|---|---|---|---|
| g  | 3.518 | 3.172 | -     | 3.343 | -0.346 | -      | -0.175 |
| r  | 2.617 | 2.271 | -     | 2.442 | -0.346 | -      | -0.175 |
| i  | 1.971 | 1.682 | -     | 1.818 | -0.289 | -      | -0.153 |
| z  | 1.549 | 1.322 | -     | 1.409 | -0.227 | -      | -0.140 |
| y  | 1.263 | 1.087 | -     | 1.145 | -0.176 | -      | -0.118 |
| J  | 0.793 | 0.709 | 0.720 | 0.704 | -0.084 | -0.073 | -0.089 |
| H  | 0.469 | 0.449 | 0.460 | 0.379 | -0.020 | -0.009 | -0.090 |
| Ks | 0.303 | 0.302 | 0.306 | 0.225 | -0.001 | +0.003 | -0.078 |
| W1 | 0.179 | -     | 0.180 | 0.114 | -      | +0.001 | -0.065 |
| W2 | 0.132 | -     | 0.160 | 0.076 | -      | +0.028 | -0.056 |

A_H/A_Ks: Green19 1.55, SF11 1.49, Yuan13 1.50, WC19 1.68.

The disagreements come in two kinds:
- **SF11 vs Green19 is a tilt** (optical vs IR shape): the IR agrees and the
  optical is 0.18-0.35 lower. Schlafly16 measured this shape with APOGEE and
  rejects F99, so this is arguably resolved rather than an open uncertainty.
- **WC19 vs Green19 is mostly gray**: changing only A_H/A_K from 1.55 to
  1.68 in Green's recipe shifts every band by -0.059. The rest of the
  WC19-G difference (optical about -0.17, IR about -0.07) looks very much
  like the optical part of a -1 sigma(x) shape change (next table).

## Sightline variation of the curve shape (Schlafly16)

Schlafly16 Sec. 4.2 measures RMS sigma(x) = 0.023 along dR/dx, which is about
sigma(R_V) = 0.18. The per-band change in extinction per unit E for a 1-sigma
change in x, re-applying Green's recipe (gray fixed by A_H/A_K = 1.55), is
below. How Bayestar's E responds to the shape change sets the normalization,
so three cases are shown:
- **E(g-r) fixed**: E tracks g-r reddening only. This is an upper bound.
- **scale fixed**: E unchanged.
- **PS1+2MASS fit**: E is the least-squares color match over the 8 Bayestar
  bands. This is probably closest to what Bayestar does.

| band | E(g-r) fixed | scale fixed | PS1+2MASS fit | gray: A_H/A_K=1.68 |
|---|---|---|---|---|
| g  | 0.159 | -0.019 | 0.010 | -0.059 |
| r  | 0.159 |  0.027 | 0.048 | -0.059 |
| i  | 0.153 |  0.053 | 0.069 | -0.059 |
| z  | 0.136 |  0.058 | 0.070 | -0.059 |
| y  | 0.118 |  0.054 | 0.065 | -0.059 |
| J  | 0.076 |  0.035 | 0.042 | -0.059 |
| H  | 0.045 |  0.021 | 0.025 | -0.059 |
| Ks | 0.029 |  0.014 | 0.016 | -0.059 |
| W1 | 0.012 |  0.003 | 0.005 | -0.059 |
| W2 | 0.002 | -0.004 | -0.003 | -0.059 |

The magnitude shift is E times these values. For the PS1+2MASS-fit case at
the 95th-percentile E = 0.39 this is at most 0.027 mag (i, z), and about
0.001 mag at the median E.

## Gaia bands on the Green19 scale

The WC19 - Green19 offset is nearly constant across the Gaia wavelengths
(0.15-0.175 per E), so the WC19 Gaia coefficients can be put on the Green19
scale by adding the offset interpolated in wavelength (WC19 lambda_eff):

| band | lambda_eff [um] | WC19 (Bayestar E) | offset | on Green19 scale | sigma(x) term, PS1+2MASS fit |
|---|---|---|---|---|---|
| BP | 0.5387 | 2.938 | 0.175 | 3.113 | 0.021 |
| G  | 0.6419 | 2.286 | 0.172 | 2.457 | 0.051 |
| RP | 0.7667 | 1.728 | 0.152 | 1.880 | 0.069 |

These inherit Green's gray assumption, which keeps them consistent with the
other SED bands.

### Why WC19 and not dustapprox's absolute values

dustapprox (0.2.0) gives k = A_X/A_0 for Gaia DR3 G/BP/RP directly, but per
A_0 in a model law (F99 or G23), so it needs an anchor to Bayestar E. Per
unit E, with both normalized by g-r:

| | g | r | i | z | BP | G | RP |
|---|---|---|---|---|---|---|---|
| Green19 | 3.518 | 2.617 | 1.971 | 1.549 | | | |
| WC19 (measured) | 3.343 | 2.442 | 1.818 | 1.409 | 2.938 | 2.286 | 1.728 |
| F99 (SF11 for PS1, dustapprox at 4700 K for Gaia) | 3.172 | 2.271 | 1.682 | 1.322 | 2.93 | 2.30 | 1.66 |
| Green19 - WC19 | 0.175 | 0.175 | 0.153 | 0.140 | | | |
| Green19 - F99 | 0.346 | 0.346 | 0.289 | 0.227 | | | |

WC19 and F99 agree in the Gaia bands but not in PS1, and PS1 is what ties
the Gaia values to Green19:
- WC19's PS1 shape matches Green's (nearly constant offset, 0.175 to 0.140),
  so shifting WC19's Gaia values by the interpolated offset is robust to
  about +-0.02 per E.
- F99's offset drifts by 0.12 over the same bands (the tilt Schlafly16
  rejects). The F99 route gives A_G of about 2.57 (Green's r plus
  dustapprox's G-r) to 2.63 (interpolated offset) per E, against 2.46 via
  WC19: 4-7% higher, about 0.04-0.07 mag at E = 0.4.

WC19 measured Gaia-PS1 color excess ratios on stars observed in both
systems, an empirical link to the bands Green's vector is built on.
dustapprox models that link with a law that is wrong in this wavelength
range by Green's and Schlafly's standards. So WC19 sets the level and
dustapprox only the Teff dependence.

Caveat: dustapprox has no PS1 model, so it was anchored with SDSS g-r using
SF11's 7000 K value (1.018 per E). This is good to 1-2%.

### Color (Teff) dependence

WC19 measured red-clump stars (Teff about 4700 K). dustapprox (Kurucz, A_0 =
0.1, R_0 = 3.1, Gaia DR3 passbands) gives k_X(Teff)/k_X(4700 K), F99 law
(G23 agrees within about 1%):

| Teff | 4000 | 4700 | 5800 | 7500 | 10000 | 15000 | 25000 |
|---|---|---|---|---|---|---|---|
| G  | 0.976 | 1 | 1.035 | 1.084 | 1.144 | 1.228 | 1.287 |
| BP | 0.987 | 1 | 1.019 | 1.045 | 1.078 | 1.125 | 1.160 |
| RP | 0.994 | 1 | 1.009 | 1.023 | 1.039 | 1.061 | 1.070 |

Valid for 3500 < Teff < 50000 K. Use the TIC Teff, or the flux-weighted
model Teff, once per target when the SED is built.

For comparison, Babusiaux18 Eq. 1 is fitted only for 3500-10000 K with DR2
passbands. It gives a stronger dependence: G at (BP-RP)_0 = 0 (about
10000 K) is 1.24 times its value at (BP-RP)_0 = 1.2, against 1.14 here. The
likely cause is the DR2 G passband. Use dustapprox.

## Distribution of E for the cached targets

There are 2545 TIC IDs in the `sed` table. E is from Bayestar19 at the
Bailer-Jones geometric distance. 276 targets (all with dec < -30) have no
Bayestar E; 253 of those have a TIC E(B-V) fallback (median 0.04).

| percentile | 25% | 50% | 75% | 90% | 95% | 99% | max |
|---|---|---|---|---|---|---|---|
| E | 0.00 | 0.02 | 0.08 | 0.22 | 0.39 | 0.94 | 1.96 |

Counts: E > 0.05: 775; > 0.1: 471; > 0.3: 153; > 1.0: 20.

## What x and sigma(x) are

Schlafly16 does a principal-component analysis of the reddening vectors of
APOGEE stars. The first component is the mean vector R0. The second, dR/dx
(their Table 2), is the dominant way the curve shape varies between
sightlines. Each sightline's curve is R0 + x dR/dx. x = +-0.04 corresponds
roughly to R_V = 3.6 / 3.0, so x is essentially a proxy for R_V. The
measured RMS is sigma(x) = 0.023 (stars with E > 0.5), about
sigma(R_V) = 0.18.

The pattern is not linear in wavelength. After Green's scale and gray are
re-applied, the change peaks in i/z/y and falls toward g and W2 (see the
sightline variation table above), so it is closer to a change in curvature
than in slope.

## Open questions

- **Absolute (gray) level.** This is a single choice for all bands:
  - A. Keep Green19's assumed gray (A_H/A_K = 1.55) and shift the WC19 Gaia
    coefficients onto it (Gaia bands section).
  - B. Shift every band, Green19 PS1/2MASS/WISE and converted Gaia alike, by
    one constant gray term times E. A constant shift leaves colors, and so
    Bayestar's E, unchanged; using the full WC19 vector would not. But the
    size of the shift is ambiguous: matching WC19's A_H/A_K = 1.68 gives
    -0.059 per E, while matching WC19 in the optical (where they actually
    measure A_BP/A_RP) gives about -0.17 per E. The difference between the
    two is curve shape, not gray.

  Preferred: A, optionally with a gray uncertainty in the SED covariance: a
  term v_gray v_gray^T with v_gray = sigma_gray E in every band.

  The gray term is best thought of as an uncertainty on A_W2. The colors
  E(b - W2) are what is measured, so A_b = E(b - W2) + A_W2 for every band b,
  and an error in A_W2 shifts all bands by the same number of magnitudes.
  It changes no colors (not even optical vs IR), so in the SED fit it acts
  exactly like extra distance-modulus uncertainty. It is not an independent
  uncertainty in the IR bands; the only IR-specific term is the curve-shape
  term, which is negligible there (0.005 E in W1, 0.003 E in W2).

  Because IR extinction is small, the gray term is bounded. A_W2 >= 0, so the
  largest possible downward shift is Green's whole A_W2 = 0.132 E (the
  A_W2 = 0 normalization of Green et al. 2018). The literature spread
  A_W2/A_Ks of about 0.34 +- 0.10 (as quoted by WC19) gives about +-0.03 E
  with Green's A_Ks. So sigma_gray of about 0.03 E is defensible; the
  0.059 E from WC19's A_H/A_K = 1.68 is about 45% of A_W2, which is generous.
  For the cached targets, 0.03 E is about 0.0006 mag at the median E, 0.007
  mag at the 90th percentile, and 0.012 mag at the 95th.
- **Curve shape.** Should a sigma(x) term enter the covariance as
  v_x v_x^T, with v_x = E sigma(x) dR/dx after Green's scale and gray are
  re-applied (the "PS1+2MASS fit" column)? It is a published measurement from
  the same data as Green's vector, so it is the easiest term to justify.
