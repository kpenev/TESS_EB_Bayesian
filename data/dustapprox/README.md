# Gaia DR3 extinction coefficients (from dustapprox)

`oct2025_a0r0_gaia_dr3_kurucz_F99_teff_A0_R0.ecsv` is copied unchanged from
`dustapprox` 0.2.0 (Fouesneau et al., https://github.com/mfouesneau/dustapprox,
PyPI `dustapprox==0.2.0`), file
`dustapprox/data/precomputed/polynomial/f99/kurucz/`. SHA-256:
`dd66ffd002c42596b0482f0e1625c6996f51a74a31dd5317217fdfd03dff4c32`.
It is redistributed under dustapprox's BSD 3-Clause license (`LICENSE`).

The file gives k_X = A_X / A_0 for the Gaia DR3 G, BP and RP passbands as a
cubic polynomial in A_0, R_0 and Teff / 5040 K, fitted to Kurucz atmospheres
reddened with the Fitzpatrick (1999) law. The validity ranges (Teff, A_0,
R_0, ...) are in the file header. Each column name gives the powers of one
polynomial term (e.g. `A0^2 R0`), so the file can be evaluated with astropy
and numpy alone. This repository does not depend on the dustapprox package.

Only the Teff dependence, k_X(Teff) / k_X(4700 K), is used. The absolute
extinction coefficients come from Wang & Chen (2019) shifted onto the Green
et al. (2019) scale; see `extinction_notes.md` in the repository root.
