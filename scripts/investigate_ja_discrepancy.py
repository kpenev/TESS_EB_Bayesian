"""Investigate discrepancies for the JA high-precision sample."""

import argparse
from os import path

import numpy as np
from astropy.table import Table
from matplotlib import pyplot
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import norm

from binary import Binary
from binary_parameters import calc_eclipse_phase_diff
from log_likelihood import LogLikelihood
from paths import data_dir
from sample_params import SampleParams


_RELEASE_DIR = path.expanduser(
    '~/Box-Box/TESS_EB_finished/first1000/release'
)
_QUANTILE_SIGMAS = list(range(-2, 3))
_QUANTILE_LABELS = ['m2sig', 'm1sig', 'med', 'p1sig', 'p2sig']


_TIC_IDS = (
    2020964, 214716930, 146761368, 41561453, 204497617, 153742549, 139732428,
    168674931, 9380768, 67719432, 88479623, 309434176, 139148071, 5092088,
    34236395, 41896812, 31303242, 160036449, 121420805, 12158853, 3921749,
    148544875, 91369561, 77158150, 160085375, 79565128, 290547036, 121568625,
    271225404, 369362290, 178857934, 10838695, 306107122, 219699532, 74528318,
    160328766, 55489734, 44890224,
)


def read_ja_high_precision(tic_ids=_TIC_IDS):
    """Return a DataFrame of the JA high-precision rows for the given TICs."""

    dataframe = Table.read(
        path.join(data_dir, 'ja_high_precision.fits')
    ).to_pandas()
    return dataframe[dataframe['TIC'].isin(tic_ids)].set_index('TIC')


def add_mcmc_columns(dataframe, release_dir=_RELEASE_DIR):
    """Add ecc/w quantile and best-log_prob columns from release chains."""

    quantiles = norm.cdf(_QUANTILE_SIGMAS)
    best_only_params = ('per', 'eclipse_time')
    new_columns = [
        f'{param}_{label}'
        for param in ('ecc', 'w')
        for label in _QUANTILE_LABELS + ['best']
    ] + [f'{param}_best' for param in best_only_params]
    for column in new_columns:
        dataframe[column] = np.nan

    for tic in dataframe.index:
        samples_fname = path.join(release_dir, f'tess{tic}.fits')
        if not path.exists(samples_fname):
            continue
        chain = Table.read(samples_fname).to_pandas()
        best = chain.loc[chain['log_prob'].idxmax()]
        for param in ('ecc', 'w'):
            for label, value in zip(
                _QUANTILE_LABELS, np.quantile(chain[param], quantiles)
            ):
                dataframe.at[tic, f'{param}_{label}'] = value
            dataframe.at[tic, f'{param}_best'] = best[param]
        for param in best_only_params:
            dataframe.at[tic, f'{param}_best'] = best[param]

    return dataframe


def _load_best_binary(tic, release_dir):
    """Return (Binary, best row) from the max-log_prob release chain entry."""

    samples_fname = path.join(release_dir, f'tess{tic}.fits')
    if not path.exists(samples_fname):
        return None, None
    chain = Table.read(samples_fname).to_pandas()
    best = chain.loc[chain['log_prob'].idxmax()]
    params = SampleParams(
        **{field: best[field] for field in SampleParams._fields}
    )
    try:
        return Binary(from_mcmc=params), best
    except ValueError:
        return None, None


def cycle_alignment_factors(
    time, flux, model_flux, period, time_reference, phase_min, phase_max
):
    """Return per-cycle factors aligning observed to model in the phase window.

    Per cycle, the factor is ``1 / median(observed / model)`` over the
    in-window points. Factors are normalized so their mean is 1; cycles with
    no in-window data keep a factor of 1.
    """

    cycle = np.floor((time - time_reference) / period).astype(int)
    phase = ((time - time_reference) % period) / period
    in_window = (phase >= phase_min) & (phase <= phase_max)

    unique_cycles = np.unique(cycle[in_window])
    if unique_cycles.size == 0:
        return np.ones_like(flux)

    per_cycle_factor = np.empty(unique_cycles.size)
    for i, k in enumerate(unique_cycles):
        mask = (cycle == k) & in_window
        per_cycle_factor[i] = 1.0 / np.median(flux[mask] / model_flux[mask])
    #per_cycle_factor /= per_cycle_factor.mean()

    factor = np.ones_like(flux)
    for k, value in zip(unique_cycles, per_cycle_factor):
        factor[cycle == k] = value
    return factor


def plot_folded_lightcurves(dataframe, pdf_fname, release_dir=_RELEASE_DIR):
    """Save folded lightcurves to a multi-page PDF, one page per TIC."""

    with PdfPages(pdf_fname) as pdf:
        for tic, row in dataframe.iterrows():
            tic = int(tic)
            period = row['per_best']
            time_reference = row['eclipse_time_best']
            if not np.isfinite(period) or not np.isfinite(time_reference):
                print(
                    f'Skipping TIC {tic}: missing per_best/eclipse_time_best'
                )
                continue
            binary, best = _load_best_binary(tic, release_dir)
            if binary is None:
                print(f'Skipping TIC {tic}: cannot build best-fit binary')
                continue

            this_work_phase = calc_eclipse_phase_diff(
                row['ecc_best'], row['w_best']
            )
            ja_ecc = np.hypot(row['ecosw'], row['esinw'])
            ja_w = np.degrees(np.arctan2(row['esinw'], row['ecosw']))
            ja_phase = calc_eclipse_phase_diff(ja_ecc, ja_w)
            phase_center = 0.5 * (this_work_phase + ja_phase)
            phase_half_span = 200.0 * abs(this_work_phase - ja_phase)
            phase_min = phase_center - phase_half_span
            phase_max = phase_center + phase_half_span

            log_likelihood = LogLikelihood(tic)
            log_likelihood.skip_ooe_poly=True
            # ``log_likelihood.get_model`` is set to ``get_eclipse_model`` when
            # an OOE/BLSOOE exclusion is present (so only points near eclipses
            # are modeled, with the out-of-eclipse variability removed) and to
            # ``get_full_model`` otherwise. Using it here keeps the plotted
            # model consistent with what was actually fit.
            lc_sys_err = float(best['lc_sys'])
            time_chunks = []
            flux_chunks = []
            model_chunks = []
            for header, lightcurve in log_likelihood.lcs:
                if lightcurve.size <= 10:
                    continue
                good = lightcurve[lightcurve['good']]
                if good.size < 10:
                    continue
                try:
                    model_lc, _, mask = log_likelihood.get_model(
                        binary, header, good, lc_sys_err
                    )
                except ValueError:
                    continue
                if mask is None:
                    selected = good
                elif mask.sum() < 10:
                    continue
                else:
                    selected = good[mask]
                time_chunks.append(selected['time'])
                flux_chunks.append(selected['flux'])
                model_chunks.append(model_lc)
            if not time_chunks:
                continue
            time_all = np.concatenate(time_chunks)
            flux_all = np.concatenate(flux_chunks)
            model_all = np.concatenate(model_chunks)
            phase_all = ((time_all - time_reference) % period) / period
            factor = cycle_alignment_factors(
                time_all,
                flux_all,
                model_all,
                period,
                time_reference,
                phase_min,
                phase_max,
            )

            pyplot.figure(figsize=(10, 5))
            pyplot.plot(phase_all, flux_all * factor, '.k', markersize=1)
            phase_order = np.argsort(phase_all)
            pyplot.plot(
                phase_all[phase_order],
                model_all[phase_order],
                '-',
                color='C2',
                linewidth=1,
                label='model',
            )

            pyplot.axvline(
                this_work_phase, color='C0', linestyle='--', label='this work'
            )
            pyplot.axvline(
                ja_phase, color='C3', linestyle='--', label='JA21'
            )
            pyplot.legend()

            pyplot.xlim(phase_min, phase_max)

            in_window = (phase_all >= phase_min) & (phase_all <= phase_max)
            window_flux = (flux_all * factor)[in_window]
            if window_flux.size > 0:
                ylo, yhi = np.percentile(window_flux, [1, 99])
                margin = 0.1 * (yhi - ylo)
                pyplot.ylim(ylo - margin, yhi + margin)
            pyplot.xlabel('Phase')
            pyplot.ylabel('Flux')
            pyplot.title(f'TIC {tic}: $P_{{orb}}$ = {period:.5f}')
            pdf.savefig(bbox_inches='tight')
            pyplot.close()


def _parse_command_line():
    """Parse command line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--tic',
        type=int,
        nargs='+',
        default=None,
        help='Plot only these TIC IDs (default: plot all).',
    )
    parser.add_argument(
        '--pdf',
        default='ja_folded_lightcurves.pdf',
        help='Output PDF filename.',
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = _parse_command_line()
    data = add_mcmc_columns(read_ja_high_precision())
    if args.tic is not None:
        data = data.loc[args.tic]
    print(data)
    print(80 * '=')
    print('Columns:\n\t' + '\n\t'.join(data.columns))
    plot_folded_lightcurves(data, args.pdf)
