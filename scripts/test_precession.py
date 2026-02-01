"""Test models of precessing binaries against PHOEBE."""

from matplotlib import pyplot
import numpy

from general_purpose_python_modules.kepler_angles import (
    E_to_nu,
    M_to_E,
    nu_to_E,
    E_to_M,
)

from binary import Binary
from sample_params import SampleParams
from phoebe_model import get_phoebe_reference_flux

if __name__ == "__main__":
    params = SampleParams(
        mtotal=2.0,
        mratio=0.8,
        age_gyr=1.0,
        meh=0.0,
        per=2.0,
        ecc=0.5,
        w=(0.0, 3e-3),
        primary_impact_param=0.0,
        eclipse_time=13.14,
        primary_limb_dark_1=0.2,
        primary_limb_dark_2=0.1,
        secondary_limb_dark_1=0.4,
        secondary_limb_dark_2=0.3,
        primary_prot=100.0,
        secondary_prot=100.0,
        primary_reflection_coef=0.0,
        secondary_reflection_coef=0.0,
        primary_beaming_coef=0.0,
        secondary_beaming_coef=0.0,
        lc_sys=1e-10,
        sed_sys=1e-10,
    )
    plot_times_list = [
        numpy.linspace(4000.0 * i, 4000.0 * i + params.per, 100)
        for i in range(10)
    ]
    binary = Binary(from_mcmc=params)
    phoebe_binary = binary.to_phoebe()
    phoebe_binary["dataset"]["lc01"]["times"].set_value(
        numpy.concatenate(plot_times_list)
    )

    print(f"Plotting phoebe binary:\n{phoebe_binary}")
    phoebe_binary.run_compute()
    phoebe_fluxes = phoebe_binary["lc01@latest@model@fluxes"].quantity
    phoebe_fluxes /= get_phoebe_reference_flux(phoebe_binary)

    label = True
    phoebe_start = 0
    for plot_times in plot_times_list:
        # fig = phoebe_binary.plot(show=False)[0]
        # pyplot.sca(fig.draw().gca())
        plot_color = pyplot.plot(
            plot_times % params.per,
            binary.get_lightcurve(plot_times),
            "-",
            label="ours",
        )[0].get_color()
        pyplot.plot(
            plot_times % params.per,
            phoebe_fluxes[phoebe_start : phoebe_start + plot_times.size],
            ":",
            color=plot_color,
            label=("PHOEBE" if label else None),
        )
        phoebe_start += plot_times.size
        label = False

    pyplot.legend()
    pyplot.show()
