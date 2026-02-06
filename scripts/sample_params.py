"""Define the sampling paramaters data type."""

from collections import namedtuple
from astropy import units as u

SampleParams = namedtuple(
    "SampleParams",
    [
        "mtotal",
        "mratio",
        "age_gyr",
        "meh",
        "per",
        "ecc",
        "w",
        "primary_impact_param",
        "eclipse_time",
        "primary_limb_dark_1",
        "primary_limb_dark_2",
        "secondary_limb_dark_1",
        "secondary_limb_dark_2",
        "primary_prot",
        "secondary_prot",
        "primary_reflection_coef",
        "secondary_reflection_coef",
        "primary_beaming_coef",
        "secondary_beaming_coef",
        "lc_sys",
        "sed_sys",
    ],
    defaults=(
        0.0,  # primary_limb_dark_1
        0.0,  # primary_limb_dark_2
        0.0,  # secondary_limb_dark_1
        0.0,  # secondary_limb_dark_2
        100.0,  # primary_prot
        100.0,  # secondary_prot
        0.01,  # primary_reflection_coef
        0.01,  # secondary_reflection_coef
        0.01,  # primary_beaming_coef
        0.01,  # secondary_beaming_coef
        1e-10,  # lc_sys
        1e-10,  # sed_sys
    ),
)

param_units = {
    "mtotal": u.M_sun,
    "mratio": u.dimensionless_unscaled,
    "age_gyr": u.Gyr,
    "meh": u.dimensionless_unscaled,
    "per": u.day,
    "ecc": u.dimensionless_unscaled,
    "w": u.deg,
    "primary_impact_param": u.dimensionless_unscaled,
    "eclipse_time": u.day,
    "primary_limb_dark_1": u.dimensionless_unscaled,
    "primary_limb_dark_2": u.dimensionless_unscaled,
    "secondary_limb_dark_1": u.dimensionless_unscaled,
    "secondary_limb_dark_2": u.dimensionless_unscaled,
    "primary_prot": u.day,
    "secondary_prot": u.day,
    "primary_reflection_coef": u.dimensionless_unscaled,
    "secondary_reflection_coef": u.dimensionless_unscaled,
    "primary_beaming_coef": u.dimensionless_unscaled,
    "secondary_beaming_coef": u.dimensionless_unscaled,
    "lc_sys": u.dimensionless_unscaled,
    "sed_sys": u.mag,
}

param_descriptions = {
    "mtotal": "Sum of primary and secondary masses",
    "mratio": "Ratio of secondary to primary mass",
    "age_gyr": "Age of the binary",
    "meh": "Metallicity of the binary: [M/H]",
    "per": "Orbital period",
    "ecc": "Orbital eccentricity",
    "w": "Orbital argument of periapsis",
    "primary_impact_param": "Semimajor * cos(inclination)",
    "eclipse_time": "Time of inferior conjunction",
    "primary_limb_dark_1": "Primary limb darkening linear coefficient",
    "primary_limb_dark_2": "Primary limb darkening quadratic coefficient",
    "secondary_limb_dark_1": "Secondary limb darkening linear coefficient",
    "secondary_limb_dark_2": "Secondary limb darkening quadratic coefficient",
    "primary_prot": "Primary rotation period",
    "secondary_prot": "Secondary rotation period",
    "primary_reflection_coef": "Scaling of primary reflection in LC",
    "secondary_reflection_coef": "Scaling of secondary reflection in LC",
    "primary_beaming_coef": "Scaling of Doppler beaming of primary",
    "secondary_beaming_coef": "Scaling of Doppler beaming of secondary",
    "lc_sys": "Systematic uncertainty in flux measurement (fractional)",
    "sed_sys": "Systematic uncertainty is SED [mag]",
}
