"""Define the sampling paramaters data type."""

from collections import namedtuple

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
