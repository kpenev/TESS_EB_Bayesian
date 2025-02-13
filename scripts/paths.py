"""Define constants containing the paths to various input files."""

from os import path

data_dir = path.join(path.dirname(path.dirname(path.abspath(__file__))), "data")

grav_dark = {"TESS": path.join(data_dir, "claret_gravity_darkening_TESS.fits")}
cmd_data_fname = path.join(data_dir, "isochrone_data_{photsys}.ssv")
broadband_data_dir = path.join(data_dir, "Green_et_al_2019_reddening")
prsa_ebs = path.join(
    data_dir, "hlsp_tess-ebs_tess_lcf-ffi_s0001-s0026_tess_v1.0_cat.csv"
)

cache_db = path.join(data_dir, "mcmc_cache.sqlite")
results_dir = path.join(path.dirname(path.dirname(path.abspath(__file__))),
                        "results")
