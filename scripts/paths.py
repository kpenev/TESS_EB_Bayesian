"""Define constants containing the paths to various input files."""

from os import path

data_dir = path.join(path.dirname(path.dirname(path.abspath(__file__))), "data")

grav_dark = {"TESS": path.join(data_dir, "claret_gravity_darkening_TESS.fits")}
cmd_data_fname = path.join(data_dir, "isochrone_data_{photsys}.ssv")
broadband_data_dir = path.join(data_dir, "Green_et_al_2019_reddening")
