"""Define constants containing the paths to various input files."""

from os import path

data_dir = path.join(path.dirname(path.dirname(path.abspath(__file__))), "data")

grav_dark = {"TESS": path.join(data_dir, "claret_gravity_darkening_TESS.fits")}
cmd_data_fname = path.join(data_dir, "isochrone_data_{photsys}.ssv")
broadband_data_dir = path.join(data_dir, "Green_et_al_2019_reddening")
prsa_ebs = path.join(data_dir, "prsa_ebs.fits")

cache_db = path.join(data_dir, "mcmc_cache.sqlite")


results_dir = path.join(
    path.dirname(path.dirname(path.abspath(__file__))), "results"
)
jobs_db = path.join(results_dir, "slurm_jobs.sqlite")

jktebob = {
    "template": path.join(data_dir, "jktebob_{mode}_template.in"),
    "inputfname": path.join(results_dir, "tess{tic_id}_jktebob_{mode}.in"),
    "inlcfname": path.join(results_dir, "tess{tic_id}_mag_v_time.dat"),
    "paramfname": path.join(results_dir, "tess{tic_id}_jktebob_{mode}.par"),
    "outlcfname": path.join(results_dir, "tess{tic_id}_jktebob_{mode}.out"),
    "fitlcfname": path.join(results_dir, "tess{tic_id}_jktebob_{mode}.fit"),
}
samples = path.join(results_dir, "tess{tic_id:d}_samples.h5")
release = path.join(results_dir, "release", "tess{tic_id:d}.fits")
convergence_pickle = path.join(
    results_dir, "archived", "tess{tic_id:d}_archived.pkl"
)

slurm_fname = path.join(
    path.dirname(data_dir), "slurm", "{hpc}", "{mode}_{job_name}.slurm"
)

launcher_fname = path.join(
    path.dirname(data_dir),
    "slurm",
    "{hpc}",
    "launcher_commands_{job_name}.txt",
)
render_dir = path.join(results_dir, "render")
