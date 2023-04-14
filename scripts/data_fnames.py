from os import path

data_dir = path.join(path.dirname(path.dirname(path.abspath(__file__))),
                     'data')

data_fnames = dict(
    lurie=path.join(data_dir, 'lurie.fits'),
    lurie_tic=path.join(data_dir, 'lurie_tic.npy'),
    w19=path.join(data_dir, 'w19_maxlike_pars.dat'),
    prsa=path.join(data_dir, 'prsa_ebs.fits'),
    ja21=path.join(data_dir, 'ja_high_precision.fits'),
    cmd_isochrone=path.join(data_dir, 'cmd_isochrone_1Gyr_with_TESSmag.dat')
)

plot_dir = path.join(path.dirname(data_dir), 'project_description', 'figures')
