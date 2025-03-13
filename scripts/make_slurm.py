#!/usr/bin/env python3

"""Use templates to fully funcional make slurm files."""

from socket import gethostname
from os import makedirs
from os.path import dirname

from configargparse import ArgumentParser, DefaultsFormatter

from paths import slurm_template, slurm_fname

_tic_per_node = {"juno": 4, "ls6": 8, "ganymede": 1}


def parse_command_line():
    """Return the command line configuration."""

    host = gethostname()
    this_hpc = None
    for candidate in _tic_per_node:
        if host.startswith(candidate):
            this_hpc = candidate

    parser = ArgumentParser(
        description=__doc__,
        default_config_files=["make_slurm.cfg"],
        args_for_writing_out_config_file=["--generate-config-file"],
        args_for_setting_config_path=["--config-file", "-c"],
        formatter_class=DefaultsFormatter,
        ignore_unknown_config_file_keys=False,
    )
    parser.add_argument(
        "tic_id_list",
        nargs="+",
        type=int,
        help="Create slurm script for sampling these TIC identifiers. If the "
        "number is not divisible by how many a given HPC node can handle, the "
        "last job file will have fewer.",
    )
    parser.add_argument(
        "--hpc",
        choices=_tic_per_node.keys(),
        default=this_hpc,
        help="The HPC system to create the slurm scripts for.",
    )
    parser.add_argument(
        '--partition',
        default='normal',
        help='The SLURM partition to set up the script for.'
    )
    return parser.parse_args()


def make_slurm(config):
    """Create the slurm scripts per the given configuration."""

    with open(
        slurm_template.format(hpc=config.hpc), "r", encoding="utf-8"
    ) as template_f:
        template_text = template_f.read()

    first_tic = 0
    while first_tic < len(config.tic_id_list):
        slurm_tics = " ".join(
            [
                str(tic)
                for tic in config.tic_id_list[
                    first_tic : first_tic + _tic_per_node[config.hpc]
                ]
            ]
        )
        worker = slurm_tics.replace(" ", "_")
        slurm_text = template_text.replace("@@TIC_LIST@@", slurm_tics).replace(
            "@@WORKER@@", worker
        ).replace('@@PARTITION@@', config.partition)
        assert '@@' not in slurm_text
        out_fname = slurm_fname.format(hpc=config.hpc, worker=worker)
        makedirs(dirname(out_fname))
        with open(out_fname, 'w', encoding='utf-8') as outf:
            outf.write(slurm_text)
        print(f'Created slurm script {out_fname} for TIC: {slurm_tics}')
        first_tic += _tic_per_node[config.hpc]


if __name__ == "__main__":
    make_slurm(parse_command_line())
