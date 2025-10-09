"""Create a new group of sampling jobs and fill it."""

import os

from sqlalchemy import update, select

from bui.db_interface import Session
from bui.select_ticids.data_model import JobGroup, get_ticid_select_table
from paths import slurm_fname
from command_line_util import (
    create_parser,
    identify_hpc,
    tic_per_node,
    add_slurm_config,
)


def parse_command_line():
    """Return the command line configuration."""

    this_hpc = identify_hpc()

    parser = create_parser()
    parser.add_argument(
        "tic_table",
        help="Create slurm scripts for sampling the selected TIC identifiers in"
        " the given table. If the number is not divisible by how many a given "
        "HPC node can handle, the last job file will have fewer.",
    )
    parser.add_argument(
        "--hpc",
        choices=tic_per_node.keys(),
        default=this_hpc,
        help="The HPC system to create the slurm scripts for.",
    )
    parser.add_argument(
        "--nodes-per-job",
        type=int,
        default=None,
        metavar="NNODES",
        help="Enable the use of launcher to allocate multiple nodes per job. "
        "If not specified, a non-launcher slurm file is created.",
    )
    parser.add_argument(
        "--num-jobs",
        type=int,
        default=None,
        metavar="NJOBS",
        help="If launcher is going to be used, this option specifies the number"
        " of jobs to create.",
    )
    parser.add_argument(
        "--restrict-status",
        type=int,
        nargs="+",
        default=None,
        help="If specified only TIC IDs with status within the given list are "
        "included in the sampling. If not specified, all positive statuses are "
        "used.",
    )
    parser.add_argument(
        "-d", "--job-description", default="", help="Description of the job."
    )
    add_slurm_config(parser)
    return parser.parse_args()


# Meant to function as callable
# pylint: disable=too-few-public-methods
class FileFromTemplate:
    """Create a file from a template given substitutions."""

    def __init__(self, fname, fixed_substitutions):
        """Read the specified template get ready to make files."""

        self._fname = fname
        self._fixed_substitutions = fixed_substitutions
        with open(
            fname.format(**fixed_substitutions, job_name="template"),
            "r",
            encoding="utf-8",
        ) as template_f:
            self._template_text = template_f.read()

    def __call__(self, var_substitution_list):
        """Create a file adding the given substitutions to the fixed ones."""

        def get_substitutions(var_substitutions):
            """Return the substitutions combining fixed and variable."""

            substitutions = self._fixed_substitutions.copy()
            substitutions.update(var_substitutions)
            if "job_group" in substitutions:
                substitutions["job_name"] = "{job_group}-{job_id}".format_map(
                    substitutions
                )
            else:
                substitutions["job_name"] = str(substitutions["job_id"])
            return substitutions

        fname = self._fname.format_map(
            get_substitutions(var_substitution_list[0])
        )
        dest_dir = os.path.dirname(fname)
        if not os.path.exists(dest_dir):
            os.makedirs(dest_dir)

        with open(fname, "w", encoding="utf-8") as outf:
            for var_substitutions in var_substitution_list:
                substitutions = get_substitutions(var_substitutions)
                file_contents = self._template_text
                for sub, value in substitutions.items():
                    file_contents = file_contents.replace(
                        f"@@{sub.upper()}@@", str(value)
                    )
                assert "@@" not in file_contents, (
                    "Not all substitutions filled in:\n" + file_contents
                )

                assert fname == self._fname.format_map(substitutions)
                outf.write(file_contents)
        return fname


# pylint: enable=too-few-public-methods


def get_file_makers(group_id, config):
    """Return instances of :class:`FileFromTemplate`_ for creating job files."""

    substitutions = {
        "hpc": config.hpc,
        "job_group": group_id,
        "partition": config.partition,
        "mode": "launcher",
        "nodes_per_job": config.nodes_per_job,
        "ntics_per_job": config.nodes_per_job * tic_per_node[config.hpc],
        "num_parallel": config.num_parallel,
        "time_limit": config.time_limit,
        "processes_per_job": tic_per_node[config.hpc] * config.nodes_per_job,
        "extra_cmdline": "",
    }
    return {
        "slurm": FileFromTemplate(slurm_fname, substitutions),
        "launcher_cmd": FileFromTemplate(
            config.launcher_commands_fname, substitutions
        ),
    }


def get_priority_tics(fname):
    """Read high priority TICs from the given file."""

    with open(fname, "r", encoding="utf-8") as priority_file:
        return set((int(tic_str) for tic_str in priority_file.read().split()))


def get_pending_tics(
    config, SelectTICTable, db_session  # pylint: disable=invalid-name
):
    """Return ordered list of all TIC IDs for which sampling is pending."""

    ticid_select = select(
        SelectTICTable.id  # pylint: disable=no-member
    ).filter_by(job_id=None)
    if config.restrict_status:
        ticid_select = ticid_select.where(
            SelectTICTable.status.in_(  # pylint: disable=no-member
                config.restrict_status
            )
        )
    else:
        ticid_select = ticid_select.where(
            SelectTICTable.status > 0  # pylint: disable=no-member
        )

    pending = db_session.scalars(ticid_select).all()
    if config.priority_tic_file:
        priority_tics = get_priority_tics(config.priority_tic_file)
        pending = set(pending)
        pending = list(pending & priority_tics) + list(pending - priority_tics)
    return pending


def create_job(  # pylint: disable=too-many-arguments, too-many-positional-arguments
    group_id,
    job_id,
    job_tic_ids,
    make_file,
    db_session,
    SelectTICTable,  # pylint: disable=invalid-name
):
    """Create a single job in the given group and return slurm filename."""

    db_session.execute(
        update(SelectTICTable),
        [
            {"id": tic_id, "job_group": group_id, "job_id": job_id}
            for tic_id in job_tic_ids
        ],
    )

    launcher_cmd = make_file["launcher_cmd"](
        [{"job_id": job_id, "ticid": tic_id} for tic_id in job_tic_ids]
    )
    return make_file["slurm"](
        [{"job_id": job_id, "launcher_cmd": launcher_cmd}]
    )


def create_job_group(config):
    """Create the new job group and return its slurm files."""

    SelectTICTable = get_ticid_select_table(config.tic_table, must_exist=True)
    num_tics = config.nodes_per_job * tic_per_node[config.hpc]
    with Session.begin() as db_session:  # pylint: disable=no-member
        job_group = JobGroup(
            select_tic_table=config.tic_table,
            hpc=config.hpc,
            nodes_per_job=config.nodes_per_job,
            num_jobs=config.num_jobs,
            description=config.job_description,
        )
        db_session.add(job_group)
        db_session.flush()
        make_file = get_file_makers(job_group.id, config)
        pending_tics = get_pending_tics(config, SelectTICTable, db_session)
        assert len(pending_tics) > num_tics * config.num_jobs
        return [
            create_job(
                job_group.id,
                job_id,
                pending_tics[num_tics * job_id : num_tics * (job_id + 1)],
                make_file,
                db_session,
                SelectTICTable,
            )
            for job_id in range(config.num_jobs)
        ]


if __name__ == "__main__":
    create_job_group(parse_command_line())
