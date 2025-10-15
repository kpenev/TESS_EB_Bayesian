"""Utilities to select file system paths for things."""

import os.path

from paths import render_dir

def get_render_dir(table_name, mode):
    """Return directory for storing plots for given table and mode."""

    return os.path.join(render_dir, table_name, mode)

def parse_render_dir(dir_path):
    """Return table name and plot type stored in given render directory."""

    prefix, mode = os.path.split(dir_path)
    table_name = os.path.basename(prefix)
    return table_name, mode
