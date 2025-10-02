"""Utilities to select file system paths for things."""

import os.path

from paths import render_dir

def get_render_dir(table_name, mode):
    """Return directory for storing plots for given table and mode."""

    return os.path.join(render_dir, table_name, mode)
