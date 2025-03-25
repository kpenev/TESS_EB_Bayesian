#!/usr/bin/env python3
"""Simple script to render plots for all TIC IDs on an HPC system."""

from sys import argv

from bui.select_ticids.plots import render_one

if __name__ == '__main__':
    render_one(argv[1], "prsa_ebs", '/scratch/05392/kpenev/prsa_ebs')
