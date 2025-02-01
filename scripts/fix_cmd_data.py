#!/usr/bin/env python3

"""Fix the order of lines in the input data to handle few discontinuities."""

from paths import cmd_data_fname
import numpy

if __name__ == "__main__":
    for photsys in ["panstarss1", "2mass"]:
        formatted_fname = cmd_data_fname.format(photsys=photsys)
        buffer = []
        with (
            open(formatted_fname, "r", encoding="ascii") as orig,
            open(formatted_fname + ".fixed", "w", encoding="ascii") as fixed,
        ):
            for line in orig:
                if line[0] == "#":
                    previous_mini = -numpy.inf
                    for buf_line in reversed(buffer):
                        fixed.write(buf_line)
                    fixed.write(line)
                    buffer = []
                else:
                    mini = float(line.split()[3])
                    if mini >= previous_mini:
                        # print('Flushing buffer:\n\t' + '\t'.join(buffer))
                        for buf_line in reversed(buffer):
                            fixed.write(buf_line)
                        buffer = [line]
                        previous_mini = mini
                    else:
                        buffer.append(line)
                        # print('Holding back:\n\t' + line)
            for buf_line in reversed(buffer):
                fixed.write(buf_line)
