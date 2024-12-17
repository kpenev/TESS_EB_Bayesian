#!/usr/bin/env python3

"""Similar to test_against_phoebe.py but zoom in on eclipses."""

import numpy

from binary import Binary
from phoebe_model import create_phoebe_binary
from test_against_phoebe import parse_command_line, CalculateScenario, run_tests


# Intended to function just as a callable for multiprocessing
# pylint: disable=too-few-public-methods
class CalculateZoomedScenario(CalculateScenario):
    """Set the times to evaluate the scenario to only be around eclipse(s)."""

    def _get_eclipse_times(self, binary, times):
        """Return times to cover eclipse dealing with period folding."""

        eclipse = binary.eclipse(times)
        assert (eclipse <= 1).all()
        out_of_eclipse = eclipse == 1
        if out_of_eclipse[0]:
            eclipsed_indices = numpy.nonzero(numpy.logical_not(out_of_eclipse))[
                0
            ]
            tmin = times[eclipsed_indices[0] - 1]
            tmax = times[eclipsed_indices[-1] + 1]
        else:
            uneclipsed_indices = numpy.nonzero(out_of_eclipse)[0]
            tmax = times[uneclipsed_indices[0]]
            tmin = times[uneclipsed_indices[-1]] - binary.per
        print(f"Time range: {tmin} - {tmax}")
        return numpy.linspace(tmin, tmax, self._ntimes)

    def _get_eval_times(self, parameters):
        """Return the required number of times in the vicinity of eclipses."""

        binary = Binary(from_phoebe=create_phoebe_binary(**parameters))
        times = numpy.linspace(0.0, binary.per, max(101, self._ntimes))
        primary_times = self._get_eclipse_times(binary, times)
        binary.swap_components()
        secondary_times = self._get_eclipse_times(binary, times)
        return primary_times, secondary_times


# pylint: enable=too-few-public-methods

if __name__ == "__main__":
    run_tests(
        parse_command_line(
            default_inclinations=[85.0, 90.0, 91.0], default_ntimes=101
        ),
        CalculateZoomedScenario,
        sub_scenario_titles=("Primary Eclipse", "Secondary Eclipse"),
    )
