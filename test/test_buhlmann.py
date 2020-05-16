# Project: diving_decompression
# File: test_buhlmann.py
# Created by BlueTrin at 15-May-20

from unittest import TestCase

import buhlmann
import numpy.testing as npt
import math


class TestBuhlmann(TestCase):
    def test_zero_time(self):
        gas = buhlmann.Gas(n2_pc=0.45, he_pc=0.40)
        tissue = buhlmann.Tissues()
        new_tissue = buhlmann.get_partial_pressures(tissue, gas, buhlmann.depth_to_pressure(0),
                                                    buhlmann.depth_to_pressure(40), 0)
        npt.assert_array_equal(tissue.n2_p, new_tissue.n2_p)
        npt.assert_array_equal(tissue.he_p, new_tissue.he_p)

    def test_flat_bottom(self):
        gas = buhlmann.Gas(n2_pc=0.79, he_pc=0.0)
        tissue = buhlmann.Tissues()
        new_tissue = buhlmann.get_partial_pressures(tissue, gas, buhlmann.depth_to_pressure(40),
                                                    buhlmann.depth_to_pressure(40), 20)
        self.assertEqual(
            math.ceil(buhlmann.pressure_to_depth(buhlmann.ceiling(new_tissue))),
            6
        )

    def test_dive_plan(self):
        dive = [
            (0, 0),
            (2, 40),  # 40 meters at 2 mins
            (22, 40),  # 40 meters at 22 mins
        ]

        tissues = buhlmann.Tissues()
        gas = buhlmann.Gas(n2_pc=0.79, he_pc=0.0)

        print("initial ceiling: {}".format(buhlmann.pressure_to_depth(buhlmann.ceiling(tissues))))
        for i_step, ((start_time, start_depth), (end_time, end_depth)) in enumerate(zip(dive[:-1], dive[1:])):
            print("Step:{}, t_start={}, depth_start={}, t_end={}, depth_end={}".format(
                i_step, start_time, start_depth, end_time, end_depth
            ))
            tissues = buhlmann.get_partial_pressures(
                tissues,  # vector for compartments
                gas,
                buhlmann.depth_to_pressure(start_depth),  # for example 0 feet
                buhlmann.depth_to_pressure(end_depth),  # for example 120 feet
                end_time - start_time,  # time for depth change
            )
            print("ceiling: {}".format(buhlmann.pressure_to_depth(buhlmann.ceiling(tissues))))

        stops = buhlmann.get_stops_to_surface(tissues, end_depth, gas, 9)
        pass

