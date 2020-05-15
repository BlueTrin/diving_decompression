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

