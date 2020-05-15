# Project: diving_decompression
# File: test_buhlmann.py
# Created by BlueTrin at 15-May-20

import numpy as np
import pandas as pd
import math
from prettytable import PrettyTable

df = pd.DataFrame([
    [1, 5.0, 1.1696, 0.5578, 1.88, 1.6189, 0.4770],
    [2, 8.0, 1.0000, 0.6514, 3.02, 1.3830, 0.5747],
    [3, 12.5, 0.8618, 0.7222, 4.72, 1.1919, 0.6527],
    [4, 18.5, 0.7562, 0.7825, 6.99, 1.0458, 0.7223],
    [5, 27.0, 0.6200, 0.8126, 10.21, 0.9220, 0.7582],
    [6, 38.3, 0.5043, 0.8434, 14.48, 0.8205, 0.7957],
    [7, 54.3, 0.4410, 0.8693, 20.53, 0.7305, 0.8279],
    [8, 77.0, 0.4000, 0.8910, 29.11, 0.6502, 0.8553],
    [9, 109.0, 0.3750, 0.9092, 41.20, 0.5950, 0.8757],
    [10, 146.0, 0.3500, 0.9222, 55.19, 0.5545, 0.8903],
    [11, 187.0, 0.3295, 0.9319, 70.69, 0.5333, 0.8997],
    [12, 239.0, 0.3065, 0.9403, 90.34, 0.5189, 0.9073],
    [13, 305.0, 0.2835, 0.9477, 115.29, 0.5181, 0.9122],
    [14, 390.0, 0.2610, 0.9544, 147.42, 0.5176, 0.9171],
    [15, 498.0, 0.2480, 0.9602, 188.24, 0.5172, 0.9217],
    [16, 635.0, 0.2327, 0.9653, 240.03, 0.5119, 0.9267],
],
    columns=['compartment',
             'n2_halflife', 'n2_a', 'n2_b',
             'he_halflife', 'he_a', 'he_b']
)


def generate_ascii_table(df):
    x = PrettyTable()
    x.field_names = df.columns.tolist()
    for row in df.values:
        x.add_row(row)
    print(x)
    return x


# in bar
SURFACE_PRESSURE = 1
WATER_VAPOR_PRESSURE_ALVEOLI = 0.0567

# in fsw
# SURFACE_PRESSURE = 33
# WATER_VAPOR_PRESSURE_ALVEOLI = 2.042

# 1st dive so 0.79 N2 and 0 HE
surface_n2_pp = 0.79


class Gas(object):
    def __init__(self, n2_pc, he_pc):
        if not 0 <= n2_pc <= 1:
            raise RuntimeError("N2 must be between 0 and 1")
        if not 0 <= he_pc <= 1:
            raise RuntimeError("He must be between 0 and 1")
        if not 0 <= n2_pc + he_pc <= 1:
            raise RuntimeError("He+N2 must be between 0 and 1")

        self.n2_pc = n2_pc
        self.he_pc = he_pc
        self.o2_pc = 1.0 - he_pc - n2_pc


class Tissues(object):
    def __init__(self, n2_p=None, he_p=None):
        # default tissues saturation for surface
        # Compute P0(He) and P0(N2)
        if n2_p is not None:
            self.n2_p = n2_p
        else:
            self.n2_p = pd.array([(SURFACE_PRESSURE - WATER_VAPOR_PRESSURE_ALVEOLI) * surface_n2_pp] * 16)

        if he_p is not None:
            self.he_p = he_p
        else:
            self.he_p = pd.array([0] * 16)


def get_partial_pressures(
        tissues,  # tissue loading vectors for compartments
        gas,  # gas composition
        start_pressure,  # in bar
        end_pressure,  # in bar
        t,  # time for depth change
):
    """

    :type tissues: Tissues
    :type gas: Gas
    """
    if t:
        rate_depth = (end_pressure - start_pressure) / t
    else:
        rate_depth = 0

    # Compute P_{i,0}(He)
    init_inspired_pp_he = (start_pressure - WATER_VAPOR_PRESSURE_ALVEOLI) * gas.he_pc
    init_inspired_pp_n2 = (start_pressure - WATER_VAPOR_PRESSURE_ALVEOLI) * gas.n2_pc

    rate_change_he_p = rate_depth * gas.he_pc
    rate_change_n2_p = rate_depth * gas.n2_pc

    # Compute k(He)
    kHe = math.log(2) / df['he_halflife']

    # P(He) = Pi,0(He) + R(HE) (t - 1/k(He)) - (Pi,0(He) - P0(He) - R(He)/k(He)) exp(-2 k(He))
    p_he = init_inspired_pp_he + rate_change_he_p * (t - 1 / kHe) \
           - (init_inspired_pp_he - tissues.he_p - rate_change_he_p / kHe) \
           * np.exp(-kHe * t)

    kN2 = math.log(2) / df['n2_halflife']
    p_n2 = init_inspired_pp_n2 + rate_change_n2_p * (t - 1 / kN2) \
           - (init_inspired_pp_n2 - tissues.n2_p - rate_change_n2_p / kN2) \
           * np.exp(-kN2 * t)

    return Tissues(n2_p=p_n2, he_p=p_he)


def ceiling(
        tissues: Tissues) -> float:
    a = (df['n2_a'] * tissues.n2_p + df['he_a'] * tissues.he_p) / (tissues.n2_p + tissues.he_p)
    b = (df['n2_b'] * tissues.n2_p + df['he_b'] * tissues.he_p) / (tissues.n2_p + tissues.he_p)
    tissue_ceilings = ((tissues.n2_p + tissues.he_p) - a) * b
    return max(tissue_ceilings)


def depth_to_pressure(depth):
    return depth / 10.0 + 1.0


def pressure_to_depth(pressure):
    return (pressure - 1.0) * 10.0


def main():
    generate_ascii_table(df)

    descent_rate = 20  # 20m / min
    ascent_rate = 9  # 9m / min
    gas = Gas(n2_pc=0.4, he_pc=0.45)

    dive = [
        (0, 0),
        (40, 2),  # 40 meters at 2 mins
        (40, 22),  # 40 meters at 22 mins
    ]
    tissues = Tissues()
    print("initial ceiling: {}".format(pressure_to_depth(ceiling(tissues))))

    i_step = 0
    ((start_depth, start_time), (end_depth, end_time)) = list(zip(dive[:-1], dive[1:]))[i_step]
    tissues = get_partial_pressures(
        tissues,  # vector for compartments
        gas,
        depth_to_pressure(start_depth),  # for example 0 feet
        depth_to_pressure(end_depth),  # for example 120 feet
        end_time - start_time,  # time for depth change
    )
    print("N2tissues: {}".format(tissues.n2_p))
    print("ceiling: {}".format(pressure_to_depth(ceiling(tissues))))
    # tissues = get_partial_pressures(
    #         tissues,  # vector for compartments
    #         gas,
    #         end_depth,    # for example 0 feet
    #         end_depth,      # for example 120 feet
    #         20,              # time for depth change
    # )
    # print("ceiling: {}".format(ceiling(tissues)))
