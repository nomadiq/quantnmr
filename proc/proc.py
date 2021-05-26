import numpy as np
import scipy.signal as signal
from scipy.optimize.minpack import curve_fit
from scipy.fftpack import fftshift, fft
import math

# from pathlib import Path
import os.path

# from numba import jit, njit, prange
import colorama

# ------------------------------------- #
#              Variables                #
# ------------------------------------- #


# ------------------------------------- #
#              Functions                #
# ------------------------------------- #


def get_aq_mod(acq):

    if isinstance(acq, int):
        aq_mod_dict = {
            0: "qf",
            1: "qsim",
            2: "qseq",
            3: "DQD",
            4: "parallelQsim",
            5: "parallelDQD",
        }

        return aq_mod_dict[acq]
    else:
        return None


def get_fn_mod(acq):

    if isinstance(acq, int):
        fn_mod_dict = {
            0: "undefined",
            1: "QF",
            2: "QSEQ",
            3: "TPPI",
            4: "States",
            5: "States-TPPI",
            6: "Echo-Antiecho",
        }

        return fn_mod_dict[acq]
    else:
        return None


def make_complex(data):

    return data[..., ::2] + data[..., 1::2] * 1.0j


def next_fourier_number(num):

    return math.ceil(math.log(num, 2))