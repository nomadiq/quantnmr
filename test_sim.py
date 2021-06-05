import maunakini.sim as mks


def test_tc_tract_algebraic():
    Ra = 23
    Rb = 80
    field = 700
    S2 = 0.86

    assert (
        mks.tc_tract_algebraic(Ra, Rb, field) - 2.3684226875850479498 / 100000000
    ) < 0.000000000000000000000001
