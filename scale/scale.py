class Scale:
    """
    A Scale Class
    """
    def __init__(self, domainrange, outrange, strict=False):
        self.d_min = domainrange[0]
        self.d_max = domainrange[1]
        self.d_scope = self.d_max - self.d_min

        self.o_min = outrange[0]
        self.o_max = outrange[1]
        self.o_scope = self.o_max - self.o_min

        self.strict = strict

    def linear(self, indomain):

        if self.strict is True and (indomain < self.d_min or indomain > self.d_max):
            raise Exception(
                f"input value {indomain} is outside the input domain for this scale"
            )

        domainfrac = (indomain - self.d_min) / self.d_scope
        outfrac = domainfrac * self.o_scope
        return self.o_min + outfrac

    def linear_r(self, inrange):

        if self.strict is True and (inrange < self.o_min or inrange > self.o_max):
            raise Exception(
                f"input value {inrange} is outside the input domain for this scale"
            )

        domainfrac = (inrange - self.o_min) / self.o_scope
        outfrac = domainfrac * self.d_scope
        return self.d_min + outfrac