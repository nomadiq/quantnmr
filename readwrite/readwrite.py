import struct
import sys
import numpy as np
import os

# dictionary of pipe parameters
fdata_dic = {
    'FDMAGIC': 0,
    'FDFLTFORMAT': 1,
    'FDFLTORDER': 2,
    'FDSIZE': 99,
    'FDREALSIZE': 97,
    'FDSPECNUM': 219,
    'FDQUADFLAG': 106,
    'FD2DPHASE': 256,
    'FDTRANSPOSED': 221,
    'FDDIMCOUNT': 9,
    'FDDIMORDER': 24,
    'FDDIMORDER1': 24,
    'FDDIMORDER2': 25,
    'FDDIMORDER3': 26,
    'FDDIMORDER4': 27,
    'FDPIPEFLAG': 57,
    'FDPIPECOUNT': 75,
    'FDSLICECOUNT': 443,
    'FDFILECOUNT': 442,
    'FDFIRSTPLANE': 77,
    'FDLASTPLANE': 78,
    'FDPARTITION': 65,
    'FDPLANELOC': 14,
    'FDMAX': 247,
    'FDMIN': 248,
    'FDSCALEFLAG': 250,
    'FDDISPMAX': 251,
    'FDDISPMIN': 252,
    'FDUSER1': 70,
    'FDUSER2': 71,
    'FDUSER3': 72,
    'FDUSER4': 73,
    'FDUSER5': 74,
    'FDLASTBLOCK': 359,
    'FDCONTBLOCK': 360,
    'FDBASEBLOCK': 361,
    'FDPEAKBLOCK': 362,
    'FDBMAPBLOCK': 363,
    'FDHISTBLOCK': 364,
    'FD1DBLOCK': 365,
    'FDMONTH': 294,
    'FDDAY': 295,
    'FDYEAR': 296,
    'FDHOURS': 283,
    'FDMINS': 284,
    'FDSECS': 285,
    'FDMCFLAG': 135,
    'FDNOISE': 153,
    'FDRANK': 180,
    'FDTEMPERATURE': 157,
    'FD2DVIRGIN': 399,
    'FDTAU': 199,
    'FDSRCNAME': 286,
    'FDUSERNAME': 290,
    'FDOPERNAME': 464,
    'FDTITLE': 297,
    'FDCOMMENT': 312,
    'FDF2LABEL': 16,
    'FDF2APOD': 95,
    'FDF2SW': 100,
    'FDF2OBS': 119,
    'FDF2ORIG': 101,
    'FDF2UNITS': 152,
    'FDF2QUADFLAG': 56,
    'FDF2FTFLAG': 220,
    'FDF2AQSIGN': 64,
    'FDF2LB': 111,
    'FDF2CAR': 66,
    'FDF2CENTER': 79,
    'FDF2OFFPPM': 480,
    'FDF2P0': 109,
    'FDF2P1': 110,
    'FDF2APODCODE': 413,
    'FDF2APODQ1': 415,
    'FDF2APODQ2': 416,
    'FDF2APODQ3': 417,
    'FDF2C1': 418,
    'FDF2ZF': 108,
    'FDF2X1': 257,
    'FDF2XN': 258,
    'FDF2FTSIZE': 96,
    'FDF2TDSIZE': 386,
    'FDF1LABEL': 18,
    'FDF1APOD': 428,
    'FDF1SW': 229,
    'FDF1OBS': 218,
    'FDF1ORIG': 249,
    'FDF1UNITS': 234,
    'FDF1FTFLAG': 222,
    'FDF1AQSIGN': 475,
    'FDF1LB': 243,
    'FDF1QUADFLAG': 55,
    'FDF1CAR': 67,
    'FDF1CENTER': 80,
    'FDF1OFFPPM': 481,
    'FDF1P0': 245,
    'FDF1P1': 246,
    'FDF1APODCODE': 414,
    'FDF1C1': 423,
    'FDF1ZF': 437,
    'FDF1X1': 259,
    'FDF1XN': 260,
    'FDF1FTSIZE': 98,
    'FDF1TDSIZE': 387,
    'FDF3LABEL': 20,
    'FDF3APOD': 50,
    'FDF3OBS': 10,
    'FDF3SW': 11,
    'FDF3ORIG': 12,
    'FDF3FTFLAG': 13,
    'FDF3AQSIGN': 476,
    'FDF3SIZE': 15,
    'FDF3QUADFLAG': 51,
    'FDF3UNITS': 58,
    'FDF3P0': 60,
    'FDF3P1': 61,
    'FDF3CAR': 68,
    'FDF3CENTER': 81,
    'FDF3OFFPPM': 482,
    'FDF3APODCODE': 400,
    'FDF3APODQ1': 401,
    'FDF3APODQ2': 402,
    'FDF3APODQ3': 403,
    'FDF3C1': 404,
    'FDF3ZF': 438,
    'FDF3X1': 261,
    'FDF3XN': 262,
    'FDF3FTSIZE': 200,
    'FDF3TDSIZE': 388,
    'FDF4LABEL': 22,
    'FDF4APOD': 53,
    'FDF4OBS': 28,
    'FDF4SW': 29,
    'FDF4ORIG': 30,
    'FDF4FTFLAG': 31,
    'FDF4AQSIGN': 477,
    'FDF4SIZE': 32,
    'FDF4QUADFLAG': 54,
    'FDF4UNITS': 59,
    'FDF4P0': 62,
    'FDF4P1': 63,
    'FDF4CAR': 69,
    'FDF4CENTER': 82,
    'FDF4OFFPPM': 483,
    'FDF4APODCODE': 405,
    'FDF4APODQ1': 406,
    'FDF4APODQ2': 407,
    'FDF4APODQ3': 408,
    'FDF4C1': 409,
    'FDF4ZF': 439,
    'FDF4X1': 263,
    'FDF4XN': 264,
    'FDF4FTSIZE': 201,
    'FDF4TDSIZE': 389,
    'FD_SEC': 1,
    'FD_HZ': 2,
    'FD_PPM': 3,
    'FD_PTS': 4,
    'FD_MAGNITUDE': 0,
    'FD_TPPI': 1,
    'FD_STATES': 2,
    'FD_IMAGE': 3,
    'FD_QUAD': 0,
    'FD_COMPLEX': 0,
    'FD_SINGLATURE': 1,
    'FD_REAL': 1,
    'FD_PSEUDOQUAD': 2
}


def fdata2dic(fdata):
    """
    Convert a fdata array to fdata dictionary.

    Converts the raw 512x4-byte NMRPipe header into a python dictionary
    with keys as given in fdatap.h

    """
    dic = {key: float(fdata[int(fdata_dic[key])]) for key in fdata_dic.keys()}

    # make the FDDIMORDER
    dic["FDDIMORDER"] = struct.unpack('8s', fdata[22:24])[0].rstrip(b'\x00')
    dic["FDSRCNAME"] = struct.unpack('16s', fdata[286:290])[0].rstrip(b'\x00')
    dic["FDUSERNAME"] = struct.unpack('16s', fdata[290:294])[0].rstrip(b'\x00')
    dic["FDTITLE"] = struct.unpack('60s', fdata[297:312])[0].rstrip(b'\x00')
    dic["FDCOMMENT"] = struct.unpack('160s', fdata[312:352])[0].rstrip(b'\x00')
    dic["FDOPERNAME"] = struct.unpack('32s', fdata[464:472])[0].rstrip(b'\x00')
    return dic


def readnmrPipe(infile):
    if infile:
        data = np.fromfile(infile, 'float32')
    else:
        stdin = sys.stdin.read()
        data = np.frombuffer(stdin, dtype=np.float32)
    if data[2] > 2.3450010000000003:  # check for byteswap
        data = data.byteswap()

    # header = data[:512]
    # data =  data[512:]

    return data[:512], data[512:]


def readserfile(path, filename):
    if (
            os.path.isfile(path+'/acqus')
            and os.path.isfile(path+'/'+filename)
        ):
        # do stuff
        with open(path+'/'+filename, "rb") as serial_file:
            if self.dtypa == 0:
                dtype_string = "i4"
            elif self.dtypa == 2:
                dtype_string = "i8"
    
    else:
        print('Not a valid Bruker Path for Serial File')
        return 0



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


def remove_bruker_delay(data, grpdly):
    n = len(data)
    integer_shift = np.floor(grpdly)
    fract_shift = grpdly - integer_shift
    data = np.roll(data, -int(integer_shift))
    data = nmr_fft(data) / n
    data = data * np.exp(2.0j * np.pi * (fract_shift) * np.arange(n) / n)
    data = i_nmr_fft(data) * n
    return data

def dd2g(dspfvs, decim):
    dspdic = {
        10: {
            2: 44.75,
            3: 33.5,
            4: 66.625,
            6: 59.083333333333333,
            8: 68.5625,
            12: 60.375,
            16: 69.53125,
            24: 61.020833333333333,
            32: 70.015625,
            48: 61.34375,
            64: 70.2578125,
            96: 61.505208333333333,
            128: 70.37890625,
            192: 61.5859375,
            256: 70.439453125,
            384: 61.626302083333333,
            512: 70.4697265625,
            768: 61.646484375,
            1024: 70.48486328125,
            1536: 61.656575520833333,
            2048: 70.492431640625,
        },
        11: {
            2: 46.0,
            3: 36.5,
            4: 48.0,
            6: 50.166666666666667,
            8: 53.25,
            12: 69.5,
            16: 72.25,
            24: 70.166666666666667,
            32: 72.75,
            48: 70.5,
            64: 73.0,
            96: 70.666666666666667,
            128: 72.5,
            192: 71.333333333333333,
            256: 72.25,
            384: 71.666666666666667,
            512: 72.125,
            768: 71.833333333333333,
            1024: 72.0625,
            1536: 71.916666666666667,
            2048: 72.03125,
        },
        12: {
            2: 46.0,
            3: 36.5,
            4: 48.0,
            6: 50.166666666666667,
            8: 53.25,
            12: 69.5,
            16: 71.625,
            24: 70.166666666666667,
            32: 72.125,
            48: 70.5,
            64: 72.375,
            96: 70.666666666666667,
            128: 72.5,
            192: 71.333333333333333,
            256: 72.25,
            384: 71.666666666666667,
            512: 72.125,
            768: 71.833333333333333,
            1024: 72.0625,
            1536: 71.916666666666667,
            2048: 72.03125,
        },
        13: {
            2: 2.75,
            3: 2.8333333333333333,
            4: 2.875,
            6: 2.9166666666666667,
            8: 2.9375,
            12: 2.9583333333333333,
            16: 2.96875,
            24: 2.9791666666666667,
            32: 2.984375,
            48: 2.9895833333333333,
            64: 2.9921875,
            96: 2.9947916666666667,
        },
    }
    return dspdic[dspfvs][decim]




class LINData3D:
    def __init__(
        self,
        data_dir=".",
        ser_file="ser",
        points=None,
        dim_status=None,
        decim=None,
        dspfvs=None,
        grpdly=None,
    ):  # sourcery no-metrics

        self.ac1 = os.path.join(data_dir, "acqus")
        self.ac2 = os.path.join(data_dir, "acqu2s")
        self.ac3 = os.path.join(data_dir, "acqu3s")
        self.ser = os.path.join(data_dir, "ser")
        self.pp = os.path.join(data_dir, "pulseprogram")
        self.ser = os.path.join(data_dir, ser_file)
        self.dir = data_dir
        self.acq = [0, 0, 0]  # acquisition modes start as undefined

        # dictionary of acquisition modes for Bruker
        self.acqDict = {
            0: "undefined",
            1: "qf",
            2: "qsec",
            3: "tppi",
            4: "states",
            5: "states-tppi",
            6: "echo-antiecho",
        }

        # check if we are a Bruker 2D data set
        if (
            os.path.isfile(self.ac1)
            and os.path.isfile(self.ac2)
            and os.path.isfile(self.ac3)
            and os.path.isfile(self.ser)
            and os.path.isfile(self.pp)
        ):
            self.valid = True

        else:
            self.valid = False
            print("Data Directory does not seem to contain Bruker 3D Data")

        p0 = p1 = p2 = 0  # we'll find these in the files
        dec = dsp = grp = 0  # we'll find these in the files

        with open(self.ac1, "r") as acqusfile:
            for line in acqusfile:
                if "##$TD=" in line:
                    (_, value) = line.split()
                    p0 = int(value)
                if "##$DECIM=" in line:
                    (_, value) = line.split()
                    dec = int(value)
                if "##$DSPFVS=" in line:
                    (_, value) = line.split()
                    dsp = int(value)
                if "##$GRPDLY=" in line:
                    (_, value) = line.split()
                    grp = float(value)

                if "##$BYTORDA=" in line:
                    (_, value) = line.split()
                    self.byte_order = float(value)

                self.acq[0] = 0  # doesnt matter we assume DQD for direct anyway

        with open(self.ac2, "r") as acqusfile:
            for line in acqusfile:
                if "##$TD=" in line:
                    (_, value) = line.split()
                    p1 = int(value)

                if "##$FnMODE=" in line:
                    (_, value) = line.split()
                    self.acq[1] = int(value)

        with open(self.ac3, "r") as acqusfile:
            for line in acqusfile:
                if "##$TD=" in line:
                    (_, value) = line.split()
                    p2 = int(value)

                if "##$FnMODE=" in line:
                    (_, value) = line.split()
                    self.acq[2] = int(value)

        if p0 and p1 and p2:  # we got # points for all three dimensions
            points = [p0, p1, p2]
        else:
            print("problem with detecting number of points in data")
            self.valid = False

        if dec != 0:
            self.decim = dec
        if dsp != 0:
            self.dspfvs = dsp
        if grp:
            self.grpdly = grp
        elif dec != 0 and dsp != 0:
            self.grpdly = dd2g(self.dspfvs, self.decim)
        else:
            print(
                "problem with detecting / determining grpdly - needed for Bruker conversion"
            )
            self.valid = False

        print("Data Points structure is: " + str(points))
        print(
            "DECIM= "
            + str(self.decim)
            + " DSPFVS= "
            + str(self.dspfvs)
            + " GRPDLY= "
            + str(self.grpdly)
        )

        
        
        # lets store the points to the class instance
        self.points = points

        # now lets load in the bruker serial file
        with open(self.ser, "rb") as serial_file:
            if self.byte_order == 0:
                self.raw_data = np.frombuffer(serial_file.read(), dtype="<i4")
            elif self.byte_order == 1:
                self.raw_data = np.frombuffer(serial_file.read(), dtype=">i4")

        # now reshape the data
        self.raw_data = np.reshape(self.raw_data, np.asarray(self.points), order="F")
        