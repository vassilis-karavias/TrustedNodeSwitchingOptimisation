import numpy as np
from scipy import optimize
import math
from rates_estimates.sources import QDsource, SPDCSource
from rates_estimates.detectors import Detectors
from rates_estimates.fibres import Fibre
from rates_estimates.entropyfunctions import binaryentropy, conditional_entropy, misalignment_criterion
from rates_estimates.networkclass import Network

efficiency_cascade = 1.2

class MisalignedTolerantBB84(Network):

    def __init__(self, purity, efficiencysource, rate, mu, cold, fibreloss, efficiencydetector, darkcountrate, deadtime, length, misalignment):
        """
               A general interface for a Network setup
               :param purity: The purity of the source, needed for cold Alice: float
               :param efficiencysource: The efficiency of the source, needed for cold Alice: float
               :param rate: The rate of the source in Hz: float
               :param mu: The mean number of photons per pulse, needed for hot Alice: float
               :param cold: If we are using cold Alice (QD source) or hot Alice (SPDC source): boolean
               :param fibreloss: The loss of the fibre in dB/km: float
               :param efficiencydetector: The efficiency of the detector: float
               :param darkcountrate: The dark count rate of the detector in s^(-1): float
               :param deadtime: The dead time of the detector in s: float
               :param length: The length of the fibre in km: float
               :param misalignment: The angle of misalignment of the devices in degrees
        """
        super().__init__(purity, efficiencysource, rate, mu, cold, fibreloss, efficiencydetector, darkcountrate, deadtime, length, misalignment)
        self.misalignment = misalignment* np.pi / 180

    def getgainparameters(self):
        """
        Get the parameters to obtain the Gain and QBER - Tamaki et al (2014)
        :return: The 4 parameters, p00, p10, p01, p11: float
        """
        efficiency = self.etaf * self.detector.getefficiency()
        p00 = self.pdc + (1-self.pdc) * (1-np.exp(-self.source.getmu() * (efficiency)))
        p10 = self.pdc
        p01 = self.pdc + (1-self.pdc) * (1-np.exp(-self.source.getmu() * (efficiency) * np.power(np.sin(self.misalignment/2), 2)))
        p11 = self.pdc + (1-self.pdc) * (1-np.exp(-self.source.getmu() * (efficiency) * np.power(np.cos(self.misalignment/2), 2)))
        return p00, p10, p01, p11


    def get_gain(self):
        """
        Obtain the Gain of the system - Tamaki et al (2014)
        :return: The total gain of the network including from Alice to Bob: float
        """
        p00, p10, p01, p11 = self.getgainparameters()
        gain = p00 *(1-p10) + (1-p00) * p10 + p00 * p10 + p01 *(1-p11) + (1-p01) * p11 + p01 * p11
        return gain / 3

    def get_qber(self):
        """
        Obtain the QBER for the system - Tamaki et al (2014)
        :return: The QBER of the system: float
        """
        p00, p10, p01, p11 = self.getgainparameters()
        wmu = p10 *(1-p00) + p10 * p00 /2 + p01 *(1-p11) + p01 * p11 /2
        gain = self.get_gain()
        return wmu/(3 * gain)

    def get_cstheta(self, s, j, delta):
        """
        Get the value for C_s,j(delta) for the parametes - Tamaki et al
        :param s: The Bit measured by Bob: int
        :param j: The Bit sent by Alice: int
        :param delta: The misalignment of the states: float
        :return: The value for C_s,j(delta): float
        """
        if s == 0 and j == 0:
            return (1+np.sin(delta / 2) + np.cos(delta / 2)) / (2* np.sqrt(1+np.sin(delta /2)))
        elif s == 1 and j == 0:
            return (1+np.sin(delta / 2) - np.cos(delta /2)) / (2* np.sqrt(1+np.sin(delta /2)))
        elif s == 0 and j == 1:
            return (1-np.sin(delta / 2) - np.cos(delta/ 2)) / (2* np.sqrt(1-np.sin(delta /2)))
        elif s == 1 and j == 1:
            return (1 - np.sin(delta / 2) + np.cos(delta / 2)) / (
                        2 * np.sqrt(1 - np.sin(delta / 2)))
        else:
            print("Bits must be 0 or 1")
            raise ValueError

    def get_bityieldsforgiveninputstate(self, s, j):
        """
        Get Y_sXjX for given s and j - Tamaki et al
        :param s: The bit measured by Bob - int
        :param j: The bit sent by Alice - int
        :return: The value for Y_sXjX: float
        """
        cstheta = self.get_cstheta(s = s, j = j, delta = 3 * self.misalignment / 2)
        csplus1 = self.get_cstheta(s = (s + 1) % 2, j =j, delta = 3 * self.misalignment /2)
        efficiency = self.etaf * self.detector.getefficiency()
        bityield = (efficiency) * np.power(cstheta,2) *(1-self.pdc) + (1-efficiency) * self.pdc *(1-self.pdc/ 2) \
                   + ((efficiency) * self.pdc * np.power(csplus1,2))
        return bityield

    def getsinglephotongain(self):
        """
        Finds the single photon gain - Tamaki et al (2014)
        :return: The single photon gain Q_1: float
        """
        y0z0 = self.get_bityieldsforgiveninputstate(s= 0, j= 0)
        y1z0 = self.get_bityieldsforgiveninputstate(s = 1, j= 0)
        y0z1 = self.get_bityieldsforgiveninputstate(s = 0, j= 1)
        y1z1 = self.get_bityieldsforgiveninputstate(s = 1, j= 1)
        sum_all_terms = (y0z0 + y1z0) * (1 + np.sin(self.misalignment / 2)) / 2  + (y0z1  + y1z1) * (1 - np.sin(self.misalignment/ 2)) / 2
        return np.exp(-2 * self.source.getmu()) * self.source.getmu()  * sum_all_terms/2

    # def getmatrix(self):
    #     return [[1/6, 0, 1/6],[1/6, -np.sin(self.misalignment)/6, -np.cos(self.misalignment)/6],
    #             [1/6, np.sin((np.pi +self.misalignment)/2)/6,np.cos((np.pi +self.misalignment)/2)/6]]
    #
    # def get_conditionalprobabilities(self, bit):
    #     ysx0z = self.get_bityieldsforgiveninputstate(s = bit, theta = 0)
    #     ysx1z = self.get_bityieldsforgiveninputstate(s = bit, theta= self.misalignment /2)
    #     ysx0x = self.get_bityieldsforgiveninputstate(s = bit, theta= (np.pi + self.misalignment)/2)
    #     conditional_probs = np.matmul(np.linalg.inv(self.getmatrix()), [ysx0z, ysx1z, ysx0x])
    #     return conditional_probs

    # def getvirtualyields(self, conditional_probs, jx):
    #     # trace is sin delta
    #     qsxid = conditional_probs[0]
    #     qsxx = conditional_probs[1]
    #     qsxz = conditional_probs[2]
    #     return np.sin(self.misalignment) *(qsxid + np.power(-1, jx) * np.cos(self.misalignment/2) * qsxx  + np.power(-1, jx +1) * np.sin(self.misalignment/2) * qsxz)


    def getphaseerror(self):
        """
        Calculate the phase error estimate from the method described in Tamaki et al (2014)
        :return: The phase error, e_phase
        """
        y1x0x = self.get_bityieldsforgiveninputstate(s = 1, j = 0)
        y0x1x = self.get_bityieldsforgiveninputstate(s = 0, j = 1)
        y0x0x = self.get_bityieldsforgiveninputstate(s = 0, j = 0)
        y1x1x = self.get_bityieldsforgiveninputstate(s = 1, j = 1)
        return (y1x0x + y0x1x) / (y0x0x + y1x0x + y0x1x + y1x1x)

        # conditional_probs_0 = self.get_conditionalprobabilities(bit = 0)
        # conditional_probs_1 = self.get_conditionalprobabilities(bit = 1)
        # y0x1x = self.getvirtualyields(conditional_probs=conditional_probs_0, jx = 1)
        # y1x0x = self.getvirtualyields(conditional_probs=conditional_probs_1, jx = 0)
        # y0x0x = self.getvirtualyields(conditional_probs=conditional_probs_0, jx = 0)
        # y1x1x = self.getvirtualyields(conditional_probs=conditional_probs_1, jx = 1)
        # return (y0x1x + y1x0x)/(y0x0x + y0x1x + y1x0x + y1x1x)

    def get_rate(self):
        """
        Calculate the rate of the setup: Tamaki et al (2014)
        :return: The rate for the standard decoy state BB84 protocol: float
        """
        phase_error = self.getphaseerror()
        singlephotongain = self.getsinglephotongain()
        gain = self.get_gain()
        qber = self.get_qber()
        lossfromerrorcorrection = gain * efficiency_cascade * binaryentropy(qber)
        lossfromprivacyamplification = singlephotongain * binaryentropy(phase_error)
        return max(0, singlephotongain - lossfromprivacyamplification - lossfromerrorcorrection)

