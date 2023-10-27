import numpy as np
from scipy import optimize
import math
from rates_estimates.sources import QDsource, SPDCSource
from rates_estimates.detectors import Detectors
from rates_estimates.fibres import Fibre
from rates_estimates.entropyfunctions import binaryentropy
from rates_estimates.networkclass import Network


efficiency_cascade = 1.2

class DPSNetwork(Network):

    def __init__(self, purity, efficiencysource, rate, mu, cold, fibreloss, efficiencydetector, darkcountrate, deadtime, length, misalignment):
        """
        A class for a setup using the BB84 protocol
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
        :param misalignment: The misalignment of the devices in degrees: float
        """
        super().__init__(purity, efficiencysource, rate, mu, cold, fibreloss, efficiencydetector, darkcountrate, deadtime, length, misalignment)

    def get_qber(self):
        """
         Calculates the QBER of the setup - formula from Eraerds et al 2010
        :return: The QBER of the setup: float
        """
        v = self.getvisibility()
        denom = self.pmu + 2 * self.pdc + self.pap
        num = self.pmu * (1-v) + 2 * self.pdc + self.pap
        return num/(2* denom) + np.power(self.misalignment,2)

    def get_gain(self):
        """
         Calculates the gain of the setup, Q_(mu): formula from Eraerds et al 2010
        :return: The gain, Q_(mu) of the setup: float
        """
        etadead = np.power(1+ self.detector.getdeadtime() * self.source.getrate() * (self.pmu + 2 * self.pdc + self.pap) ,-1)
        return (self.pmu + 2 * self.pdc + self.pap) * etadead


    def getpc(self, qber):
        if qber > 6/38:
            return 1- np.power(6/38, 2) - np.power(1-6 * 6/38, 2)/2
        else:
            return 1- np.power(qber, 2) - np.power(1-6 * qber, 2)/2

    def get_rate(self):
        """
        Calculate the rate of the setup: from GLLP formula Gottesman et al 2004, Lo et al 2005b
        :return: The rate for the standard BB84 protocol: float
        """
        if isinstance(self.source, QDsource):
            print("DPS protocol requires an SPDC source: warm Alice only")
            raise ValueError
        else:
            gain = self.get_gain()
            qber = self.get_qber()
            pc = self.getpc(qber)
            rawrate = gain * np.log2(pc)
            lossfromerrorcorrection = gain * efficiency_cascade * binaryentropy(qber)
            lossfromprivacyamplification = gain * 2 * self.source.getmu() * np.log2(pc)
            return max(0, rawrate - lossfromerrorcorrection - lossfromprivacyamplification)
