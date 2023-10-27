import numpy as np
from scipy import optimize
import math
from rates_estimates.sources import QDsource, SPDCSource
from rates_estimates.detectors import Detectors
from rates_estimates.fibres import Fibre
from rates_estimates.entropyfunctions import binaryentropy, conditional_entropy, misalignment_criterion
from rates_estimates.networkclass import Network

efficiency_cascade = 1.2

class SixStateNetwork(Network):

    def __init__(self, purity, efficiencysource, rate, mu, cold, fibreloss, efficiencydetector, darkcountrate, deadtime, length, misalignment):
        """
              A class for a setup using the six state protocol
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

    def get_omega(self):
        """
        Calculates the fraction of single photon states there are - formula from Lo et al 2005b
        :return: The fraction of single photon states: float
        """
        return 1-(self.source.getmultiphotonprob()/self.get_gain())

    def get_rate(self):
        """
        Calculate the rate of the setup: from GLLP formula Fung et al 2006: in particular difference with BB84 is the
        mutual information contribution. Here we assume errors are equally distributed in X, Y, Z directions i.e.
        ex=ey=ez
        :return: The rate for the standard six state protocol: float
        """
        omega = self.get_omega()
        if omega < 0.000000000000001:
            return 0
        else:
            # the rate here is the same as the rate of the BB84 except for the additional mutual information term
            # given by Q_mu Omega H(e_z+e_y) - H(Z|X)
            gain = self.get_gain()
            qber = self.get_qber()
            # assume that e_x = e_y = e_z
            e_x = qber / 2
            rawrate = omega * gain / 2
            lossfromerrorcorrection = gain * efficiency_cascade * binaryentropy(qber) / 2
            errorfrommisalignment = optimize.root_scalar(misalignment_criterion,
                                                         args=(np.power(self.misalignment, 2) * np.exp(1)), bracket=[0, 1000], method = 'bisect').root
            lossfromprivacyamplification = gain * omega * binaryentropy(qber + errorfrommisalignment) / 2
            # extra term from BB84 calculated here
            additionalmutualinfo = gain * omega * (binaryentropy(qber) - conditional_entropy(ex=e_x, ey = e_x, ez = e_x)) / 2
            return max(0, rawrate - lossfromerrorcorrection - lossfromprivacyamplification + additionalmutualinfo)

    def get_rate_efficient(self):
        """
        Calculate the rate of the setup: from GLLP formula Fung et al 2006: in particular difference with BB84 is the
        mutual information contribution. Here we assume errors are equally distributed in X, Y, Z directions i.e.
        ex=ey=ez
        :return: The rate for the efficient six state protocol - sifting removes a negligible amount of states: float
        """
        return 2 * self.get_rate()