import numpy as np
from scipy import optimize
import math
from rates_estimates.entropyfunctions import binaryentropy, misalignment_criterion
from rates_estimates.networkclass import Network

efficiency_cascade = 1.2

class BB84Network(Network):

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

    def get_omega(self):
        """
        Calculates the fraction of single photon states there are - formula from Lo et al 2005b
        :return: The fraction of single photon states: float
        """
        return 1-(self.source.getmultiphotonprob()/self.get_gain())

    def get_rate(self):
        """
        Calculate the rate of the setup: from GLLP formula Gottesman et al 2004, Lo et al 2005b
        :return: The rate for the standard BB84 protocol: float
        """
        # if the fraction of single photons in the states measured by Bob are 0 then the rate must be 0
        omega = self.get_omega()
        if omega < 0.000000000000001:
            return 0
        else:
            ## R = Q_mu/2 * [omega(1-H_2(E_mu/omega)) - f(E_mu)H_2(E_mu)]
            gain = self.get_gain()
            qber = self.get_qber()
            rawrate = omega * gain / 2
            lossfromerrorcorrection = gain * efficiency_cascade * binaryentropy(qber) / 2
            errorfrommisalignment = optimize.root_scalar(misalignment_criterion, args = (np.power(self.misalignment,2)* np.exp(1)), bracket=[0, 1000], method = 'bisect').root
            lossfromprivacyamplification = gain * omega * binaryentropy(qber / omega + errorfrommisalignment) / 2
            return max(0, rawrate - lossfromerrorcorrection - lossfromprivacyamplification)

    def get_rate_efficient(self):
        """
        Calculate the rate of the setup: from GLLP formula Gottesman et al 2004, Lo et al 2005b - here we use the
        efficient BB84 which changes the sifting to improve rates by a factor of up to 2 in asymptotic case
        :return: The rate for the efficient BB84 protocol - sifting removes a negligible amount of states: float
        """
        return 2 * self.get_rate()