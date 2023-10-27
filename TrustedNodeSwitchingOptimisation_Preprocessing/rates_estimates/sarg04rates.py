import numpy as np
from scipy import optimize
import math
from rates_estimates.sources import QDsource, SPDCSource
from rates_estimates.detectors import Detectors
from rates_estimates.fibres import Fibre
from rates_estimates.entropyfunctions import binaryentropy, conditional_entropy, misalignment_criterion
from rates_estimates.networkclass import Network

efficiency_cascade = 1.2

class SARG04Network(Network):

    def __init__(self, purity, efficiencysource, rate, mu, cold, fibreloss, efficiencydetector, darkcountrate, deadtime, length, misalignment):
        """
               A class for a setup using the SARG04 protocol
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
        denom = self.pmu * (2-v)/2 + 2 * self.pdc + self.pap
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
        return 1-self.source.getmultiphotonprob()/self.get_gain()

    def get_omega_2(self):
        """
        Calculates the fraction of two photon states Bob recieves - removes the single photon contribution accounted
        for in omega
        :return: The fraction of two photon states: float
        """
        if isinstance(self.source, SPDCSource):
            # for SPDC source omega_2 = 1- (sum_n=3 ^ inf P(n) - P(1))/Q_mu
            return 1- (self.source.getmultiphotonprob() -self.source.getnphotonprobability(2) + self.source.getnphotonprobability(1)) / self.get_gain()
        else:
            # for QD source omega_2 = 1-P(1)/Q_mu
            return 1-self.source.getnphotonprobability(1)/ self.get_gain()

    def function_to_minimise(self, x):
        """
        Defines the function to be minimised to calculate the binary entropy information Eve has with the 2-photon
        states Fung et al 2006
        :param x: the variable for the function to be minimised: [float]
        :return: The result of the evaluation of the function: float
        """
        return x[0] * self.get_gain() + (3-2* x[0] + np.sqrt(6-6*np.sqrt(2) * x[0] + 4 * np.power(x[0],2)))/6

    def geterrors(self, qber):
        """
        Gets the values of ex, ey, ez for the system using formulas in Fung et al 2006
        :param qber: Quantum bit error rate
        :return: ex, ey, ez: floats
        """
        # returns ex, ey,ez for the worse case scenario
        return qber/2, qber/2, qber

    def get_rate(self):
        """
        Calculate the rate of the setup: formula from Fung et al 2006
        :return: The rate for the SARG04 protocol: float
        """
        # estimate contribution of single photons to Bob's output
        omega = max(self.get_omega(), 0)
        # estimate contribution of two photon states to Bob's output
        omega_2 = max(self.get_omega_2(), 0)
        # if no single or 2 photon states in output => no rate
        if omega + omega_2 < 0.000000000000001:
            return 0
        else:
            # calculate the rate as in the equation
            # R = Q_mu[-f(E_mu)H_2(E_mu) + omega[1-H_2(Z1|X1)] +omega_2[1-H_2(Z2|X2)]]
            gain = self.get_gain()
            qber = self.get_qber()
            rawrate = omega * gain / 2
            lossfromerrorcorrection = gain * efficiency_cascade * binaryentropy(qber) / 2
            ex, ey, ez = self.geterrors(qber)
            lossfromprivacyamplification = gain * omega * conditional_entropy(ex= ex, ey = ey, ez = ez) / 2
            rawrate2photon = omega_2 * gain / 2
            minx = optimize.minimize(self.function_to_minimise, [1])
            errorfrommisalignment = optimize.root_scalar(misalignment_criterion,
                                                         args=(np.power(self.misalignment, 2) * np.exp(1)), bracket=[0, 1000], method = 'bisect').root
            lossfromprivacyamplification2photon = gain * omega_2 * binaryentropy(self.function_to_minimise(minx.x) +
                                                                                 errorfrommisalignment) / 2
            return max(0, rawrate - lossfromerrorcorrection - lossfromprivacyamplification + rawrate2photon - lossfromprivacyamplification2photon)
