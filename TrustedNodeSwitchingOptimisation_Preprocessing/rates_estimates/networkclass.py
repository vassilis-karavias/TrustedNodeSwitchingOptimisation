import numpy as np
import scipy as sp
import math
from rates_estimates.sources import QDsource, SPDCSource
from rates_estimates.detectors import Detectors
from rates_estimates.fibres import Fibre

class Network():

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
        if cold:
            self.source = QDsource(purity = purity, efficiency=efficiencysource, rate=rate)
        else:
            self.source = SPDCSource(mu=mu, rate=rate)
        self.fibre = Fibre(loss=fibreloss)
        self.detector = Detectors(efficiency=efficiencydetector, darkcountrate=darkcountrate, deadtime= deadtime)
        # get the efficiency of the fibre
        self.etaf = self.fibre.getfibreefficiency(length)
        self.length = length
        # define clocktime as in Gobby et al 2004
        self.clocktime = 3.5 * (10 ** -9)
        # get the probability of a photon being detected at Bob given Alice triggered: p_mu
        self.pmu = self.source.getpmu(detector=self.detector, fibre = self.fibre, length = self.length, clocktime = self.clocktime)
        # get the dark count probability in the clocktime: p_dc
        self.pdc = self.detector.getprobabilityofdarkcount(self.clocktime)
        # get the afterpulse probability: p_ap
        self.pap = self.detector.getafterpulse(source=self.source, fibre=self.fibre, length=self.length, clocktime= self.clocktime)
        self.misalignment = np.sin(misalignment * np.pi/180)
        super().__init__()


    def getvisibility(self):
        """
        Calculate the visibility of the setup: V = mu etaf etad / mu etaf etad + 2* perror
        :return: The visibitity: float
        """
        coefficient = self.source.getmu() * self.etaf * self.detector.getefficiency()
        prob_error = self.detector.getproberror(self.clocktime)
        return coefficient/(coefficient + 2 * prob_error)

    def set_mu(self, mu):
        """
          sets the mean photon number of the network to new length and recalculates parameters that depend on mu
        :param mu: new mean photon number of network: float
        """
        if isinstance(self.source, QDsource):
            print("QD source does not use mean photon number parameter")
        else:
            # get a new source
            self.source = SPDCSource(mu=mu, rate=self.source.getrate())
            # get the probability of a photon being detected at Bob given Alice triggered: p_mu
            self.pmu = self.source.getpmu(detector=self.detector, fibre=self.fibre, length=self.length,
                                          clocktime=self.clocktime)
            # get the afterpulse probability: p_ap
            self.pap = self.detector.getafterpulse(source=self.source, fibre=self.fibre, length=self.length,
                                                   clocktime=self.clocktime)

    def set_length(self, length):
        """
        sets the length of the network to new length and recalculates parameters that depend on the length
        :param length: new length of the fibre: float
        """
        self.length = length
        # get new pmu, pdc, pap and etaf with this new length
        self.pmu = self.source.getpmu(detector=self.detector, fibre=self.fibre, length=self.length,
                                      clocktime=self.clocktime)
        self.pdc = self.detector.getprobabilityofdarkcount(self.clocktime)
        self.pap = self.detector.getafterpulse(source=self.source, fibre=self.fibre, length=self.length,
                                               clocktime=self.clocktime)
        self.etaf = self.fibre.getfibreefficiency(length)

    def get_qber(self):
        """
        Calculates the QBER of the setup- method should be overridden in subclass
        :return: The QBER: float
        """
        pass

    def get_gain(self):
        """
        Calculates the gain of the setup, Q_(mu) - method should be overridden in subclass
        :return: The gain: float
        """
        pass

    def get_omega(self):
        """
        Calculates the fraction of single photon states there are - method should be overridden in subclass
        :return: The fraction of single photon states, omega: float
        """
        pass

    def get_rate(self):
        """
        Calculate the rate of the setup - method should be overridden in subclasss
        :return: rate of the setup R: float
        """
        pass