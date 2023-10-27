import numpy as np
import scipy as sp
import math

class Detectors():

    def __init__(self, efficiency, darkcountrate, deadtime):
        """
        Class for a Detector
        :param efficiency: The efficiency of the detector: float
        :param darkcountrate: The dark count rate of the detector in Hz: float
        :param deadtime: The dead time of the detector in s: float
        """
        self.efficiency = efficiency
        self.darkcountrate = darkcountrate
        self.deadtime = deadtime
        super().__init__()

    def getefficiency(self):
        """
        :return: The efficiency of the detector: float
        """
        return self.efficiency

    def getdarkcountrate(self):
        """
        :return: The dark count rate of the detector: float
        """
        return self.darkcountrate

    def getdeadtime(self):
        """
        :return: The dead time of the detector: float
        """
        return self.deadtime

    def getprobabilityofdarkcount(self, clocktime):
        """
        Calculates the probability of a dark count occurring in clocktime : 1-e^(-darkcountrate * clocktime): assume in
        small interval that the dark count rate can be modelled with a poisson distribution:
        P(mu = darkcountrate * clocktime)
        :param clocktime: The time resolution of the detector: float
        :return: The probability of obtaining a dark count rate
        """
        # dark count rate give the mean number of events per second. mean number of events per unit clocktime is
        # darkcountrate * clocktime -> p(2) events is negligible in small clocktime - assumed here
        # and thus binomial p(1) = 1- e^(-darkcountrate * clocktime)
        return 1- np.exp(-self.darkcountrate * clocktime)

    def getproberror(self, clocktime):
        """
        Calculates the probability of an error in the clocktime: P_e moddeled here as component from Gobby et al 2004
        and dark count rates from modern devices
        :param clocktime: The time resolution of the detector: float
        :return: The probability of error: float
        """
        # probability of error P_e as given in notes takes the stray light contribution from Gobby et al and adds dark
        # count from new detectors
        return 5.3 * (10 ** -7) * clocktime / (3.5 * (10 ** -9))  + self.getprobabilityofdarkcount(clocktime)

    def getafterpulse(self, source, fibre, length, clocktime):
        """
        Calculates the probability of the detector afterpulse: modelled as 0.008(pmu + 2pDC) as Eraerds et al 2010
        :param source: The source of the setup: Source (one of the subclasses)
        :param fibre: The fibre of the setup: Fibre
        :param length: The length of the connection: float
        :param clocktime: The time resolution of the detector: float
        :return: The probability of an afterpulse in the clocktime: float
        """
        return 0.008 * (source.getpmu(self, fibre, length, clocktime) + 2 * self.getprobabilityofdarkcount(clocktime))
