import numpy as np
import scipy as sp
import math
from scipy import optimize

class Source():

    def __init__(self, efficiency, rate):
        """
        General Base class for a Source
        :param efficiency: The efficiency of the source
        :param rate: The rate of triggering the source in Hz
        """
        self.efficiency = efficiency
        self.rate = rate
        super().__init__()

    def getefficiency(self):
        """
        :return: The efficiency of the source
        """
        return self.efficiency

    def getrate(self):
        """
        :return: the rate of the source (Hz)
        """
        return self.rate

    def getmultiphotonprob(self):
        """
        Method for getting the probability of obtaining multiphoton probability- should be overriden in subclasses
        :return: The multiphoton probability
        """
        # p_(multi)
        pass

    def getpmu(self, detector, fibre, length, clocktime):
        """
        Method for obtaining the probability of a detection at Bob's detector given Alice triggered a photon - should
        be overriden in subclasses
        :param detector: The detector for the setup: of type Detectors
        :param fibre: The fibre connecting Alice to Bob: of type Fibre
        :param length: The length of the fibre connecting Alice to Bob: float
        :param clocktime: The clocktime for the error probability: float
        :return: The probability of a detection at Bob's detector given Alice triggered: pmu
        """
        # p_(mu)
        pass



class QDsource(Source):

    def __init__(self, purity, efficiency, rate):
        """
        Class for the QD source- requires cold Alice and has different statistics to SPDC source
        :param purity: The purity of the source g^(2) (0): float
        :param efficiency: The efficiency of the source etas: float
        :param rate: The rate of triggering of the source in Hz
        """
        self.purity = purity
        super().__init__(efficiency, rate)

    def getpurity(self):
        """
        :return: The purity of the source g^(2) (0): float
        """
        return self.purity

    def getmu(self):
        """
        Mean photon number for a QD source is sum (P(n) n) where n is the number of photons in the state = p1 + 2p2
        :return: mean photon number, mu: float
        """
        p1, p2 = self.getphotonprob()
        return  p1 + 2* p2

    def equationsforprobabilities(self, inputs):
        """
        Function defining the constraints for the probabilities: p1 + p2 = etas, i.e. the efficiency = sum of
        probabilities of emitting at least one photon, g^(2) (0) = 2p2/(p1+2p2)^2 as derived in the summary
        :param inputs: The input parameters for p1,p2 in form [p1, p2]: array of float
        :return: The values for the equations rearranged to yield p1+p2-etas, 2p2/(p1+2p2)^2 - g^(2) (0)
        """
        p1, p2 = inputs
        return (p1 + p2 - self.efficiency, ((2* p2) / np.power((p1+2* p2), 2)) - self.purity)

    def getphotonprob(self):
        """
        obtains the fits of p1, p2 to the equations defined in equationsforprobabilities by using scipy.optimize.fsolve
        :return: The probabilities p1 and p2 fit to the parameters: float, float
        """
        # p(1), p(2)
        if (self.purity < 0.00000000001):
            # if the putrity is 0 then p(1) is the efficiency and there is no probability of generating 2 photon states
            return self.efficiency, 0
        else:
            p1, p2 = optimize.fsolve(self.equationsforprobabilities, (1, 0))
            return p1, p2

    def getmultiphotonprob(self):
        """
        :return: the probability of getting multiple photons, p2: float
        """
        p1, p2 = self.getphotonprob()
        return p2

    def getnphotonprobability(self, n):
        """
        gets the probability of obtaining n photons, n=0 1-p1-p2, n=1  p1, n=2: p2 else 0
        :param n: number of photons in state desired: int
        :return: probability of getting that state: float
        """
        p1, p2 = self.getphotonprob()
        if n == 0:
            return 1-p1 -p2
        elif n == 1:
            return p1
        elif n == 2:
            return p2
        else:
            return 0

    def getpmu(self, detector, fibre, length, clocktime):
        """
        Method for obtaining the probability of a detection at Bob's detector given Alice triggered a photon - should
        be overriden in subclasses
        :param detector: The detector for the setup: of type Detectors
        :param fibre: The fibre connecting Alice to Bob: of type Fibre
        :param length: The length of the fibre connecting Alice to Bob: float
        :param clocktime: The clocktime for the error probability: float
        :return: The probability of a detection at Bob's detector given Alice triggered: pmu
        """
        # p_(mu)
        p1, p2 = self.getphotonprob()
        etad = detector.getefficiency()
        etaf = fibre.getfibreefficiency(length)
        return p1 * ( etad * etaf + (1-etad* etaf) * detector.getprobabilityofdarkcount(clocktime)) +\
               p2 * ((1-np.power(1-etaf * etad ,2))  + np.power(1-etaf * etad ,2) * detector.getprobabilityofdarkcount(clocktime))


class SPDCSource(Source):

    def __init__(self, rate, mu):
        """
        Class for the SPDC source - Poissonian statistics for photon number
        :param rate: Rate of the laser trigger in Hz
        :param mu: Mean photon number: float
        """
        self.mu = mu
        efficiency = 1-np.exp(-mu)
        super().__init__(efficiency, rate)

    def getmu(self):
        """
        :return: Mean photon number: float
        """
        # mu
        return self.mu

    def getnphotonprobability(self, n):
        """
        gets the probability of obtaining n photons: p(n) = e^(-mu) mu^n / n!
        :param n: number of photons in the desired state
        :return: The probability of generating this state
        """
        # P_n(mu)
        return (np.exp(-self.mu) * np.power(self.mu , n))/ math.factorial(n)

    def getmultiphotonprob(self):
        """
        Method for getting the probability of obtaining multiphoton probability
        :return: The multiphoton probability
        """
        # p_(multi)
        return 1- self.getnphotonprobability(0) - self.getnphotonprobability(1)

    def getpmu(self, detector, fibre, length, clocktime):
        """
        Method for obtaining the probability of a detection at Bob's detector given Alice triggered a photon - should
        be overriden in subclasses
        :param detector: The detector for the setup: of type Detectors
        :param fibre: The fibre connecting Alice to Bob: of type Fibre
        :param length: The length of the fibre connecting Alice to Bob: float
        :param clocktime: The clocktime for the error probability: float
        :return: The probability of a detection at Bob's detector given Alice triggered: pmu
        """
        # p_(mu)
        pmu = 0
        probtransmission = 1- detector.getefficiency() * fibre.getfibreefficiency(length)
        for n in range(0,100):
            nprobabilityemission = self.getnphotonprobability(n)
            pmu += nprobabilityemission * ((1-np.power(probtransmission, n)) +
                                           np.power(probtransmission, n) * detector.getprobabilityofdarkcount(clocktime))
        return pmu
