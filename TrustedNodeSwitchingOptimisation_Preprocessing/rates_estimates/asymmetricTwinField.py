import numpy as np
from scipy import optimize
from scipy import special
import math
from rates_estimates.sources import QDsource, SPDCSource
from rates_estimates.detectors import Detectors
from rates_estimates.fibres import Fibre
from rates_estimates.entropyfunctions import binaryentropy, conditional_entropy, misalignment_criterion
from rates_estimates.networkclass import Network
# from platypus import NSGAII, Problem, Real
from rates_estimates.symmetricTwinField import SymmetricTwinField
import time

efficiency_cascade = 1.2

class AsymmetricTwinField(SymmetricTwinField):

    def __init__(self, purity, efficiencysource, rate, mu, cold, fibreloss, efficiencydetector, darkcountrate, deadtime,
                 length, misalignment, phasemisalignment, length_1):
        """
               A class for a setup using the Asymmetric Twin Field Rates - Wang et al (2020)
               :param purity: The purity of the source, needed for cold Alice: float
               :param efficiencysource: The efficiency of the source, needed for cold Alice: float
               :param rate: The rate of the source in Hz: float
               :param mu: The mean number of photons per pulse, needed for hot Alice: float
               :param cold: If we are using cold Alice (QD source) or hot Alice (SPDC source): boolean
               :param fibreloss: The loss of the fibre in dB/km: float
               :param efficiencydetector: The efficiency of the detector: float
               :param darkcountrate: The dark count rate of the detector in s^(-1): float
               :param deadtime: The dead time of the detector in s: float
               :param length: The length of the fibre between Alice and Bob in km: float
               :param misalignment: The misalignment of the devices in degrees: float
               :param phasemisalignment: The phase misalignment: float
               :param length_1: The distance between Alice and Charlie length - length_1 is the length of the Bob to
               Charlie : float
        """
        if length < length_1:
            print("Length of Alice to Charlie must be less than length from Alice to Bob")
            raise ValueError
        super().__init__(purity, efficiencysource, rate, mu, cold, fibreloss, efficiencydetector, darkcountrate,
                         deadtime, length / 2, misalignment, phasemisalignment)
        self.misalignment = misalignment * np.pi / 180
        self.phasemisalignment = phasemisalignment * np.pi / 180
        self.length = length
        self.papac = self.detector.getafterpulse(source=self.source, fibre=self.fibre, length=length_1, clocktime= self.clocktime)
        self.pmuac = self.source.getpmu(detector=self.detector, fibre = self.fibre, length = length_1, clocktime = self.clocktime)
        self.papbc = self.detector.getafterpulse(source=self.source, fibre=self.fibre, length=length - length_1, clocktime= self.clocktime)
        self.pmubc = self.source.getpmu(detector=self.detector, fibre = self.fibre, length = length - length_1, clocktime = self.clocktime)
        self.etafa = self.fibre.getfibreefficiency(length_1)
        self.etafb = self.fibre.getfibreefficiency(length - length_1)
        self.sqrtetaa = self.etafa * self.detector.getefficiency()
        self.sqrtetab = self.etafb * self.detector.getefficiency()

    def set_intensities(self):
        """
        Sets the intensities of the states to appropriate values to minimise phase error rates: etaa * sa = etab * sb
        Here we assume sqrt(sa * sb) = 0.1
        """
        precision = 0.00000000000000001
        # sa = 0.1 * np.sqrt(self.etafa/ (self.etafb+ precision))
        # sb = 0.1 * np.sqrt(self.etafb / (self.etafa + precision))
        if self.etafa > self.etafb:
            sb = 0.01
            sa = self.etafb * sb / self.etafa
        else:
            sa = 0.01
            sb = self.etafa * sa / self.etafb
        self.sa = sa
        self.sb = sb

    def set_length(self, length, length_1):
        """
        set the lengths between Alice and Bob and Alice and Charlie - recalculates all the parameters
        :param length: The length of the fibre between Alice and Bob in km: float
        :param length_1: The distance between Alice and Charlie length - length_1 is the length of the Bob to
               Charlie : float
        """
        self.papac = self.detector.getafterpulse(source=self.source, fibre=self.fibre, length=length_1,
                                                 clocktime=self.clocktime)
        self.pmuac = self.source.getpmu(detector=self.detector, fibre=self.fibre, length=length_1,
                                        clocktime=self.clocktime)
        self.papbc = self.detector.getafterpulse(source=self.source, fibre=self.fibre, length=length - length_1,
                                                 clocktime=self.clocktime)
        self.pmubc = self.source.getpmu(detector=self.detector, fibre=self.fibre, length=length - length_1,
                                        clocktime=self.clocktime)
        self.etafa = self.fibre.getfibreefficiency(length_1)
        self.etafb = self.fibre.getfibreefficiency(length - length_1)
        self.length = length
        self.sqrtetaa = self.etafa * self.detector.getefficiency()
        self.sqrtetab = self.etafb * self.detector.getefficiency()
        self.set_intensities()

    def getpxxtotal(self, kc, kd):
        """
        Calculate the total bit probability pxx(kc,kd)
        :param kc: The bit for whether detector C triggered or not: int
        :param kd: The bit for whether detector D triggered or not: int
        :return: pxx(kc,kd)
        """
        gammaa = self.sa * self.etafa
        gammab = self.sb * self.etafb
        omega = np.cos(self.misalignment) * np.cos(self.phasemisalignment)
        return (1-self.pdc) * np.exp(-1/2 * (gammaa + gammab)) * (np.exp(-np.sqrt(gammaa * gammab) * omega) +
                    np.exp(np.sqrt(gammaa * gammab) * omega)) / 2 - np.power(1-self.pdc, 2) * np.exp(-(gammaa + gammab))

    def getexx(self, kc, kd):
        """
        Get the bit error rate for the case kc, kd: eX,kckd - Wang et al (2020)
        :param kc: The bit for whether detector C triggered or not: int
        :param kd: The bit for whether detector D triggered or not: int
        :return: eX,kckd
        """
        gammaa = self.sa * self.etafa
        gammab = self.sb * self.etafb
        omega = np.cos(self.misalignment) * np.cos(self.phasemisalignment)
        denom = np.exp(-np.sqrt(gammaa * gammab) * omega) + np.exp(np.sqrt(gammaa * gammab) * omega) \
                - 2 * (1-self.pdc) * np.exp(-1/2 * (gammaa + gammab))
        return (np.exp(-np.sqrt(gammaa * gammab) * omega) - (1-self.pdc) * np.exp(-1/2 * (gammaa + gammab))) / denom

    def coeff_tosumqzz(self, q, m, p, l, k, na, nb):
        """
        The calculation of the exact pzz(kc,kd|na,nb) requires qzz(kc,kd|na,nb): this has multiple sums:
        This function calculates the inner coefficient of the sum - Wang et al (2020)
        The parameters are just parameters that get summed over
        :param na: The number of photons in the state sent by Alice: int
        :param nb: The number of photons in the state sent by Bob: int
        :return: The coefficient in the inner sum
        """
        # assuming theta_a = theta_B = misalignment/2
        theta = self.misalignment /2
        return special.binom(k, q) * special.binom(l, m + p - q) * math.factorial(m+p) * math.factorial(k+l-m-p) *\
                np.power(np.cos(theta), m + q) * np.power(np.cos(theta), m + p- q) * np.power(np.sin(theta), 2*k-m-q)\
                * np.power(np.sin(theta), 2 * l - m - 2 * p + q) - np.power((1-self.sqrtetaa), na) * np.power((1-self.sqrtetab), nb)

    def gettermforfourthsum(self, l, k, na, nb):
        """
        The calculation of the exact pzz(kc,kd|na,nb) requires qzz(kc,kd|na,nb): this has multiple sums:
        This function calculates the coefficient after the third sum - Wang et al (2020)
        The parameters are just parameters that get summed over
        :param na: The number of photons in the state sent by Alice: int
        :param nb: The number of photons in the state sent by Bob: int
        :return: The coefficient after the third sum
        """
        coeff = special.binom(nb, l) * np.power(self.sqrtetaa, k) * np.power(self.sqrtetab, l) * \
                np.power(1 - self.sqrtetaa, na - k) * np.power(1-self.sqrtetab, nb - l) / (
                            np.power(2, k + l) * math.factorial(k) * math.factorial(l))
        sum_term = 0
        for m in range(0, k):
            sum_term += self.gettermforthirdsum(m=m, l=l, k=k, na=na, nb=nb)
        return coeff * sum_term

    def getpzzforinfinitedecoycase(self, kc, kd, na, nb):
        """
        Gets the analytical value for pzz(kc,kd|na,nb) for the case of infinite decoy states used - Wang et al (2020)
        :param kc: The bit for whether detector C triggered or not: int
        :param kd: The bit for whether detector D triggered or not: int
        :param na: The number of photons in the state sent by Alice: int
        :param nb: The number of photons in the state sent by Bob: int
        :return: pzz(kc,kd|na,nb) for infinite decoy states
        """
        qzz = self.getqzzforinfinitedecoycase(kc =kc, kd = kd, na = na, nb = nb)
        coeff = 1- self.pdc
        return coeff * (qzz + self.pdc * np.power(1-self.sqrtetaa, na) * np.power(1-self.sqrtetab, nb))

    def getcatcoefficientsna(self, n, j):
        """
        Get the cat coefficients for the states Alice sent for n photons using the intensity of photons sent by Alice (sa)
        :param n: The number of photons in the state: int
        :param j: The cat coefficient state (either 0 or 1): int
        :return: c^(A,j)(n): float
        """
        if (n + j) %2 == 1:
            return 0
        else:
            return np.exp(-self.sa/2) * np.power(np.power(self.sa, 1/2), n) / np.sqrt(math.factorial(n))

    def getcatcoefficientsnb(self, n, j):
        """
        Get the cat coefficients for the states Bob sent for n photons using the intensity of photons sent by Bob (sb)
        :param n: The number of photons in the state: int
        :param j: The cat coefficient state (either 0 or 1): int
        :return: c^(B,j)(n): float
        """
        if (n + j) % 2 == 1:
            return 0
        else:
            return np.exp(-self.sb / 2) * np.power(np.power(self.sb, 1 / 2), n) / np.sqrt(math.factorial(n))

    def getphaseerrorforinfinitedecoycase(self, kc, kd):
        """
        Calculate the phase error for infinite decoy states case: no need to optimise as there is an analytic solution
        :param kc: The bit for whether detector C triggered or not: int
        :param kd: The bit for whether detector D triggered or not: int
        :return: eZ,kckd for infinite decoy states
        """
        # Get the total pxx
        pxxtotal = self.getpxxtotal(kc=kc, kd=kd)
        # keeps track of inner sums
        totalinnersumforeventerms = 0.0
        totalinnersumforoddterms = 0.0
        for na in range(0, 2):
            for nb in range(0, 2):
                # only estimate states for 5 states
                # get the cat states for 2na, 2nb, 2na+1,2nb+1
                # j = 0 term
                c2na = self.getcatcoefficientsna(2 * na, 0)
                c2nb = self.getcatcoefficientsnb(2 * nb, 0)
                # j = 1 term
                c2naplus1 = self.getcatcoefficientsna(2 * na + 1, 1)
                c2nbplus1 = self.getcatcoefficientsnb(2 * nb + 1, 1)
                # j = 0 pzz(kc,kd|2ma,2mb)
                pzzkckd2na2nb = self.getpzzforinfinitedecoycase(kc=kc, kd=kd, na=2 * na, nb=2 * nb)
                # j = 1 pzz(kc,kd|2ma+1,2mb+1)
                pzzkckd2naplus12nbplus1 = self.getpzzforinfinitedecoycase(kc=kc, kd=kd, na=2 * na + 1, nb=2 * nb + 1)
                # if negative then set to 0
                if pzzkckd2na2nb < 0:
                    pzzkckd2na2nb = 0
                if pzzkckd2naplus12nbplus1 < 0:
                    pzzkckd2naplus12nbplus1 = 0
                # j = 0 sum tracker
                totalinnersumforeventerms += c2na * c2nb * np.sqrt(pzzkckd2na2nb)
                # j = 1 sum tracker
                totalinnersumforoddterms += c2naplus1 * c2nbplus1 * np.sqrt(pzzkckd2naplus12nbplus1)

        # add the rest of the terms where we do not estimate pzz(kc,kd|2ma+j,2mB+j) -> just set to =1 for large n as
        # cat states virtually 0
        for na in range(0, 10):
            for nb in range(0, 10):
                # already calculated these terms above need cases where one of na and nb is larger than this
                if na<  3 and nb < 3:
                    continue
                else:
                    # j =0 additional term is c_(2na) * c_(2nb)
                    c2na = self.getcatcoefficientsna(2 * na, 0)
                    c2nb = self.getcatcoefficientsnb(2 * nb, 0)
                    # j = 1 additional term is c_(2na+1) * c_(2nb+1)
                    c2naplus1 = self.getcatcoefficientsna(2 * na + 1, 1)
                    c2nbplus1 = self.getcatcoefficientsnb(2 * nb + 1, 1)
                    totalinnersumforeventerms += c2na * c2nb
                    totalinnersumforoddterms += c2naplus1 * c2nbplus1
        return (np.power(totalinnersumforeventerms, 2) + np.power(totalinnersumforoddterms, 2)) / pxxtotal

    def get_rate_infinite_decoys(self):
        """
        Obtain the secure key rate for the setup with infinite decoy states: NOTE: this assumes infinite decoy states
        :return: rate of the setup R: float
        """
        self.set_intensities()
        pxxtotal10 = self.getpxxtotal(kc=1, kd=0)
        biterror10 = self.getbiterrorrate(kc=1, kd=0)
        # t_4 = time.time()
        phaseerror10 = self.getphaseerrorforinfinitedecoycase(kc=1, kd=0)
        # t_5 = time.time()
        # print("time to get phase error: " + str(t_5 - t_4))
        rlow10 = pxxtotal10 * (1 - binaryentropy(biterror10) - efficiency_cascade * binaryentropy(phaseerror10))
        pxxtotal01 = self.getpxxtotal(kc=0, kd=1)
        biterror01 = self.getbiterrorrate(kc=0, kd=1)
        phaseerror01 = self.getphaseerrorforinfinitedecoycase(kc=0, kd=1)
        rlow01 = pxxtotal01 * (1 - binaryentropy(biterror01) - efficiency_cascade * binaryentropy(phaseerror01))
        return max(rlow10, 0) + max(rlow01, 0)