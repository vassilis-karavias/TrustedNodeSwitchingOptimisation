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

efficiency_cascade = 1.2

class SymmetricTwinField(Network):

    def __init__(self, purity, efficiencysource, rate, mu, cold, fibreloss, efficiencydetector, darkcountrate, deadtime,
                 length, misalignment, phasemisalignment):
        """
               A class for a setup using the Symmetric Twin Field Rates - Curty et al (2019)
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
               :param phasemisalignment: The phase misalignment: float
        """
        super().__init__(purity, efficiencysource, rate, mu, cold, fibreloss, efficiencydetector, darkcountrate,
                         deadtime, length / 2, misalignment)
        self.misalignment = misalignment * np.pi / 180
        self.phasemisalignment = phasemisalignment * np.pi / 180
        self.length = length
        self.sqrteta = self.etaf * self.detector.getefficiency()

    def set_length(self, length):
        """
        sets the length of the network to new length and recalculates parameters that depend on the length - needs to
        add calculation for sqrteta
        :param length: new length of the fibre: float
        """
        super().set_length(length / 2)
        self.sqrteta = self.etaf * self.detector.getefficiency()

    def getf(self, sign):
        """
        gets the function f +- (theta, phi, sqrtetaalpha^2) for a given sign as in Curty et al (2019)
        :param sign: The sign of the parameter == 1 for -1: == 0 for +1 : int
        :return: f+-: float
        """
        omega = np.cos(self.misalignment) * np.cos (self.phasemisalignment)
        return np.exp(- self.sqrteta * np.power(self.source.getmu(), 2) * (1 + np.power(-1,sign) * omega)) - \
               np.exp(-2 * self.sqrteta * np.power(self.source.getmu(), 2))

    def getqxx(self, kc, kd, ba, bb):
        """
        Calculates qxx for the fibre model given by Curty et al (2019)  - depends on kc, kd, ba, bb
        :param kc: The bit for whether detector C triggered or not: int
        :param kd: The bit for whether detector D triggered or not: int
        :param ba: The bit chosen by Alice to send to Charlie: int
        :param bb: The bit chosen by Bob to send to Charlie: int
        :return: The value qxx(kc,kd|ba,bb): float
        """
        if ((kc + ba + bb) % 2 == 1):
            return self.getf(sign= 1)
        else:
            return self.getf(sign = 0)

    def getpxx(self, kc, kd, ba, bb):
        """
        Calculates pxx for the fibre model given by Curty et al (2019)  - depends on kc, kd, ba, bb
        :param kc: The bit for whether detector C triggered or not: int
        :param kd: The bit for whether detector D triggered or not: int
        :param ba: The bit chosen by Alice to send to Charlie: int
        :param bb: The bit chosen by Bob to send to Charlie: int
        :return: pxx(kc,kd|ba,bb) : float
        """
        coeff = 1- self.pdc
        qxx = self.getqxx(kc, kd, ba, bb)
        return coeff * (self.pdc * np.exp(-2 * self.sqrteta * np.power(self.source.getmu(), 2)) + qxx)

    def getqzz(self, kc, kd, betaa, betab):
        """
        Calculates qzz for the fibre model given by Curty et al (2019)  - depends on kc, kd, betaa, betab
        :param kc: The bit for whether detector C triggered or not: int
        :param kd: The bit for whether detector D triggered or not: int
        :param betaa: The intensity of the decoy state sent by Alice: float
        :param betab: The intensity of the decoy state sent by Bob: float
        :return: qzz(kc,kd|betaa, betab)
        """
        besselterm = special.iv(0, betaa * betab * self.sqrteta * np.cos(self.misalignment))
        power = -(np.power(betaa, 2) + np.power(betab, 2)) * self.sqrteta
        return np.exp(power/2) * besselterm - np.exp(power)

    def getpzz(self, kc, kd, betaa, betab):
        """
        Calculates pzz for the fibre model given by Curty et al (2019)  - depends on kc, kd, betaa, betab
        :param kc: The bit for whether detector C triggered or not: int
        :param kd: The bit for whether detector D triggered or not: int
        :param betaa: The intensity of the decoy state sent by Alice: float
        :param betab: The intensity of the decoy state sent by Bob: float
        :return: pzz(kc,kd|betaa, betab)
        """
        qzz = self.getqzz(kc, kd, betaa, betab)
        coeff = 1-self.pdc
        return coeff *(self.pdc * np.exp(- (np.power(betaa, 2)+ np.power(betab, 2)) * self.sqrteta) + qzz)

    def getphotonprobabilities(self, mu, n):
        """
        Gets the n photon probability for a poissonian photon source with a mean photon number mu
        :param mu: mean photon number of pulse: float
        :param n: phtoton number of desired state: int
        :return: The probability of the n photon state: float
        """
        return np.exp(-mu) * np.power(mu, n) / math.factorial(n)


    # def optimise_constraints(self, pzzarray):
    #     """
    #     Optimisation problem for the kc=1, kd =0 case to get pzz(kc=1,kd=0|2ma+1,2mb+1)
    #     :param pzzarray: Guess for  pzz(kc=1,kd=0|na,nb)
    #     :return: the objective functionspzz(kc=1,kd=0|2ma+1,2mb+1) and the constraints
    #     """
    #     # generate the set of decoys used to estimate the probabilities
    #     betaa = [0.0, 0.01, 0.05, 0.1, 0.15]
    #     betab = [0.0, 0.01, 0.05, 0.1, 0.15]
    #     pzzbeta_10array = []
    #     # pzzbeta_01array = []
    #     # for each of the decoys get pzz(kc = 1, kd = 0|betaa, betab): this will be needed in the constraints
    #     for i in range(len(betaa)):
    #         for j in range(len(betab)):
    #             pzzbeta_10 = self.getpzz(kc = 1, kd = 0, betaa=betaa[i], betab=betab[j])
    #             # pzzbeta_01 = self.getpzz(kc = 0, kd = 1, betaa=betaa[i], betab=betab[i])
    #             pzzbeta_10array.append(pzzbeta_10)
    #             # pzzbeta_01.append((pzzbeta_01, betaa[i], betab[i]))
    #     # keep track of the constraints
    #     constraint_1 = []
    #     constraint_2 = []
    #     objective_functions = []
    #     for k in range(len(betaa)):
    #         for j in range(len(betab)):
    #             righthandofconstraintone = 0
    #             righthandofconstrainttwo = 0
    #             for i in range(len(pzzarray)):
    #                 # we only estimate the probabilities for n >6: so we need to get the correct values for n_A and n_B
    #                 # for each element in the array: as we can only use 1D arrays
    #                 n_B = i % 6
    #                 n_A = int(np.floor(i/6))
    #                 # only add this constraint once: if both nA and n_B are odd then they are part of the objective
    #                 # function to be maximised
    #                 if (n_A + n_B) % 2 == 0 and k == 0 and j == 0:
    #                     objective_functions.append(pzzarray[i])
    #                 # get photon probabilities for n_A, n_B - needed in constraints
    #                 pna = self.getphotonprobabilities(mu = betaa[k], n = n_A)
    #                 pnb = self.getphotonprobabilities(mu = betab[j], n = n_B)
    #                 # keep track of the sums in the constraints
    #                 righthandofconstraintone += pzzarray[i] * pna * pnb
    #                 righthandofconstrainttwo += (1-pzzarray[i]) * pna * pnb
    #             # add all the constraints for the set of data
    #             constraint_1.append(pzzbeta_10array[k*len(betab)+j] - righthandofconstraintone)
    #             constraint_2.append(1-pzzbeta_10array[k*len(betab)+j] - righthandofconstrainttwo)
    #     constraints = np.concatenate((constraint_1, constraint_2))
    #     return objective_functions, constraints
    #
    # def optimise_10(self):
    #     """
    #     Carries out the optimisation of the kc = 1, kd = 0 case to get pzz(kc=1, kd = 0|2m_A+j, 2m_B+j)
    #     :return: Optimisation result
    #     """
    #     problem = Problem(36, 18, 50)
    #     # The final constraint: pzz(kc,kd|na,nb) is in range [0,1]
    #     problem.types[:] = Real(0, 1)
    #     problem.constraints[:] = ">=0"
    #     problem.directions[:] = problem.MAXIMIZE
    #     problem.function = self.optimise_constraints
    #     algorithm = NSGAII(problem)
    #     algorithm.run(10000)
    #     return algorithm.result
    #
    # def optimise_constraints_01(self, pzzarray):
    #     """
    #     Optimisation problem for the kc=0, kd=1 case to get pzz(kc=0,kd=1|2ma+1,2mb+1)
    #     :param pzzarray: Guess for  pzz(kc=0,kd=1|na,nb)
    #     :return: the objective functionspzz(kc=0,kd=1|2ma+1,2mb+1) and the constraints
    #     """
    #     # generate the set of decoys used to estimate the probabilities
    #     betaa = [0.0, 0.01, 0.05, 0.1, 0.15]
    #     betab = [0.0, 0.01, 0.05, 0.1, 0.15]
    #     pzzbeta_10array = []
    #     # pzzbeta_01array = []
    #     # for each of the decoys get pzz(kc = 0, kd = 1|betaa, betab): this will be needed in the constraints
    #     for i in range(len(betaa)):
    #         for j in range(len(betab)):
    #             pzzbeta_10 = self.getpzz(kc = 0, kd = 1, betaa=betaa[i], betab=betab[j])
    #             # pzzbeta_01 = self.getpzz(kc = 0, kd = 1, betaa=betaa[i], betab=betab[i])
    #             pzzbeta_10array.append(pzzbeta_10)
    #     # keep track of the constraints
    #     constraint_1 = []
    #     constraint_2 = []
    #     objective_functions = []
    #     for k in range(len(betaa)):
    #         for j in range(len(betab)):
    #             righthandofconstraintone = 0
    #             righthandofconstrainttwo = 0
    #             for i in range(len(pzzarray)):
    #                 # we only estimate the probabilities for n >6: so we need to get the correct values for n_A and n_B
    #                 # for each element in the array: as we can only use 1D arrays
    #                 n_B = i % 6
    #                 n_A = int(np.floor(i/6))
    #                 # only add this constraint once: if both nA and n_B are odd then they are part of the objective
    #                 # function to be maximised
    #                 if (n_A + n_B) % 2 == 0 and k == 0 and j == 0:
    #                     objective_functions.append(pzzarray[i])
    #                 # get photon probabilities for n_A, n_B - needed in constraints
    #                 pna = self.getphotonprobabilities(mu = betaa[k], n = n_A)
    #                 pnb = self.getphotonprobabilities(mu = betab[j], n = n_B)
    #                 # keep track of the sums in the constraints
    #                 righthandofconstraintone += pzzarray[i] * pna * pnb
    #                 righthandofconstrainttwo += (1-pzzarray[i]) * pna * pnb
    #             # add all the constraints for the set of data
    #             constraint_1.append(pzzbeta_10array[k*len(betab)+j] - righthandofconstraintone)
    #             constraint_2.append(1-pzzbeta_10array[k*len(betab)+j] - righthandofconstrainttwo)
    #     constraints = np.concatenate((constraint_1, constraint_2))
    #     return objective_functions, constraints
    #
    # def optimise_01(self):
    #     """
    #     Carries out the optimisation of the kc = 0, kd = 1 case to get pzz(kc=0, kd = 1|2m_A+j, 2m_B+j)
    #     :return: Optimisation result
    #     """
    #     problem = Problem(36, 18, 50)
    #     # The final constraint: pzz(kc,kd|na,nb) is in range [0,1]
    #     problem.types[:] = Real(0, 1)
    #     problem.directions[:] = problem.MAXIMIZE
    #     problem.constraints[:] = ">=0"
    #     problem.function = self.optimise_constraints_01
    #     algorithm = NSGAII(problem)
    #     algorithm.run(10000)
    #     return algorithm.result

    def getpxxbabbkckd(self, ba, bb, kc, kd):
        """
        Get the probability pxx(ba, bb|kc, kd) from Bayes theorem - Curty et al (2019)
        :param ba: The bit chosen by Alice to send to Charlie: int
        :param bb: The bit chosen by Bob to send to Charlie: int
        :param kc: The bit for whether detector C triggered or not: int
        :param kd: The bit for whether detector D triggered or not: int
        :return: pxx(ba,bb|kc, kd)
        """
        pxxkckdbabb = self.getpxx(kc = kc, kd = kd, ba = ba, bb = bb)
        pxxtotal = self.getpxxtotal(kc = kc, kd = kd)
        return pxxkckdbabb/ (4 * pxxtotal)

    def getbiterrorrate(self, kc, kd):
        """
        Get the bit error rate for the case kc, kd: eX,kckd - Curty et al (2019)
        :param kc: The bit for whether detector C triggered or not: int
        :param kd: The bit for whether detector D triggered or not: int
        :return: eX,kckd
        """
        return self.getpxxbabbkckd(kc=kc, kd=kd, ba=0 + kc % 2,bb = 0) + self.getpxxbabbkckd(kc=kc, kd=kd, ba= 1 + kc % 2,bb = 1)

    def getpxxtotal(self, kc, kd):
        """
        Calculate the total bit probability pxx(kc,kd)
        :param kc: The bit for whether detector C triggered or not: int
        :param kd: The bit for whether detector D triggered or not: int
        :return: pxx(kc,kd)
        """
        pxx = 0
        for ba in [0,1]:
            for bb in [0,1]:
               pxx += self.getpxx(kc = kc, kd = kd, ba = ba, bb = bb)
        return pxx / 4

    def getcatcoefficients(self, n, j):
        """
        Calculate the cat state coefficients c^(j)(n) for photon number n: note if n+j %2 == 1 the cat states are 0
        otherwise they are e^(-mu^2/2) mu^n/n!
        :param n: The number of photons in the state: int
        :param j: The cat coefficient state (either 0 or 1): int
        :return: c^(j)(n): float
        """
        if (n + j) %2 == 1:
            return 0
        else:
            return np.exp(-np.power(self.source.getmu(),2) /2) * np.power(self.source.getmu(), n) / np.sqrt(math.factorial(n))


    def coeff_tosumqzz(self, q, m, p, l, k, na, nb):
        """
        The calculation of the exact pzz(kc,kd|na,nb) requires qzz(kc,kd|na,nb): this has multiple sums:
        This function calculates the inner coefficient of the sum - Curty et al
        The parameters are just parameters that get summed over
        :param na: The number of photons in the state sent by Alice: int
        :param nb: The number of photons in the state sent by Bob: int
        :return: The coefficient in the inner sum
        """
        # assuming theta_a = theta_B = misalignment/2
        theta = self.misalignment /2
        return special.binom(k, q) * special.binom(l, m + p - q) * math.factorial(m+p) * math.factorial(k+l-m-p) *\
                np.power(np.cos(theta), m + q) * np.power(np.cos(theta), m + 2*p- q) * np.power(np.sin(theta), 2*k-m-q)\
                * np.power(np.sin(theta), 2 * l - m - 2 * p + q) - np.power((1-self.sqrteta), na + nb)

    def gettermforsecondsum(self, m , p, l, k, na, nb):
        """
         The calculation of the exact pzz(kc,kd|na,nb) requires qzz(kc,kd|na,nb): this has multiple sums:
        This function calculates the coefficient after the first sum - Curty et al
        The parameters are just parameters that get summed over
        :param na: The number of photons in the state sent by Alice: int
        :param nb: The number of photons in the state sent by Bob: int
        :return: The coefficient after the first sum
        """
        coeff = special.binom(l, p)
        sum_term = 0
        for q in range(max(0, m+p -l), min(k, m + p)):
            sum_term += self.coeff_tosumqzz(q = q, m = m, p = p, l = l, k = k, na = na, nb = nb)
        return coeff * sum_term

    def gettermforthirdsum(self, m, l, k, na, nb):
        """
        The calculation of the exact pzz(kc,kd|na,nb) requires qzz(kc,kd|na,nb): this has multiple sums:
        This function calculates the coefficient after the second sum - Curty et al
        The parameters are just parameters that get summed over
        :param na: The number of photons in the state sent by Alice: int
        :param nb: The number of photons in the state sent by Bob: int
        :return: The coefficient after the second sum
        """
        coeff = special.binom(k, m)
        sum_term = 0
        for p in range(0, l):
            sum_term += self.gettermforsecondsum(m=m, p=p, l =l, k=k, na = na, nb =nb)
        return coeff * sum_term

    def gettermforfourthsum(self, l, k, na, nb):
        """
        The calculation of the exact pzz(kc,kd|na,nb) requires qzz(kc,kd|na,nb): this has multiple sums:
        This function calculates the coefficient after the third sum - Curty et al
        The parameters are just parameters that get summed over
        :param na: The number of photons in the state sent by Alice: int
        :param nb: The number of photons in the state sent by Bob: int
        :return: The coefficient after the third sum
        """
        coeff = special.binom(nb, l) * np.power(self.sqrteta, k +l) *\
                np.power(1-self.sqrteta, na+nb - k - l) / (np.power(2, k + l) * math.factorial(k) * math.factorial(l))
        sum_term = 0
        for m in range(0, k):
            sum_term += self.gettermforthirdsum(m= m, l = l, k =k, na= na, nb = nb)
        return coeff * sum_term

    def gettermforfinalsum(self, k, na, nb):
        """
        The calculation of the exact pzz(kc,kd|na,nb) requires qzz(kc,kd|na,nb): this has multiple sums:
        This function calculates the coefficient after the fourth sum - Curty et al (2019)
        The parameters are just parameters that get summed over
        :param na: The number of photons in the state sent by Alice: int
        :param nb: The number of photons in the state sent by Bob: int
        :return: The coefficient after the fourth sum
        """
        coeff = special.binom(na, k)
        sum_term = 0
        for l in range(0,nb):
            sum_term += self.gettermforfourthsum(l = l, k =k, na = na, nb = nb)
        return coeff * sum_term

    def getqzzforinfinitedecoycase(self, kc, kd, na, nb):
        """
        Gets the analytical value for qzz(kc,kd|na,nb) for the case of infinite decoy states used - Curty et al (2019)
        :param kc: The bit for whether detector C triggered or not: int
        :param kd: The bit for whether detector D triggered or not: int
        :param na: The number of photons in the state sent by Alice: int
        :param nb: The number of photons in the state sent by Bob: int
        :return: qzz(kc,kd|na,nb) for infinite decoy states
        """
        qzz = 0
        for k in range(0, na):
            qzz += self.gettermforfinalsum(k = k, na = na, nb =nb)
        return qzz

    def getpzzforinfinitedecoycase(self, kc, kd, na, nb):
        """
        Gets the analytical value for pzz(kc,kd|na,nb) for the case of infinite decoy states used - Curty et al (2019)
        :param kc: The bit for whether detector C triggered or not: int
        :param kd: The bit for whether detector D triggered or not: int
        :param na: The number of photons in the state sent by Alice: int
        :param nb: The number of photons in the state sent by Bob: int
        :return: pzz(kc,kd|na,nb) for infinite decoy states
        """
        coeff = 1-self.pdc
        return coeff *(self.pdc * np.power(1-self.sqrteta, na +nb) + self.getqzzforinfinitedecoycase(kc= kc, kd = kd, na = na, nb = nb))

    def getphaseerrorforinfinitedecoycase(self, kc, kd):
        """
        Calculate the phase error for infinite decoy states case: no need to optimise as there is an analytic solution
        :param kc: The bit for whether detector C triggered or not: int
        :param kd: The bit for whether detector D triggered or not: int
        :return: eZ,kckd for infinite decoy states
        """
        # get pxx(kc,kd)
        pxxtotal = self.getpxxtotal(kc= kc, kd = kd)
        # keeps track of inner sums
        totalinnersumforeventerms = 0.0
        totalinnersumforoddterms = 0.0
        for na in range(0,5):
            for nb in range(0,5):
                # only estimate states for 5 states
                # get the cat states for 2na, 2nb, 2na+1,2nb+1
                # j = 0 term
                c2na = self.getcatcoefficients(2 * na, 0)
                c2nb = self.getcatcoefficients(2 * nb, 0)
                # j = 1 term
                c2naplus1 = self.getcatcoefficients(2 * na + 1, 1)
                c2nbplus1 = self.getcatcoefficients(2 * nb + 1, 1)
                # j = 0 pzz(kc,kd|2ma,2mb)
                pzzkckd2na2nb = self.getpzzforinfinitedecoycase(kc =kc, kd = kd, na = 2 * na, nb = 2 * nb)
                # j = 1 pzz(kc,kd|2ma+1,2mb+1)
                pzzkckd2naplus12nbplus1 = self.getpzzforinfinitedecoycase(kc =kc, kd = kd, na = 2 * na + 1, nb = 2 * nb + 1)
                # if negative then set to 0
                if pzzkckd2na2nb < 0:
                    pzzkckd2na2nb = 0
                if pzzkckd2naplus12nbplus1 <0:
                    pzzkckd2naplus12nbplus1 = 0
                # j = 0 sum tracker
                totalinnersumforeventerms += c2na * c2nb * np.sqrt(pzzkckd2na2nb)
                # j = 1 sum tracker
                totalinnersumforoddterms += c2naplus1 * c2nbplus1 * np.sqrt(pzzkckd2naplus12nbplus1)
        # add the rest of the terms where we do not estimate pzz(kc,kd|2ma+j,2mB+j) -> just set to =1 for large n as
        # cat states virtually 0
        for na in range(0, 10):
            for nb in range(0,10):
                # already calculated these terms above need cases where one of na and nb is larger than this
                if na <6 and nb < 6:
                    continue
                else:
                    # j =0 additional term is c_(2na) * c_(2nb)
                    c2na = self.getcatcoefficients(2 * na, 0)
                    c2nb = self.getcatcoefficients(2 * nb, 0)
                    # j = 1 additional term is c_(2na+1) * c_(2nb+1)
                    c2naplus1 = self.getcatcoefficients(2 * na + 1, 1)
                    c2nbplus1 = self.getcatcoefficients(2 * nb + 1, 1)
                    totalinnersumforeventerms += c2na * c2nb
                    totalinnersumforoddterms += c2naplus1 * c2nbplus1
        return (np.power(totalinnersumforeventerms,2) + np.power(totalinnersumforoddterms,2)) / pxxtotal

    # def getphaseerror(self, kc, kd):
    #     """
    #     Calculate the phase error for finite decoy states case
    #     :param kc: The bit for whether detector C triggered or not: int
    #     :param kd: The bit for whether detector D triggered or not: int
    #     :return: eZ,kckd for finite decoy states
    #     """
    #     # get pxx(kc,kd)
    #     pxxtotal = self.getpxxtotal(kc = kc, kd = kd)
    #     # for kc = 1, kd = 0 need to carry out the appropriate optimisation to estimate pzz(kc,kd|2mA+j,2mB+j)
    #     if kc == 1 and kd == 0:
    #         solutions = self.optimise_10()
    #         pzzs = solutions[-1].objectives
    #     # for kc = 0, kd = 1 need to carry out the appropriate optimisation to estimate pzz(kc,kd|2mA+j,2mB+j)
    #     elif kc == 0 and kd == 1:
    #         solutions = self.optimise_01()
    #         pzzs = solutions[-1].objectives
    #     # keeps track of the sum for j=0, j=1 inside the bracket
    #     totalsumodd = 0
    #     totalsumeven = 0
    #     for i in range(len(pzzs)):
    #         # we only estimate the probabilities for n >6: so we need to get the correct values for n_A and n_B
    #         # for each element in the array: as we can only use 1D arrays
    #         m_B = i % 6
    #         m_A = int(np.floor(i / 6))
    #         # coefficient are 0 therefor no need to add to the term
    #         if (m_A + m_B) % 2 == 1:
    #             continue
    #         elif (m_A %2 == 1):
    #             # both m_A and m_B are odd so j = 1: add term to the odd sum
    #             c_A =self.getcatcoefficients(n = m_A, j =1)
    #             c_B =self.getcatcoefficients(n = m_B, j =1)
    #             totalsumodd += c_A * c_B * np.sqrt(pzzs[i])
    #         else:
    #             # both m_A and m_B are even so j = 0: add term to the even sum
    #             c_A =self.getcatcoefficients(n = m_A, j =0)
    #             c_B =self.getcatcoefficients(n = m_B, j =0)
    #             totalsumeven += c_A * c_B * np.sqrt(pzzs[i])
    #     #### need to add higher order terms:
    #     for i in range(6,30):
    #         for j in range(6,30):
    #             # there is a sum
    #             if (i + j % 2 == 0):
    #                 if i % 2 == 0:
    #                     # both are even: add term to even sum
    #                     c_A = self.getcatcoefficients(n = i, j =0)
    #                     c_B = self.getcatcoefficients(n = j, j =0)
    #                     totalsumeven += c_A * c_B
    #                 else:
    #                     # both are odd: add term to odd sum
    #                     c_A = self.getcatcoefficients(n=i, j=1)
    #                     c_B = self.getcatcoefficients(n=j, j=1)
    #                     totalsumeven += c_A * c_B
    #             else:
    #                 # coefficients are 0
    #                 continue
    #     totalsum = np.power(totalsumodd, 2) + np.power(totalsumeven, 2)
    #     return totalsum/ pxxtotal
    #
    # def get_rate(self):
    #     """
    #     Obtain the secure key rate for the setup with finite decoy states: NOTE: this will be slow due to optimisation
    #     required
    #     :return: rate of the setup R: float
    #     """
    #     pxxtotal10 = self.getpxxtotal(kc = 1, kd = 0)
    #     biterror10 = self.getbiterrorrate(kc = 1, kd = 0)
    #     phaseerror10 = self.getphaseerror(kc = 1, kd = 0)
    #     rlow10 = pxxtotal10 * (1-binaryentropy(biterror10) - efficiency_cascade * binaryentropy(phaseerror10))
    #     pxxtotal01 = self.getpxxtotal(kc = 0, kd = 1)
    #     biterror01 = self.getbiterrorrate(kc = 0, kd = 1)
    #     phaseerror01 = self.getphaseerror(kc = 0, kd = 1)
    #     rlow01 = pxxtotal01 * (1-binaryentropy(biterror01) - efficiency_cascade * binaryentropy(phaseerror01))
    #     if rlow01 == np.nan:
    #         rlow01 = 0
    #     elif rlow10 == np.nan:
    #         rlow10 = 0
    #     return max(rlow01, 0) + max(rlow10, 0)

    def get_rate_infinite_decoys(self):
        """
        Obtain the secure key rate for the setup with infinite decoy states: NOTE: this assumes infinite decoy states
        :return: rate of the setup R: float
        """
        pxxtotal10 = self.getpxxtotal(kc = 1, kd = 0)
        biterror10 = self.getbiterrorrate(kc = 1, kd = 0)
        phaseerror10 = self.getphaseerrorforinfinitedecoycase(kc =1, kd = 0)
        rlow10 = pxxtotal10 *(1-binaryentropy(biterror10) - efficiency_cascade * binaryentropy(phaseerror10))
        pxxtotal01 = self.getpxxtotal(kc = 0, kd = 1)
        biterror01 = self.getbiterrorrate(kc = 0, kd =1)
        phaseerror01 = self.getphaseerrorforinfinitedecoycase(kc = 0, kd = 1)
        rlow01 = pxxtotal01 * (1- binaryentropy(biterror01) - efficiency_cascade * binaryentropy(phaseerror01))
        return max(rlow10,0) + max(rlow01, 0)
