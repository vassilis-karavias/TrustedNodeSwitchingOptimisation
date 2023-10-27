import numpy as np
from scipy import optimize
import math
from rates_estimates.sources import QDsource, SPDCSource
from rates_estimates.entropyfunctions import binaryentropy, misalignment_criterion
from rates_estimates.networkclass import Network

efficiency_cascade = 1.2

class DecoyBB84Network(Network):

    def __init__(self, purity, efficiencysource, rate, mu, cold, fibreloss, efficiencydetector, darkcountrate, deadtime, length, misalignment):
        """
               A class for a setup using the Decoy State BB84 protocol
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

    def get_single_photon_gain(self):
        """
        Gets the single photon gain. In this case this can be taken as the expected value Y1 * e^-mu mu^n/ n!  for SPDC
        as deviations from standard is detected by decoys: from Lo et al 2005b. For QD it is Y1 * p(1)
        :return: Single photon gain Q_1: float
        """
        probofdetection = self.fibre.getfibreefficiency(self.length)* self.detector.getefficiency()
        # Y1 = pd + (1-pd) * pdc: the probability of getting a detection given Alice sent a photon, in the case there
        # was no detection there is the chance of getting a dark count
        y1 = probofdetection + (1-probofdetection) * self.pdc
        etadead = np.power(
            1 + self.detector.getdeadtime() * self.source.getrate() * (self.pmu + 2 * self.pdc + self.pap), -1)
        if isinstance(self.source, SPDCSource):
            return y1 * self.source.getmu() * np.exp(-self.source.getmu()) * etadead
        else:
            # source is QD source
            return y1 * self.source.getnphotonprobability(n=1) * etadead

    def get_gain_mu(self, mu):
        """
        gets total gain Q_(mu) for a given mu used as decoys: Lo et al 2005b  - this is for SPDC sources only
        :param mu: mean photon number of the decoy state: float
        :return: The gain: float
        """
        if isinstance(self.source, SPDCSource):
            # create a source with the given mu to get the rate from the source
            source = SPDCSource(mu=mu, rate=self.source.getrate())
            # get the probability of a detection for this new mu
            pmu = source.getpmu(detector=self.detector, fibre = self.fibre, length = self.length, clocktime = self.clocktime)
            # calculate the fraction of time the source spends in the dead time
            etadead = np.power(
                1 + self.detector.getdeadtime() * self.source.getrate() * (pmu + 2 * self.pdc + self.pap), -1)
            # calculate gain
            return (pmu + 2 * self.pdc + self.pap) * etadead
        else:
            print("Decoy state must be used with SPDC source")
            raise ValueError

    def get_qber_mu(self, mu):
        """
        gets total QBER E_(mu) for a given mu used as decoys: Lo et al 2005b - this is for SPDC sources only
        :param mu: mean photon number of the decoy state: float
        :return: The quantum bit error rate: float
        """
        if isinstance(self.source, SPDCSource):
            # create a source with the given mu to get the rate from the source
            source = SPDCSource(mu=mu, rate=self.source.getrate())
            # get the probability of a detection for this new mu
            pmu = source.getpmu(detector=self.detector, fibre=self.fibre, length=self.length, clocktime=self.clocktime)
            # calculate the visibility for the new mu
            coefficient = source.getmu() * self.etaf * self.detector.getefficiency()
            prob_error = self.detector.getproberror(self.clocktime)
            v =coefficient / (coefficient + prob_error)
            # calculate the qber using formula
            denom = pmu + 2 * self.pdc + self.pap
            num = pmu * (1-v) + 2 * self.pdc + self.pap
            return num/(2* denom)
        else:
            print("Decoy state must be used with SPDC source")
            raise ValueError

    def get_gain_qd(self, purity):
        """
        gets total gain Q_(mu) for a given purity used as decoys: - this is for a QD source only
        :param mu: mean photon number of the decoy state: float
        :return: The gain: float
        """
        if isinstance(self.source, QDsource):
            # create a source with the given purity to get the rate from the source
            source = QDsource(purity = purity, efficiency = self.source.getefficiency(), rate=self.source.getrate())
            # get the probability of a detection for this new mu
            pmu = source.getpmu(detector=self.detector, fibre = self.fibre, length = self.length, clocktime = self.clocktime)
            # calculate the fraction of time the source spends in the dead time
            etadead = np.power(
                1 + self.detector.getdeadtime() * self.source.getrate() * (pmu + 2 * self.pdc + self.pap), -1)
            # calculate gain
            return (pmu + 2 * self.pdc + self.pap) * etadead
        else:
            print("Decoy state must be used with QD source")
            raise ValueError

    def get_qber_qd(self, purity):
        """
        gets total QBER E_(mu) for a given purity used as decoys - this is for a QD source only
        :param mu: mean photon number of the decoy state: float
        :return: The quantum bit error rate: float
        """
        if isinstance(self.source, QDsource):
            # create a source with the given purity to get the rate from the source
            source = QDsource(purity =purity, efficiency = self.source.getefficiency(), rate=self.source.getrate())
            # get the probability of a detection for this new purity
            pmu = source.getpmu(detector=self.detector, fibre=self.fibre, length=self.length, clocktime=self.clocktime)
            # calculate the visibility for the new purity
            coefficient = source.getmu() * self.etaf * self.detector.getefficiency()
            prob_error = self.detector.getproberror(self.clocktime)
            v =coefficient / (coefficient + prob_error)
            # calculate QBER for new purity
            denom = pmu + 2 * self.pdc + self.pap
            num = pmu * (1-v) + 2 * self.pdc + self.pap
            return num/(2* denom)
        else:
            print("Decoy state must be used with QD source")
            raise ValueError

    def fit_errors_spdc(self, mu, edet):
        """
        The calculation that will yield Q_(mu)E_(mu) for SPDC source using the errors for n particles as defined in
        Lo et al 2005b. The fitting is done by fitting edet (section 4.3.1)
        The optimisation fits the detector errors including misalignment- no need to reinclude it
        :param mu: mean photon numbers for the optimisation: Array(float)
        :param edet: The detector error from calculation: float
        :return: The calculation of Q_(mu)E_(mu) using mu and edet as inputs on the LHS of equation: Array(float)
        """
        # Q_muE_mu = (sum Y_n e-mu mu^n/n fact  * en + 1/2 (2pdc + pap))eta_dead
        # get the parameters used often in the formulas
        probofdetection = self.etaf * self.detector.getefficiency()
        pdark = self.detector.getprobabilityofdarkcount(self.clocktime)
        # errors are given by edet (1-(1-probdet)^n) + 1/2 pdark as described in Lo et al 2005b
        en = [(edet*(1-(1-probofdetection))+ 1/2 * pdark), (edet*(1-(1-probofdetection) ** 2)+ 1/2 * pdark),
              (edet*(1-(1-probofdetection) ** 3)+ 1/2 * pdark), (edet*(1-(1-probofdetection) ** 4)+ 1/2 * pdark),
              (edet*(1-(1-probofdetection) ** 5)+ 1/2 * pdark), (edet*(1-(1-probofdetection) ** 6)+ 1/2 * pdark)]
        # counter to hold the sum of each photon number yield in equation
        distribution = 0.0
        # generate the sources for each of the mu's
        sources = [SPDCSource(mu=m, rate=self.source.getrate()) for m in mu]
        # get the probability of detection and fraction of time in dead time for the source for each of the mu's
        pmu = [source.getpmu(detector=self.detector, fibre=self.fibre, length=self.length, clocktime=self.clocktime) for source in sources]
        etadead = [np.power(
            1 + self.detector.getdeadtime() * self.source.getrate() * (p + 2 * self.pdc + self.pap), -1) for p in pmu]
        # carry out the sum in the formula for the E_mu Q_mu
        for n in range(1,len(en)+1):
            probabilityofgeneration = np.exp(-mu) * np.power(mu, n) / math.factorial(n)
            yieldn = (1-np.power(1-probofdetection, n)) + np.power(1-probofdetection, n * pdark)
            distribution += probabilityofgeneration * en[n-1] * yieldn
        return (distribution + 1/2 * (2 * self.pdc + self.pap)) * etadead

    def fit_errors_qd(self, purity, e1, e2):
        """
        The calculation that will yield Q_(mu)E_(mu) for QD source using the errors for n particles as defined in
        Lo et al 2005b
        :param purity: set of purities to act as decoys: Array(float_
        :param e1: The detector error for single photons: float
        :param e2: The detector error for 2 photons: float
        :return: The calculation of Q_(mu)E_(mu) using purity and e1, e2 as inputs on the LHS of equation: Array(float)
        """
        # Q_muE_mu = (sum Y_n e-mu mu^n/n fact  * en + 1/2 (2pdc + pap))eta_dead
        # get the parameters used often in the formulas
        probofdetection = self.etaf * self.detector.getefficiency()
        pdark = self.detector.getprobabilityofdarkcount(self.clocktime)
        # only two parameters to fit e1 and e2 here
        en = [e1, e2]
        # counter to hold the sum of each photon number yield in equation
        distribution = 0.0
        # generate the sources for each of the purities
        sources = [QDsource(purity=pr, efficiency = self.source.getefficiency(), rate=self.source.getrate()) for pr in purity]
        # get the probability of detection and fraction of time in dead time for the source for each of the purities
        pmu = [source.getpmu(detector=self.detector, fibre=self.fibre, length=self.length, clocktime=self.clocktime) for source in sources]
        etadead = [np.power(
            1 + self.detector.getdeadtime() * self.source.getrate() * (p + 2 * self.pdc + self.pap), -1) for p in pmu]
        # carry out the sum in the formula for the E_mu Q_mu
        for n in range(1,len(en)+1):
            probabilityofgeneration = [source.getnphotonprobability(n) for source in sources]
            yieldn = (1-np.power(1-probofdetection, n)) + np.power(1-probofdetection, n * pdark)
            distribution += np.asarray(probabilityofgeneration) * en[n-1] * yieldn
        return (distribution + 1/2 * (2 * self.pdc + self.pap)) * etadead

    def get_error_estimates(self):
        """
        Performs the Decoy step: estimates a better bound for the single photon error using decoys and parameter fitting
        :return: the value of the fitting in an array:
        (output[0] * (1-(1-probability of detection)) + 1/2 * probability of dark count)/ probability of detection is the
        single photon error: [float]
        """
        if isinstance(self.source, SPDCSource):
            # for SPDC use a series of mu and carry out optimisation
            mu = [0.0, 0.01, 0.05, 0.1, 0.15, 0.2]
            e_mu = [self.get_qber_mu(m) for m in mu]
            q_mu = [self.get_gain_mu(m) for m in mu]
            yparams = [e_mu[m] * q_mu[m] for m in range(len(mu))]
            popt,  pcov = optimize.curve_fit(self.fit_errors_spdc, xdata= mu, ydata= yparams, bounds= (0, 1), p0 = 0.05)
        else:
            # QDsource - use a series of purities
            purity = [0.001, 0.003, 0.02]
            e_mu = [self.get_qber_qd(m) for m in purity]
            q_mu = [self.get_gain_qd(m) for m in purity]
            yparams = [e_mu[m] * q_mu[m] for m in range(len(purity))]
            popt, pcov = optimize.curve_fit(self.fit_errors_qd, xdata=purity, ydata=yparams, bounds=(0, 1), p0=(0.05,0.05))
        return popt

    def get_rate(self):
        """
        Calculate the rate of the setup: Lo et al 2005b
        :return: The rate for the standard decoy state BB84 protocol: float
        """
        q1 = self.get_single_photon_gain()
        # if this is too small then the rate is 0 as there are no secure photons reaching Bob anyway
        if q1 < 0.000000000000001:
            return 0
        else:
            # the probability of a detection
            probofdetection = self.etaf * self.detector.getefficiency()
            # dark count probability
            pdark = self.detector.getprobabilityofdarkcount(self.clocktime)
            # optimisation parameters output from Decoys state calculation
            popt = self.get_error_estimates()
            # if it is SPDC the optimisation outcome is edet, need e1 = (edet * probofdetection + 1/2 * pdark) / probofdetection
            # else QD optimisation outputs e1 directly
            if isinstance(self.source, SPDCSource):
                e1 = (popt[0]*probofdetection+ 1/2 * pdark) / probofdetection
            else:
                e1 = popt[0]
            # get gain and QBER
            gain = self.get_gain()
            qber = self.get_qber()
            # calculation R = q1/2[1-H_2(e1)] - Q_mu/2f(E_mu)H_2(E_mu)
            rawrate = q1 / 2
            lossfromerrorcorrection = gain * efficiency_cascade * binaryentropy(qber) / 2
            errorfrommisalignment = optimize.root_scalar(misalignment_criterion,
                                                         args=(np.power(self.misalignment, 2) * np.exp(1)),
                                                         bracket=[0, 1000], method='bisect').root
            lossfromprivacyamplification = q1 * binaryentropy(e1 +errorfrommisalignment) / 2
            return max(0, rawrate - lossfromerrorcorrection - lossfromprivacyamplification)

    def get_rate_efficient(self):
        """
        Calculate the rate of the setup: Lo et al 2005b - here we use the
        efficient BB84 which changes the sifting to improve rates by a factor of up to 2 in asymptotic case
        :return: The rate for the efficient Decoy state BB84 protocol - sifting removes a negligible amount of states:
        float
        """
        return 2 * self.get_rate()