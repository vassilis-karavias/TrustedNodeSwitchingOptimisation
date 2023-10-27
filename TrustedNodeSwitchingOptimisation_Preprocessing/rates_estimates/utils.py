import numpy as np
import scipy as sp
import math
from rates_estimates.bb84rates import BB84Network
from rates_estimates.decoybb84rates import DecoyBB84Network
from rates_estimates.sixstaterates import SixStateNetwork
from rates_estimates.sarg04rates import SARG04Network
from rates_estimates.dpsrates import DPSNetwork
from rates_estimates.sixstatedecoyrates import SixStateDecoyNetwork
from rates_estimates.misalignedtolerantBB84 import MisalignedTolerantBB84
from rates_estimates.symmetricTwinField import SymmetricTwinField
from rates_estimates.asymmetricTwinField import AsymmetricTwinField
from scipy import optimize

def optimisation_function(mu, network):
    """
    Function that gives the rates for different values of mu to be optimised
    :param mu: the average photon number of the laser source: float
    :param network: The network used: Network
    :return: -The rate of the setup for the input mu: float
    """
    network.set_mu(mu = mu[0])
    return - network.get_rate()

def optimisemu(network):
    """
    Optimiser to optimise the value of mu for the given network
    :param network: The network used: Network
    :return: The optimisation outcome
    """
    return optimize.minimize(optimisation_function, x0 = 0.1, args = network, method = "CG").x[0]

def get_rate(length, protocol, coldalice, coldbob, length_1 = None):
    """
    Gives the maximum theoretical rate for a given length of fibre, the protocol defined and whether to use cold Alice
    and Cold Bob
    :param length: the length of the connection in km: float
    :param protocol: the protocol used: one of "BB84", "SARG04", "Decoy", "SixState",  'DecoySixState',
                    'MisalignmentTolerant', 'SymmetricTwinField', 'AsymmetricTwinField' or 'DPS': string
    :param coldalice: whether to use cold Alice conditions (QDs): boolean
    :param coldbob: whether to use cold Bob conditions (SNSPDs): boolean
    :param length_1: only needed for AsymmetricTwinField Length of Alice to Charlie: float
    :return: The rate of the point to point connection: float (take floor of result to get it in integers)
    """
    # for each case of protocol, coldalice, coldbob - set up the network
    if protocol == "BB84":
        if coldalice:
            if coldbob:
                ### using ID281 Superconducting nanowire system (SNSPD)
                # -https://marketing.idquantique.com/acton/attachment/11868/f-023b/1/-/-/-/-/ID281_Brochure.pdf
                # using data from H. Want et al 2019 for source, 0.2dB/km fibre loss and detector from Kahl et al 2015
                network = BB84Network(purity= 0.025, efficiencysource = 0.60, rate = np.power(10, 8), mu = 0.0, cold =
                    True, fibreloss = 0.2, efficiencydetector = 0.85, darkcountrate = 100, deadtime = 1000 * (10 ** -9),
                                      length = length, misalignment= 2)
            else:
                # # Using ID Qube Series NIR Gated version
                # https://marketing.idquantique.com/acton/attachment/11868/f-926db6fe-7c84-4bed-92fa-6d90a2612e03/1/-/-/-/-/ID%20Qube%20NIR%20Gated%20Brochure.pdf
                # cold alice hot bob: Will use InGaAs/InGaAsP as in Sun et al 2009 and Meng et al 2015 - these work at
                # telecom wavelengths as required. Here we use data from Sun et al 2009 (here we will assume 250K is
                # easy to achieve)
                network = BB84Network(purity = 0.025, efficiencysource=0.60, rate= np.power(10,8), mu = 0.0, cold = True,
                                      fibreloss = 0.2, efficiencydetector= 0.2, darkcountrate= 200,
                                      deadtime= 50000* (10 ** -9), length = length, misalignment= 2)
        else:
            if coldbob:
                # hot Alice, cold Bob, for Hot Alice use average photon number of 0.1 as done in many experiments (can
                # change this to use optimal average photon number)
                network = BB84Network(purity = 0.001, efficiencysource= 1.0, rate= np.power(10, 8), mu = 0.1, cold =
                                      False, fibreloss= 0.2, efficiencydetector= 0.85, darkcountrate= 100,
                                      deadtime= 1000 * (10 ** -9), length = length, misalignment= 2)
            else:
                # cold Alice, hot Bob
                network = BB84Network(purity = 0.001, efficiencysource= 1.0, rate= np.power(10, 8), mu = 0.1, cold =
                                      False, fibreloss= 0.2, efficiencydetector=0.2, darkcountrate= 200,
                                      deadtime= 50000* (10 ** -9), length = length, misalignment= 2)
    elif protocol == "SARG04":
        if coldalice:
            if coldbob:
                # using data from H. Want et al 2019 for source, 0.2dB/km fibre loss and detector from Kahl et al 2015
                network = SARG04Network(purity=0.025, efficiencysource=0.60, rate=np.power(10, 8), mu=0.0, cold=
                True, fibreloss=0.2, efficiencydetector=0.85, darkcountrate=100, deadtime=1000 * (10 ** -9),
                                      length=length, misalignment=2)
            else:
                # cold alice hot bob: Will use InGaAs/InGaAsP as in Sun et al 2009 and Meng et al 2015 - these work at
                # telecom wavelengths as required. Here we use data from Sun et al 2009 (here we will assume 250K is
                # easy to achieve)
                network = SARG04Network(purity=0.025, efficiencysource=0.60, rate=np.power(10, 8), mu=0.0, cold=True,
                                      fibreloss=0.2, efficiencydetector=0.2, darkcountrate=200,
                                      deadtime= 50000* (10 ** -9), length=length, misalignment=2)
        else:
            if coldbob:
                # hot Alice, cold Bob, for Hot Alice use average photon number of 0.1 as done in many experiments (can
                # change this to use optimal average photon number)
                network = SARG04Network(purity=0.001, efficiencysource=1.0, rate=np.power(10, 8), mu=0.1, cold=
                False, fibreloss=0.2, efficiencydetector=0.85, darkcountrate=100,
                                      deadtime=1000 * (10 ** -9), length=length, misalignment=2)
            else:
                # hot Alice, hot Bob
                network = SARG04Network(purity=0.001, efficiencysource=1.0, rate=np.power(10, 8), mu=0.1, cold=
                False, fibreloss=0.2, efficiencydetector=0.2, darkcountrate=200,
                                      deadtime=50000* (10 ** -9), length=length, misalignment=2)
    elif protocol == "SixState":
        if coldalice:
            if coldbob:
                # using data from H. Want et al 2019 for source, 0.2dB/km fibre loss and detector from Kahl et al 2015
                network = SixStateNetwork(purity=0.025, efficiencysource=0.60, rate=np.power(10, 8), mu=0.0, cold=
                True, fibreloss=0.2, efficiencydetector=0.85, darkcountrate=100, deadtime=1000 * (10 ** -9),
                                      length=length, misalignment=2)
            else:
                # cold alice hot bob: Will use InGaAs/InGaAsP as in Sun et al 2009 and Meng et al 2015 - these work at
                # telecom wavelengths as required. Here we use data from Sun et al 2009 (here we will assume 250K is
                # easy to achieve)
                network = SixStateNetwork(purity=0.025, efficiencysource=0.60, rate=np.power(10, 8), mu=0.0, cold=True,
                                      fibreloss=0.2, efficiencydetector=0.2, darkcountrate=200,
                                      deadtime=50000* (10 ** -9), length=length, misalignment=2)
        else:
            if coldbob:
                # hot Alice, cold Bob, for Hot Alice use average photon number of 0.1 as done in many experiments (can
                # change this to use optimal average photon number)
                network = SixStateNetwork(purity=0.001, efficiencysource=1.0, rate=np.power(10, 8), mu=0.1, cold=
                False, fibreloss=0.2, efficiencydetector=0.85, darkcountrate=100,
                                      deadtime=1000 * (10 ** -9), length=length, misalignment=2)
            else:
                # cold Alice, cold Bob
                network = SixStateNetwork(purity=0.001, efficiencysource=1.0, rate=np.power(10, 8), mu=0.1, cold=
                False, fibreloss=0.2, efficiencydetector=0.2, darkcountrate=200,
                                      deadtime=50000* (10 ** -9), length=length, misalignment=2)
    elif protocol == "Decoy":
        if coldalice:
            if coldbob:
                # using data from H. Want et al 2019 for source, 0.2dB/km fibre loss and detector from Kahl et al 2015
                network = DecoyBB84Network(purity=0.025, efficiencysource=0.60, rate=np.power(10, 8), mu=0.0, cold=
                True, fibreloss=0.2, efficiencydetector=0.85, darkcountrate=100, deadtime=1000 * (10 ** -9),
                                          length=length, misalignment=2)
            else:
                # cold alice hot bob: Will use InGaAs/InGaAsP as in Sun et al 2009 and Meng et al 2015 - these work at
                # telecom wavelengths as required. Here we use data from Sun et al 2009 (here we will assume 250K is
                # easy to achieve)
                network = DecoyBB84Network(purity=0.025, efficiencysource=0.60, rate=np.power(10, 8), mu=0.0, cold=True,
                                          fibreloss=0.2, efficiencydetector=0.2, darkcountrate=200,
                                          deadtime=50000* (10 ** -9), length=length, misalignment=2)
        else:
            if coldbob:
                # hot Alice, cold Bob, for Hot Alice use average photon number of 0.1 as done in many experiments (can
                # change this to use optimal average photon number)
                network = DecoyBB84Network(purity=0.001, efficiencysource=1.0, rate=np.power(10, 8), mu=0.1, cold=
                False, fibreloss=0.2, efficiencydetector=0.85, darkcountrate=100,
                                          deadtime=1000 * (10 ** -9), length=length, misalignment=2)
            else:
                # cold Alice, cold Bob
                network = DecoyBB84Network(purity=0.001, efficiencysource=1.0, rate=np.power(10, 8), mu=0.1, cold=
                False, fibreloss=0.2, efficiencydetector=0.2, darkcountrate=200,
                                          deadtime=50000 * (10 ** -9), length=length, misalignment=2)
    elif protocol == "DPS":
        if coldalice:
            print("DPS must use warm Alice setup")
            raise ValueError
        else:
            if coldbob:
                # hot Alice, cold Bob, for Hot Alice use average photon number of 0.1 as done in many experiments (can
                # change this to use optimal average photon number)
                network = DPSNetwork(purity=0.001, efficiencysource=1.0, rate=np.power(10, 8), mu=1, cold=
                False, fibreloss=0.2, efficiencydetector=0.85, darkcountrate=100,
                                           deadtime=1000 * (10 ** -9), length=length, misalignment=2)
            else:
                # cold Alice, cold Bob
                network = DPSNetwork(purity=0.001, efficiencysource=1.0, rate=np.power(10, 8), mu=1, cold=
                False, fibreloss=0.2, efficiencydetector=0.2, darkcountrate=200,
                                           deadtime=50000 * (10 ** -9), length=length, misalignment=2)
    elif protocol == "DecoySixState":
        if coldalice:
            if coldbob:
                # using data from H. Want et al 2019 for source, 0.2dB/km fibre loss and detector from Kahl et al 2015
                network = SixStateDecoyNetwork(purity=0.025, efficiencysource=0.60, rate=np.power(10,8), mu=0.0, cold=
                True, fibreloss=0.2, efficiencydetector=0.85, darkcountrate=100, deadtime=1000 * (10 ** -9),
                                           length=length, misalignment=2)
            else:
                # cold alice hot bob: Will use InGaAs/InGaAsP as in Sun et al 2009 and Meng et al 2015 - these work at
                # telecom wavelengths as required. Here we use data from Sun et al 2009 (here we will assume 250K is
                # easy to achieve)
                network = SixStateDecoyNetwork(purity=0.025, efficiencysource=0.60, rate=np.power(10, 8), mu=0.0, cold=True,
                                           fibreloss=0.2, efficiencydetector=0.2, darkcountrate=200,
                                           deadtime=50000 * (10 ** -9), length=length, misalignment=2)
        else:
            if coldbob:
                # hot Alice, cold Bob, for Hot Alice use average photon number of 0.1 as done in many experiments (can
                # change this to use optimal average photon number)
                network = SixStateDecoyNetwork(purity=0.001, efficiencysource=1.0, rate=np.power(10, 8), mu=0.1, cold=
                False, fibreloss=0.2, efficiencydetector=0.85, darkcountrate=100,
                                           deadtime=1000 * (10 ** -9), length=length, misalignment=2)
            else:
                # cold Alice, cold Bob
                network = SixStateDecoyNetwork(purity=0.001, efficiencysource=1.0, rate=np.power(10, 8), mu=0.1, cold=
                False, fibreloss=0.2, efficiencydetector=0.2, darkcountrate=200,
                                           deadtime=50000 * (10 ** -9), length=length, misalignment=2)
    elif protocol == "MisalignmentTolerant":
        if coldalice:
            if coldbob:
                # using data from H. Want et al 2019 for source, 0.2dB/km fibre loss and detector from Kahl et al 2015
                network = MisalignedTolerantBB84(purity=0.025, efficiencysource=0.60, rate=np.power(10, 8), mu=0.0, cold=
                True, fibreloss=0.2, efficiencydetector=0.85, darkcountrate=100, deadtime=1000 * (10 ** -9),
                                           length=length, misalignment=2)
            else:
                # cold alice hot bob: Will use InGaAs/InGaAsP as in Sun et al 2009 and Meng et al 2015 - these work at
                # telecom wavelengths as required. Here we use data from Sun et al 2009 (here we will assume 250K is
                # easy to achieve)
                network = MisalignedTolerantBB84(purity=0.025, efficiencysource=0.60, rate=np.power(10, 8), mu=0.0, cold=True,
                                           fibreloss=0.2, efficiencydetector=0.2, darkcountrate=200,
                                           deadtime=50000 * (10 ** -9), length=length, misalignment=2)
        else:
            if coldbob:
                # hot Alice, cold Bob, for Hot Alice use average photon number of 0.1 as done in many experiments (can
                # change this to use optimal average photon number)
                network = MisalignedTolerantBB84(purity=0.001, efficiencysource=1.0, rate=np.power(10, 8), mu=0.1, cold=
                False, fibreloss=0.2, efficiencydetector=0.85, darkcountrate=100,
                                           deadtime=1000 * (10 ** -9), length=length, misalignment=2)
            else:
                # cold Alice, cold Bob
                network = MisalignedTolerantBB84(purity=0.001, efficiencysource=1.0, rate=np.power(10, 8), mu=0.1, cold=
                False, fibreloss=0.2, efficiencydetector=0.2, darkcountrate=200,
                                           deadtime=50000 * (10 ** -9), length=length, misalignment=2)
    elif protocol == "SymmetricTwinField":
        if coldalice:
            print("Symmetric Twin Field only implemented for warm Alice setup")
            raise ValueError
        else:
            if coldbob:
                # hot Alice, cold Bob, for Hot Alice use average photon number of 0.1 as done in many experiments (can
                # change this to use optimal average photon number)
                network = SymmetricTwinField(purity=0.001, efficiencysource=1.0, rate=np.power(10, 8), mu=0.1, cold=
                False, fibreloss=0.2, efficiencydetector=0.85, darkcountrate=100,
                                                 deadtime=1000 * (10 ** -9), length=length, misalignment=2, phasemisalignment=0)
            else:
                # cold Alice, cold Bob
                network = SymmetricTwinField(purity=0.001, efficiencysource=1.0, rate=np.power(10, 8), mu=0.1, cold=
                False, fibreloss=0.2, efficiencydetector=0.2, darkcountrate=200,
                                                 deadtime=50000 * (10 ** -9), length=length, misalignment=2, phasemisalignment=0)
    elif protocol == "AsymmetricTwinField":
        if coldalice:
            print("Asymmetric Twin Field only implemented for warm Alice setup")
            raise ValueError
        else:
            if coldbob:
                # hot Alice, cold Bob, for Hot Alice use average photon number of 0.1 as done in many experiments (can
                # change this to use optimal average photon number)
                network = AsymmetricTwinField(purity=0.001, efficiencysource=1.0, rate=np.power(10, 8), mu=0.1, cold=
                False, fibreloss=0.2, efficiencydetector=0.85, darkcountrate=100,
                                                 deadtime=1000 * (10 ** -9), length=length, misalignment=2, phasemisalignment=0, length_1= length_1)
            else:
                # cold Alice, cold Bob
                network = AsymmetricTwinField(purity=0.001, efficiencysource=1.0, rate=np.power(10, 8), mu=0.1, cold=
                False, fibreloss=0.2, efficiencydetector=0.2, darkcountrate=200,
                                                 deadtime=50000 * (10 ** -9), length=length, misalignment=2, phasemisalignment=0, length_1= length_1)
    else:
        print("Protocol must be one of 'BB84', 'SARG04, 'Decoy', 'SixState', 'DecoySixState', 'MisalignmentTolerant', 'SymmetricTwinField', 'AsymmetricTwinField' or 'DPS'")
        raise ValueError
    # if not coldalice:
    #     # if not cold alice we will perform an optimisation on the laser power
    #     if not network.get_rate() == 0:
    #         # if the rate is not 0 then the optimisation can go on
    #         mu_1 = optimisemu(network)
    #         print("Optimal mu for network is " + str(mu_1))
    #         network.set_mu(mu_1)
    #     else:
    #         found_efficiency = False
    #         # else try to find a value of mu which gives non-zero rates: if this does not exist then rate = 0
    #         # else optimise.
    #         for m in np.linspace(0.0001, 1, 500):
    #             network.set_mu(mu = m)
    #             if not network.get_rate() == 0:
    #                 found_efficiency = True
    #                 mu_1 = optimisemu(network)
    #                 print("Optimal mu for network 1 is " + str(mu_1))
    #                 network.set_mu(mu_1)
    #                 break
    #         if not found_efficiency:
    #             return 0 # no rate if no efficiency found
    # obtain the rates of the protocols
    if protocol == "SARG04" or protocol == "DPS" or protocol == "MisalignmentTolerant":
        rate = network.get_rate()
    elif protocol == "SymmetricTwinField" or protocol == "AsymmetricTwinField":
        rate = network.get_rate_infinite_decoys()
    else:
        rate = network.get_rate_efficient()
    if coldalice:
        return np.power(10, 8) * rate
    else:
        return np.power(10, 8) * rate

def getratesofnetwork(lengths, protocols, coldalices, coldbobs):
    """
    Gives the maximum theoretical rate for a set of lengths of fibre, the protocold defined and whether to use cold Alice
    and Cold Bob
    :param lengths: An array of lengths of the connections in km: Array(float)
    :param protocols: Either a string defining the protocol or an array of strings for each of the links (must be same
    length as lengths array): string or Array(string). Strings must be one of 'BB84', 'SARG04, 'Decoy' or 'SixState'
    :param coldalices: Either a boolean defining the use of a cold Alice or not, or an array of booleans for each of
    the links (must be same length as lengths array): boolean or Array(boolean)
    :param coldbobs: Either a boolean defining the use of a cold Alice or not, or an array of booleans for each of
    the links (must be same length as lengths array): boolean or Array(boolean)
    :return: An array of the upper bound of the capacities of the links based on the information: float (if you want the
    rate in int take the floor of the elements of the array)
    """
    rates = []
    # for each length set up the networks and get rates: need to check cases where protocols, coldalices, coldbobs are
    # arrays or not arrays
    if isinstance(protocols, list):
        if len(lengths) != len(protocols):
            print("Length of arrays must be the same")
            raise ValueError
        else:
            if isinstance(coldalices, list):
                if len(lengths) != len(coldalices):
                    print("Length of arrays must be the same")
                    raise ValueError
                else:
                    if isinstance(coldbobs, list):
                        if len(lengths) != len(coldbobs):
                            print("Length of arrays must be the same")
                            raise ValueError
                        else:
                            # protocols, coldalices and coldbobs are arrays
                            for i in range(len(lengths)):
                                ratesi = get_rate(length=lengths[i], protocol= protocols[i], coldalice=coldalices[i],
                                                  coldbob= coldbobs[i])
                                rates.append(ratesi)
                    else:
                        # protocols, coldalices are arrays, coldbobs is a boolean
                        for i in range(len(lengths)):
                            ratesi = get_rate(length=lengths[i], protocol=protocols[i], coldalice=coldalices[i],
                                              coldbob=coldbobs)
                            rates.append(ratesi)
            else:
                if isinstance(coldbobs, list):
                    if len(lengths) != len(coldbobs):
                        print("Length of arrays must be the same")
                        raise ValueError
                    else:
                        # protocols and coldbobs are arrays coldalices is a boolean
                        for i in range(len(lengths)):
                            ratesi = get_rate(length=lengths[i], protocol=protocols[i], coldalice=coldalices,
                                              coldbob=coldbobs[i])
                            rates.append(ratesi)
                else:
                    # protocols is an array, coldalices and coldbobs are booleans
                    for i in range(len(lengths)):
                        ratesi = get_rate(length=lengths[i], protocol=protocols[i], coldalice=coldalices,
                                          coldbob=coldbobs)
                        rates.append(ratesi)
    else:
        if isinstance(coldalices, list):
            if len(lengths) != len(coldalices):
                print("Length of arrays must be the same")
                raise ValueError
            else:
                if isinstance(coldbobs, list):
                    if len(lengths) != len(coldbobs):
                        print("Length of arrays must be the same")
                        raise ValueError
                    else:
                        # coldalices and coldbobs are arrays, protocols is a string
                        for i in range(len(lengths)):
                            ratesi = get_rate(length=lengths[i], protocol=protocols, coldalice=coldalices[i],
                                              coldbob=coldbobs[i])
                            rates.append(ratesi)
                else:
                    # coldalices is an array, coldbobs is boolean, protocols is a string
                    for i in range(len(lengths)):
                        ratesi = get_rate(length=lengths[i], protocol=protocols, coldalice=coldalices[i],
                                          coldbob=coldbobs)
                        rates.append(ratesi)
        else:
            if isinstance(coldbobs, list):
                if len(lengths) != len(coldbobs):
                    print("Length of arrays must be the same")
                    raise ValueError
                else:
                    # coldbobs is an array, coldalices is a boolean, protocols is a string
                    for i in range(len(lengths)):
                        ratesi = get_rate(length=lengths[i], protocol=protocols, coldalice=coldalices,
                                          coldbob=coldbobs[i])
                        rates.append(ratesi)
            else:
                # none of them are arrays
                for i in range(len(lengths)):
                    ratesi = get_rate(length=lengths[i], protocol=protocols, coldalice=coldalices,
                                      coldbob=coldbobs)
                    rates.append(ratesi)
    return rates