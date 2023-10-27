import numpy as np

def binaryentropy(p):
    """
    Calculates the binary entropy H_2(p) for a given binomial distribution with probability of outcomes (p,1-p)
    :param p: Probability of an outcome in the binomial distribution
    :return: the binary entropy H_2(p) of the distribution
    """
    return - p * np.log2(p) - (1-p) * np.log2(1-p)

def conditional_entropy(ex, ey, ez):
    """
    Calculates the conditional entropy of the bit errors conditioned on the phase errors defined in Fung et al. 2006 as
    p_bit = e_y+e_z, p_phase = e_x + e_y. The conditional entropy is defined as
     H(Z|X) = -sum_(x in X, z in Z) p(x,z) log_2 (p(x,z)/p(x)):
     p(x) = (ex+ey, 1-ex-ey) is the probability distribution of a phase error and no phase error
     p(x,z) = (1-ex-ey-ez, ex, ez, ey) corresponds to no phase or bit error, a phase error only, a bit error only,
     and both a phase and bit error
     Now we note that if there is no error then there must be no phase errors so the sum is only over compatible
     probability parameters
    :param ex: error in the X value: float
    :param ey: error in the Y value: float
    :param ez: error in the Z value: float
    :return: The conditional entropy: float
    """
    entropy = - (1-ex-ey-ez) * np.log2((1-ex-ey-ez)/(1-ex-ey))
    entropy += - ex * np.log2(ex/(ex + ey))
    entropy += -ez * np.log2(ez/(1-ex-ey))
    entropy += -ey * np.log2(ey/(ex+ey))
    return entropy

def misalignment_criterion(y, misalignment):
    """
    The misalignment criterion in Gottesmann et al (2004) requires f(esin^2theta) such that H_2(1/2 - f(x)) + H_2(x) = 1
    find f(esin^2 theta)
    :param y: the guess for f(esin^2 theta)
    :param misalignment: The value of esin^2 theta
    :return: The outcome of H_2(1/2 - f(x)) + H_2(x) - 1
    """
    # add the result of this root finding problem to the phase error term for GLLP method
    return binaryentropy(1/2 - y) + binaryentropy(misalignment) -1
