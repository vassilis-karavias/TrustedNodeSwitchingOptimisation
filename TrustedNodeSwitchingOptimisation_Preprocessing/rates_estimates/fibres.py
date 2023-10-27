import numpy as np
import scipy as sp
import math

class Fibre():

    def __init__(self, loss):
        """
        Class for a Fibre
        :param loss: loss of the fibre in dB/km: float
        """
        self.loss = loss
        super().__init__()

    def getloss(self):
        """
        :return: the loss of the fibre in dB/km: float
        """
        return self.loss

    def getfibreefficiency(self, length):
        """
        Calculate the fibre efficiency for a given length: etaf = 10^(-loss * length /10)
        :param length: Length of connection
        :return: fibre efficiency: float
        """
        return np.power(10, -(self.loss * length) / 10)