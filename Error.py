# Authors: Marco Urbano & Ciro Brandi.
from abc import ABCMeta, abstractmethod
import numpy as np

# La classe Error descrive oggetti che hanno il metodo per il calcolo della
# derivata della funzione di errore. PATTERN STRATEGY


class Error(metaclass=ABCMeta):

    @abstractmethod
    def fun(self, Y, T):
        pass

    @abstractmethod
    def compute_error(self, Y, T):
        pass


class TSS(Error):
    def fun(self, Y, T):
        return Y - T

    def compute_error(self, Y, T):
        return np.power(np.sum(Y - T), 2)

# https://stackoverflow.com/questions/42599498/numercially-stable-softmax
class CrossEntropy(Error):
    def softmax(self, Y):
        Y = Y - np.max(Y)
        t_sum = sum(np.exp(Y))
        z_exp = np.exp(Y)
        return np.divide(z_exp, t_sum)

    def fun(self, Y, T):
        return np.nan_to_num(self.softmax(Y) - T)

    def compute_error(self, Y, T):
        return -1 * np.sum(T * np.nan_to_num(np.log10(Y)))

