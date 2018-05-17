# Authors: Marco Urbano & Ciro Brandi.

# Classe che descrive un layer della rete neurale.
# E' la classe astratta che va poi istanziata con una delle sue sottoclassi
# che specificano la funzione di attivazione adatta.


from abc import ABCMeta, abstractmethod
import math


class Layer(metaclass=ABCMeta):

    def print_weights_matrix(self):
        print(self.weights_matrix)

    def print_bias(self):
        print(self.b)

    # Funzione di attivazione da implementare.
    @abstractmethod
    def actfun(self, x):
        pass
    # Derivata della funzione di attivazione da implementare .
    @abstractmethod
    def actfun_der(self, x):
        pass


# Sottoclasse di Layer con funzione di attivazione Sigmoide.
class Layer_s(Layer):

    def __init__(self, weights_matrix, b):

        self.weights_matrix = weights_matrix
        self.b = b

    # Sigmoide = 1 / (1 + e ^ - x)
    def actfun(self, x):
        return 1 / (1 + math.exp(-x))

    # Derivata = s(x) * (1 - s(x))
    def actfun_der(self, x):
        return self.actfun(x)*(1 - self.actfun(x))


