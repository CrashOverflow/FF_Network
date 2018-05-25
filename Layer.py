# Authors: Marco Urbano & Ciro Brandi.

# Classe che descrive un layer della rete neurale.
# E' la classe astratta che va poi istanziata con una delle sue sottoclassi
# che specificano la funzione di attivazione adatta.


from abc import ABCMeta, abstractmethod
import numpy as np


class Layer(metaclass=ABCMeta):
    # Array degli input (a)
    a = np.array([])
    # Array degli output (z)
    z = np.array([])

    # LEARNING ATTRIBUTES #

    # Array dei delta
    delta = np.array([])
    # Derivata matrice dei pesi.
    der_w = np.array([])
    # Derivata array bias.
    der_b = np.array([])

    def __init__(self, n_neurons, n_connections):
        # Costruisci la matrice dei pesi in maniera random tra 0 e 1.
        self.weights_matrix = 1 - 2 * np.random.rand(n_neurons, n_connections)
        # Array dei pesi per il bias random tra 0 e 1
        self.b = 1 - 2 * np.random.rand(1, n_neurons)

    def print_weights_matrix(self):
        print(self.weights_matrix)

    def print_bias(self):
        print(self.b)

    def print_delta(self):
        print(self.delta)


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


    # Sigmoide = 1 / (1 + e ^ - x)
    def actfun(self, x):
        return 1 / (1 + np.exp(-x))

    # Derivata = s(x) * (1 - s(x))
    # Implementato usando il prodotto punto-punto con numpy
    def actfun_der(self, x):
        return np.dot(self.actfun(x), (1 - self.actfun(x)))

# Sottoclasse di Layer con funzione di attivazione Identità.
# Dato che la sua derivata è uguale a 1 non è mai usata come funzione
# di attivazione degli hidden layers.
class Layer_i(Layer):


    # Identità = x
    def actfun(self, x):
        return x
    # Derivata = 1
    def actfun_der(self, x):
        return 1