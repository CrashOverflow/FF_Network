# Authors: Marco Urbano & Ciro Brandi.

# Classe che descrive un layer della rete neurale.
# E' la classe astratta che va poi istanziata con una delle sue sottoclassi
# che specificano la funzione di attivazione adatta.


from abc import ABCMeta, abstractmethod
import numpy as np
from scipy.stats import logistic

class Layer(metaclass=ABCMeta):
    # Array degli input (a)
    a = np.array([[]])
    # Array degli output (z)
    z = np.array([[]])

    # LEARNING ATTRIBUTES #

    # Array dei delta
    delta = np.array([])
    # Derivata matrice dei pesi.
    der_w = np.ndarray([])
    # Derivata array bias.
    der_b = np.ndarray([])

    # RPROP Parameters.
    # Derivata epoca precedente dei pesi.
    der_w_prev_epoch = 0
    # Derivata epoca precedente bias.
    der_b_prev_epoch = 0
    # Matrice dei fattori di aggiornamento (UPDATE VALUES)
    update_values_rprop = 0
    # Vettori dei fattori di aggiornamento per i bias (BIAS UPDATE VALUES)
    update_values_b_rprop = 0
    # Matrice della differenza dei pesi di RPROP. (Weight diff)
    weight_diff = 0
    # Vettore della differenza dei bias di RPROP. (Bias diff)
    bias_diff = 0


    def __init__(self, n_neurons, n_connections):
        # Costruisci la matrice dei pesi in maniera random tra 0 e 1.
        self.weights_matrix = 1 - 2 * np.random.rand(n_neurons, n_connections)
        # Array dei pesi per il bias random tra 0 e 1.
        # E' un vettore colonna.
        self.b = 1 - 2 * np.random.rand(n_neurons, 1)

    def print_weights_matrix(self):
        print(self.weights_matrix)

    def print_bias(self):
        print(self.b)

    def print_delta(self):
        print(self.delta)

    # Funzione di inizializzazione on-demand
    # della matrice per aggiornare i pesi con RPROP.
    # Tutti a 0.1 per update values e tutti a 0 per la differenza dei pesi.
    def init_layer_rprop(self):
        # Inizializza anche le matrici dei pesi e dei bias perchè si
        # andrà a sommare e si deve partire da una matrice e da un vettore
        # con valori settati a 0.
        self.der_w = np.full(np.shape(self.weights_matrix), 0.0)
        self.der_b = np.full(np.shape(self.b), 0.0)
        self.der_w_prev_epoch = np.full(np.shape(self.weights_matrix), 0.0)
        self.der_b_prev_epoch = np.full(np.shape(self.b), 0.0)

        # Inizializzazione dei valori per la RPROP.
        self.update_values_rprop = np.full(np.shape(self.weights_matrix), 0.1)
        self.weight_diff = np.full(np.shape(self.weights_matrix), 0.0)
        self.update_values_b_rprop = np.full(np.shape(self.b), 0.1)
        self.bias_diff = np.full(np.shape(self.b), 0.0)


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
        #return logistic.pdf(x)
        return logistic.cdf(x)
        #return 1 / (1 + np.exp(-x))

    # Derivata = s(x) * (1 - s(x))
    # Implementato usando il prodotto punto-punto con numpy
    def actfun_der(self, x):
        act = self.actfun(x)
        act_1 = 1 - act
        prod = act * act_1
        return self.actfun(x) * (1 - self.actfun(x))

# Sottoclasse di Layer con funzione di attivazione Identità.
# Dato che la sua derivata è uguale a 1 non è mai usata come funzione
# di attivazione degli hidden layers.
class Layer_i(Layer):


    # Identità = x
    def actfun(self, x):
        return x
    # Derivata = 1
    def actfun_der(self, x):
        return np.ones((len(x), 1))