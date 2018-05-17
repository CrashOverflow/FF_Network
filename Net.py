from workdir import Layer as l
import numpy as np

# Classe che rappresenta una singola rete Feed-Forward Full Connected.
class Net:

    # L'array di layers sarà aggiunto dopo l'input dell'utente.
    # w_layers = null.

    # Costruttore con parametri:

    # Lunghezza del feature vector in input alla rete.
    # (x1, x2, ..., xN)
    # - n_f -> n_features

    # Numero dei layers di PESI della rete.
    # - n_l -> n_layers

    def __init__(self, n_f, n_l):

        self.n_features = n_f
        self.n_layers = n_l
        self.array_layers  = []

        """All'inizio n_prev, ovvero il numero dei nodi dello strato
           precedente corrisponderà al numero degli input X1, ..., Xd"""

        n_prev = self.n_features

        for i in range(0, self.n_layers):

            print("Number of neurons for layer " + str(i) + ":")
            n_nodes = int(input())

            # Switch case hand made per scegliere la funzione di attivazione
            # del layer (Python non ha gli switch :( )

            print("Which activation function will have this layer? :")
            print("1. Logistic sigmoid")
            print("_______________________")

            while True:
                print("Choice: ")
                choice = int(input())
                if choice == 1:
                    self.array_layers.append(l.Layer_s(n_nodes, n_prev))
                    break

            # Aggiorno il numero dei nodi precedenti per le connessioni.
            n_prev = n_nodes

    # Stampa della rete
    def print(self):

        # Stampa le matrici del layer
        print("Printing the weight matrixes for every layer: \n")
        i = 0
        for lay in self.array_layers:
            print("Layer " + str(i) + " weights matrix : \n")
            lay.print_weights_matrix()
            i = i + 1

    # Forward propagation per l'input x
    def forward(self, x):
        z_prev = x
        for l in self.array_layers:
            a = np.sum([l.weights_matrix.dot(z_prev), l.b], axis=0)
            z = l.actfun(a)
            l.a = a
            l.z = z
            z_prev = z



