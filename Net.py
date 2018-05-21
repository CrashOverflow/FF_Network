from workdir import Layer as l
import numpy as np

# Classe che rappresenta una singola rete Feed-Forward Full Connected.
# La funzione di errore "Somma di quadrati" è quella scelta per questa classe.
# Vanno implementate sia TSS che Cross Entropy.
class Net:

    # L'array di layers sarà aggiunto dopo l'input dell'utente.
    # w_layers = null.

    # Costruttore con parametri:

    # Lunghezza del feature vector in input alla rete.
    # (x1, x2, ..., xN)
    # - n_f -> n_features

    # neurons_list è la lista che contiene il numero di neuroni
    # per ogni layer: es. [5, 4, 3] avrò:
    # - 5 neuroni per il 1° layer
    # - 4 neuroni per il 2° layer
    # - 3 neuroni per l'ultimo layer

    # actfun_list è la scelta della funzione di attivazione per ogni layer
    # ad esempio: [1, 1, 0]
    # se i tipi di layer disponibili sono:
    # - 0: layer con funzione di attivazione IDENTITA'
    # - 1: layer con funzione di attivazione SIGMOIDE
    # Avrò i layer 1 e 2 (hidden layers) con sigmoide e
    # il layer di uscita con identità.

    # Raises ValueError exception if values of neurons_list
    # or actfun_list contain errors. (e.g. activation function
    # not existent)
    def __init__(self, n_f, neurons_list, actfun_list):
        try:
            if(len(neurons_list) == len(actfun_list)):

                self.n_features = n_f
                self.array_layers = []
                self.n_layers = len(neurons_list)

                """All'inizio n_prev, ovvero il numero dei nodi dello strato
                precedente corrisponderà al numero degli input X1, ..., Xd"""
                n_prev = self.n_features

                for i in range(0, len(neurons_list)):

                    n_nodes = neurons_list[i]

                    # Switch case hand made per scegliere la funzione di attivazione
                    # del layer (Python non ha gli switch :( )

                    if actfun_list[i] == 1:
                            self.array_layers.append(l.Layer_s(n_nodes, n_prev))
                    else:
                        raise ValueError('Function with code ' + str(actfun_list[i]) +
                                         'doesn\'t exists!')

                    # Aggiorno il numero dei nodi precedenti per le connessioni.
                    n_prev = n_nodes
            else:
                  raise ValueError('actfun_list and neurons_list have different lengths!!')

        except ValueError:
            raise



    # Stampa della rete
    def print(self):

        # Stampa le matrici del layer
        print("Printing the weight matrixes for every layer: \n")
        i = 0
        for lay in self.array_layers:
            print("Layer " + str(i) + " weights matrix : \n")
            lay.print_weights_matrix()
            print("bias: ")
            lay.print_bias()
            i = i + 1

    # Forward propagation per l'input x
    def forward(self, x):
        z_prev = x

        for layer in self.array_layers:

            # visto che la somma tra ndarray e array è un wrapper di
            # ndarray si prende il primo elemento, che è a sua volta
            # l'array risultante tra la somma del bias con la combinazione lineare.

            # (W^T * Z) + b
            a = np.dot(z_prev, np.transpose(layer.weights_matrix)) + layer.b[0]
            z = layer.actfun(a)
            layer.a = a
            layer.z = z
            z_prev = z

        # Restituisce l'array di output Y
        return z_prev


    # Funzione che calcola la i delta per ogni layer
    # per la rete con funzione di errore TSS (Total Sum of Squares.)
    def backpro_tss(self, X, T):

        # Effettua la forward per l'input X
        Y = self.forward(X)

        # Il delta per lo strato di output è
        # uguale a -------> f'(a) * (Y - T)
        # Y è l'array di output, T quello dei labels (e.g. [0, 0, 0, 1])
        self.array_layers[self.n_layers - 1].delta = \
            np.dot(self.array_layers[self.n_layers - 1].actfun_der(Y), (Y - T))

        # Calcola il delta per i layer a ritroso.
        for i in range(self.n_layers -2, -1, -1):

            # delta_temporaneo = W ^ i .* D ^ i+1
            # calcolato con prodotto tra matrici.

            delta = np.dot(self.array_layers[i + 1].delta,
                           self.array_layers[i + 1].weights_matrix)

            # D ^ i = delta_temporaneo .* f'(a)
            self.array_layers[i].delta = np.dot(self.array_layers[i].actfun_der(self.array_layers[i].z),
                                                delta)




