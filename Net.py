from workdir import Layer as l
from workdir import Error as Error
import numpy as np
import matplotlib.pyplot as plt
import copy as cp


# Classe che rappresenta una singola rete Feed-Forward Full Connected.
# La funzione di errore è stata implementata utilizzando il Behavioral Pattern STRATEGY:
# per scegliere una particolare funzione di errore basta fornire un oggetto di classe ErrorF.
class Net():

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
    def __init__(self, n_f, neurons_list, actfun_list, error):
        try:
            if(len(neurons_list) == len(actfun_list)):
                # Strategia per il calcolo dell'errore.
                self.error = error
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
                    if actfun_list[i] == 0:
                        self.array_layers.append(l.Layer_i(n_nodes, n_prev))
                    elif actfun_list[i] == 1:
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
            print("delta: ")
            lay.print_delta()
            i = i + 1

    # Forward propagation per l'input x
    def forward(self, x):
        # Aggiunge una dimensione al vettore e lo traspone per farlo diventare
        # vettore colonna.
        z_prev = np.transpose(np.expand_dims(x, axis=0))

        for layer in self.array_layers:


            # l'array risultante tra la somma del bias con la combinazione lineare.

            # (W * Z ^ T) + b
            # Ora Z sarà visto come vettore colonna.
            a = np.dot(layer.weights_matrix, z_prev) + layer.b
            z = layer.actfun(a)
            layer.a = a
            layer.z = z
            z_prev = z

        # Restituisce l'array di output Y
        return z_prev

    # Funzione che calcola la i delta per ogni layer.
    # E' indipendente dalla funzione di Errore e di Attivazione scelta.
    def backprop(self, X, T):

        # Traspongo T poichè è visto come vettore riga
        T = np.transpose(np.expand_dims(T, axis=0))
        # Effettua la forward per l'input X
        Y = self.forward(X)

        # Il delta per lo strato di output è
        # uguale a -------> f'(a) * (Y - T)
        # Y è l'array di output, T quello dei labels (e.g. [0, 0, 0, 1])
        a = self.array_layers[self.n_layers - 1].a
        self.array_layers[self.n_layers - 1].delta = \
            self.array_layers[self.n_layers - 1].actfun_der(a) * self.error.fun(Y, T)


        # Calcola il delta per i layer a ritroso.
        for i in range(self.n_layers -2, -1, -1):

            # delta_temporaneo = W(i+1) ^ T  .* D(i+1) ---> trasposto perchè deve tornare vettore riga
            # calcolato con prodotto tra matrici.
            delta = np.dot(
                np.transpose(self.array_layers[i + 1].weights_matrix), self.array_layers[i+1].delta)

            # D ^ i = delta_temporaneo .* f'(a)
            #attiv = self.array_layers[i].actfun_der(self.array_layers[i].a)
            self.array_layers[i].delta = self.array_layers[i].actfun_der(self.array_layers[i].a) * delta


    # Calcola le derivate per ogni livello.
    def compute_derivatives(self, X):
        for i in range(0, self.n_layers - 1):
            # W' = delta * Z (nel primo layer Z = X)
            if i == 0:
                self.array_layers[i].der_w = \
                        np.dot(self.array_layers[i].delta, np.expand_dims(X, axis = 0))
            else:
                self.array_layers[i].der_w = \
                    np.dot(np.transpose(np.expand_dims(self.array_layers[i].delta, axis=0)),
                           np.expand_dims(self.array_layers[i - 1].z, axis = 0))

            # b' = delta
            self.array_layers[i].der_b = self.array_layers[i].delta


    # Aggiornamento dei pesi
    def update_weights(self, eta):
        for l in self.array_layers:
            l.weights_matrix = l.weights_matrix - (eta * l.der_w)
            l.b = l.der_b - (eta * l.der_b)



    # Train della rete.
    # Parametri:
    #    - X : training set
    #    - T : labels del training set
    #    - V : validation test
    #    - eta: learning rate
    #    - epoch: numero di epoche.
    def online_train(self, X, x_label, V, v_label, eta, epoch):
        x_size = np.size(X, 0) // 100
        v_size = np.size(V, 0) // 50

        x_err_array = []
        v_err_array = []

        curr_net = cp.deepcopy(self);
        prev_net = cp.deepcopy(self);

        for i in range(0, epoch-1):
            print("Current status: EPOCH " + str(i) + "\n")
            for j in range(0, x_size):
                curr_net.backprop(X[j], x_label[j])
                curr_net.compute_derivatives(X[j])
                curr_net.update_weights(eta)

            # Calcolo dell'errore sul training set.
            err_X = 0
            for j in range(0, x_size):
                Y = curr_net.forward(X[j])

                # Controllo per applicare softmax in caso di CrossEntropy.
                if(isinstance(curr_net.error, Error.CrossEntropy)):
                    Y = curr_net.error.softmax(Y)

                err_X = err_X + self.error.compute_error(Y, np.transpose(np.expand_dims(x_label[j],axis=0)))
            # Aggiorna l'errore nel vettore degli errori per stamparlo
            x_err_array.append(err_X)

            # Calcolo dell'errore sul validation set
            err_V = 0
            for j in range(0, v_size):
                Y = curr_net.forward(V[j])
                # Controllo per applicare softmax in caso di CrossEntropy.
                if (isinstance(curr_net.error, Error.CrossEntropy)):
                    Y = curr_net.error.softmax(Y)

                err_V = err_V + self.error.compute_error(Y, np.transpose(np.expand_dims(v_label[j],axis=0)))

            # Aggiorna l'errore nel vettore degli errori per stamparlo
            v_err_array.append(err_V)

            if i != 0:
                if err_V > v_err_array[i-1]:
                    break
                else:
                    print(err_V)
                    print(v_err_array[i-1])
                    prev_net = cp.deepcopy(curr_net)




        # Plotta
        plt.plot(x_err_array)
        plt.xlabel('Epoch')
        plt.ylabel('Error')
        plt.show()

        plt.plot(v_err_array)
        plt.xlabel('Epoch')
        plt.ylabel('Error')
        plt.show()

        # Ritorna la rete neurale con errore minore.
        return prev_net

    # Funzione di testing della rete
    def test(self, X, X_labels):
        x_len = len(X) // 100
        print(x_len)
        correct = 0
        for i in range(0, x_len):
            Y_p = self.forward(X[i])
            Y_p = self.error.softmax(Y_p)
            # Mette 1 nella posizione con probabilità massima e
            # tutti 0 nelle altre.
            Y_v = np.zeros(np.shape(Y_p))
            Y_v[np.argmax(Y_p)] = 1

            if np.all(np.transpose(np.expand_dims(X_labels[i], axis=0)) == Y_v):
                correct = correct + 1

        return correct





