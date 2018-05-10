from net import *
from layer import *
from array import array
import numpy as np

print("CREAZIONE DELLA RETE")

print("Inserire la dimensione dell'input della rete")
x = int(input())

print("Inserire i layer della rete")
n_layer = int(input())

# creo n_layer di pesi
n_node_prec = x
array_layers = []
for i in range(0, n_layer):
    print("Inserire il numero di nodi per layer")
    n_node = int(input())
    # Matrice dei pesi per ogni layer random tra 0 e 1
    weights_matrix = 1 - 2 * np.random.rand(n_node, n_node_prec)
    # Array dei pesi per il bias random tra 0 e 1
    b = 1 - 2 * np.random.rand(1, n_node_prec)
    # TODO ASSOCIARE FUNZIONE DI ATTIVAZIONE

    # Salvo il numero di nodi del layer precendete
    n_node_prec = n_node


    array_layers.append(layer(weights_matrix, b, "null"))

# STAMPA LE MATRICI dei layer
for j in array_layers:
    j.print_weights_matrix()

#print(b)
#m = np.array([[1, 2, 3]])
#print(m.transpose())
