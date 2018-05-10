# Authors: Ciro Brandi & Marco Urbano.

from Net import *
from Layer import *
from array import array
import numpy as np

print("Started building the Feed Forward Full Connected Neural Network! \n")

print("Insert number of features the input will have: ")
x = int(input())

print("Insert number of layers: ")
n_layers = int(input())

# Creazione di n layers di pesi.
n_node_prec = x
array_layers = []

for i in range(0, n_layers):
    print("Number of neurons for layer " + str(i) + ":")
    n_nodes = int(input())

    # Matrice dei pesi per ogni layer random tra 0 e 1
    weights_matrix = 1 - 2 * np.random.rand(n_nodes, n_node_prec)

    # Array dei pesi per il bias random tra 0 e 1
    b = 1 - 2 * np.random.rand(1, n_node_prec)


    # Salvo il numero di nodi del layer precendete
    n_node_prec = n_nodes

    # Aggiungo il layer appena creato al vettore dei layer che sarà
    # aggiunto alla rete.
    array_layers.append(Layer(weights_matrix, b, "null"))


# Stampa le matrici del layer
print("Printing the weight matrixes for every layer: \n")

for j in array_layers:
    print("Weight matrix for layer " + str(i) + ": \n")
    j.print_weights_matrix()
    print("\n")

print("Closing the script, bye bye! \n")

