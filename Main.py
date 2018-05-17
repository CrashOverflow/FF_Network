# Authors: Ciro Brandi & Marco Urbano.

from workdir import Net as N
import numpy as np


print("Started building the Feed Forward Full Connected Neural Network! \n")

print("How many features will have the input? : ")
dim_input = int(input())

print("Insert number of layers: ")
n_layers = int(input())

my_net = N.Net(dim_input, n_layers)

my_net.print()
my_net.forward(np.array([1, 2, 3, 4, 5]))

#print(my_net.array_layers[n_layers - 1].z)


print("Closing the script, bye bye! \n")

