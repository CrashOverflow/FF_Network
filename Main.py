# Authors: Ciro Brandi & Marco Urbano.

from workdir import Net as N

print("Started building the Feed Forward Full Connected Neural Network! \n")

print("How many features will have the input? : ")
dim_input = int(input())

print("Insert number of layers: ")
n_layers = int(input())

my_net = N.Net(dim_input, n_layers)

my_net.print()

print("Closing the script, bye bye! \n")

