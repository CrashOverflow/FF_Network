# Authors: Ciro Brandi & Marco Urbano.

from workdir import Net as N
import numpy as np
import tensorflow.examples.tutorials.mnist.input_data

print("Started building the Feed Forward Full Connected Neural Network! \n")

my_net = N.Net(5, [5, 4], [1, 1])

#my_net.print()
my_net.forward(np.array([1, 2, 3, 4, 5]))

#print(my_net.array_layers[2 - 1].z)


print("Closing the script, bye bye! \n")

