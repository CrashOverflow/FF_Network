# Authors: Ciro Brandi & Marco Urbano.

from workdir import Net as N
from workdir import Error as E
import numpy as np


#print("Started building the Feed Forward Full Connected Neural Network! \n")
my_net = N.Net(5, [5, 4], [1, 0], E.TSS())
#my_net.print()



test_X = np.array([1, 2, 3, 4, 5])
test_T = 1 - 2 * np.random.rand(1, 4)

my_net.backprop(test_X, test_T)
my_net.comp_der()

my_net.print()


#print("Closing the script, bye bye! \n")

