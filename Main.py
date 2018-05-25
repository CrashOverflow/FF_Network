# Authors: Ciro Brandi & Marco Urbano.

from workdir import Net as N
from workdir import Error as E
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data


mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

train_set = mnist.train.images
label_train_set = mnist.train.labels
v = mnist.validation
validation_set = mnist.validation.images
label_validation_set = mnist.validation.labels
t = mnist.test

n_f = np.size(train_set, 1)
print(n_f)
label_num = len(label_train_set[0])

#print("Started building the Feed Forward Full Connected Neural Network! \n")
my_net = N.Net(n_f, [5, label_num], [1, 0], E.CrossEntropy())
my_net.online_train(train_set, label_train_set, validation_set, label_validation_set, 0.1, 10)



#test_X = np.array([1, 2, 3, 4, 5])
#test_T = 1 - 2 * np.random.rand(1, 4)

#my_net.backprop(test_X, test_T)
#my_net.compute_derivatives()

#my_net.print()


#print("Closing the script, bye bye! \n")

