# Authors: Ciro Brandi & Marco Urbano.

from workdir import Net as N
from workdir import Error as E
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from workdir import StopCriteria as Stop


mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

train_set = mnist.train.images
label_train_set = mnist.train.labels
validation_set = mnist.validation.images
label_validation_set = mnist.validation.labels
#t = mnist.test

test_set = mnist.test.images
label_test_set = mnist.test.labels

n_f = np.size(train_set, 1)
print(n_f)
label_num = len(label_train_set[0])

#print("Started building the Feed Forward Full Connected Neural Network! \n")
for i in range(0,10):
    print("NETWORK"+str(i)+"\n")
    my_net = N.Net(n_f, [8, label_num], [1, 0], E.CrossEntropy())
    #new_net = my_net.online_train(train_set[:1000,], label_train_set[:1000,], validation_set[:500,], label_validation_set[:500,], 0.1, 100)
    new_net = my_net.rprop_train(train_set[:500,], label_train_set[:500,], validation_set[:250,], label_validation_set[:250,], 100,  Stop.GL(5, 5))
    print("Network " + str(i) + ": \n")
    print("Accuracy of trained network on Validation: " + str(new_net.test(validation_set[:250,], label_validation_set[:250,])) + "\n")
    print("Accuracy of trained network on Test: "+ str(new_net.test(test_set[:250,], label_test_set[:250,])) + "\n")


#my_net.print()
#print("Accuracy of trained network on Validation: "+ str(new_net.test(validation_set, label_validation_set)) +"\n")
#print("Accuracy of trained network on Training: "+ str(new_net.test(train_set, label_train_set)) +"\n")


#my_net = N.Net(5, [5, 4], [1, 0], E.CrossEntropy())
#test_X = np.array([1, 2, 3, 4, 5])
#print(test_X)
#test_T = np.array([0.1, 0.9, -0.3, 0.5])

#my_net.backprop(test_X, test_T)
#my_net.compute_derivatives()
#my_net.update_weights(0.1)
#my_net.print()


#print("Closing the script, bye bye! \n")

