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
test_set = mnist.test.images
label_test_set = mnist.test.labels
n_f = np.size(train_set, 1)
label_num = len(label_train_set[0])

# Fornisco 10 valori diversi di alpha da fornire sia a GL che PQ per vedere come si comportano.
# per ogni alpha si effettua il test su rete e si restituisce la coppia (accuratezza media, media epoche).


#print("Started building the Feed Forward Full Connected Neural Network! \n")
for i in range(0,10):
    #print("NETWORK"+str(i)+"\n")
    my_net = N.Net(n_f, [8, label_num], [1, 0], E.CrossEntropy())
    #new_net = my_net.online_train(train_set[:1000,], label_train_set[:1000,], validation_set[:500,], label_validation_set[:500,], 0.1, 100)
    new_net = my_net.rprop_train(train_set[:500,], label_train_set[:500,], validation_set[:250,], label_validation_set[:250,], 100,  Stop.PQ(0.001, 2))
    new_net.test(validation_set[:250,], label_validation_set[:250,])
    print("Network " + str(i) + " accuracy : " + str(new_net.accuracy) + ", Epochs: " + str(new_net.epoch) + "\n")
    #print("Network " + str(i) + ": \n")
    #print("Accuracy of trained network on Validation: " + str(new_net.test(validation_set[:250,], label_validation_set[:250,])) + "\n")
    #print("Accuracy of trained network on Test: "+ str(new_net.test(test_set[:250,], label_test_set[:250,])) + "\n")



