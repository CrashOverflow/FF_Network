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
GL_alpha = [2, 4, 6, 8, 10, 11, 12, 14, 16, 20]
PQ_alpha = [0.001, 0.002, 0.003, 0.004, 0.005, 0.008, 0.010, 0.011, 0.015, 0.018]
# Al variare dei neuroni.
Net_neurons = [2, 4, 8, 16, 24, 32]


#print("Started building the Feed Forward Full Connected Neural Network! \n")

# Liste per salvare accuratezza media e numero di epoche medie per ogni parametro alpha scelto.
# E' una pair dove (mean_accuracy, mean_epochs) per ogni alpha.

Mean_epochs_accuracy_GL = []
Mean_epochs_accuracy_PQ = []




for i in range(len(GL_alpha)):
    tot_accuracy_GL = 0.0
    tot_epochs_GL = 0.0
    tot_accuracy_PQ = 0.0
    tot_epochs_PQ = 0.0
    for n_neurons in Net_neurons:

        print("GL Alpha = " + str(GL_alpha[i]) + ", PQ Alpha = " + str(PQ_alpha[i]) + ", Neurons = " + str(n_neurons) + "\n")

        for j in range(0, 2):
            print("NETWORK"+str(j)+"\n")
            my_net = N.Net(n_f, [n_neurons, label_num], [1, 0], E.CrossEntropy())
            #new_net = my_net.online_train(train_set[:1000,], label_train_set[:1000,], validation_set[:500,], label_validation_set[:500,], 0.1, 100)
            GL_net = my_net.rprop_train(train_set[:500,], label_train_set[:500,], validation_set[:250,], label_validation_set[:250,], 100,  Stop.GL(GL_alpha[i]))
            PQ_net = my_net.rprop_train(train_set[:500,], label_train_set[:500,], validation_set[:250,], label_validation_set[:250,], 100,  Stop.PQ(PQ_alpha[i], 5))
            GL_net.test(validation_set[:250,], label_validation_set[:250,])
            PQ_net.test(validation_set[:250,], label_validation_set[:250,])
            print("Network GL accuracy : " + str(GL_net.accuracy) + ", Epochs: " + str(GL_net.epoch) + "\n")
            print("Network PQ accuracy : " + str(PQ_net.accuracy) + ", Epochs: " + str(PQ_net.epoch) + "\n")

            # Somma delle epoche e dell'accuratezza per la combinazione n_neurons e in totale su 10 reti.
            tot_accuracy_GL = tot_accuracy_GL + GL_net.accuracy
            tot_accuracy_PQ = tot_accuracy_PQ + PQ_net.accuracy
            tot_epochs_GL = tot_epochs_GL + GL_net.epoch
            tot_epochs_PQ = tot_epochs_PQ + PQ_net.epoch

    Mean_epochs_accuracy_GL.append((tot_accuracy_GL / (len(Net_neurons) * 2), tot_epochs_GL / (len(Net_neurons) * 2)))
    Mean_epochs_accuracy_PQ.append((tot_accuracy_PQ / (len(Net_neurons) * 2), tot_epochs_PQ / (len(Net_neurons) * 2)))

    print("GL: Mean epochs accuracy = " + str(Mean_epochs_accuracy_GL[i]) + "\n")
    print("PQ: Mean epochs accuracy = " + str(Mean_epochs_accuracy_PQ[i]) + "\n")



