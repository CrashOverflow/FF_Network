import tkinter
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
from workdir import Net as N

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

x = mnist.train.images
l = mnist.train.labels
v = mnist.validation
t = mnist.test

# Numero di feature del training set
n_f = np.size(x, 1)
# numero dei neuroni sull'ultimo layer pari alla dimensione delle label
last_num_layer = len(l[0])

my_net = N.Net(n_f, [5, last_num_layer], [1, 1])
n_ts = np.size(x,0)

for i in range(0, n_ts):
    my_net.backpro_tss(x[i], l[i])

#print(my_net.forward(X[0]))
#print(my_net.backpro_tss(x[0], l[0]))
#print(my_net.array_layers[0].delta)
#print(my_net.array_layers[1].delta)

#print(len(X.flatten()))
#print(X.flatten())
#print("Training set\n")
#dim = X.shape
#print(dim)
#print(np.size(X,1))
#print(X)
#print("Label\n")
#print(Y)
#print("Validation set\n")
#print(V)
#print("Test set\n")
#print(T)
"""sn = 1000
amount = 20
lines = 4
columns = 5
image = np.zeros((amount, 28, 28))
number = np.zeros(amount)

for i in range(amount):
    image[i] = mnist.train.images[sn + i].reshape(28, 28)
    label = mnist.train.labels[sn + i]
    number[i] = int(np.where(label == 1)[0])
    # print(number[0])

fig = plt.figure()

for i in range(amount):
    ax = fig.add_subplot(lines, columns, 1 + i)
    plt.imshow(image[i], cmap='binary')
    plt.sca(ax)

plt.xlabel(number)
plt.show() """