# Authors: Marco Urbano & Ciro Brandi.

from abc import ABCMeta, abstractmethod
import numpy as np

class StopCriteria(metaclass=ABCMeta):
    strip = 1


class GL(StopCriteria):
    alpha = 0

    def __init__(self, alpha, strip=1):
        self.alpha = alpha
        self.strip = strip

    def stop(self, curr_err, opt_err):
        # Calcolo la generalization loss per ogni epoca
        GL_epoch = 100 * ((curr_err / opt_err) - 1.0)
        print("GL_epoch" + str(GL_epoch) + "\n")

        if GL_epoch > self.alpha:
            return 1

        return 0


class PQ(StopCriteria):
    alpha = 0

    def __init__(self, alpha, strip):
        self.alpha = alpha
        self.strip = strip

    def stop(self, curr_err, opt_err, curr_epoch, err_vect):

        # Calcola Pk
        print(err_vect[curr_epoch - self.strip :curr_epoch])
        print("minimo : " + str(np.min(err_vect[curr_epoch - self.strip: curr_epoch])) + "\n")
        numerator = np.sum(err_vect[curr_epoch - self.strip:curr_epoch])
        denominator = self.strip * np.min(err_vect[curr_epoch - self.strip: curr_epoch])

        PK_epoch = 1000 * (numerator/denominator)

        # Calcola GL
        GL_epoch = np.round(100 * ((curr_err / opt_err) - 1.0), 10)

        curr_PQ = (GL_epoch / PK_epoch)
        print("Curr_GL: " + str(GL_epoch) + "\n")
        print("Curr_PQ: " + str(curr_PQ) + "\n")
        if curr_PQ > self.alpha:
            return 1

        return 0


