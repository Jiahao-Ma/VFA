import numpy as np
import math

def gaussian_label(label, num_class, u=0, sigma=4.0):
    label = int(label)
    x = np.array(range(math.floor(-num_class/2), math.ceil(num_class/2)))
    y = np.exp( -(x - u)**2 / (2 * sigma**2) )
    return np.concatenate([y[math.ceil(num_class/2)-label:],
                          y[:math.ceil(num_class/2)-label]], axis=0)

def rectangular_label(label, num_class, raduius=4):
    label = int(label)
    x = np.zeros([num_class])
    x[:raduius+1] = 1
    x[-raduius:] = 1
    y_sig = np.concatenate([x[-label:], x[:-label]], axis=0)
    return y_sig


def pulse_label(label, num_class):
    label = int(label)
    x = np.zeros([num_class])
    x[label] = 1
    return x


def triangle_label(label, num_class, raduius=4):
    label = int(label)
    y_sig = np.zeros([num_class])
    x = np.array(range(raduius+1))
    y = -1/(raduius+1) * x + 1
    y_sig[:raduius+1] = y
    y_sig[-raduius:] = y[-1:0:-1]

    return np.concatenate([y_sig[-label:], y_sig[:-label]], axis=0)

if __name__ == '__main__':
    cls_label = gaussian_label(30, 360)
    print(cls_label.shape)