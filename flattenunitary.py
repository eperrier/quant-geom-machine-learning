import qutip as qt
import numpy as np
def flattenunitary(x1a):
    x1a_re = x1a.data.toarray().real
    x1a_im = x1a.data.toarray().imag
    complex_x1a_top = np.concatenate((x1a_re, -x1a_im), axis=1)
    complex_x1a_bottom = np.concatenate((x1a_im, x1a_re), axis=1)
    complex_x1a = np.vstack((complex_x1a_top, complex_x1a_bottom))
    return complex_x1a.flatten()

def realise(x1a):
    x1a_re = x1a.data.toarray().real
    x1a_im = x1a.data.toarray().imag
    complex_x1a_top = np.concatenate((x1a_re, -x1a_im), axis=1)
    complex_x1a_bottom = np.concatenate((x1a_im, x1a_re), axis=1)
    complex_x1a = np.vstack((complex_x1a_top, complex_x1a_bottom))
    return complex_x1a