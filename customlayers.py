from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers,optimizers,constraints,initializers,Model,backend
import pickle
import time
import zipfile    
import os
from itertools import accumulate

import pickle

import qutip as qt
import pandas as pd
import scipy as sc
import random as rn
import os
import itertools as itr
from itertools import combinations, accumulate
from operator import itemgetter
import operator
from functools import reduce
import csv

from tensorflow.python.ops import control_flow_ops

from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, LSTM, LSTMCell, GRU
from tensorflow.keras.callbacks import CSVLogger
from numpy import array
from tensorflow.keras import backend as K
from tensorflow.keras.regularizers import l2
from tensorflow.keras import Model


from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits import mplot3d
from matplotlib import animation
from IPython.display import HTML, Image
from flattenunitary import flattenunitary
from flattenunitary import realise
from dataprocess import dataprocess
from simulation import GenData

'''========================CUSTOM LAYERS===================
Contains custom layers called by various QML models.'''

'''========================Layer: [] ========================
Description: 
Location used:
'''

class unitary_est(layers.Layer):
    
    def __init__(self, su2ndelta, segments, **kwargs):
        super(unitary_est, self).__init__(**kwargs)
        self.su2ndelta = su2ndelta
        self.su2n_dim = self.su2ndelta[0].shape[0]
    
    def build(self, input_shape):
        super(unitary_est, self).build(input_shape)
    
    def call(self, x):
        self.Hlist = []
        for i in range(len(self.su2ndelta)):
            sigma_x = self.su2ndelta[i]
            temp_shape = tf.concat( [tf.shape(x)[0:1],tf.constant(np.array([1,1],dtype=np.int32))],0 )
            sigma_x = tf.expand_dims(sigma_x,0)
            sigma_x = tf.tile(sigma_x, temp_shape)
            h_x = tf.cast(x[:,i:(i+1)], dtype=tf.complex128)
            temp_shape = tf.constant(np.array([1,self.su2n_dim,self.su2n_dim],dtype=np.int32))
            h_x = tf.expand_dims(h_x,-1)
            h_x = tf.tile(h_x, temp_shape)
            Htmp = tf.multiply(sigma_x, h_x) 
            self.Hlist.append(Htmp)
        
        H = sum(self.Hlist)
        return tf.linalg.expm(-H) 


'''========================Layer: Hamiltonian estimation layer ========================
Description: custom layer that estimates \Lambda_0+
Location used: QMLcontrolmodel - SubRiemannian model
Inputs:
- positive control coefficients
- \Delta (distribution) of generators
Outputs: estimated \Lambda_0+ applying controls to generators

Note: in FC Greybox, initial Hamiltonian is just U0 @ \Lambda_0 @ U0.dag(),
but U0 is just the identity, so the Hamiltonian is just \Lambda_0.
As such, generators are drawn from su2n not just \Delta'''

class ham_est(layers.Layer):
    '''Note: while su2n is named as an argument here, in the code the input will be either su2n or su2ndelta
    depending on whether the full Lie algebra or distribution is used to generate \Lambda_0.'''
    
    def __init__(self, su2n, segments, **kwargs):
        super(ham_est, self).__init__(**kwargs)
        self.su2n = su2n
        self.su2n_dim = self.su2n[0].shape[0]
    
    def build(self, input_shape):
        super(ham_est, self).build(input_shape)
    
    def call(self, x):
        self.Hlist = []
        for i in range(len(self.su2n)):
            sigma_x = self.su2n[i]
            temp_shape = tf.concat( [tf.shape(x)[0:1],tf.constant(np.array([1,1],dtype=np.int32))],0 )
            sigma_x = tf.expand_dims(sigma_x,0)
            sigma_x = tf.tile(sigma_x, temp_shape)
            h_x = tf.cast(x[:,i:(i+1)], dtype=tf.complex128)
            temp_shape = tf.constant(np.array([1,self.su2n_dim,self.su2n_dim],dtype=np.int32))
            h_x = tf.expand_dims(h_x,-1)
            h_x = tf.tile(h_x, temp_shape)
            Htmp = tf.multiply(sigma_x, h_x) 
            self.Hlist.append(Htmp)
        
        H = sum(self.Hlist)
        return H 

'''========================Layer: Hamiltonian estimation layer ========================
Description: custom layer that estimates \Lambda_0-
Location used: QMLcontrolmodel - SubRiemannian model
Inputs:
- negative control coefficients
- \Delta (distribution) of generators
Outputs: estimated \Lambda_0- applying controls to generators'''

class ham_est_neg(layers.Layer):
    '''Note: while su2n is named as an argument here, in the code the input will be either su2n or su2ndelta
    depending on whether the full Lie algebra or distribution is used to generate \Lambda_0.'''
    def __init__(self, su2n, segments, **kwargs):
        super(ham_est_neg, self).__init__(**kwargs)
        self.su2n = su2n
        self.su2n_dim = self.su2n[0].shape[0]
    
    def build(self, input_shape):
        super(ham_est_neg, self).build(input_shape)
    
    def call(self, x):
        self.Hlist = [] 
        for i in range(len(self.su2n)):
            sigma_x = self.su2n[i]
            temp_shape = tf.concat( [tf.shape(x)[0:1],tf.constant(np.array([1,1],dtype=np.int32))],0 )
            sigma_x = tf.expand_dims(sigma_x,0)
            sigma_x = tf.tile(sigma_x, temp_shape)
            h_x = tf.cast(x[:,i:(i+1)], dtype=tf.complex128)
            temp_shape = tf.constant(np.array([1,self.su2n_dim,self.su2n_dim],dtype=np.int32))
            h_x = tf.expand_dims(h_x,-1)
            h_x = tf.tile(h_x, temp_shape)
            Htmp = tf.multiply(sigma_x, h_x) 
            self.Hlist.append(Htmp)
        
        H = sum(self.Hlist)
        return -H 

'''========================Layer: Hamiltonian estimation layer ========================
Description: custom layer that combines \Lambda_0+ + \Lambda_0- into \Lambda_0
Location used: QMLcontrolmodel - SubRiemannian model
Inputs:
-  \Lambda_0+, \Lambda_0-
Outputs: estimated \Lambda_0'''

class hamcomb(layers.Layer):
    
    def __init__(self, **kwargs):
        super(hamcomb,self).__init__(**kwargs)
        
    def build(self, input_shape):
        super(hamcomb, self).build(input_shape)
    
    def call(self, x):
        H1 = x[0]
        H2 = x[1]
        H = H1 + H2
        return H


    
'''========================Layer: Hamiltonian estimation layer ========================
Description: custom layer that estimates Hamiltonian
Location used: deprecated (previous iteration of model)
Hamiltonian estimation layers for FC simple:
ham_est: estimates Hamiltonian
ham_est_neg: provides estimate of Hamiltonian with negative coefficients
ham_comb: combines ham_est and ham_est_neg for final Hamiltonian
Note: in FC simple, Hamiltonian is just c_i \tau^i i.e. the generators themselves.
The layers work in the NN by just replicating \Delta generators in a tensor, converting the
coefficients to identically-shaped tensors, multiplying (elementwise) all at once,
then exponentiating.'''


class ham_est_neg_simple(layers.Layer):
   
    def __init__(self, su2n, segments, **kwargs):
        super(ham_est_neg, self).__init__(**kwargs)
        self.su2n = su2n
        self.su2n_dim = self.su2n[0].shape[0]
    
    def build(self, input_shape):
        super(ham_est_neg, self).build(input_shape)
    
    def call(self, x):
        self.Hlist = []
        
        for i in range(len(self.su2n)):
            sigma_x = self.su2n[i]
            temp_shape = tf.concat( [tf.shape(x)[0:1],tf.constant(np.array([1,1],dtype=np.int32))],0 )
            sigma_x = tf.expand_dims(sigma_x,0)
            sigma_x = tf.tile(sigma_x, temp_shape)
            h_x = tf.cast(x[:,i:(i+1)], dtype=tf.complex128)
            temp_shape = tf.constant(np.array([1,self.su2n_dim,self.su2n_dim],dtype=np.int32))
            h_x = tf.expand_dims(h_x,-1)
            h_x = tf.tile(h_x, temp_shape)
            Htmp = tf.multiply(sigma_x, h_x) 
            self.Hlist.append(Htmp)
        
        H = sum(self.Hlist)
        return -H 


'''========================Layer: Hamiltonian estimation layer ========================
Description: custom layer that estimates Hamiltonian
Location used: deprecated (previous iteration of model)'''

class hamcomb_simple(layers.Layer):
    
    def __init__(self, **kwargs):
        super(hamcomb,self).__init__(**kwargs)
        
    def build(self, input_shape):
        super(hamcomb, self).build(input_shape)
    
    def call(self, x):
        H1 = x[0]
        H2 = x[1]
        H = H1 + H2
        return H

    
'''========================Layer: Unitary estimation layer ========================
Description: Layer to generate unitaries from Hamiltonians.
Location used: deprecated (previous iteration of model)'''

class unitarycalc(layers.Layer):
    '''
    x[i]: input Hamiltonians from Hamiltonian estimation layer(s)
    '''
    
    def __init__(self, **kwargs):
        super(unitarycalc,self).__init__(**kwargs)
        
    def build(self, input_shape):
        super(unitarycalc, self).build(input_shape)
    
    def call(self, x):
        H1 = x[0]
        H2 = x[1]
        H = H1 + H2
        U = tf.linalg.expm(-H)
        return U

'''========================Layer: Unitary estimation layer ========================
Description: Layer to generate unitaries from Hamiltonians.
Location used: deprecated (previous iteration of model)'''

class unitarycalcproj(layers.Layer):
    '''Layer to generate unitaries from Hamiltonians.'''
    
    '''
    x[i]: input Hamiltonians from Hamiltonian estimation layer(s)
    '''
    
    def __init__(self, **kwargs):
        super(unitarycalcproj,self).__init__(**kwargs)
        
    def build(self, input_shape):
        super(unitarycalcproj, self).build(input_shape)
    
    def call(self, x):
        U = tf.linalg.expm(-x)
        return U

'''========================Layer: Unitary estimation layer ========================
Description: Layer to generate unitaries from Hamiltonians.
Location used: deprecated (previous iteration of model)'''

class unitary_est_comb(layers.Layer):
    
    def __init__(self, su2ndelta, segments, **kwargs):
        super(unitary_est_comb, self).__init__(**kwargs)
        self.su2ndelta = su2ndelta
        self.su2n_dim = self.su2ndelta[0].shape[0]
    
    def build(self, input_shape):
        super(unitary_est_comb, self).build(input_shape)
    
    def call(self, x):
        self.Hlist = []
        for i in range(2):
            sigma_x = self.su2ndelta[i]
            temp_shape = tf.concat( [tf.shape(x)[0:1],tf.constant(np.array([1,1],dtype=np.int32))],0 )
            sigma_x = tf.expand_dims(sigma_x,0)
            sigma_x = tf.tile(sigma_x, temp_shape)
            h_x = tf.cast(x[:,i:(i+1)], dtype=tf.complex128)
            temp_shape = tf.constant(np.array([1,self.su2n_dim,self.su2n_dim],dtype=np.int32))
            h_x = tf.expand_dims(h_x,-1)
            h_x = tf.tile(h_x, temp_shape)
            Htmp = tf.multiply(sigma_x, h_x) #+ tf.multiply(sigma_y, h_y) #+ self.sigma_z
            self.Hlist.append(Htmp)
        
        H = sum(self.Hlist)
        U1 = tf.linalg.expm(-H)
        self.Hlist = []
        for i in range(2,4):
            sigma_x = self.su2ndelta[i]
            temp_shape = tf.concat( [tf.shape(x)[0:1],tf.constant(np.array([1,1],dtype=np.int32))],0 )
            sigma_x = tf.expand_dims(sigma_x,0)
            sigma_x = tf.tile(sigma_x, temp_shape)
            h_x = tf.cast(x[:,i:(i+1)], dtype=tf.complex128)
            temp_shape = tf.constant(np.array([1,self.su2n_dim,self.su2n_dim],dtype=np.int32))
            h_x = tf.expand_dims(h_x,-1)
            h_x = tf.tile(h_x, temp_shape)
            Htmp = tf.multiply(sigma_x, h_x) #+ tf.multiply(sigma_y, h_y) #+ self.sigma_z
            self.Hlist.append(Htmp)
        H = sum(self.Hlist)
        U2 = tf.linalg.expm(-H)
        return U2 @ U1 



'''Note: input layer should be real, input .real and .imag and then add inside layer'''

'''========================Layer: Fidelity estimation layer ========================
Description: Layer to estimate fidelities from unitarys.
    x[0]: unitary estimated from ham_est using the learnt control amplitudes multiplied
    by the generators (complex-valued)
    x[1]: the real part of the related Hamiltonian from the training set z0
    x[2]: the imaginary part of the related Hamiltonian from the training set z0
Location used: deprecated (previous iteration of model).'''

class fidmulti(layers.Layer):
    def __init__(self, su2ndim, **kwargs):
        super(fidmulti,self).__init__(**kwargs)
        self.su2ndim = su2ndim
    
    def build(self, input_shape):
        super(fidmulti, self).build(input_shape)
    
    def call(self, x):
        U_z0real = tf.cast(x[2],dtype=tf.complex128)
        U_z0imag = tf.cast(x[3],dtype=tf.complex128)
        U1 = x[0]
        U2 = x[1]
        Umulti = tf.multiply(U2,U1)
        Umulti = tf.linalg.normalize(Umulti)
        U = tf.math.add(U_z0real, 1j*U_z0imag)
        fid = tf.square(tf.abs(tf.linalg.trace(tf.matmul(Umulti,U, adjoint_a=True))))
        return fid/(self.su2ndim ** 2)



'''========================Layer: Fidelity estimation layer ========================
Description: Layer to estimate fidelities from unitarys.
Location used: deprecated (previous iteration of model)'''

class fidelity(layers.Layer):
    '''Layer to calculate fidelity of estimate versus target unitary.'''
    
    '''
    x[0]: unitary estimated from ham_est using the learnt control amplitudes multiplied
    by the generators (complex-valued)
    x[1]: the real part of the related Hamiltonian from the training set z0
    x[2]: the imaginary part of the related Hamiltonian from the training set z0
    '''
    
    def __init__(self, su2ndim, **kwargs):
        super(fidelity,self).__init__(**kwargs)
        self.su2ndim = su2ndim
    def build(self, input_shape):
        super(fidelity, self).build(input_shape)
    
    def call(self, x):
        U_z0real = tf.cast(x[1],dtype=tf.complex128)
        U_z0imag = tf.cast(x[2],dtype=tf.complex128)
        u_est = x[0]
        U = tf.math.add(U_z0real, 1j*U_z0imag)
        fid = tf.square(tf.abs(tf.linalg.trace(tf.matmul(U,u_est, adjoint_a=True))))
        tf.print('fid dim')
        tf.print(fid.shape)
        return fid/(self.su2ndim ** 2)


'''========================Layer: Fidelity estimation layer ========================
Description: Layer to estimate fidelities from unitarys.'''
'''Layer to calculate time (amplitude) of control pulses - using 
bang bang control approximation (where we can trade off duration and amplitude
of pulses without penalty) i.e. we use partitioned non-time dependent
Schrodinger equation approxmations

Idea is to minimise the sum of the coefficients i.e. controls by doing so
we are minimising the evolution time (see Boozer (2012); Khaneja (2001) etc).'''

'''The layer outputs the sum of coefficients '''

'''
x[0]: unitary estimated from ham_est using the learnt control amplitudes multiplied
by the generators (complex-valued)
x[1]: the real part of the related Hamiltonian from the training set z0
x[2]: the imaginary part of the related Hamiltonian from the training set z0
Location used: deprecated (previous iteration of model).
    '''
class pulsetime(layers.Layer):
    
    
    def __init__(self,coef_numb):
        super(pulsetime,self).__init__()
        self.coef_numb = coef_numb
        
    def build(self, input_shape):
        super(pulsetime, self).build(input_shape)
    
    def call(self, x):
        self.csumlist = []
        for i in range(self.coef_numb):
            h_x = tf.cast(x[:,i:(i+1)], dtype=tf.float64)
            self.csumlist.append(h_x)
        coef_sum = sum(self.csumlist)
        return coef_sum


'''========================Layer: Conjugation layer ========================
Description: Layer to conjugate \Lambda_0.
Location used: deprecated (previous iteration of model)'''



class conjugacyinitial(layers.Layer):
    def __init__(self, su2nidentity):
    
        super(conjugacyinitial, self).__init__()
        self.su2nidentity = su2nidentity
        
    
    def build(self, input_shape):
        super(conjugacyinitial, self).build(input_shape)
    
    def call(self, x):
        Uj = tf.expand_dims(self.su2nidentity,0)
        Lambda0 = x
        conjL = tf.matmul(Lambda0, Uj)
        conjL = tf.matmul(Uj,conjL)
        return conjL

'''========================Layer: Conjugation layer ========================
Description: Layer to conjugate \Lambda_0.
Location used: deprecated (previous iteration of model)'''


class conjugacy(layers.Layer):
    def __init__(self, **kwargs):
    
        super(conjugacy, self).__init__(**kwargs)
        
    
    def build(self, input_shape):
        super(conjugacy, self).build(input_shape)
    
    def call(self, x):
        Lambda0 = x[0]
        Uj = x[1]
        Ujdag = tf.linalg.adjoint(Uj)
        conjL = tf.matmul(Lambda0, Ujdag)
        conjL = tf.matmul(Uj,conjL)
        return conjL



'''========================Layer: Projection layer ========================
Description: Layer to implement projection function (see paper).
Location used: deprecated (previous iteration of model)'''


class projection(layers.Layer):
    def __init__(self, su2ndelta, segments, **kwargs):
        super(projection, self).__init__(**kwargs)
        self.su2ndelta = su2ndelta
        self.su2n_dim = self.su2ndelta[0].shape[0]
    
    def build(self, input_shape):
        super(projection, self).build(input_shape)
    
    def call(self, x):
        self.Hlist = []
        for i in range(len(self.su2ndelta)):
            sigma_x = self.su2ndelta[i]
            sigma_x = tf.expand_dims(sigma_x,0)
            mmul = tf.linalg.matmul(x,sigma_x)
            tautrace = tf.linalg.trace(mmul)
            dim = tf.shape(tautrace)[0]
            sigma_x = self.su2ndelta[i]
            sigma_x = tf.expand_dims(sigma_x,-1)
            Htmp = tf.transpose(x, perm=[0, 1 ,2])
            self.Hlist.append(Htmp)
        
        H = sum(self.Hlist)
        return -H 


'''========================Layer: Projection layer ========================
Description: Layer to implement recursive construction of (U_j).
Location used: SubRiemannian model.
Inputs: 
- su2ndelta: generator set
- su2nidentity: identity operator
- segments: scalar number of segments
- h: time-step

Outputs:
- Ujlist: list (Uj)
- taulistmain: list of coefficients for each Hamiltonian used to generate each Uj
'''


class identitylayer(layers.Layer):
    def __init__(self, su2ndelta, su2nidentity, segments, h,**kwargs):
        super(identitylayer, self).__init__(**kwargs)
        self.su2ndelta = su2ndelta
        self.su2n_dim = self.su2ndelta[0].shape[0]
        self.su2nidentity = su2nidentity #'''INCLUDE IN MODEL CLASS'''
        self.currentUj = su2nidentity
        self.segments = segments #'''FIX UP TO REMOVE SELFS DEFINED IN HERE, JUST USE CLASS VARS'''
        self.h = h
    def build(self, input_shape):
        super(identitylayer, self).build(input_shape)
    
    def conjugacy(self, Lambda0, Uj):
        '''Description: Takes Lambda0 and Uj as inputs, returns conjugation
        Uj @ Lambda0 @ Uj.dag (where @ indicates matmul). This becomes input to
        projection function.'''
        Lambda0 = tf.cast(Lambda0, dtype = tf.complex128)
        Ujdag = tf.linalg.adjoint(Uj)
        conjL = Uj @ Lambda0 @ Ujdag
        return conjL
    
    def project(self, x):
        Hlist = []
        taulist = []
        for i in range(len(self.su2ndelta)):
            sigma_x = tf.convert_to_tensor(self.su2ndelta[i])
            sigma_x = tf.expand_dims(sigma_x,0)
            mmul = tf.linalg.matmul(x,sigma_x)
            tautrace = tf.linalg.trace(mmul)
            '''Note: taulist here saves the coefficients from the projection function
            which are the control amplitudes (which need to be multiplied by delta_t i.e.
            h in order to render the control pulse over time delta_t)'''
            taulist.append(tautrace)
            temp_shape = tf.constant(np.array([1,self.su2n_dim,self.su2n_dim],dtype=np.int32))
            tautrace = tf.expand_dims(tautrace,-1)
            tautrace = tf.expand_dims(tautrace,-1)
            tautrace = tf.tile(tautrace, temp_shape)
            dim = tf.shape(tautrace)[0]
            sigma_x = tf.convert_to_tensor(self.su2ndelta[i])
            sigma_x = tf.expand_dims(sigma_x,0)
            Htmp = tf.multiply(tautrace,  sigma_x)
            Hlist.append(Htmp)
        H = sum(Hlist)
        return [H,taulist]
    
    def evolve(self, x):
        '''Note: recall imaginary unit is already included when the simulation
        generates the distribution \Delta'''
        self.currentUj =  tf.linalg.expm(-self.h * x)
        return self.currentUj
    
    def call(self, x):
        Lambda0 = tf.cast(x, dtype=tf.complex128)
        U = self.su2nidentity
        Ujlist = []
        taulistmain = []
        for i in range(self.segments):
            conjL = self.conjugacy(Lambda0, U)
            proj = self.project(conjL)
            projH = proj[0]
            tau1 = proj[1]
            taulistmain.append(tau1)
            Uj = self.evolve(projH)
            Ujlist.append(Uj)
            U = Uj @ U
            
        Ujlist = tf.convert_to_tensor(Ujlist)
        Ujlist = tf.transpose(Ujlist, [1,0,2,3])
        Ujlist = tf.reverse(Ujlist, [1])
        
        taulistmain = tf.convert_to_tensor(taulistmain)
        taulistmain = tf.transpose(taulistmain, [2,0,1])
        taulistmain = tf.reverse(taulistmain,[1])
        #tf.print(taulistmain.shape)
        return [Ujlist,taulistmain]
        #return Ujlist




'''========================Layer: Fidelity estimation layer ========================
Description: Layer to calculate batch fidelity of estimated (\hat{U}_j) v. ground-truth (U_j).
Location used: SubRiemannian model, GRU RNN Greybox model, Fully-connected (FC) Greybox model.
Inputs: 
- x[0]: unitary (\hat{U}_j) estimated from quantum evolution 
- x[1]: the real part of the related (Uj) from the training set (Uj)
- x[2]: the imaginary part of the related (Uj) from the training set (Uj)

Outputs:
- Ujlist: list (Uj)
- taulistmain: list of coefficients for each Hamiltonian used to generate each Uj
'''

class fidmulti_unitary(layers.Layer):
    
    def __init__(self, su2ndim, **kwargs):
        super(fidmulti_unitary,self).__init__(**kwargs)
        self.su2ndim = su2ndim
    def build(self, input_shape):
        super(fidmulti_unitary, self).build(input_shape)
    
    def call(self, x):
        Ure = tf.cast(x[0], dtype = tf.complex128)
        Uim = tf.cast(x[1], dtype = tf.complex128)
        Utarget = Ure + 1j*Uim
        Uest = x[2]
        fid = tf.square(tf.abs(tf.linalg.trace(tf.matmul(Uest,Utarget, adjoint_a=True))))/((self.su2ndim ** 2))
        return fid
        

'''========================Layer: Hamiltonian estimation layer ========================
Description: Layer to generate Hamiltonians in FC Greybox model.
Location used: FC Greybox model'''

class ham_est_simple(layers.Layer):
    '''Description: output estimate of Hamiltonian.
    Takes as input coefficients of generators in su2n in order to generate \Lambda_0'''
    def __init__(self, su2ndelta, segments, **kwargs):
        super(ham_est_simple, self).__init__(**kwargs)
        self.su2ndelta = tf.convert_to_tensor(su2ndelta)
        self.segments = segments
        self.su2nda = tf.stack([self.su2ndelta] * self.segments)
        self.su2n_dim = self.su2ndelta[0].shape[0]
        self.su2ndeltashp = self.su2ndelta.shape[0]
        self.dimseg = self.su2ndeltashp * self.segments
        
    
    def build(self, input_shape):
        super(ham_est_simple, self).build(input_shape)
    
    def call(self, x):
        su2nda = tf.reshape(self.su2nda,(self.dimseg,self.su2n_dim,self.su2n_dim))
        su2nda = tf.expand_dims(su2nda,0)
        temp_shape = tf.constant(np.array([1,1,self.su2n_dim,self.su2n_dim],dtype=np.int32))
        a1 = tf.cast(x, dtype=tf.complex128)
        a1 = tf.expand_dims(a1,-1)
        a1 = tf.expand_dims(a1,-1)
        a1tile = tf.tile(a1, temp_shape)
        su2ndmult = tf.multiply(a1tile, su2nda)
        return su2ndmult 


'''========================Layer: Unitary estimation layer ========================
Description: Layer to generate Unitaries in FC Greybox model.
Location used: FC Greybox model'''



class unitary_simple(layers.Layer):
    
    def __init__(self, segments, su2nidentity,su2ndim, **kwargs):
        super(unitary_simple,self).__init__(**kwargs)
        self.segments = segments
        self.su2nidentity = su2nidentity
        self.su2ndim = su2ndim
    
    def build(self, input_shape):
        super(unitary_simple, self).build(input_shape)
    
    def call(self, x):
        maintensor = tf.linalg.expm(-x)
        maintensor = tf.reverse(maintensor, [0])
        Ujlistsplit = tf.split(maintensor, num_or_size_splits=self.segments, axis=1)
        '''Makes a list of tensors'''
        Ulistmain = []
        for i in range(self.segments):
            Ulist = Ujlistsplit[i]
            U = self.su2nidentity
            for j in range(self.su2ndim):
                Uj = Ulist[:,j]
                U = Uj @ U
            Ulistmain.append(U)    
        Ujlist = tf.convert_to_tensor(Ulistmain)
        Ujlist = tf.transpose(Ujlist, [1,0,2,3])
        return Ujlist




'''========================Layer: GRU layer ========================
Description: Layer to estimate coefficients of generators for GRU RNN Greybox model.
Location used: GRU RNN Greybox'''
class ham_est_GRU(layers.Layer):
    '''
    Description: output estimate of Hamiltonian.
    Detail: generates Hamiltonians using su2ndelta and GRU-generated coefficients
    '''
    def __init__(self, su2ndelta, segments, **kwargs):
        super(ham_est_GRU, self).__init__(**kwargs)
        self.su2ndelta = tf.convert_to_tensor(su2ndelta)
        self.segments = segments
        self.su2nda = tf.stack([self.su2ndelta] * self.segments)
        self.su2n_dim = self.su2ndelta[0].shape[0]
        self.su2ndeltashp = self.su2ndelta.shape[0]
        self.dimseg = self.su2ndeltashp * self.segments
        
    def build(self, input_shape):
        super(ham_est_GRU, self).build(input_shape)
    
    def call(self, x):
        su2nda = tf.reshape(self.su2nda,(self.dimseg,self.su2n_dim,self.su2n_dim))
        '''Add two extra dimensions for batch and time'''
        su2nda = tf.expand_dims(su2nda,0)
        su2nda = tf.expand_dims(su2nda,0)
        '''construct a tensor in the form of a row vector whose elements are [d1,d2,1,1], 
        where d1 and d2 correspond to the batch size (number of examples) and 
        number of time steps of the input. Here time vector is segment length'''
        temp_shape = tf.concat( [tf.shape(x)[0:2],tf.constant(np.array([1,1],dtype=np.int32))],0 )
        temp_shape = tf.constant(np.array([1,1,1,self.su2n_dim,self.su2n_dim],dtype=np.int32))
        a1 = tf.cast(x, dtype=tf.complex128)
        a1 = tf.expand_dims(a1,-1)
        a1 = tf.expand_dims(a1,-1)
        a1tile = tf.tile(a1, temp_shape)
        su2ndmult = tf.multiply(a1tile, su2nda)
        return su2ndmult 

'''========================Layer: GRU layer ========================
    Description: takes batch, time and Hamiltonian estimates for each Uj and
    sums them
    Detail: inputs Hamiltonian tensor and sums up for each H_j in Uj = exp(-Hj)
    '''
class ham_sum_GRU(layers.Layer):
    
    def __init__(self, su2ndelta, segments, **kwargs):
        super(ham_sum_GRU, self).__init__(**kwargs)
        self.su2ndelta = tf.convert_to_tensor(su2ndelta)
        self.segments = segments
        self.su2nda = tf.stack([self.su2ndelta] * self.segments)
        self.su2n_dim = self.su2ndelta[0].shape[0]
        self.su2ndeltashp = self.su2ndelta.shape[0]
        self.dimseg = self.su2ndeltashp * self.segments
        
    
    def build(self, input_shape):
        super(ham_sum_GRU, self).build(input_shape)
    
    def call(self, x):
        '''Note: can also be done with segment sum'''
        x = tf.math.reduce_sum(x, axis=2, keepdims=True)
        x = tf.squeeze(x, axis=2)
        return x
        
'''========================Layer: Unitary layer ========================'''
'''Description: takes Hamiltonian estimates for GRU layer and evolves them.
Location: GRU RNN Greybox model.'''
class unitary_GRU(layers.Layer):
    
    def __init__(self, **kwargs):
        super(unitary_GRU,self).__init__(**kwargs)
    def build(self, input_shape):
        super(unitary_GRU, self).build(input_shape)
    def call(self, x):
        Ujlist = tf.linalg.expm(-x)
        return Ujlist
    
'''========================Layer: Unitary layer ========================'''
'''Description: layer to reconstruct real and imaginary unitaries in redesigned original model.'''

class recombcomplex(layers.Layer):
    
    def __init__(self, **kwargs):
        super(recombcomplex,self).__init__(**kwargs)
    def build(self, input_shape):
        super(recombcomplex, self).build(input_shape)
    def call(self, x):
        xreal = tf.cast(x[0], dtype=tf.complex128)
        ximag = tf.cast(x[1], dtype=tf.complex128)
        U = xreal + 1j*ximag
        return U