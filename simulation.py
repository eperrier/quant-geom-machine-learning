'''FILE: simulation.py'''

import qutip as qt
import numpy as np
import pandas as pd
import sympy as sp
import scipy as sc
import random as rn
import os
import itertools as itr
from itertools import combinations
from operator import itemgetter
import operator
from functools import reduce
import csv

from colorama import init, Fore, Back, Style
from holonomy import holonomy

'''Class: gendata'''

'''==============Simulation: generating normal subRiemannian geodesic datasets=============='''

'''Description: this code adapts Mathematica code from Swaddle (2017) which generates training datasets
comprising discretised approximations to normal subRiemannian geodesics and target unitaries. The basis of
the formulation is set-out in the paper, however in short we have that:

(U_j) = U_n....U_0 = U_T

that is, the target unitary U_T is the forward-cumulant of the intermediate subunitaries U_j.

These unitaries are generated using a first-order integrator that solves the differential form of the
variational equations i.e. the normal subRiemannian geodesic equations whose solutions are
normal subRiemannian geodesics. 

Various iterations and attributes of the data are provided, including realised forms of the
sequences (U_j), hyperparameters and so on. These are accessed as attributes of the class.'''


'''==============GenData class=============='''
'''Description: class for generating data.'''

'''Arguments:
- su_dim: n in dimension SU(2^n);
- segments: number of segments;
- ntraining: the number of training examples;
- time: used when want a time-step h based on evolution time;
- holo: holo: option to as to whether first unitary in loop generating $(U_j)$ is to be unitary generated
via the implementation of method in Boozer (2012) (used in Appendix to paper to compare 
subRiemannian method of generating geodesic approximations to the analytic method in Boozer (2012)).
This works by calling another class in holonomy.py which implements Boozer (2012);
- su2ngen: Specify generators for initial condition $\Lambda_0$: 
    * 0 $: \Lambda_0 \in \Delta$ (the distribution) and \\
    * 1$: \Lambda_0 \in su(2^n)$ (full Lie algebra);
- gens: for the case of SU(2), we can select three different distributions xy, xz and xy. To select
which one, we set gens to 'xy', 'xz' and 'xy' respective. Used in method: su2ndelta(); and
- hscale: Set $\Delta t_j$ (i.e. h in code or duration of pulse).
'''


class GenData(object):
    '''paulis: a list of pauli operators, called throughout the code.'''
    paulis = [qt.identity(2), qt.sigmax(), qt.sigmay(), qt.sigmaz()]
    def __init__(self, su_dim, segments, ntraining,time,holo=None,su2ngen=None, gens=None, hscale=None):
        '''self.gens: for su(2), string input to select generators: xy, xz, xy'''
        self.gens = gens
        '''self.su2ngen: controls whether z0 is generated using su2ndelta or full su2n generators'''
        self.su2ngen = su2ngen
        self.holo = holo
        '''Attribute calling Boozer (2012) implementation (holonomy) code'''
        self.holotest = holonomy()
        '''Set n in SU(2^n) dimension as attribute'''
        self.su_dim = su_dim
        '''Set full dim(SU(2^n)) as attribute'''
        self.su2n_dim = 2 ** su_dim
        '''Set training number as attribute'''
        self.ntraining = ntraining
        '''Set segment number as attribute'''
        self.segments = segments
        self.time = time
        #self.h = self.time/self.segments
        '''Set time-step hscale: if None then defaults to 1/segment number (as per original code).'''
        if hscale is None: 
            self.h = 1/self.segments
        else:
            self.h = hscale
#         self.h = 1
        '''Set number of elements in distribution'''
        self.nAllowed = self.nAllowed()        
        '''List for use in generating distribution'''
        self.tuplist = self.tuplist_calc()
        '''List for use in generating distribution (maps paulis to numbered list)'''
        self.tuplist_delta = [''.join(map(str,x)) for x in self.tuplist[:self.nAllowed]]
        #self.tuplist_delta = self.paulistrings(self.tuplist_delta)
        '''List for use in generating distribution to relabel in effect paulis with x,y,z and i 
        for identity'''
        self.tuplist_delta = [w.replace('1', 'i') for w in self.tuplist_delta]
        self.tuplist_delta = [w.replace('2', 'x') for w in self.tuplist_delta]
        self.tuplist_delta = [w.replace('3', 'y') for w in self.tuplist_delta]
        self.tuplist_delta = [w.replace('4', 'z') for w in self.tuplist_delta]
        
        '''List for use in generating distribution'''
        self.nums_list = self.nums_list()
        '''Attribute that calls function to generate Lie algebra su(2^n)'''
        self.su_2n = self.su2n_fn()
        '''Identity operator for use in generating data'''
        self.su_dim_identity = self.input_identity()
        '''Identity operator used in solution to normal subRiemannian equations (see paper)'''
        self.x0 = self.input_identity()
        '''Identity operator for su(2^n)'''
        self.su_2n_id = np.append(1j * 1/(np.sqrt(2) **  self.su_dim) *  self.su_dim_identity, self.su_2n)
        '''Attribute that calls function which creates distribution \Delta'''
        self.su_2n_delta = self.su2ndelta()
        
        self.su_2n_target = self.su_2n[0]
        self.fidones = np.asarray([1] * self.ntraining)
        self.fidzeros = np.asarray([0] * self.ntraining)
        '''Assign main attributes (see below for description)'''
        self.dtrain,self.Uj_master, self.Uj_master_array, self.x0_master, self.coefficients_list, self.Uj_final_list, self.Uj_final_list_array, self.Uj_master_realised, self.Uj_final_list_realised, self.eta_list, self.hol_ujlist, self.z0_list, self.z0_array_list, self.z0_list_real_flat, self.coeflist, self.z0_list_realpart, self.z0_list_imagpart, self.U_z0_list, self.U_z0_array_list, self.U_z0_list_real_flat, self.U_z0_list_realpart, self.U_z0_list_imagpart, self.Uj_re_final_list_array, self.Uj_im_final_list_array,self.Uj_master_list_array_re, self.Uj_master_list_array_im, self.Uj_master_coef_list, self.Hj_master_list, self.Uj_final_list_array_cuml,  self.Uj_final_list_realised_cuml = self.mastergen()
        self.deltadict = dict(zip(self.tuplist_delta, self.su_2n_delta))
        
    
    
    ''' tuplist_calc: calculates a list of tuples for creation of distribution'''
    def tuplist_calc(self):
        self.tuplist = list(itr.product([1,2,3,4], repeat=self.su_dim))
        self.tuplist = sorted(self.tuplist[1::], key=(lambda y: y in [item for item in self.tuplist[1::] if 1 in item]))
        self.tuplist.reverse()
        return self.tuplist
    
    ''' nums_list: maps pauli operators to the tuple list in order to create a list of pauli operators
    as part of generation of distribution or su(2^n)'''
    def nums_list(self):
        self.nums_list = []
        for i in range(len(self.tuplist)):
            self.nums_list.append(list(map(lambda x: self.paulis[x-1], self.tuplist[i])))
        return self.nums_list
    
    ''' tuplist_calc: generates Kronecker (tensor) products of the paulis in order to generate su(2^n)'''
    def su2n_fn(self):
        ''''''
        self.nfold_kronecker = []
        for i in range(0,len(self.nums_list)):
            self.nfold_kronecker_tuple = []
            self.qt1 = self.nums_list[i][0]
            '''code differs here su_dim-1 (i.e. want to remove identity at the start) - fix'''
        
            for j in range(1,self.su_dim):
                self.qt1 = qt.tensor(self.qt1,self.nums_list[i][j])
            self.nfold_kronecker.append(self.qt1)


        '''Create the bracket generating set (\Delta in Swaddle)'''
        '''Note: we include the imaginary unit here in the set, so it is not included when evolving
        according to Schrodinger's equation'''
        self.su_2n = [(1/(np.sqrt(2 ** self.su_dim )) ) * 1j * np.asarray(i) for i in self.nfold_kronecker]

        return self.su_2n
    
    '''input_identity: input identity for use in iterative projection function used to generate
    the geodesics'''
    def input_identity(self):
        self.x0 = qt.identity(2)
        if self.su_dim > 1:
            for i in range(self.su_dim-1):
                self.x0 = (qt.tensor(self.x0, qt.identity(2)))       
        self.x0 = np.asarray(self.x0)
        return self.x0
    
    '''nAllowed: for n=1,2 we need to ensure the braket-generating set
    is smaller than the su(2n) Lie algebra, else the generation
    mechanism simply reproduces the same unitary Uj, so we manually set nAllowed=2 in that case.'''
    def nAllowed(self):
        if self.su_dim == 1:
            self.nAllowed = 2
        elif self.su_dim == 2:
            self.nAllowed = 7
        elif self.su_dim > 2:
            self.nAllowed = int(9/2 * self.su_dim * ( self.su_dim - 1) + 3 * self.su_dim) 
        return self.nAllowed
    
    '''paulistrings: relabelling paulis so clearer. Here 'i' indexes the identity.'''
    def paulistrings(self):
        #self.tuplist_delta = [''.join(map(str,x)) for x in self.tuplist[:self.nAllowed]]
        self.tuplist_delta = [w.replace('1', 'i') for w in self.tuplist_delta]
        self.tuplist_delta = [w.replace('2', 'x') for w in self.tuplist_delta]
        self.tuplist_delta = [w.replace('3', 'y') for w in self.tuplist_delta]
        self.tuplist_delta = [w.replace('4', 'z') for w in self.tuplist_delta]
        return self.tuplist_delta
    
    
    '''su2ndelta: function that generates distribution \Delta. For SU(2), different
    distributions can be set via setting self.gens as appropriate (see above).'''
    def su2ndelta(self):
        if self.su_dim == 1: 
            if self.gens == 'yz' or self.gens == None:
                self.su_2n_delta = self.su_2n[0:2]
            elif self.gens == 'xy':
                self.su_2n_delta = self.su_2n[1::]
            elif self.gens == 'xz':
                self.su_2n_delta = [gentest.su_2n[0],gentest.su_2n[2]]
        
        elif self.su_dim == 2:
            self.twobod = rn.sample(list(self.su_2n[6::]),1)[0]
            self.su_2n_delta = list(self.su_2n[:6])
            self.su_2n_delta.append(self.twobod)
        elif self.su_dim > 2:
            self.su_2n_delta = self.su_2n[:self.nAllowed]
        return self.su_2n_delta
            

    
    '''projAcoef: extracts coefficients of generators for \Delta from projection function.'''
    def projAcoef(self,mat):
        self.projAcoef_list = []
        for i in range(0,self.nAllowed):
            su_2n_op = (self.su_2n_delta[i])
            projAcoef = qt.Qobj(-mat @ su_2n_op).tr()
            self.projAcoef_list.append(projAcoef)
        return self.projAcoef_list
    
    '''projA: implements projection function from paper. Takes input matrix (operator) and
    projects it onto the distribution \Delta.'''
    def projA(self, mat):
        self.proj_list = []
        '''Loop over each generator in distribution then after loop sum projection onto each generator.'''
        for i in range(0,self.nAllowed):
            self.su_2n_op = (self.su_2n_delta[i])
            '''Note: using trace usually requires complex conjugate but for Pauli's they are self
            conjugate'''
            '''project: projects input matrix (note @ in numpy for matrices)'''
            project = qt.Qobj(-(mat @ self.su_2n_op)).tr() * self.su_2n_op
            self.proj_list.append(project)
        proj_list_sum = sum(self.proj_list)
        return proj_list_sum
    
    def tupletest(self):
        self.a = 1
        self.b = [1,2]
        self.c = qt.identity(2)
        return self.a, self.b, self.c
    
    '''rand_unitary: generate \Lambda(0) randomly (option in code).'''
    def rand_sunitary(self):
        a1 = qt.rand_unitary(self.su2n_dim).data.toarray() 
        det = np.linalg.det(a1)
        z0 = z0 = qt.Qobj(a1/(det**2))
        return z0
    
    '''mastergen: the main code sequence that generates the training data geodesics.'''
    def mastergen(self):
        '''dtrain stores coefficients and unitaries (flattened see below)
        for use in neural network, this recreates form in which data stored in Swaddle (2017).'''
        self.dtrain = []
        '''List of lists: to store each different sequence (Uj) generated'''
        self.Uj_master = []
        '''Store Uj_master as array'''
        self.Uj_master_array = []
        '''Store realised form of Uj_master as array'''
        self.Uj_master_realised = []
        
        '''Cumulative List of lists: stores cumulative evolved unitary i.e.
        Uk where Uk = \prod_j=1^j=k U_j'''
        self.Uj_master_cuml = []
        '''Store Uj_master_cuml as array'''
        self.Uj_master_array_cuml = []
        '''Store realised form of Uj_master_cuml as array'''
        self.Uj_master_realised_cuml = []
        
        '''Save initial unitaries x_0'''
        self.x0_master = []

        '''List to store list of coefficients of Lie algebra elements.'''
        self.coefficients_list = []
        
        '''Final lists (list of targets U_T)'''
        '''Store list of final U_j (which should be cumulant i.e. U_T).
        Note: in the QMLcontrolmodel code we recreate U_T by accumulating from Uj_master directly.'''
        self.Uj_final_list = []
        '''Store Uj_final_list as array'''
        self.Uj_final_list_array = []
        '''Store real part of Uj_final_list'''
        self.Uj_re_final_list = []
        '''Store real part of Uj_final_list as array'''
        self.Uj_re_final_list_array = []
        '''Store imaginary part of Uj_final_list'''
        self.Uj_im_final_list = []
        '''Store imaginary part of Uj_final_list as array'''
        self.Uj_im_final_list_array = []
        '''Store realised form of Uj_final_list'''
        self.Uj_final_list_realised = []
        
        '''Final lists - cumulative'''
        
        self.Uj_final_list_cuml = []
        self.Uj_final_list_array_cuml = []
        
        self.Uj_final_list_realised_cuml = []
        
        '''List for saving angle eta (if used)'''
        self.eta_list = []
        '''Save list of unitaries generated in relation to Boozer (2012)'''
        self.hol_ujlist = []
        '''Save initial list of \Lambda_0 (this is z0)'''
        self.z0_list = []
        '''Save initial list of \Lambda_0 (this is z0) as array'''
        self.z0_array_list = []
        '''Save realised flattened initial list of \Lambda_0 (this is z0)'''
        self.z0_list_real_flat = []
        '''U_z0 is unitary generated by \Lambda_0, save list'''
        self.U_z0_list = []
        '''U_z0 is unitary generated by \Lambda_0, save list as array'''
        self.U_z0_array_list = []
        '''U_z0 is unitary generated by \Lambda_0, save realised list'''
        self.U_z0_list_real_flat = []
        
        '''Save list of coefficients'''
        self.coeflist = []
        '''Save real part of \Lambda_0 (list)'''
        self.z0_list_realpart = []
        '''Save imag part of \Lambda_0 (list)'''
        self.z0_list_imagpart = []
        '''Save real part of U_z0 (list)'''
        self.U_z0_list_realpart = []
        '''Save imaginary part of U_z0 (list)'''
        self.U_z0_list_imagpart = []
        '''Save real part of Uj_master as array'''
        self.Uj_master_list_array_re = []
        '''Save imaginary part of Uj_master as array'''
        self.Uj_master_list_array_im = []
        '''Save list of all coefficients for Uj_master'''
        self.Uj_master_coef_list = []
        '''Save list of all Hamiltonians used to generate unitaries in Uj_master'''
        self.Hj_master_list = []
        for j in range(0,self.ntraining):
            '''Loop: generate list of random numbers between 0 and 1 
            to act as coefficients for each element in su_2n or su2ndelta.
            For \Lambda_0 in \Delta, needs to be su2ndelta.
            For \Lambda_0 in su(2^n), need to initialise with su2n.'''
            '''Call holonomy.py code'''
            holtest = holonomy()
            '''Generate \Lambda_0 depending on generator set chosen'''
            if self.su2ngen == 1:
                L = np.asarray([ rn.uniform(-1,1) for i in range(len(self.su_2n)) ])
                z0 = sum([x*y for x,y in zip(L,self.su_2n)])
            
            else:
                L = np.asarray([ rn.uniform(-1,1) for i in range(len(self.su_2n_delta)) ])
                z0 = sum([x*y for x,y in zip(L,self.su_2n_delta)])
            
            '''Unitary generated by \Lambda_0 as Qutip object'''
            self.z0 = qt.Qobj(z0)
            '''Append to lists etc'''
            self.z0_list.append(qt.Qobj(z0))
            self.z0_array_list.append(z0)
            self.z0_list_realpart.append(np.real(z0))
            self.z0_list_imagpart.append(np.imag(z0))
            '''Generate Uz0 using Qutip'''
            Uz0 = qt.Qobj(-z0).expm()
            self.U_z0_list.append(Uz0)
            '''Generate Uz0 using SciPy'''
            Uz0ar = sc.linalg.expm(-z0)
            '''Append to lists'''
            self.U_z0_array_list.append(Uz0ar)
            self.U_z0_list_realpart.append(np.real(Uz0ar))
            self.U_z0_list_imagpart.append(np.imag(Uz0ar))
            
            '''Realise \Lambda_0: set real and imaginary parts of \Lambda_0
            ultimately to form realised form of matrix.'''
            z1a_re = self.z0.data.toarray().real
            z1a_im = self.z0.data.toarray().imag
            complex_z1a_top = np.concatenate((z1a_re, -z1a_im), axis=1)
            complex_z1a_bottom = np.concatenate((z1a_im, z1a_re), axis=1)
            complex_z1a = np.vstack((complex_z1a_top, complex_z1a_bottom))
            self.z0_list_real_flat.append(complex_z1a.flatten())
            
            '''Realise Uz0: set real and imaginary parts of \Lambda_0
            ultimately to form realised form of matrix.'''
            Uz1a_re = Uz0.data.toarray().real
            Uz1a_im = Uz0.data.toarray().imag
            Ucomplex_z1a_top = np.concatenate((Uz1a_re, -Uz1a_im), axis=1)
            Ucomplex_z1a_bottom = np.concatenate((Uz1a_im, Uz1a_re), axis=1)
            Ucomplex_z1a = np.vstack((Ucomplex_z1a_top, Ucomplex_z1a_bottom))
            self.U_z0_list_real_flat.append(Ucomplex_z1a.flatten())
            
            '''Commented out: option for using holomic code, takes first unitary generated from
            the implementation of Boozer and inputs it into the subRiemannian sequence, idea is to
            test how similar both methods are at generating approximate geodesic.'''
#             if holo is None:
#                 self.z0 = qt.Qobj(z0)
#             else:
#                 self.z0 = self.holtest.Uj_list_hol[0]
#                 print(self.z0)
            


            
            '''Empty object for Uj and Uj_realised'''
            Uj = []
            Uj_realised = []
            '''List to store each Uj in various forms'''
            Uj_list = []
            Uj_list_array = []
            Uj_list_realised = []
            Uj_list_array_re = []
            Uj_list_array_im = []
            
            
            '''Cumulative empty objects for Uj'''
            Uj_cuml = []
            Uj_realised_cuml = []
            '''Cumulative lists to store each Uj'''
            Uj_list_cuml = []
            Uj_list_array_cuml = []
            Uj_list_realised_cuml = []
            Uj_coef_list = []
            '''List to store Hamiltonians generated by the system'''
            Hj_list = []
            '''List for x_0'''
            x0_list = []
            self.x0a = qt.Qobj(self.x0)
            '''Saves parameter eta from Boozer (2012) code in object.'''
            self.eta_list.append(holtest.eta)
            self.hol_ujlist.append(holtest.Uj_list_hol)
            for i in range(0,self.segments):
                x0_list.append(self.x0a)
                '''Note: in Qutip, matmul is via * not @'''
                U1 = qt.Qobj((self.h *self.projA( self.x0a * self.z0 * self.x0a.dag()))).expm()
                Ujcf = self.projAcoef(self.x0a * self.z0 * self.x0a.dag())
                Uj_coef_list.append(Ujcf)
                Hj = self.projA( self.x0a * self.z0 * self.x0a.dag())
                Hj_list.append(Hj)
                U1a = U1           
                '''Note: this part of activating the holonomy (Boozer) unitary initially
                was rearranged (switched)'''
                if self.holo == 1 and i == 1:
                    U1 = holtest.Uj_list_hol[0]
                else:
                    U1 = U1a
                '''Check if intermediate unitaries Uj being generated are in fact unitary.'''
                if U1.check_isunitary() == False:
                    print(Fore.RED + 'Non-unitary generated')
                    print(Style.RESET_ALL)
                    break
                '''Append to list of Uj'''
                Uj_list.append(U1)
                '''Convert to array'''
                Uj_list_array.append(U1.data.toarray())
                '''Update x1 for loop (see paper)'''
                x1 = U1 * self.x0a
                '''Divide the unitary into real and imaginary parts'''
                '''X = Re(U)'''
                U1_re = U1.data.toarray().real 
                '''Y = Im(U)'''
                U1_im = U1.data.toarray().imag 
                
                Uj_list_array_re.append(U1_re)
                Uj_list_array_im.append(U1_im)
                '''Create realisation of U as: [[X, -Y],[Y,X]]'''
                complex_U1_top = np.concatenate((U1_re, -U1_im), axis=1)
                complex_U1_bottom = np.concatenate((U1_im, U1_re), axis=1)
                '''Stack them both together'''
                complex_U1 = np.vstack((complex_U1_top, complex_U1_bottom))
                '''Now flatten the realised unitary'''
                Uj = np.concatenate([Uj,complex_U1.flatten()])
                '''Append the realised form of Uj to list'''
                Uj_list_realised.append(complex_U1)
                '''Update x0 to be x1'''
                self.x0a = x1
                Uj_list_array_cuml.append(x1.data.toarray())
                
                '''=============Realise x1 (the cumulant)'''
                x1_re = x1.data.toarray().real 
                
                x1_im = x1.data.toarray().imag
                complex_x1_top = np.concatenate((x1_re, -x1_im), axis=1)
                complex_x1_bottom = np.concatenate((x1_im, x1_re), axis=1)
                '''Stack them both together'''
                complex_x1 = np.vstack((complex_x1_top, complex_x1_bottom))
                
                Uj_list_realised_cuml.append(complex_x1)
                

            '''Append various unitaries and sequences of unitaries to lists.'''
            self.Uj_final_list_realised_cuml.append(Uj_list_realised_cuml)
            self.Uj_final_list_array_cuml.append(Uj_list_array_cuml)
            self.Uj_master_coef_list.append(Uj_coef_list[::-1])
            self.Hj_master_list.append(Hj_list[::-1])
            '''Forward-solve cumulant (note need to revers list as want U_T = U_n...U0 not U_0...U_n.'''
            x1a = reduce(operator.mul, Uj_list[::-1], 1)
            x1a_re = x1a.data.toarray().real
            x1a_im = x1a.data.toarray().imag
            complex_x1a_top = np.concatenate((x1a_re, -x1a_im), axis=1)
            complex_x1a_bottom = np.concatenate((x1a_im, x1a_re), axis=1)
            complex_x1a = np.vstack((complex_x1a_top, complex_x1a_bottom))
            self.dtrain.append(Uj[::-1])
            self.dtrain.append(complex_x1a.flatten())
            
            '''Master list, which contains lists of Uj lists'''
            self.Uj_master.append(Uj_list[::-1])
            self.Uj_master_realised.append(Uj_list_realised[::-1])

            '''Uj_master to array list'''
            self.Uj_master_array.append(Uj_list_array[::-1])
            
            '''Real part of unitaries in Uj_master'''
            self.Uj_master_list_array_re.append(Uj_list_array_re[::-1])
            '''Imaginary part of unitaries in Uj_master'''
            self.Uj_master_list_array_im.append(Uj_list_array_im[::-1])

            '''U_final_list contains the final Uj obtained by multiplying
            each of the Ujs together'''
            self.Uj_final_list.append(x1a)
            self.Uj_final_list_realised.append(complex_x1a)
            '''U_final converted to array'''

            self.Uj_final_list_array.append(self.Uj_final_list[j].data.toarray())

            self.x0_master.append(x0_list)
           
            '''Saving coefficients for the bracket generating set. For n=1,2 some of these
            are intentionally zero as we are taking the whole of su_dim as the bracket-generating
            set (see above)'''
            if (self.su_dim == 1) or (self.su_dim == 2):
                self.coefficients_list.append(L[:self.nAllowed].flatten())
            else:
                self.coefficients_list.append(L[:self.nAllowed].flatten())
            '''Note: Uj_master is a list of the Uj_lists. The first element of each
            Uj_list is the initial Uj generated using z0 (\Lambda_0 in the paper).
            This is the Uj that is generated using the randomly sampled coefficients
            c_i because for the first Uj, exp(h projA(x0z0x0.dag())) = exp(h projA(z0))
            as x0 is the identity. We repeat the process above of realising the
            complex matrix by converting it into the (X -Y; Y X) form and flattening.
            We take the element at the end of the list because of the reversed order.
            We have the reversed order because we act from the left'''
            self.coeflist.append(L[:self.nAllowed].flatten())
            Ujm_re = self.Uj_master[j][self.segments-1].data.toarray().real
            self.Uj_re_final_list_array.append(Ujm_re)
            Ujm_im = self.Uj_master[j][self.segments-1].data.toarray().imag
            self.Uj_im_final_list_array.append(Ujm_im)
            Ujm_top = np.concatenate((Ujm_re, -Ujm_im), axis=1)
            Ujm_bottom = np.concatenate((Ujm_im, Ujm_re), axis=1)
            Ujm = np.vstack((Ujm_top, Ujm_bottom))
            self.coefficients_list.append(Ujm.flatten())
            
        return self.dtrain,self.Uj_master, self.Uj_master_array, self.x0_master, self.coefficients_list, self.Uj_final_list, self.Uj_final_list_array, self.Uj_master_realised, self.Uj_final_list_realised, self.eta_list, self.hol_ujlist, self.z0_list, self.z0_array_list, self.z0_list_real_flat, self.coeflist, self.z0_list_realpart, self.z0_list_imagpart, self.U_z0_list, self.U_z0_array_list, self.U_z0_list_real_flat, self.U_z0_list_realpart, self.U_z0_list_imagpart, self.Uj_re_final_list_array, self.Uj_im_final_list_array, self.Uj_master_list_array_re, self.Uj_master_list_array_im, self.Uj_master_coef_list, self.Hj_master_list, self.Uj_final_list_array_cuml,  self.Uj_final_list_realised_cuml
        