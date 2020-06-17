'''FILE: holonomy.py'''

import numpy as np
import random as rn
import qutip as qt
from itertools import accumulate
from matplotlib import pyplot as plt
from scipy.linalg import expm,sqrtm
import operator
# %matplotlib notebook
'''Description: implementation of Boozer (2012) results that analytically determine time-optimal
set of controls to synthesise unitary z-rotations. Here we generate discretised approximations:
- eta: (eta parameter from paper)
- segments: number of segments for discretised geodesic
- time: evolution time
- bigtmulti: is overall time T in Boozer (2012)
- evolve-steps: number of steps 
'''


class holonomy(object):
    def __init__(self, eta=None, segments=None, time=None, bigtmulti=None, evolvesteps=None):
        """
        ==========================Class constructor=========================
        eta: rotation about z-axis by angle eta, if None randomly generates eta between 0 and 2*pi
        rotate: qutip unitary operator for rotating entire sequence of unitaries by a random unitary in order
        to generate holonomic rotations about arbitrary axes with orthogonal controls
        segments:  number of segments Uj (each cumulative)
        
        time:   if 0, evolution time is bigt
                else, evolution time is input 'time'
        
        bigtmulti: if 0, evolution time is bigt
                    else: evolution time is bigt * bigtmulti
        """
        self.pi = np.pi
        if segments is None:
            self.segments = 100
        else:
            self.segments = segments
        
        if eta is None:
            self.eta = rn.uniform(0,2*self.pi)
        else:
            self.eta = eta

        self.nu = (1-(self.eta/(2*self.pi)))
        self.bigom = (1 - self.nu**2)**(1/2)

        self.omega = (2 * self.nu) / self.bigom

        self.bigt = np.abs(self.pi * self.bigom)

        if bigtmulti is None:
            self.bigt = self.bigt
        else:
            self.bigt = bigt * bigtmulti
        
        if time is None:
            self.time = self.bigt
        else:
            self.time = time
        
        if evolvesteps is None:
            self.evolvesteps = 100
        else:
            self.evolvesteps = evolvesteps
        
        self.t = np.linspace(0,self.time,self.segments)

        self.psi0= qt.basis(2,0)
        self.delta_t = self.time/self.segments
        self.H1 = qt.sigmax()
        self.H2 = qt.sigmay()
        self.Hcomb = [self.H1,self.H2]
        self.H3 = qt.sigmaz() * rn.uniform(0,1)
        self.H1_coeff = np.cos(self.omega * self.t)
        self.H2_coeff = np.sin(self.omega * self.t)
        self.H_coeflist = [[np.cos(self.omega * k),np.sin(self.omega * k)] for k in self.t]
        self.H_j = [sum([x*y for x,y in zip(self.H_coeflist[j],self.Hcomb)]) for j in range(len(self.H_coeflist))]
        self.Uj_list_hol = [( -1j*self.delta_t* (H + H.dag())/2 ).expm() for H in self.H_j]
        self.op_list = list(accumulate(self.Uj_list_hol, operator.mul) )
        self.state_list = [q * qt.basis(2,0) for q in self.op_list]
        self.state_fidelity = self.statefidelity()

        self.U0 = qt.identity(2)

        self.rand_rotate_op_list, self.rand_rotate_state_list, self.rand_rotate_Uj_list = self.rand_rotate_hol()
    
    
    def evolve_state(self):
        '''Description: runs Qutip sesolve, evolves, outputs evolved state for each timestep'''  
        output1 = qt.sesolve(self.H,self.psi0,self.t)
        evolvst = output1.states
        return evolvst
    
    
    def evolve_op(self):  
        '''Description: runs sesolve, evolves, outputs evolved (cumulative) 
        unitary operators Uj for each timestep'''
        output2 = qt.sesolve(self.H,self.U0,self.t)
        evolvop = output2.states
        return evolvop
    
    def rotate_hol(self,rotate):
        '''Description: rotates holonomic paths'''
        self.rotate_op_list = [rotate * i for i in self.op_list]
        return self.rotate_op_list
    
    
    def rand_rotate_hol(self):
        '''Description: rotates evolved states and operators by 
        random unitaries (one for each item in a list)'''
        randunitary = qt.rand_unitary(2, density=0.75)
        self.rand_rotate_op_list = [randunitary * i for i in self.op_list]
        self.rand_rotate_state_list = [randunitary * i for i in self.state_list]
        self.rand_rotate_Uj_list = [randunitary * i for i in self.Uj_list_hol]
        return self.rand_rotate_op_list, self.rand_rotate_state_list, self.rand_rotate_Uj_list
    
    def qtbloch(self,statelist):
        '''Description: for use in visualising on Bloch sphere using Qutip'''
        a = qt.Bloch()
        for i in range(len(statelist)):
            a.add_states(statelist[i])
        return a.show()

#     def statefidelity(self):
#         fin = self.state_list[-1]
#         target = (qt.rz(self.eta) * qt.basis(2,0))
#         return (np.abs((fin.dag() * target).data.toarray()) **2)[0][0] 
    
    