import numpy as np

class dataprocess(object):
    def __init__(self, gentest):
        self.segments = gentest._segments
        self.realdim = gentest._su2n_dim * 2
        self.sunrealdim = (self.realdim) ** 2
        self.ntraining = gentest._ntraining
        self.flatdim = self.segments * self.sunrealdim
        self.dtrain = gentest._dtrain
        #converting from list to array, this is the label data (the Uj sequence)
        self.input_data = np.asarray(gentest._Uj_master_realised)
#         self.input_data = self.input_data.reshape(self.ntraining, self.sunrealdim)
        #converting from list to array, this is the input data, the target U_T
        self.lab = np.asarray(gentest._Uj_final_list_realised)
        self.lab1 = self.lab.reshape(self.ntraining,self.sunrealdim)
        self.Uj_dim = self.realdim * self.segments