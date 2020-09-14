try:
    from .CUQBase import UQBase
except():
    from CUQBase import UQBase
import numpy as np


class UQEn(UQBase):
    def __init__(self, nm=1, nd=1, nr=1):
        super(UQEn, self).__init__(nm, nd, nr)
        # Nm, Nd defined is UQBase
        # Nr - ensemble size
        self.np_ = 0 # Size of the state vector (0 for smoothers and non-zero for filters)
        # Cd defined in UQBase
        self.cmd_ = np.zeros((self.nm_, self.nd_))
        self.cdd_ = np.zeros((self.nd_, self.nd_))
        # self.duc_ = np.zeros((self.nd_, self.nr_))
        self.dn_ = np.zeros((self.nd_, self.nr_))

    def initialize(self):
        # initialize algorithm (virtual)
        return True

    def forecast(self):
        # forecast step
        self.d_k_ = self.sim_master_.run_list_sim(self.m_k_)
        return True

    def update(self):
        # update step (virtual)
        return True

    def solve(self):
        # virtual solve
        # d_list = self.sim_master_.run_list_sim(self.m_prior_)
        # print(d_list)
        return True




