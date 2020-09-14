try:
    from .CUQEn import UQEn
except():
    from CUQEn import UQEn
import numpy as np


class UQEnS(UQEn):
    def __init__(self, nm=1, nd=1, nr=1):
        super(UQEnS, self).__init__(nm, nd, nr)
        self.num_iter = 1

    def initialize(self):
        self.m_k_ = np.copy(self.m_prior_)

    def update(self):
        m_ave = np.mean(self.m_k_, 1, keepdims=True)
        d_ave = np.mean(self.d_k_, 1, keepdims=True)
        ones = np.ones((1, self.nr_))
        self.cmd_ = np.dot(self.m_k_ - np.dot(m_ave, ones),
                           np.transpose(self.d_k_ - np.dot(d_ave, ones))) / (self.nr_ - 1)
        self.cdd_ = np.dot(self.d_k_ - np.dot(d_ave, ones),
                           np.transpose(self.d_k_ - np.dot(d_ave, ones)))/ (self.nr_ - 1)
        cd_inv = np.linalg.inv(self.cdd_ + self.cd_)
        for i in range(0, self.nr_):
            self.m_k_[:, i] = np.copy(self.m_k_[:, i]) + np.dot(np.dot(self.cmd_, cd_inv), self.d_uc_[:, i] - self.d_k_[:, i])

    def save_result(self):
        self.d_posterior_ = np.copy(self.d_k_)
        self.m_posterior_ = np.copy(self.m_k_)

    def solve(self):
        self.initialize()
        self.forecast()
        self.update()
        self.save_result()
        self.sim_master_.stop()