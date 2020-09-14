try:
    from .CUQEn import UQEn
except():
    from CUQEn import UQEn
import numpy as np


class UQESMDA(UQEn):
    def __init__(self, nm=1, nd=1, nr=1):
        super(UQESMDA, self).__init__(nm, nd, nr)
        self.na_ = 1
        self.i_na_ = 0
        self.alpha_ = np.ones((self.na_, 1))*self.na_

    def set_na(self, na: int):
        self.na_ = na
        if self.na_ < 1:
            print('[ERROR] Invalid Na value')
            exit(1)

    def set_alpha(self, alpha: np.ndarray):
        self.alpha_ = alpha
        if np.abs(np.sum(1/self.alpha_) - 1) > 1e-3:
            print('[ERROR] Sum of multiplication coefficients not equal to 1.')
            exit(1)

    def initialize(self):
        self.m_k_ = np.copy(self.m_prior_)

    def perturb_observation(self):
        for i in range(0, self.nr_):
            self.d_uc_[:, i] = self.d_obs_ + \
                               np.random.multivariate_normal(np.zeros(self.nd_), self.alpha_[self.i_na_]*self.cd_, 1)

    def update(self):
        self.perturb_observation()
        m_ave = np.mean(self.m_k_, 1, keepdims=True)
        d_ave = np.mean(self.d_k_, 1, keepdims=True)
        ones = np.ones((1, self.nr_))
        self.cmd_ = np.dot(self.m_k_ - np.dot(m_ave, ones),
                           np.transpose(self.d_k_ - np.dot(d_ave, ones))) / (self.nr_ - 1)
        self.cdd_ = np.dot(self.d_k_ - np.dot(d_ave, ones),
                           np.transpose(self.d_k_ - np.dot(d_ave, ones)))/ (self.nr_ - 1)
        cd_inv = np.linalg.inv(self.cdd_ + self.alpha_[self.i_na_]*self.cd_)
        for i in range(0, self.nr_):
            self.m_k_[:, i] = np.copy(self.m_k_[:, i]) + np.dot(np.dot(self.cmd_, cd_inv), self.d_uc_[:, i] - self.d_k_[:, i])
        self.i_na_ += 1

    def save_result(self):
        self.d_posterior_ = np.copy(self.d_k_)
        self.m_posterior_ = np.copy(self.m_k_)

    def solve(self):
        self.initialize()
        while self.i_na_ < self.na_:
            self.forecast()
            self.update()
            self.sim_master_.callback_iter(self.i_na_)
        self.forecast()
        self.save_result()
        self.sim_master_.stop()