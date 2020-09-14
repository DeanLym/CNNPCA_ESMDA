try:
    from .CUQEn import UQEn
except():
    from CUQEn import UQEn
import numpy as npy


class UQEnKF(UQEn):
    def __init__(self, nm=1, nd=1, nr=1, np=1):
        super(UQEnKF, self).__init__(nm, nd, nr)
        self.num_iter = 1
        self.np_ = np
        self.ny_ = nm + np
        self.p_k_ = npy.zeros((self.np_, self.nr_))
        self.p_prior_ = npy.zeros((self.np_, self.nr_))
        self.p_posterior_ = npy.zeros((self.np_, self.nr_))
        self.y_k_ = npy.zeros((self.ny_, self.nr_))
        self.cyd_ = npy.zeros((self.ny_, self.nd_))

    def initialize(self):
        self.m_k_ = npy.copy(self.m_prior_)
        self.p_k_ = npy.copy(self.p_prior_)
        self.y_k_[0:self.nm_, :] = npy.copy(self.m_k_)
        self.y_k_[self.nm_:, :] = npy.copy(self.p_k_)

    def forecast(self):
        data = self.sim_master_.run_list_sim(self.y_k_)
        self.d_k_ = data[0:self.nd_, :]
        self.p_k_ = data[self.nd_:, :]
        self.y_k_[self.nm_:, :] = npy.copy(self.p_k_)

    def input_p_prior(self, p_prior):
        self.p_prior_ = p_prior
        if self.p_prior_.shape[0] != self.np_ or self.p_prior_.shape[1] != self.nr_:
            print("[ERROR] Dimension mismatch for m_prior.")
            exit(1)

    def update(self):
        y_ave = npy.mean(self.y_k_, 1, keepdims=True)
        d_ave = npy.mean(self.d_k_, 1, keepdims=True)
        ones = npy.ones((1, self.nr_))
        self.cyd_ = npy.dot(self.y_k_ - npy.dot(y_ave, ones),
                            npy.transpose(self.d_k_ - npy.dot(d_ave, ones))) / (self.nr_ - 1)
        self.cdd_ = npy.dot(self.d_k_ - npy.dot(d_ave, ones),
                            npy.transpose(self.d_k_ - npy.dot(d_ave, ones))) / (self.nr_ - 1)
        cd_inv = npy.linalg.inv(self.cdd_ + self.cd_)
        for i in range(0, self.nr_):
            self.y_k_[:, i] = npy.copy(self.y_k_[:, i]) + npy.dot(npy.dot(self.cyd_, cd_inv), self.d_uc_[:, i] - self.d_k_[:, i])
        self.m_k_ = npy.copy(self.y_k_[0:self.nm_, :])
        self.p_k_ = npy.copy(self.y_k_[self.nm_:, :])

    def save_result(self):
        self.d_posterior_ = npy.copy(self.d_k_)
        self.m_posterior_ = npy.copy(self.m_k_)
        self.p_posterior_ = npy.copy(self.p_k_)

    def solve(self):
        self.initialize()
        self.forecast()
        self.update()
        self.save_result()
        self.sim_master_.stop()