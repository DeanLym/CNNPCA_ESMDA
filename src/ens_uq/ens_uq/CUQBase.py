import numpy as npy


class UQBase(object):
    def __init__(self, nm=1, nd=1, nr=1):
        self.nm_ = nm  # number of model parameters
        self.nd_ = nd  # number of data
        self.nr_ = nr  # number of realizations
        self.d_obs_ = npy.zeros((nd, 1))  # observed data
        self.d_uc_ = npy.zeros((nd, nr))  # observed data
        self.m_prior_ = npy.zeros((nm, nr))  # prior models
        self.d_prior_ = npy.zeros((nd, nr))  # prior predictions
        self.m_posterior_ = npy.zeros((nm, nr))  # posterior models
        self.d_posterior_ = npy.zeros((nd, nr))  # posterior predictions
        self.m_k_ = npy.zeros((nm, 1))  # models at k-th iteration
        self.d_k_ = npy.zeros((nd, 1))  # prediction at k-th iteration
        self.cd_ = npy.zeros((nd, nd))  # covariance of data error
        self.cm_ = npy.zeros((nm, nm))  # covariance of model parameters
        self.sim_master_ = None  # a class for performing simulation
        self.mpi_run = False  # whether or not use mpi_run

    def input_m_prior(self, m_prior):
        self.m_prior_ = m_prior
        if self.m_prior_.shape[0] != self.nm_ or self.m_prior_.shape[1] != self.nr_:
            print("[ERROR] Dimension mismatch for m_prior.")
            exit(1)

    def input_d_obs(self, d_obs):
        self.d_obs_ = d_obs
        if self.d_obs_.shape[0] != self.nd_:
            print("[ERROR] Dimension mismatch for d_obs.")
            exit(1)

    def input_d_uc(self, d_uc):
        self.d_uc_ = d_uc
        if self.d_uc_.shape[0] != self.nd_ or self.d_uc_.shape[1] != self.nr_:
            print("[ERROR] Dimension mismatch for d_uc.")
            exit(1)

    def input_cd(self, cd):
        self.cd_ = cd
        if self.cd_.shape[0] != self.nd_ or self.cd_.shape[1] != self.nd_:
            print("[ERROR] Dimension mismatch for C_d.")
            exit(1)

    def input_cm(self, cm):
        self.cm_ = cm
        if self.cm_.shape[0] != self.nm_ or self.cm_.shape[1] != self.nm_:
            print("[ERROR] Dimension mismatch for C_m.")
            exit(1)

    def solve(self):
        print("UQBase virtual solve is called.")

    def stop(self):
        self.sim_master_.stop_slaves()

    # def solve(self):
    #     self.m_posterior_ = np.zeros((self.nm_, self.nr_))
    #     self.d_posterior_ = np.zeros((self.nd_, self.nr_))

    # def simulate(self, m, nr):
    #     # m: matrix of nm_ * nr_
    #     d = np.zeros(self.nd_, nr)
    #     return d
    #
    # def data_mismatch(self):



# uq = UQBase(10,10)
# print(uq.alg_)


