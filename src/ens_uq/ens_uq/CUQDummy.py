
class UQDummy:
    def __init__(self, nm=1, nd=1, nr=1):
        self.sim_master_ = None
        self.mpi_run = False

    def input_m_prior(self, fn):
        pass

    def solve(self):
        self.sim_master_.slave_run()
