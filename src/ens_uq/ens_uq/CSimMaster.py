try:
    from .UQGlobal import *
except():
    from UQGlobal import *
import numpy as np
from enum import IntEnum
import os

UQ_USE_MPI = {
        None: False,
        "True": True,
        "TRUE": True,
        "FALSE": False,
        "False": False,
}.get(os.getenv('UQ_USE_MPI'), False)

if UQ_USE_MPI:
    from mpi4py import MPI


# class SIGNAL(IntEnum):
#     SIM = 0
#     WAIT = 1
#     STOP = 2


class SimMaster:
    def __init__(self, nm=1, nd=1):
        # self.num_run_ = 1
        self.nm_ = nm
        self.nd_ = nd
        self.comm = None
        self.num_procs = 1
        self.rank = 0
        self.verbose = 1
        self.iteration = 0

        if UQ_USE_MPI:
            self.comm = MPI.COMM_WORLD
            self.num_procs = self.comm.Get_size()
            self.rank = self.comm.Get_rank()

    def run_list_sim(self, m_list):
        """
        Run a list of simulations.

        Input:
            m_list (np.ndarray): shape (num_params, num_run)

        return:
            d_list (np.ndarray): shape (num_data, num_run)
        """
        num_run = m_list.shape[1]
        d_list = np.zeros((self.nd_, num_run))
        if UQ_USE_MPI:
            if self.rank == 0:
                if self.verbose: print("[MESSAGE] Run list of simulations in parallel.")
                d_list = self.master_run(m_list)
        else:
            if self.verbose: print("[MESSAGE] Run list of simulations in serial.")
            for i_run in range(0, num_run):
                if self.verbose: print("[MESSAGE] Run sim {}/{}".format(i_run+1, num_run))
                d_list[:, i_run] = self.sim(m_list[:, i_run], "%d_%d" % (self.iteration, i_run))
        return d_list

    # def run(self):
    #     if self.rank != 0:
    #         self.slave_run()

    def sim(self, m, tag):
        # The black-box simulator d=g(m)
        # Overwrite this function for your problem
        print("SimBase sim is called.")
        d = np.zeros((self.nd_, 1))
        return d

    def master_run(self, m_list):
        num_run = m_list.shape[1]
        print("Number of run: " + str(num_run))
        i_run = 0
        i_data = 0
        d_list = np.zeros((self.nd_, num_run))
        while i_run < num_run:
            print("-------------------------------Master sending Run signal...")
            for k in range(1, self.num_procs):
                if i_run < num_run:
                    msg = {
                        'signal': "SIM",
                        'data': m_list[:, i_run],
                        'tag': "%d_%d" % (self.iteration, i_run),
                    }
                else:
                    msg = {
                        'signal': "WAIT"
                    }
                self.comm.send(msg, dest=k)
                i_run += 1

            print("-------------------------------Master receiving data...")
            for k in range(1, self.num_procs):
                d = self.comm.recv(source=k)
                if i_data < num_run:
                    d_list[:, i_data] = d
                else:
                    pass
                i_data += 1
                print("d from slave #"+str(k))
                # print(d)

        print("-------------------------------Master run..")
        return d_list

    def slave_run(self):
        # while True:
            # self.comm.receive()
        cpt = 0
        while True:
            cpt += 1
            print("Slave #" + str(self.rank) + "loop #" + str(cpt))
            msg = self.comm.recv(source=0)
            print("Slave #" + str(self.rank) + " received signal " + msg.get("signal") + "...")
            if msg.get("signal") == "SIM":
                m = msg.get("data")
                tag = msg.get("tag")
                d = self.sim(m, tag)
                self.comm.send(d, dest=0)
            if msg.get("signal") == "WAIT":
                self.comm.send('', dest=0)
            if msg.get("signal") == "STOP":
                break
        print("Slave #" + str(self.rank) + " breaked from while loop...")

    def stop(self):
        if UQ_USE_MPI:
            print("-------------------------------Master sending Stop signal...")
            # Stop the slaves
            for k in range(1, self.num_procs):
                msg = {
                    'signal': "STOP",
                }
                self.comm.send(msg, dest=k)

    def callback_iter(self, iteration):
        self.iteration = iteration
        



