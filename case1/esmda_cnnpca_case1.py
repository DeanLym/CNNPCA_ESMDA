#!/usr/bin/env python
# coding: utf-8

import numpy as np
# import standard modules
import pandas as pd
import os
import h5py
import sys
import shutil

# Import appuqa
sys.path.append('/data3/Astro/global/unified_uq_framework/')
from appuqa.UQGlobal import *
UQ_USE_MPI = True
from appuqa import SimMaster

from appuqa import UQMain
# Global option for using MPI or not
from mpi4py import MPI


# Import torch modules
import torch
from torch import FloatTensor, cat, from_numpy
from torch.autograd import Variable
from torchsummary import summary
sys.path.append('/data3/Astro/personal/yiminliu/cnnpca3d/')
# Import transform net
from transformer_net import TransformerNet

sys.path.append('/data3/Astro/personal/yiminliu/3d_cnnpca/')
from opca_base import OpcaBase


class ADGPRSSim(SimMaster):
    def __init__(self, options={}):
        super().__init__(options['dim'], 1)
        self.options = options
        self.nx = options['nx']
        self.ny = options['ny']
        self.nz = options['nz']
        self.load_transform_net(options)
        self.pca_facies = self.load_pca_model(options['pca_model_facies'])
        self.pca_sand = self.load_pca_model(options['pca_model_sand'])
        self.pca_mud = self.load_pca_model(options['pca_model_mud'])
        self.load_target_hist(options)
        self.load_obs_data(options)
        self.num_iter = 0

    def load_transform_net(self, options):
        self.transformer = TransformerNet().to('cpu')
        self.transformer.load_state_dict(torch.load(options['transform_net'], map_location=torch.device('cpu')))
        self.transformer.eval()

    def load_pca_model(self, fn):
        nx, ny, nz, nr = 60, 60, 40, 3000
        pca_model = OpcaBase(nc=nx*ny*nz, nr=nr, l=nr)
        fid = h5py.File(fn, 'r')
        pca_model.usig = np.array(fid['usig'])
        pca_model.xm = np.array(fid['xm'])
        fid.close()
        return pca_model

    def load_target_hist(self, options):
        fn = options['target_hist_file']
        with h5py.File(fn, 'r') as fid:
            self.bins = int(np.array(fid['bins']))
            self.cdf_target = np.array(fid['cdf_target'])
            self.bins_target = np.array(fid['bins_target'])

    def get_hist_data(self, fn, options):
        df = pd.read_csv(fn, delim_whitespace=True)
        df = df.set_index("Day")
        df = df.loc[options['hist_time'], options['hist_data']].abs()
        data = df.to_numpy().flatten()
        return data, df

    def load_obs_data(self, options):
        self.true_data, self.true_data_df = self.get_hist_data(options['hist_file'], options)
        self.data_std_df = self.true_data_df * options['rate_std']
        self.data_std_df[self.data_std_df < options['rate_std_min']] = options['rate_std_min']
        # Perturb hist data
        np.random.seed(0)
        self.obs_data_df = self.true_data_df + np.random.normal(0, self.data_std_df)
        self.obs_data = self.obs_data_df.to_numpy().flatten()
        self.data_std = self.data_std_df.to_numpy().flatten()
        self.nd_ = self.obs_data.shape[0]

    def hist_trans(self, data):
        hist, bins = np.histogram(data.flatten(), self.bins)
        cdf = hist.cumsum()
        cdf = cdf / cdf.max()
        cdf = np.concatenate(([0], cdf))
        # Histogram transformation
        cdf_values = np.interp(data.flatten(), bins, cdf)
        data_ht = np.interp(cdf_values, self.cdf_target, self.bins_target)
        data_ht = data_ht.reshape(data.shape)
        return data_ht

    def cookie_cutter(self, facies, logk_sand, logk_mud):
        return facies * logk_sand + (1 - facies) * logk_mud

    def sim(self, xi, tag):
        #
        xi_facies = xi[:400]
        xi_sand = xi[400:600]
        xi_mud = xi[600:]
        # make subdir
        topdir = self.options['topdir']
        casedir = self.options['casedir']
        subdir = os.path.join(casedir, 'run_{}'.format(tag))
        # copy ADGPRS input file

        shutil.copytree(self.options['model_dir'], subdir)
        # Change into subdir
        os.chdir(subdir)
        # Save xi to file
        np.savetxt(os.path.join(subdir, 'xi.txt'), xi)
        # Convert xi to facies
        m_pca = self.pca_facies.generate_pca_realization(xi_facies, dim=400)
        m_pca = m_pca.reshape((1, 1, self.nz, self.nx, self.ny))
        # Normalize data
        max_, min_ = 1., 0.
        m_pca = (m_pca - min_) / (max_ - min_) * 255.
        # CNN-PCA
        m_cnn = self.transformer(Variable(torch.from_numpy(m_pca).float()).to('cpu')).data.numpy()
        # Histogram transform
        m_cnn = self.hist_trans(m_cnn).round()
        # Get logk within each facies
        m_sand = self.pca_sand.generate_pca_realization(xi_sand, dim=200)
        m_sand = m_sand.reshape((1, 1, self.nz, self.nx, self.ny))
        m_mud = self.pca_mud.generate_pca_realization(xi_mud, dim=200)
        m_mud = m_mud.reshape((1, 1, self.nz, self.nx, self.ny))
        # Cookie cutter to get bimodal
        logk = self.cookie_cutter(m_cnn, m_sand, m_mud)
        # Write perm/poro file
        self.save_data_to_file(logk, os.path.join(subdir, self.options['perm_file']), os.path.join(subdir, self.options['poro_file']))
        # Launch ADGPRS
        cmd = "{} {} {} 0".format(self.options['simulator'], self.options['sim_input_file'], self.options['num_thread'])
        print(cmd)
        os.system(cmd)
        # Get data
        data, _ = self.get_hist_data(os.path.join(subdir, self.options['rates_file']), self.options)
        # Save d_obs and dsim
        np.savetxt(os.path.join(subdir, 'dsim.txt'), data)
        np.savetxt(os.path.join(subdir, 'dobs.txt'), self.obs_data)
        # Remove some files
        os.remove("OUTPUT.res_partition.bin")
        os.remove("OUTPUT.solver_partition.bin")

        # Change back into topdir
        os.chdir(topdir)
        return data

    def save_data_to_file(self, logk, perm_file, poro_file):
        perm = np.exp(logk)
        np.savetxt(perm_file, perm.flatten(), fmt='%.2f', header='MULTPERM', footer='/', comments='')
        poro = logk / 40.
        np.savetxt(poro_file, poro.flatten(), fmt='%.3f', header='PORO', footer='/', comments='')


##
def main():
    data_dir = '/data3/Astro/personal/yiminliu/models/3d_chan_60x60x40_cond4wfar_wellsonly/'

    options = {}
    options['nx'] = 60
    options['ny'] = 60
    options['nz'] = 40
    options['dim'] = 800
    options['target_hist_file'] = os.path.join(data_dir, 'hm/target_hist.h5')
    options['transform_net'] = os.path.join(data_dir, 'saved_models/cnnpca_chan_less_60x60x40_cond4wfar_ptb40_std1_l400_sw100.0_rw500.0_hw10.0_9ep.model')
    options['pca_model_facies'] = os.path.join(data_dir, 'pca_model_chan_wellsonly_60x60x40_cond4wfar_l3000.h5')
    options['pca_model_sand'] = os.path.join(data_dir, 'pca_model_bimodal_chan_logksand_wellsonly_60x60x40_cond4wfar_l3000.h5')
    options['pca_model_mud'] = os.path.join(data_dir, 'pca_model_bimodal_chan_logkmud_wellsonly_60x60x40_cond4wfar_l3000.h5')
    options['topdir'] = os.path.join(data_dir, 'hm')
    options['casedir'] = os.path.join(options['topdir'], 'case2_nr200_na4_std0.01_5timestep')
    options['model_dir'] = os.path.join(options['topdir'], 'model')
    options['perm_file'] = 'perm.dat'
    options['poro_file'] = 'poro.dat'
    options['sim_input_file'] = 'GPRS.txt'
    options['num_thread'] = 8
    options['simulator'] = 'ADGPRS_CentOS6'
    options['verbose'] = 2
    options['rates_file'] = 'OUTPUT.rates.txt'
    options['hist_file'] = os.path.join(options['topdir'], 'true_model', options['rates_file'])
    options['hist_time'] = np.linspace(100, 500, num=5)
    # options['hist_file'] = os.path.join(options['topdir'], 'test_true/run_0', options['rates_file'])
    # options['hist_time'] = [1., 2.]
    options['hist_data'] = ['I%d:WIR' % i for i in range(1,3)] + ['P%d:OPR' % i for i in range(1,3)] + ['P%d:WPR' % i for i in range(1,3)]
    options['rate_std'] = 0.01
    options['rate_std_min'] = 2.0
    options['ensemble_size'] = 200


    # =========================ES-MDA========================== #
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    my_sim = ADGPRSSim(options=options)
    nd, nr, nm = my_sim.nd_, options['ensemble_size'], options['dim']
    print(nd, nr, nm)
    es_mda = UQMain(nm, nd, nr, alg=UQAlg.ES_MDA, sim_master=my_sim)

    print("Rank %d" % rank)
    if rank == 0:
        if not os.path.exists(options['casedir']):
            os.mkdir(options['casedir'])

        cd = np.diag(my_sim.data_std ** 2)
        cm = np.eye(nm)
        print(cd.shape, cm.shape)

        na = 4
        alpha = np.array([9.333, 7.0, 4.0, 2.0])

        np.random.seed(0)
        xi_prior = np.random.normal(0, 1, (nm, nr))
        print(xi_prior.shape)
        es_mda.solver.input_m_prior(xi_prior)
        es_mda.solver.input_d_obs(my_sim.obs_data)
        es_mda.solver.input_cm(cm)
        es_mda.solver.input_cd(cd)
        es_mda.solver.set_na(na)
        es_mda.solver.set_alpha(alpha)

    es_mda.solve()

if __name__ == "__main__":
    main()
