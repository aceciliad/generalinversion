#!/usr/bin/env python
# -*- coding: utf-8 -*-


#==========================================================================
# Inversion module
#
# Classes:
# - PRMS: read initial parameters from file and generate depth and T nodes
# - FM: compute forward model
# - FM_AR
# - FM_RF
# - INV: inversion module
#==========================================================================

import numpy as np
import os, csv, sys, glob, copy
from depthloop.perplex import depth_loop
from coreeos.outercore import outercore
from obspy.core import read, Stream, Trace
from obspy.taup import taup_create, TauPyModel
from rf import deconvolve
from scipy.integrate import solve_ivp
from scipy.optimize import minimize, root_scalar
#from mypackage import myfunctions as mf
from multiprocessing import Pool
from scipy.interpolate import interp1d
from specnm_lsl import rayleigh_fg

from pebble import ProcessPool, concurrent
from concurrent.futures import TimeoutError
import resource
from threadpoolctl import threadpool_limits


import warnings
warnings.filterwarnings("ignore", message="using Voigt average")

sys.path.insert(1, '../')
#==========================================================================
# some random functions
def find_nearest(arr, val):
    '''
    find nearest value to val in arr
    return index and value
    '''
    idx = np.abs(arr-val).argmin()
    return arr[idx], idx


def radians(val):
    return val*np.pi/180.

#==========================================================================

class PRMS():

    def __init__(self):
        self.read_prms()
        return

    def read_prms(self, fname='input_parameters_lvl'):
        '''
        read input parameters from file
        '''

        print(' Read input file {}'.format(fname))
        prms    = __import__(fname)
        # read depth grid info
        self.depth_ini  = prms.z0
        self.zdisc      = prms.zdisc
        self.nzlit      = prms.nzlit
        self.nzmant     = prms.nzmant
        self.nzdisc     = prms.nzdisc
        self.nzlvl      = prms.nzlvl
        self.ntemp      = prms.ntemp

        # Input files and folders
        self.input_fol  = prms.input_fol
        self.input_picks= prms.input_picks
        self.RF_obs_fileps  = prms.RF_obsps
        self.RF_obs_filesp  = prms.RF_obssp
        self.tmp        = prms.output_tmp
        self.file_tmp   = prms.file_tmp
        self.file_tmp_lvl   = prms.file_tmp_lvl

        self.phases_lvl = prms.phases_lvl

        self.input_ref  = prms.input_ref
        self.ref_fol    = prms.ref_fol
        self.ref_executable = prms.ref_exec
        self.output_fol = prms.output_fol.format(os.getcwd().split('/')[-1])
        self.temp_file  = os.path.join(self.input_fol, prms.temp_file)

        os.makedirs(self.tmp, exist_ok=True)
        os.makedirs(self.output_fol, exist_ok=True)

        self.exc        = prms.exc
        self.events_pp  = prms.events_pp
        self.fixed_loc  = prms.fixed_loc

        # observed values (to fit)
        self.dens_obs   = prms.dens_obs
        self.dens_sigma = prms.dens_sigma
        self.moi_obs    = prms.moi_obs
        self.moi_sigma  = prms.moi_sigma

        # perplex parameters
        self.cfmasna    = prms.cfmasna
        self.tsurf      = prms.tsurf
        self.tstart     = prms.tstart
        self.adiabatic  = prms.adiabatic

        # attenuation parameters
        self.grain_sz   = prms.grain_sz
        self.activ_vol  = prms.activ_vol
        self.period_att = prms.period_at
        self.qkp        = prms.qkp
        self.qs_constant= prms.qs_contant

        # core parameters
        self.ncore      = prms.ncore

        # planetary settings
        self.m_p    = prms.m_p
        self.r_p    = prms.r_p
        self.Gnewt  = 6.6743e-11
        # RFs
        self.run_sp     = prms.run_sp
        self.events_ps  = prms.events_ps
        self.events_sp  = prms.events_sp
        self.events_rf  = list(dict.fromkeys(self.events_ps + self.events_sp))
        self.motherps   = prms.motherps
        self.mothersp   = prms.mothersp

        self.win_resps  = prms.win_resps
        self.win_srcps  = prms.win_srcps

        self.win_ressp  = prms.win_ressp
        self.win_srcsp  = prms.win_srcsp

        self.taper_win  = prms.taper_win
        self.taper_per  = prms.taper_per
        self.filter_rf  = prms.filter_rf
        self.filter_order   = prms.filter_order
        self.misf_winps = prms.misf_winps
        self.misf_winps2= prms.misf_winps2
        self.misf_winsp = prms.misf_winsp

        self.unc_level  = prms.unc_level
        self.unc_level2 = prms.unc_level2

        # inversion parameters
        self.phases_diff= prms.phases_diff
        self.max_ite    = prms.max_ite
        self.count_accepted = 0
        self.accepted_block = 0
        self.count_all  = 0
        self.nblock     = 0
        self.count_block= 0
        self.ite_read   = prms.step_read
        self.max_step   = prms.max_step
        self.step_num   = prms.step_num

        # parameters that are Perturbed
        self.param_pert = prms.param    # dictionary of parameters
        self.param_group= prms.param_group

        self.ini_model()

        self.ncrust = len(self.zcrust_pert)

        self.read_picks()

        # finde index of mother phase (to read from time arrays)
        self.idx_motherps = self.phases_obs.index(self.motherps)
        self.idx_mothersp = self.phases_obs.index(self.mothersp)

        return


    def ini_model(self):
        '''
        generate random initial model given a priori distribution
        of parameters
        '''

        self.zcmb1_pert = self.param_pert['zcmb1']['val']
        self.zlit_pert  = self.param_pert['zlit']['val']
        self.tlit_pert  = self.param_pert['tlit']['val']
        self.alpha_pert = self.param_pert['alpha']['val']
        self.beta_pert  = self.param_pert['beta']['val']
        self.alphaT_pert  = self.param_pert['alphatemp']['val']

        # Low velocity layer (first virtual core layer) parameters
        ini_fixed=True
        ini_param= False
        if ini_fixed:
            self.dlvl_pert  = self.param_pert['dlvl']['val']
            self.k0_pert    = self.param_pert['k0']['val']
            self.k0p_pert   = self.param_pert['k0p']['val']
            self.rho0_pert  = self.param_pert['rho0']['val']

            # Outer core parameters
            self.k1_pert    = self.param_pert['k1']['val']
            self.k1p_pert   = self.param_pert['k1p']['val']
            self.rho1_pert  = self.param_pert['rho1']['val']
        elif not ini_fixed and ini_param:

            vals_core   = np.array(([80., 4.6, 3.75e3, 160., 5., 6.63e3, 190],
                                    [88., 3.3, 3.54e3, 186., 5.8, 7.e3, 218],
                                    [71., 3.5, 4.4e3, 194., 5.3, 6.7e3, 146],
                                    [77., 4.0, 3.7e3, 161., 5.17, 6.6e3, 191]))


            idx_core= int(int("".join(filter(str.isdigit, self.output_fol)))-1)
            print(f'     > Index is {idx_core}')

            self.k0_pert    = vals_core[idx_core][0]
            self.k0p_pert   = vals_core[idx_core][1]
            self.rho0_pert  = vals_core[idx_core][2]
            self.k1_pert    = vals_core[idx_core][3]
            self.k1p_pert   = vals_core[idx_core][4]
            self.rho1_pert  = vals_core[idx_core][5]
            self.dlvl_pert  = vals_core[idx_core][6]

        else:
            # Get parameters given initial prior info
            self.dlvl_pert  = np.random.uniform(self.param_pert['dlvl']['bd'][0],
                                                self.param_pert['dlvl']['bd'][1])

            # K0' and K1'
            self.k0p_pert   = np.random.uniform(self.param_pert['k0p']['bd'][0],
                                                self.param_pert['k0p']['bd'][1])
            self.k1p_pert   = np.random.uniform(self.param_pert['k1p']['bd'][0],
                                                self.param_pert['k1p']['bd'][1])

            # K0 and K1
            self.k0_pert    = np.random.uniform(self.param_pert['k0']['bd'][0],
                                                self.param_pert['k0']['bd'][1])
            bd_min          = np.max([self.param_pert['k1']['bd'][0], self.k0_pert])
            self.k1_pert    = np.random.uniform(bd_min, self.param_pert['k1']['bd'][1])

            # rho0 and rho1
            self.rho0_pert  = np.random.uniform(self.param_pert['rho0']['bd'][0],
                                                self.param_pert['rho0']['bd'][1])
            bd_min          = np.max([self.param_pert['rho1']['bd'][0], self.rho0_pert])
            self.rho1_pert  = np.random.uniform(bd_min, self.param_pert['rho1']['bd'][1])

        #------------------------------------------------------------------
        # others
        self.zcrust_pert= self.param_pert['zcrust']['val']
        self.vs_crpert  = self.param_pert['vs_crust']['val']

        self.vp_crpert  = self.alpha_pert * self.vs_crpert
        self.rho_crpert = self.beta_pert * self.vs_crpert

        # assign to dictionary in case they were generated somehow else
        self.param_pert['zlit']['val']  = self.zlit_pert
        self.param_pert['tlit']['val']  = self.tlit_pert
        self.param_pert['zcmb1']['val'] = self.zcmb1_pert
        self.param_pert['alpha']['val'] = self.alpha_pert
        self.param_pert['beta']['val']  = self.beta_pert
        self.param_pert['k0']['val']    = self.k0_pert
        self.param_pert['k0p']['val']   = self.k0p_pert
        self.param_pert['rho0']['val']  = self.rho0_pert
        self.param_pert['k1']['val']    = self.k1_pert
        self.param_pert['k1p']['val']   = self.k1p_pert
        self.param_pert['rho1']['val']  = self.rho1_pert
        self.param_pert['dlvl']['val']  = self.dlvl_pert
        self.param_pert['vs_crust']['val']  = self.vs_crpert
        self.param_pert['zcrust']['val']= self.zcrust_pert

        return

    def read_picks(self):
        '''
        read the observed picks
        return data as a dictionary data_obs[event][info]
        input file must have
        events - phases - sigphases - depth - bazi - dist
        '''
        data    = []
        events  = []
        path    = os.path.join(self.input_fol, self.input_picks)
        with open(path) as f:
            reader  = csv.reader(f)
            row1    = next(reader) # Get first row with categories
            for row in f:
                data.append(row.split(',')[1:]) # Get obs data
                events.append(row.split(',')[0])# Get event name

        cat = list(filter(None, row1))
        cat.remove('Event')
        data_obs = {}
        for ii,_ in enumerate(events):
            data_obs[events[ii]] = {}
            for jj,_ in enumerate(cat):
                data_obs[events[ii]][cat[jj]] = float(data[ii][jj])

        # Read event info like dict data_obs[event][info]
        phases_obs = []
        for cc in cat:
            if not any(exc_cat in cc for exc_cat in self.exc):
                phases_obs.append(cc)
        sig_obs = []
        for pha in phases_obs:
            sig_obs.append('sig'+pha)

        self.data_obs   = data_obs
        self.phases_obs = phases_obs    # observed phases
        self.sigma_obs  = sig_obs       # uncertanties
        self.events     = events        # events

        # epicentral distances
        self.dist_ev    = np.zeros_like(self.events, dtype=float)
        self.depth_ev   = np.zeros_like(self.events, dtype=float)
        self.bazi_ev    = np.zeros_like(self.events, dtype=float)

        self.idx_RFps = np.array(())
        self.idx_RFsp = np.array(())
        self.idx_depth  = []

        for i,ev in enumerate(self.events):
            self.dist_ev[i] = self.data_obs[ev]['Dist']
            self.depth_ev[i]= self.data_obs[ev]['Depth']
            self.bazi_ev[i] = self.data_obs[ev]['Baz']

            if ev in self.events_ps:
                self.idx_RFps = np.append(self.idx_RFps, i)
            if ev in self.events_sp:
                self.idx_RFsp = np.append(self.idx_RFsp, i)

            if np.any(~np.isnan([self.data_obs[ev]['pP'], self.data_obs[ev]['sS']])):
                self.idx_depth.append(i)

        # indexes of events for which compute RFs
        self.idx_RFps   = self.idx_RFps.astype(int)
        self.idx_RFsp   = self.idx_RFsp.astype(int)

        self.idx_RF = np.unique(np.concatenate((self.idx_RFsp, self.idx_RFps)))

        self.dist_evpert= np.copy(self.dist_ev)
        self.depth_evpert   = np.copy(self.depth_ev)
        self.param_pert['distances']['val'] = self.dist_evpert
        self.param_pert['depths']['val'] = self.depth_evpert

        return

#==========================================================================

class FM(PRMS):
    '''
    class for forward modelling of structural model
    '''

    def __init__(self):
        PRMS.__init__(self)
        self.fm_run()
        return

    def fm_run(self):
        '''
        generate complete forward model
        '''
        self.depth_nodes()
        self.temp_nodes()
        self.comps_ak()
        self.get_structure()
        if self.ier:
            return
        self.taup_file_mantle()
        self.taup_file_lvl()

        return

    def depth_nodes(self):
        '''
        based on prms, generate depth nodes with different resolution
        depending on region for the crust and mantle
        '''
        moho_depth  = self.zcrust_pert[-1]

        repeat_idx = np.ones(len(self.zcrust_pert))
        repeat_idx[1:-1]    *= 2

        self.depth_pert = np.concatenate((np.repeat(self.zcrust_pert, repeat_idx.astype(int)),\
                np.linspace(moho_depth, self.zlit_pert, self.nzlit, endpoint=False), \
                np.linspace(self.zlit_pert,self.zdisc,self.nzmant, endpoint=False),\
                np.linspace(self.zdisc, self.zcmb1_pert,self.nzdisc, endpoint=True)))
        self.nlayers    = self.depth_pert.shape[0]
        self.ncrust     = np.where(self.depth_pert==moho_depth)[0][0]

        return


    def temp_lithosphere(self,
                         gradient='conductivity'):

        tlit_pert   = np.copy(self.tlit_pert)
        tlit_pert   -= 273.15

        if gradient=='linear':
            temp_lith = np.interp(self.depth_pert,\
                              np.array([self.depth_ini, self.zlit_pert]),
                              np.array([self.tsurf, tlit_pert+273.15]))

            temp_lith[self.depth_pert>self.zlit_pert]=0


        if gradient=='conductivity':

            temp_lith   = - (self.alphaT_pert*(self.depth_pert**2)) + (tlit_pert/self.zlit_pert  + self.alphaT_pert*self.zlit_pert) * self.depth_pert +273.15

            temp_lith[self.depth_pert>self.zlit_pert]=0

        return temp_lith


    def temp_nodes(self):
        '''
        create array of temperature nodes
        conductive lithosphere
        '''

        if self.count_all==0:

            self.temp_pert  = np.zeros(self.depth_pert.shape)
            mask_depth  = self.depth_pert>self.zlit_pert

            if not self.adiabatic:

                depth, temp = np.loadtxt(self.temp_file, unpack=True)
                ff  = interp1d(depth, temp)
                temp_inter  = ff(self.depth_pert)
                self.temp_pert[mask_depth]   = temp_inter[mask_depth]

                self.tlit_pert   = self.temp_pert[np.argwhere(mask_depth)[0][0]]-10

                # CAREFUL, CHECK BEFORE THAT INITIAL MODEL OF
                # TEMPERATURE IS EQUAL OR HIGHER THAN THE CURRENT
                # TLIT/ZLIT COMBINATION

            temp_lith  = self.temp_lithosphere()

            self.temp_pert[~mask_depth] = temp_lith[~mask_depth]

            # only in first iteration read file
            # Downsample temperature to perturb it
            self.depth_down = np.linspace(self.zlit_pert,
                                          self.zcmb1_pert,
                                          self.ntemp)
            fdown       = interp1d(self.depth_pert, self.temp_pert,
                                   fill_value='extrapolate')
            self.temp_down_pert = ff(self.depth_down)



        else:
            # after perturbing temperature model is constructed differently
            self.tlit_pert  = self.temp_down_pert[0]

            mask_depth  = self.depth_pert<self.zlit_pert
            depth_lit   = self.depth_pert[mask_depth]

            temp_lit    = self.temp_lithosphere()

            fup = interp1d(self.depth_down, self.temp_down_pert,
                           fill_value='extrapolate')

            self.depth_down = np.linspace(self.zlit_pert,
                                          self.zcmb1_pert,
                                          self.ntemp)

            self.temp_pert  = fup(self.depth_pert)
            self.temp_pert[mask_depth]  = temp_lit[mask_depth]



        return

    def comps_ak(self):
        '''
        asign compositions based on AK21
        comp: CaO, FeO, MgO, Al2O3, SiO3, Na2O
        '''
        ncomp   = 6

        CaO     = np.array([7, self.cfmasna[0]])
        FeO     = np.array([18.8, self.cfmasna[1]])
        MgO     = np.array([9.2, self.cfmasna[2]])
        Al2O3   = np.array([10.9, self.cfmasna[3]])
        SiO2    = np.array([50.7, self.cfmasna[4]])
        Na2O    = np.array([3.3, self.cfmasna[5]])

        self.comp   = np.ones((self.nlayers, ncomp))

        idx = 0
        for jj in range(self.nlayers):
            if jj>self.ncrust:
                idx=1
            self.comp[jj]   = np.array([CaO[idx], FeO[idx], MgO[idx], \
                                        Al2O3[idx], SiO2[idx], Na2O[idx] ])
        return


    def odes_pg(self, r, z, rho):
        """
        Definition of the relevant ODE's as in Rivoldini et al.
        """
        [P, g] = z
        return [- rho * g,
                4. * np.pi * self.Gnewt * rho - 2. * g / r]

    def core_structure(self, pg, radius, K0, K0prime, V0, core_param=False):
        """
        Calls the EoS and integrates the radius up
        """
        def vinet_eos(x, p, K0S, K0S_prime, V0):
            p_gpa = p / 1e9
            eta = 1.5 * (K0S_prime - 1.0)

            f = 3.0 * K0S * (x / V0) ** (- 2.0 / 3.0) *\
                (1.0 - (x / V0) ** (1.0 / 3.0)) *\
                np.exp(eta * (1.0 - (x / V0) ** (1.0 / 3.0))) - p_gpa

            return f

        def vinet_get_Vrho(p, K0S, K0S_prime, V0, M=0.05):
            root_obj = root_scalar(vinet_eos,
                                   args=(p, K0S, K0S_prime, V0),
                                   bracket=[0.1 * V0, 1.5 * V0],
                                   method='brentq')
            if (root_obj.converged):
                V = root_obj.root
            else:
                raise RuntimeError(
                    'Getting volume from the Vinet EoS did not converge.')

            return V, M / V

        def vinet_get_ksvp(V, V0, K0S, K0S_prime, rho):
            """
            :input:
            volume in m**3, molar volume in m**3, K0S in GPa, K0S_prime
            density in kg / m**3
            :output:
            p-wave velocity in m/s
            """
            K0S_pa = K0S * 1e9
            Vbar = V / V0
            eta = 1.5 * (K0S_prime - 1.0)

            Vbarp13 = Vbar ** (1.0 / 3.0)

            KS = K0S_pa * Vbar ** (- 2.0 / 3.0) *\
                (1.0 + (eta * Vbarp13 + 1.0) * (1.0 - Vbarp13)) *\
                np.exp(eta * (1.0 - Vbarp13))

            return KS, np.sqrt(KS / rho)

        ps = [pg[0]]
        gs = [pg[1]]
        [V_i, rho_i] = vinet_get_Vrho(ps[-1],
                                      K0,
                                      K0prime,
                                      V0)
        Ks_i, vp_i = vinet_get_ksvp(V_i, V0, K0, K0prime, rho_i)
        V = [V_i]
        rho = [rho_i]
        vp = [vp_i]

        for k, _ in enumerate(radius[:-1]):
            rint = radius[k:k + 2]
            sol = solve_ivp(self.odes_pg,
                            rint,
                            [ps[-1], gs[-1]],
                            args=(rho[-1],))

            pgsol = sol.y
            ps.append(pgsol[0][-1])
            gs.append(pgsol[1][-1])

            [V_i, rho_i] = vinet_get_Vrho(ps[-1],
                                          K0,
                                          K0prime,
                                          V0)

            Ks_i, vp_i = vinet_get_ksvp(V_i, V0, K0, K0prime, rho_i)

            rho.append(rho_i)
            vp.append(vp_i)
            V.append(V_i)

        # globalize
        ps = np.array(ps)
        rho = np.array(rho)
        vp = np.array(vp)
        vs = np.zeros_like(vp)
        gs = np.array(gs)

        if core_param:
            k0ref   = Ks_i
            rho0ref = rho_i
            return rho, vp, vs, ps, gs, k0ref, rho0ref

        return rho, vp, vs, ps, gs

    def min_core(self, pg, bc, radius, K0, K0prime, V0):
        """
        Minimization of the CMB boundary conditions based on
        L2-norm of pressure and gravity constant
        """
        out = self.core_structure([pg[0], 0.0], radius, K0, K0prime, V0)
        po = out[-2][-1]
        go = out[-1][-1]
        return np.linalg.norm([bc[0] - po, bc[1] - go])

    def get_core_structure(self, pcmb):
        '''
        core structure based on an EPOC implementation
        (changed to BM of Rivoldini)
        '''

        try:
            # do domething
            rcmb    = 1.e3*(self.r_p - self.depth_pert[-1])  # limit cmb [km]
            depths  = 1.e3*np.copy(self.depth_pert)

            radius, ps, gs, mass_core, iner, rho, vp, \
            vs, coreier  = outercore(self.ncore, rcmb, pcmb,
                                     self.rho1_pert, self.k1_pert*1.e9,
                                     self.k1p_pert)

            depths  = self.r_p - radius
            self.ier= bool(coreier)

            if np.isclose(vp[0], 0, atol=0.5):
                self.ier    = True

            # insert core
            self.vs_pert   = np.append(self.vs_pert,vs)
            self.vp_pert   = np.append(self.vp_pert, vp)
            self.rho_pert  = np.append(self.rho_pert, rho)
            self.depth_pert= np.append(self.depth_pert, depths)
            self.pres_pert = np.append(self.pres_pert, ps)

            print('K1=', self.k1_pert, ', K1p=', self.k1p_pert, ', rho1=', self.rho1_pert)

        except:
            print('  Core problems')
            self.ier    = True

        return

    def get_lvl_structure(self, pcmb):
        '''
        partially melted structure based on an EPOC implementation
        (changed to BM of Rivoldini)
        '''

        try:
            # do domething
            rcmb    = 1.e3*(self.r_p - self.zcmb1_pert)  # limit cmb [km]
            depths  = 1.e3*np.copy(self.depth_pert)

            radius, ps, gs, mass_core, iner, rho, vp, \
            vs, coreier  = outercore(self.ncore, rcmb, pcmb,
                                     self.rho0_pert, self.k0_pert*1.e9,
                                     self.k0p_pert)

            depths  = self.r_p - radius
            self.ier= bool(coreier)

            if np.isclose(vp[0], 0, atol=0.5):
                self.ier= True

            # interpolate to better resolve the layer
            depths_lvl  = np.linspace(self.zcmb1_pert,
                                      self.zcmb1_pert+self.dlvl_pert,
                                      self.nzlvl,
                                      endpoint=True)
            vp_lvl  = np.interp(depths_lvl, depths, vp)
            vs_lvl  = np.interp(depths_lvl, depths, vs)
            rho_lvl = np.interp(depths_lvl, depths, rho)
            pres_lvl= np.interp(depths_lvl, depths, ps)

            # insert lvl
            self.vs_pert   = np.append(self.vs_pert,vs_lvl)
            self.vp_pert   = np.append(self.vp_pert, vp_lvl)
            self.rho_pert  = np.append(self.rho_pert, rho_lvl)
            self.depth_pert= np.append(self.depth_pert, depths_lvl)
            self.pres_pert = np.append(self.pres_pert, pres_lvl)

            print('K0=', self.k0_pert, ', K0p=', self.k0p_pert, ', rho0=', self.rho0_pert)

        except:
            print('  Core problems')
            self.ier    = True

        return



    def check_perplex(self, ier):
        '''
        check for error in perplex models
        skips model if it's broken
        '''

        if ier:
            self.ier    = bool(ier)
        elif np.any(np.diff(self.depth_pert)<0):
            self.ier    = True
        elif np.any(np.diff(self.pres_pert)<0):
            self.ier    = True
        elif np.any(np.diff(self.temp_pert)<0):
            self.ier    = True
        elif np.any(self.vp_pert<0) or np.any(self.vs_pert<0):
            self.ier    = True
        else:
            self.ier    = False

        return

    def get_structure(self):
        '''
        run perplex, core and attenuation to get structure
        '''

        print('  > Run perplex')

        grainsize_att   = self.grain_sz*np.ones_like(self.depth_pert)
        avolume_att     = self.activ_vol*np.ones_like(self.depth_pert)

        ier, rho_min, kappa, mu, vp_min, vs_min, pres, alpha, \
        cp, phase_prop, phase_list, nph_vec, xring, xppv, xmg, \
        xcpx, xwu, xpv, xwads, tad, S_lit, qsmant, qpmant, vseff = \
            depth_loop(comp = self.comp, T_start_K = self.tstart, \
                       zlit = self.zlit_pert, z = self.depth_pert, \
                       T = self.temp_pert, \
                       adiabatic = self.adiabatic, dt_in=-50 , \
                       period   = self.period_att, \
                       grainsize=grainsize_att, \
                       avolume  = avolume_att, \
                       m_planet = self.m_p, r_planet = self.r_p*1.e3)

        self.vs_pert    = vs_min
        self.vp_pert    = vp_min
        self.rho_pert   = rho_min/1.e3
        #self.temp_pert  = tad
        self.pres_pert  = pres
        self.depth_pert[0]  = 0.

        # Check for perplex error
        self.check_perplex(ier)

        if self.ier:
            print('----------------------------------------')
            print('----------------------------------------')
            print('-------- PERPLEX BROKE, RETURN ---------')
            return

        # replace crust
        vs_cr   = np.repeat(self.vs_crpert, 2)
        vp_cr   = np.repeat(self.vp_crpert, 2)
        rho_cr  = np.repeat(self.rho_crpert, 2)

        self.vs_pert[:vs_cr.shape[0]]   = vs_cr
        self.vp_pert[:vp_cr.shape[0]]   = vp_cr
        self.rho_pert[:rho_cr.shape[0]] = rho_cr

        # compute core structure
        print('  > Core computation')
        tcmb = tad[-1]      # adiabatic temperature at cmb [k]
        pcmb = pres[-1]     # presure at cmb [pa]

        # Get first core structure for partially melted layer
        self.get_lvl_structure(pcmb)
        if self.ier:
            return

        # Get core structure - new "pcmb" is pressure at bottom of partially
        # melted layer
        self.get_core_structure(self.pres_pert[-1])
        if self.ier:
            return

        self.qs_pert    = np.concatenate((self.qs_constant*np.ones_like(qsmant),
                                    1.e-5*np.ones(self.nzlvl), 1.e-5*np.ones(self.ncore)))

        self.qp_pert    = np.zeros_like(self.qs_pert)
        self.qp_pert  = 1./ ( (4. * self.vs_pert**2) / (3. * self.qs_pert * self.vp_pert**2) + \
                        (1. - ((4. * self.vs_pert**2) / (3. * self.vp_pert**2))) * (1./self.qkp))

        self.qs_pert[self.qs_pert<1.e-4]    = 0

        self.mass_planet()
        self.inertia_planet()

        return

    def mass_planet(self):
        '''
        compute synthetic mass/density of the planet
        '''

        radius  = (self.r_p - self.depth_pert)*1.e3
        dens    = self.rho_pert*1.e3

        mass_i  = np.zeros(len(radius)-1)
        press_i = np.zeros_like(mass_i)
        grav_i  = np.zeros_like(mass_i)

        sum_mass= 0.0
        p_bef   = 0
        for i in range(len(radius)-1):
            mass_i[i]   = 4./3. * np.pi * (dens[i] + dens[i+1])/2. *\
                (radius[i]**3 - radius[i+1]**3)
            sum_mass    += mass_i[i]
            grav_i[i]   = self.Gnewt * (self.m_p - sum_mass) / ((radius[i+1]+radius[i])/2.)**2
            press_i[i]  = p_bef + grav_i[i] * ((dens[i]+dens[i+1])/2.) * (radius[i] - radius[i+1])
            p_bef       = np.copy(press_i[i])

        self.pcmb_calc  = press_i[-1]
        self.gcmb_calc  = grav_i[-1]

        self.mass_calc  = np.sum(mass_i)    # synthetic mean mass in kg
        self.dens_calc  = 1.e-3 * self.mass_calc / (4./3. * np.pi * (self.r_p*1.e3)**3) # synthetic mean density in g/cm3

        return

    def inertia_planet(self):
        '''
        compute synthetic moment of inertia of the planet (sphere)
        '''
        radius  = (self.r_p - self.depth_pert)*1.e3
        dens    = self.rho_pert*1.e3

        moi_i   = np.zeros(len(radius)-1)

        for i in range(len(radius)-1):
            moi_i[i]    = 8./15. * np.pi * (dens[i] + dens[i+1])/2. *\
                (radius[i]**5 - radius[i+1]**5)

        self.moi_calc   = np.sum(moi_i) / (self.mass_calc * (self.r_p*1.e3)**2)
        return

    def taup_file_mantle(self):
        '''
        generate file with taup format
        '''

        print('  > Generating taup model ')

        with open(os.path.join(self.tmp, self.file_tmp.format('.nd')), 'w') as f:
            for ii,_ in enumerate(self.depth_pert):
                f.write('{:10.3f}{:10.4f}{:10.4f}{:10.4f}\n'.format(self.depth_pert[ii], self.vp_pert[ii], self.vs_pert[ii], self.rho_pert[ii]))
                if ii==self.ncrust:
                    f.write('mantle\n')
                elif ii==self.nlayers-1:
                    f.write('outer-core\n')
                elif ii==len(self.depth_pert)-2:    # quite dirty, fix
                    f.write('{:10.3f}{:10.4f}{:10.4f}{:10.4f}\n'.format(3389., self.vp_pert[ii], self.vs_pert[ii], self.rho_pert[ii]))
                    f.write('inner-core\n')
                    f.write('{:10.3f}{:10.4f}{:10.4f}{:10.4f}\n'.format(3389., 8., 5., 7.))
                    f.write('{:10.3f}{:10.4f}{:10.4f}{:10.4f}\n'.format(3389.5, 8., 5., 7.))

                    break
        return

    def taup_file_lvl(self):
        '''
        generate file with taup format
        '''

        print('  > Generating taup model with partially melted layer')

        vs_virtual  = np.copy(self.vs_pert)
        cond_depth  = np.full(len(self.depth_pert), False)
        cond_depth[:self.nlayers+self.nzlvl]    = True

        vs_virtual[(vs_virtual==0) & cond_depth]= 0.1

        with open(os.path.join(self.tmp, self.file_tmp_lvl.format('.nd')), 'w') as f:
            for ii,_ in enumerate(self.depth_pert):
                f.write('{:10.3f}{:10.4f}{:10.4f}{:10.4f}\n'.format(self.depth_pert[ii],
                                                                    self.vp_pert[ii],
                                                                    vs_virtual[ii],
                                                                    self.rho_pert[ii]))
                if ii==self.ncrust:
                    f.write('mantle\n')
                elif ii==self.nlayers+self.nzlvl-1:
                    f.write('outer-core\n')
                elif ii==len(self.depth_pert)-2:    # quite dirty, fix
                    f.write('{:10.3f}{:10.4f}{:10.4f}{:10.4f}\n'.format(3389., self.vp_pert[ii],
                                                                        vs_virtual[ii],
                                                                        self.rho_pert[ii]))
                    f.write('inner-core\n')
                    f.write('{:10.3f}{:10.4f}{:10.4f}{:10.4f}\n'.format(3389., 8., 5., 7.))
                    f.write('{:10.3f}{:10.4f}{:10.4f}{:10.4f}\n'.format(3389.5, 8., 5., 7.))

                    break
        return

#=========================================================================

class FM_AR(FM):
    '''
    compute travel times for forward model and RF stack
    for the latter it will run FM_RF in a parallel process
    '''

    def __init__(self):
        FM.__init__(self)
        if self.ier:
            return
        self.compute_times()
        return

    def run_taup(self, file, phases, idx_pha, arrival_pert,
                 incident_angle, zturn_pert):
        '''
        run taup for the given model, events and phases
        '''
        phases  = [''.join((x for x in pha if not x.isdigit())) for pha in phases]

        try:
            taup_create.build_taup_model(os.path.join(self.tmp,
                                                      file.format('.nd')),
                                         output_folder=self.tmp, verbose=False)

            model_TauP = TauPyModel(os.path.join(self.tmp,
                                                 file.format('.npz')))

            for j, ev in enumerate(self.events):
                arr_ev = model_TauP.get_travel_times(
                                    source_depth_in_km = self.depth_evpert[j],
                                    distance_in_degree = self.dist_evpert[j],
                                    phase_list = phases)
                path_ev= model_TauP.get_ray_paths(
                                    source_depth_in_km = self.depth_evpert[j],
                                    distance_in_degree = self.dist_evpert[j],
                                    phase_list = phases)
                # initialize array of arrival times
                for jj, pha in zip(idx_pha, phases):
                    arrival = [ele for ele in arr_ev if ele.name == pha]
                    path_arr= [ele for ele in path_ev if ele.name == pha]
                    if len(arrival)>0:
                        arrival_pert[j,jj]  = arrival[0].time
                        if pha==self.motherps:
                            incident_angle[j]   = arrival[0].incident_angle
                        if pha==self.mothersp:
                            incident_angle[j]   = arrival[0].incident_angle
                        zturn   = np.max(path_arr[0].path['depth'])
                        zturn_pert[j,jj]    = np.copy(zturn)
            ier_taup   = False
        except:
            print('ERROR IN TAUP')
            ier_taup   = True
            dict_data   = {'arrivals':arrival_pert,
                           'inc_angle':incident_angle,
                           'ier_taup':ier_taup,
                           'zturns':zturn_pert}
            return dict_data

        dict_data   = {'arrivals':arrival_pert,
                       'inc_angle':incident_angle,
                       'ier_taup':ier_taup,
                       'zturns':zturn_pert}

        return dict_data


    def compute_times(self):
        '''
        compute differential travel times based on observed data
        '''
        self.ier_taup   = False

        # Find indexes of phases that must be computed with one or another model
        idx_phalvl  = [self.phases_obs.index(pha) for pha in self.phases_lvl \
                       if pha in self.phases_obs]

        idx_phaman  = [self.phases_obs.index(pha) for pha in self.phases_obs \
                       if pha not in self.phases_lvl]
        # check that its done correctly
        inter_idx   = set(idx_phalvl).intersection(idx_phaman)
        if len(inter_idx)>0:
            print('     SETS HAVE ELEMENTS IN COMMON, CHECK CODE')
            self.ier= True
            return

        # Initialize arrays
        self.arrival_pert       = np.empty((len(self.events),
                                        len(self.phases_obs)))
        self.arrival_pert[:]    = np.nan
        self.incident_angleps   = np.empty(len(self.events))
        self.incident_angleps[:]= np.nan
        self.incident_anglesp   = np.empty(len(self.events))
        self.incident_anglesp[:]= np.nan

        self.zturn_pert         = np.empty((len(self.events),
                                        len(self.phases_obs)))
        self.zturn_pert[:]      = np.nan

        incident_angles = np.copy(self.incident_angleps)
        arrival_pert    = np.copy(self.arrival_pert)
        zturn_pert      = np.copy(self.zturn_pert)

        # run models in parallel and combine
        ncore   = 2

        files       = [self.file_tmp, self.file_tmp_lvl]
        phases_man  = [self.phases_obs[j] for j in idx_phaman]
        phases_lvl  = [self.phases_obs[j] for j in idx_phalvl]
        phases_list = [phases_man, phases_lvl]
        phases_idx  = [idx_phaman, idx_phalvl]

        with Pool(processes=ncore) as pool:
            data = pool.starmap(self.run_taup,
                                [(files[p], phases_list[p], phases_idx[p],
                                  arrival_pert, incident_angles, zturn_pert) for p in range(ncore)])

        # check errors first
        iers    = [data[k]['ier_taup'] for k in range(ncore)]
        if np.any(iers):
            self.ier_taup   = True
            return

        barr    = np.stack((data[0]['arrivals'], data[1]['arrivals']))
        arrivals= np.nansum(barr, axis=0)
        arrivals[np.all(np.isnan(barr), axis=0)] = np.nan
        self.arrival_pert   = np.copy(arrivals)

        carr    = np.stack((data[0]['inc_angle'], data[1]['inc_angle']))
        incident_angle  = np.nansum(carr, axis=0)
        incident_angle[np.all(np.isnan(carr), axis=0)] = np.nan
        self.incident_angleps   = np.copy(incident_angle)

        barr    = np.stack((data[0]['zturns'], data[1]['zturns']))
        zturns  = np.nansum(barr, axis=0)
        zturns[np.all(np.isnan(barr), axis=0)] = np.nan
        self.zturn_pert = np.copy(zturns)

        return

#=========================================================================
class FM_NM(FM):
    '''
    forward model for computing normal modes employing SPECNM
    '''

    def __init__(self):
        FM.__init__(self)
        if self.ier:
            return
        self.get_freqs()
        return
   
    def get_freqs(self):
        future  = self.compute_frequencies()

        try:
            ray_out = future.result()
        except TimeoutError:
            future.cancel()
            self.ier    = True
            print('     > skip model')
            return
        except (MemoryError, Exception):
            self.ier    = True
            print('     > skip model')
            return

        self.mode    = 'Rayleigh'
        self.modes_array    = {self.mode:{}}

        mode_type   = [key for key in self.modes_array.keys()]


        l_unique= np.unique(ray_out['angular orders'])

        for ll, ang in enumerate(l_unique):

            freqs   = ray_out['frequencies'][ray_out['angular orders']==ang]

            for overtone, freq in enumerate(freqs):
                if ang==1:  # for these specific models we don't get (0,1) and counting
                    overtone    +=1 # needs to be corrected

                mode_id = (overtone,ang)
                #if mode_id not in self.nm_obs:
                #    continue

                self.modes_array[self.mode].update({mode_id:freq*1.e3})

        #print(self.modes_array[self.mode])
        return

    @concurrent.process(timeout=300)
    def compute_frequencies(self,*,
                           fmax=0.005,
                           fmin=1.2e-3,
                           lmax=100,
                           att='eigenvector continuation stepwise',
                           maxmemory=6.0e9):

        # set memory limit for mp purposes
        resource.setrlimit(resource.RLIMIT_AS, (int(maxmemory), int(maxmemory)))

        file_model  = os.path.join(self.tmp, self.file_tmp.format('.bm'))

        with threadpool_limits(limits=1, user_api='blas'):

            print(' Run specnm')
            ray = rayleigh_fg(model=file_model,
                              fmax=3*fmax)

            ray_out = ray.rayleigh_problem(attenuation_mode=att,
                                           #llist=self.list_ll,
                                           fmin=fmin, fmax=fmax,
                                           lmax=lmax)
        return ray_out



#=========================================================================

class FM_RF(FM_AR):
    '''
    forward modelling of receiver functions
    writen to run in parallel and return each receiver function
    '''

    def __init__(self):
        '''
        idx_RF: index of event for which compute RF
        core: number of running core
        '''
        FM_AR.__init__(self)
        if self.ier:
            return
        # run Ps RF
        self.ref_fileps()
        if self.ier:
            return
        self.run_reflectivityps()

        # run Sp RF
        if self.run_sp:
            self.ref_filesp()
            if self.ier:
                return
            self.run_reflectivitysp()

        return

    def roundup(self, x):
        return int(np.ceil(x / 100.0)) * 100

    def ref_fileps(self):
        '''
        generate file with reflectivity format
        will read epicentral distances from perturbed distances for
        each event
        EVENTS FOR PS RF
        '''

        print('  > Read in input-file for reflectivity parameters Ps')

        self.epidegps   = np.copy(self.dist_evpert[self.idx_RFps])
        depth_RF        = np.mean(self.depth_evpert[self.idx_RFps])

        zs  = depth_RF

        z   = self.depth_pert
        vp  = self.vp_pert
        vs  = self.vs_pert
        rho = self.rho_pert
        qp  = self.qp_pert
        qs  = self.qs_pert

        prms_ref    = __import__(self.input_ref)

        freq 	= prms_ref.freq
        dt 	    = prms_ref.dt
        Tsec 	= prms_ref.Tsec
        mt 	    = prms_ref.mtps
        vred 	= prms_ref.vred
        tmin 	= prms_ref.tmin

        rp 	= self.r_p

        string  = prms_ref.string
        l1      = prms_ref.l1
        l2      = prms_ref.l2
        zr      = prms_ref.zr
        xs      = prms_ref.xs
        ys      = prms_ref.ys
        ts      = prms_ref.ts
        es      = prms_ref.es
        azi     = prms_ref.azi
        c2      = prms_ref.c2
        cwil    = prms_ref.cwil
        cwir    = prms_ref.cwir
        c1      = prms_ref.c1
        fr      = prms_ref.fr
        na      = prms_ref.na
        nextr   = prms_ref.nextr
        perc    = prms_ref.perc

        depth_near, layer_source = find_nearest(z, zs)

        l2  = l2.format(layer_source)
        fu  = freq[0]
        fwil= freq[1]
        fwir= freq[2]
        fo  = freq[3]

        epikm   = self.epidegps * 2. * np.pi * rp / 360.
        if hasattr(epikm, '__len__'):
            sort_RF = np.argsort(epikm)
            self.idx_RFps   = self.idx_RFps[sort_RF]
            epikm   = np.sort(epikm)
            nsta    = len(epikm)
            depi    = (np.max(epikm) - np.min(epikm))/nsta
            epimax  = np.max(epikm)
            azi     = np.tile(azi, nsta)
        else:
            nsta    = 1
            depi    = 0
            epimax  = np.copy(epikm)
            self.sort_RFps    = None

        fN = 1./(2.*dt)
        if (fo>fN):
            fo   = fN
            fwir = fo-.05
            print('!  too high frequency. Will be set to {} instead of {}'.format(fo,fN))

        npts    = Tsec / dt
        npts    = np.ceil(np.log2(npts))
        npts    = int(2.**float(npts))
        tsigma  = perc * dt * float(npts)

        nprm    = self.roundup((epimax*fwir)/c2)

        file_name   = 'crfl.dat'
        with open(file_name,'w') as f:
            f.write(string+'\n')
            f.write(l1+'\n')
            f.write(l2+'\n')

            for i in range((z.size)): # silicate earth
                if z[i]>2800.:  # CAREFUL, ONLY FOR MARS
                    break
                if (i <= self.ncrust and z[i] != z[i+1]):
                    ix = 30.
                else:
                    ix = 1.
                f.write('%10.4f%10.4f%10.4f%10.4f%10.4f%10.4f%10.0f\n' \
                        % (z[i], vp[i], qp[i], vs[i], qs[i], rho[i], ix))

            f.write('\n')
            f.write('%10.4f\n' % zr)
            f.write('%10.4f%10.4f%10.4f%10.4f%10.4f\n'\
                    % (xs,ys,zs,ts,es))
            f.write('%10.4f%10.4f%10.4f%10.4f%10.4f%10.4f\n'\
                    % (mt[0],mt[1],mt[2],mt[3],mt[4],mt[5]))

            # different receiver distances
            if hasattr(epikm, '__len__'):
                f.write('%10.4f%10.4f%10.3f%10.3f%10d\n'\
                    % (epikm[0],epikm[-1],depi,azi[0],nsta))

                for j in range(nsta):
                    f.write('%10.4f' % (epikm[j]))
                    if ((j+1) % 8 == 0):
                        f.write('\n')
                if nsta % 8 != 0:
                    f.write('\n')

                for j in range(nsta):
                    f.write('%10.4f' % (azi[j]))
                    if ((j+1) % 8 == 0):
                        f.write('\n')
                if nsta % 8 != 0:
                    f.write('\n')

            else:   # if it's single receiver
                f.write('%10.4f%10.4f%10.3f%10.3f%10d\n'\
                    % (epikm,epikm,depi,azi,nsta))

                f.write('%10.4f \n' % (epikm))

                f.write('%10.4f \n' % (azi))

            frmt = '%10.4f\n'
            f.write('%10.4f%10.4f\n' % (vred,tmin))
            f.write('%10.4f%10.4f%10.4f%10.4f%10d\n'\
                    % (c2,cwil,cwir,c1,nprm))
            f.write('%10.4f%10.4f%10.4f%10.4f%10.4f\n'\
                    % (fu,fwil,fwir,fo,fr))
            f.write('%10.4f%10d%10d%10d%10.4f%10.4f\n'\
                    % (dt,npts,na,nextr,dt,tsigma))
            f.close()

        return

    def ref_filesp(self):
        '''
        generate file with reflectivity format
        will read epicentral distances from perturbed distances for
        each event
        EVENTS FOR SP RF
        '''

        print('  > Read in input-file for reflectivity parameters Sp')

        self.epidegsp   = np.copy(self.dist_evpert[self.idx_RFsp])
        depth_RF        = np.mean(self.depth_evpert[self.idx_RFsp])

        zs  = depth_RF

        z   = self.depth_pert
        vp  = self.vp_pert
        vs  = self.vs_pert
        rho = self.rho_pert
        qp  = self.qp_pert
        qs  = self.qs_pert

        prms_ref    = __import__(self.input_ref)

        freq 	= prms_ref.freq
        dt 	= prms_ref.dt
        Tsec 	= prms_ref.Tsec
        mt 	= prms_ref.mtsp
        vred 	= prms_ref.vred
        tmin 	= prms_ref.tmin

        rp 	= self.r_p

        string  = prms_ref.string
        l1      = prms_ref.l1
        l2      = prms_ref.l2
        zr      = prms_ref.zr
        xs      = prms_ref.xs
        ys      = prms_ref.ys
        ts      = prms_ref.ts
        es      = prms_ref.es
        azi     = prms_ref.azi
        c2      = prms_ref.c2
        cwil    = prms_ref.cwil
        cwir    = prms_ref.cwir
        c1      = prms_ref.c1
        fr      = prms_ref.fr
        na      = prms_ref.na
        nextr   = prms_ref.nextr
        perc    = prms_ref.perc

        depth_near, layer_source = find_nearest(z, zs)

        l2  = l2.format(layer_source)
        fu  = freq[0]
        fwil= freq[1]
        fwir= freq[2]
        fo  = freq[3]

        epikm   = self.epidegsp * 2. * np.pi * rp / 360.
        if hasattr(epikm, '__len__'):
            sort_RF = np.argsort(epikm)
            self.idx_RFsp   = self.idx_RFsp[sort_RF]
            epikm   = np.sort(epikm)
            nsta    = len(epikm)
            depi    = (np.max(epikm) - np.min(epikm))/nsta
            epimax  = np.max(epikm)
            azi     = np.tile(azi, nsta)
        else:
            nsta    = 1
            depi    = 0
            epimax  = np.copy(epikm)
            self.sort_RFsp    = None

        fN = 1./(2.*dt)
        if (fo>fN):
            fo   = fN
            fwir = fo-.05
            print('!  too high frequency. Will be set to {} instead of {}'.format(fo,fN))

        npts    = Tsec / dt
        npts    = np.ceil(np.log2(npts))
        npts    = int(2.**float(npts))
        tsigma  = perc * dt * float(npts)

        nprm    = self.roundup((epimax*fwir)/c2)

        file_name   = 'crfl.dat'
        with open(file_name,'w') as f:
            f.write(string+'\n')
            f.write(l1+'\n')
            f.write(l2+'\n')

            for i in range((z.size)): # silicate earth
                if z[i]>2800.:  # CAREFUL, ONLY FOR MARS
                    break
                if (i <= self.ncrust and z[i] != z[i+1]):
                    ix = 30.
                else:
                    ix = 1.
                f.write('%10.4f%10.4f%10.4f%10.4f%10.4f%10.4f%10.0f\n' \
                        % (z[i], vp[i], qp[i], vs[i], qs[i], rho[i], ix))

            f.write('\n')
            f.write('%10.4f\n' % zr)
            f.write('%10.4f%10.4f%10.4f%10.4f%10.4f\n'\
                    % (xs,ys,zs,ts,es))
            f.write('%10.4f%10.4f%10.4f%10.4f%10.4f%10.4f\n'\
                    % (mt[0],mt[1],mt[2],mt[3],mt[4],mt[5]))

            # different receiver distances
            if hasattr(epikm, '__len__'):
                f.write('%10.4f%10.4f%10.3f%10.3f%10d\n'\
                    % (epikm[0],epikm[-1],depi,azi[0],nsta))

                for j in range(nsta):
                    f.write('%10.4f' % (epikm[j]))
                    if ((j+1) % 8 == 0):
                        f.write('\n')
                if nsta % 8 != 0:
                    f.write('\n')

                for j in range(nsta):
                    f.write('%10.4f' % (azi[j]))
                    if ((j+1) % 8 == 0):
                        f.write('\n')
                if nsta % 8 != 0:
                    f.write('\n')

            else:   # if it's single receiver
                f.write('%10.4f%10.4f%10.3f%10.3f%10d\n'\
                    % (epikm,epikm,depi,azi,nsta))

                f.write('%10.4f \n' % (epikm))

                f.write('%10.4f \n' % (azi))

            frmt = '%10.4f\n'
            f.write('%10.4f%10.4f\n' % (vred,tmin))
            f.write('%10.4f%10.4f%10.4f%10.4f%10d\n'\
                    % (c2,cwil,cwir,c1,nprm))
            f.write('%10.4f%10.4f%10.4f%10.4f%10.4f\n'\
                    % (fu,fwil,fwir,fo,fr))
            f.write('%10.4f%10d%10d%10d%10.4f%10.4f\n'\
                    % (dt,npts,na,nextr,dt,tsigma))
            f.close()

        return

    def run_reflectivityps(self):
        '''
        run reflectivity and read traces and info
        '''
        os.system('{}/{}'.format(self.ref_fol, self.ref_executable))

        # read traces for each event and sort them by component
        comp_sort = ['Z', 'R']

        self.RF_allps     = Stream()

        for nrf in range(self.epidegps.size):
            ref_files   = glob.glob('st{0:03d}.*'.format(nrf+1))

            ref_files = [f for c in comp_sort for f in ref_files \
                     if f[-1].upper() == c]

            stream = Stream()
            for file, comp in zip(ref_files, comp_sort):
                trace   = read(file)[0]
                trace.stats.channel = comp

                if comp=='Z' or comp=='R':
                    trace.data *= -1.

                stream.append(trace)

            B2          = stream[0].stats.sac.b         # beginning time
            tstart2     = stream[0].stats.starttime     # absolute starttime
            reftime     = tstart2 - B2          # absolute time of event
            dt          = stream[0].stats.delta

            # put origin time as starttime
            stream.trim(starttime=reftime,
                        endtime=stream[0].stats.endtime)

            self.compute_rfps(stream, reftime, nrf, dt)

            if self.ier:
                return

        self.RF_allps.filter('bandpass', freqmin=self.filter_rf[0],\
                        freqmax=self.filter_rf[1],\
                        corners=self.filter_order,\
                        zerophase=True)

        self.RF_stackps   = self.RF_allps.copy()

        try:
            self.RF_stackps.stack()
            self.RF_stackps.normalize()
        except:
            self.ier = True

        return

    def run_reflectivitysp(self):
        '''
        run reflectivity and read traces and info
        '''
        os.system('{}/{}'.format(self.ref_fol, self.ref_executable))

        # read traces for each event and sort them by component
        comp_sort = ['Z', 'R']

        self.RF_allsp     = Stream()

        for nrf in range(self.epidegsp.size):
            ref_files   = glob.glob('st{0:03d}.*'.format(nrf+1))

            ref_files = [f for c in comp_sort for f in ref_files \
                     if f[-1].upper() == c]

            stream = Stream()
            for file, comp in zip(ref_files, comp_sort):
                trace   = read(file)[0]
                trace.stats.channel = comp

                if comp=='Z' or comp=='R':
                    trace.data *= -1.

                stream.append(trace)

            B2          = stream[0].stats.sac.b         # beginning time
            tstart2     = stream[0].stats.starttime     # absolute starttime
            reftime     = tstart2 - B2          # absolute time of event
            dt          = stream[0].stats.delta

            # put origin time as starttime
            stream.trim(starttime=reftime,
                        endtime=stream[0].stats.endtime)

            self.compute_rfsp(stream, reftime, nrf, dt)

            if self.ier:
                return

        self.RF_allsp.filter('bandpass', freqmin=self.filter_rf[0],\
                        freqmax=self.filter_rf[1],\
                        corners=self.filter_order,\
                        zerophase=True)

        self.RF_stacksp   = self.RF_allsp.copy()

        try:
            self.RF_stacksp.stack()
            self.RF_stacksp.normalize()
        except:
            self.ier = True

        return


    def compute_rfps(self, stream, reftime, nrf, dt):
        '''
        compute receiver functions Ps
        '''

        idx_pha     = self.phases_obs.index(self.motherps)
        if np.isnan(self.arrival_pert[self.idx_RFps[nrf],idx_pha]):
            self.ier    = True
            return

        for st in stream:
            tt = st.times()
            ttval, idx_cut = find_nearest(tt, self.arrival_pert[self.idx_RFps[nrf],idx_pha]-5)
            st.data[:idx_cut] = 0

        t_mother    = reftime + self.arrival_pert[self.idx_RFps[nrf],idx_pha]

        stream_lqt  = self.rot_lqtps(stream, nrf)

        # compute RF
        if self.motherps=='P':
            source_pha  = 'L'
            resp_pha    = 'Q'
        else:
            print('NOT PROGRAMMED FOR Sp RFs')

        # initialize source and response streams
        stream_rf_src = stream_lqt.select(component=source_pha)[0].copy()
        stream_rf_res = stream_lqt.copy()

        stream_rf_src.trim(t_mother - self.win_srcps[0],\
                                t_mother + self.win_srcps[1])
        stream_rf_res.trim(t_mother - self.win_resps[0],\
                                t_mother + self.win_resps[1])

        stream_rf_src.taper(type=self.taper_win,\
                              max_percentage=self.taper_per)
        stream_rf_res.taper(type=self.taper_win,\
                              max_percentage=self.taper_per)

        mid_src = -1.*self.win_srcps[0]+np.mean(self.win_srcps)
        shift   = int(-mid_src / dt)

        # deconvolution
        try:
            rec_fct = deconvolve.deconv_time(rsp_list = [stream_rf_res.select(component=resp_pha)[0]],\
                                             src    = stream_rf_src, \
                                             shift  = shift)
        except:
            self.ier = True
            return

        norm = np.max(np.abs(rec_fct[0]))
        rec_fct = rec_fct[0] / norm

        if self.motherps == 'P':
            ileft   = 0
            irght   = 1
            switch  = 1.
            rec_fct *= -1.

        trace_rf    = Trace(data=switch*rec_fct)
        trace_rf.stats.delta    = dt
        trace_rf.stats.sampling_rade = 1./dt
        trace_rf.stats.location = str(self.win_resps[ileft])
        self.RF_allps.append(trace_rf)

        return

    def compute_rfsp(self, stream, reftime, nrf, dt):
        '''
        compute receiver functions Sp
        '''

        idx_pha     = self.phases_obs.index(self.mothersp)
        if np.isnan(self.arrival_pert[self.idx_RFsp[nrf],idx_pha]):
            self.ier    = True
            return

        idx_p     = self.phases_obs.index(self.motherps)
        for st in stream:
            tt = st.times()
            ttval, idx_cut = find_nearest(tt, self.arrival_pert[self.idx_RFsp[nrf],idx_p]-5)
            st.data[:idx_cut] = 0

        t_mother    = reftime + self.arrival_pert[self.idx_RFsp[nrf],idx_pha]

        stream_lqt  = self.rot_lqtsp(stream, nrf)

        # compute RF
        if self.mothersp=='S':
            source_pha  = 'Q'
            resp_pha    = 'L'
        else:
            print('NOT PROGRAMMED FOR PS RFs')

        # initialize source and response streams
        stream_rf_src = stream_lqt.select(component=source_pha)[0].copy()
        stream_rf_res = stream_lqt.copy()

        stream_rf_src.trim(t_mother - self.win_srcsp[0],\
                                t_mother + self.win_srcsp[1])
        stream_rf_res.trim(t_mother - self.win_ressp[0],\
                                t_mother + self.win_ressp[1])

        stream_rf_src.taper(type=self.taper_win,\
                              max_percentage=self.taper_per)
        stream_rf_res.taper(type=self.taper_win,\
                              max_percentage=self.taper_per)

        mid_src = -1.*self.win_srcsp[0]+np.mean(self.win_srcsp)
        shift   = int(-mid_src / dt)

        # deconvolution
        try:
            rec_fct = deconvolve.deconv_time(rsp_list = [stream_rf_res.select(component=resp_pha)[0]],\
                                             src    = stream_rf_src, \
                                             shift  = shift)
        except:
            self.ier = True
            return

        norm = np.max(np.abs(rec_fct[0]))
        rec_fct = rec_fct[0] / norm

        if self.mothersp == 'S':
            rec_fct = np.flip(rec_fct,0) # flip RF
            ileft   = 1
            irght   = 0
            switch  = 1. # flip sign of arrivals

        trace_rf    = Trace(data=switch*rec_fct)
        trace_rf.stats.delta    = dt
        trace_rf.stats.sampling_rade = 1./dt
        trace_rf.stats.location = str(self.win_ressp[ileft])
        self.RF_allsp.append(trace_rf)

        self.times_rfsp = np.linspace(-1.*self.win_ressp[ileft],\
                            -1.*self.win_ressp[ileft]+trace_rf.stats.delta*float(len(rec_fct)), len(rec_fct))

        return

    def rot_lqtps(self, stream, nrf):
        '''
        rotate from ZRT to LQT using incident angle
        '''
        # look for incident angle of mother phase in synthetic

        stream_lqt  = stream.copy()
        inc_angle   = self.incident_angleps[self.idx_RFps[nrf]]

        inc_rad = radians(inc_angle)

        # rotate clockwise
        L = np.cos(inc_rad)*stream[0].data + np.sin(inc_rad)*stream[1].data
        Q = np.sin(inc_rad)*stream[0].data - np.cos(inc_rad)*stream[1].data

        stream_lqt[0].data = L; stream_lqt[0].stats.channel = 'L'
        stream_lqt[1].data = Q; stream_lqt[1].stats.channel = 'Q'

        return stream_lqt

    def rot_lqtsp(self, stream, nrf):
        '''
        rotate from ZRT to LQT using incident angle
        '''
        # look for incident angle of mother phase in synthetic

        stream_lqt  = stream.copy()
        inc_angle   = self.incident_anglesp[self.idx_RFsp[nrf]]

        inc_rad = radians(inc_angle)

        # rotate clockwise
        L = np.cos(inc_rad)*stream[0].data + np.sin(inc_rad)*stream[1].data
        Q = np.sin(inc_rad)*stream[0].data - np.cos(inc_rad)*stream[1].data

        stream_lqt[0].data = L; stream_lqt[0].stats.channel = 'L'
        stream_lqt[1].data = Q; stream_lqt[1].stats.channel = 'Q'

        return stream_lqt

#==========================================================================

class INV(FM_RF):

    def __init__(self):
        FM_RF.__init__(self)
        if self.ier:
            return
        self.rf_uncertainty()
        self.misfits()

        return

    def inversion(self):
        '''
        all functions necessary for inversion
        '''
        self.count_all  += 1
        self.count_block+=1
        self.pert_prms()
        self.fm_run()
        if self.ier:
            return
        self.ref_fileps()
        self.compute_times()
        if self.ier_taup:
            return
        self.run_reflectivityps()

        # add Sp
        if self.run_sp:
            self.ref_filesp()
            self.run_reflectivitysp()

        self.misfits()
        self.accept_model()
        return

    def rf_uncertainty(self):
        '''
        define observed RF uncertainty
        '''
        # Define uncertainty for observed RF
        # get arrays of data and time with the correct shift
        self.stream_obsps = read(os.path.join(self.input_fol, self.RF_obs_fileps))
        reftimeps         = float(self.stream_obsps[0].stats.location)

        self.RFdata_obsps = self.stream_obsps[0].data
        self.RFtime_obsps = self.stream_obsps[0].times() - \
                    float(self.stream_obsps[0].stats.location)

        if self.run_sp:
            self.stream_obssp = read(os.path.join(self.input_fol, self.RF_obs_filesp))
            reftimesp         = float(self.stream_obssp[0].stats.location)

            self.RFdata_obssp = self.stream_obssp[0].data
            self.RFtime_obssp = self.stream_obssp[0].times() - \
                        float(self.stream_obssp[0].stats.location)


        return

    def misfits(self):
        '''
        compute misfits of arrival times and RFs
        '''

        # Misfit of RF
        reftimeps = float(self.RF_stackps[0].stats.location)

        self.data_saveps  = self.RF_stackps[0].data
        self.time_saveps  = self.RF_stackps[0].times() - float(reftimeps)

        dt      = self.RF_stackps[0].stats.delta

        # window in samples in which compute misfit
        samps   = (reftimeps+self.misf_winps)/dt
        samps   = samps.astype(int)

        self.RF_synps     = self.data_saveps[samps[0]:samps[1]]
        self.time_synps   = self.time_saveps[samps[0]:samps[1]]

        self.RF_obsps = np.interp(self.time_synps, self.RFtime_obsps, self.RFdata_obsps)
        self.unc_obsps= self.unc_level * np.mean(np.abs(self.RF_obsps))
        # normalize arrays to make them comparable
        self.RF_obsps = self.RF_obsps/np.max(np.abs(self.RF_obsps))
        self.RF_synps = self.RF_synps/np.max(np.abs(self.RF_synps))

        misf_cteps    = len(self.RF_obsps)*self.unc_obsps    # normalized misfit
        misf_RFps= -1.*np.sum(np.abs(self.RF_obsps-self.RF_synps)) / misf_cteps

        if self.run_sp:
            dt      = self.RF_stacksp[0].stats.delta
            reftimesp = float(self.RF_stacksp[0].stats.location)
            self.data_savesp  = self.RF_stacksp[0].data
            self.time_savesp  = self.RF_stacksp[0].times() - float(reftimesp)
            # window in samples in which compute misfit
            samsp   = (reftimesp+self.misf_winsp)/dt
            samsp   = samsp.astype(int)
            self.RF_synsp     = self.data_savesp[samsp[0]:samsp[1]]
            self.time_synsp   = self.time_savesp[samsp[0]:samsp[1]]

            self.RF_obssp = np.interp(self.time_synsp, self.RFtime_obssp, self.RFdata_obssp)
            self.unc_obssp= self.unc_level * np.mean(np.abs(self.RF_obssp))

            # normalize arrays to make them comparable
            self.RF_obssp = self.RF_obssp/np.max(np.abs(self.RF_obssp))
            self.RF_synsp = self.RF_synsp/np.max(np.abs(self.RF_synsp))

            misf_ctesp    = len(self.RF_obssp)*self.unc_obssp    # normalized misfit
            misf_RFsp= -1.*np.sum(np.abs(self.RF_obssp-self.RF_synsp)) / misf_ctesp


        #----------------------------------------------------
        # Misfit of differential travel times

        loglike_at  = np.array(())
        count   = 0
        for i, ev in enumerate(self.events):
            if ev in self.events_pp:
                continue
            phases_obs  = self.phases_obs.copy()
            for pha_ref in self.phases_diff:
                # remove from list so it does not compare with itself
                idx_ref = self.phases_obs.index(pha_ref)
                phases_obs.remove(pha_ref)
                arr_ref = self.arrival_pert[i, idx_ref]

                for pha in phases_obs:
                    # differential travel time for synthetic data
                    idx_pha = self.phases_obs.index(pha)
                    arr     = self.arrival_pert[i, idx_pha]
                    dt_syn  = arr - arr_ref
                    # differential travel time for observed data
                    dt_obs  = self.data_obs[ev][pha] - self.data_obs[ev][pha_ref]
                    if not np.isnan(dt_obs):
                        count   +=1
                    sigma_obs   = np.sqrt(self.data_obs[ev]['sig'+pha]**2 +
                                        self.data_obs[ev]['sig'+pha_ref]**2)

                    if np.isnan(dt_syn) and not np.isnan(dt_obs):
                        print(' for event {} there is no {}'.format(ev, pha))
                        loglike_at  = np.append(loglike_at, 10000)
                    else:
                        loglike = np.abs(dt_syn - dt_obs)/sigma_obs
                        loglike_at  = np.append(loglike_at, loglike)


        # Especial for event S1000a
        for i, ev in enumerate(self.events_pp):
            phases_obs  = self.phases_obs.copy()
            pha_ref = 'PP'
            idx_ref = self.phases_obs.index(pha_ref)
            phases_obs.remove(pha_ref)
            save_i  = self.events.index(ev)

            arr_ref = self.arrival_pert[save_i, idx_ref]

            for pha in phases_obs:
                idx_pha = self.phases_obs.index(pha)
                arr     = self.arrival_pert[save_i, idx_pha]
                dt_syn  = arr - arr_ref

                dt_obs  = self.data_obs[ev][pha] - self.data_obs[ev][pha_ref]
                count   +=1
                sigma_obs   = np.sqrt(self.data_obs[ev]['sig'+pha]**2 +
                                self.data_obs[ev]['sig'+pha_ref]**2)
                if np.isnan(dt_syn) and not np.isnan(dt_obs):
                    print(' for event {} there is no {}'.format(ev, pha))
                    loglike_at  = np.append(loglike_at, 10000)
                else:

                    loglike = np.abs(dt_syn - dt_obs)/sigma_obs
                    # dirty weighting for testing
                    extra_weight    = 1
                    if pha=='SKS':
                        extra_weight    = 2
                    loglike_at  = np.append(loglike_at, extra_weight*loglike)
                if ev=='S1000a':
                    if pha=='Pdiff':
                        print(' Synthetic diff Pdiff-PP:', dt_syn )
                    if pha=='SS':
                        print(' Synthetic diff SS-PP:', dt_syn )


        misf_times   = (-1./count)*np.nansum(loglike_at)   # normalized misfit
        #misf_times  = -1.*np.nansum(loglike_at)

        #==================================================================
        # Misfit of mean density and moment of inertia

        misf_dens   = -1.* np.abs(self.dens_obs - self.dens_calc) / self.dens_sigma
        misf_moi    = -1. * np.abs(self.moi_obs - self.moi_calc) / self.moi_sigma

        #==================================================================
        # Sum of all misfits
        cte_max = 5.
        we_RFps = 2.*cte_max
        we_RFsp = 2.*cte_max
        we_time = 2.*cte_max
        we_dens = 1.*cte_max
        we_moi  = 1.*cte_max

        if self.run_sp:
            print('Misfit RF_Ps: {}, RF_Sp: {}, AT: {}, Rho: {}, MoI: {}'.format(we_RFps*misf_RFps, we_RFsp*misf_RFsp, we_time*misf_times, we_dens*misf_dens, we_moi*misf_moi))
            self.loglike_pert   = we_RFps*misf_times + we_RFsp*misf_RFps + we_time*misf_RFsp + we_dens*misf_dens + we_moi*misf_moi

        else:
            print('Misfit RF_Ps: {}, AT: {}, Rho: {}, MoI: {}'.format(we_RFps*misf_RFps, we_time*misf_times, we_dens*misf_dens, we_moi*misf_moi))
            self.loglike_pert   = we_time*misf_times + we_RFps*misf_RFps + we_dens*misf_dens + we_moi*misf_moi
            #self.loglike_pert   = we_time*misf_times + we_RFps*misf_RFps

        return

    def find(self, key):
        '''
        get keys of parameters to change depending on group
        '''
        idx = []

        for k in self.param_pert.keys():
            if self.param_pert[k]['group']==key:
                idx.append(k)
        return idx

    def pert_prms(self):
        '''
        perturb parameters
        '''
        self.param_pert = copy.deepcopy(self.param_curr)

        # which events have fixed location
        idx_nopert  = [self.events.index(el) for el in self.fixed_loc]

        # Which parameter perturb? select between crust, mantle, core
        ngroups = len(self.param_group)
        group_idx   = np.random.randint(ngroups)
        group   = self.param_group[group_idx]
        print(self.param_group)
        print(group_idx)
        print('     Perturb {}'.format(group))
        idx     = self.find(group)

        if group=='crust':
            idx2 = []
            newidx = idx[np.random.randint(len(idx))]
            idx2.append(newidx)

            while newidx in idx2:
                newidx = idx[np.random.randint(len(idx))]
            idx2.append(newidx)
        else:
            idx2 = idx.copy()

        for el in idx2:
            if el=='vs_crust':
                self.vs_crpert    = np.copy(self.param_curr[el]['val'])
                for i, vs in enumerate(self.vs_crcurr):
                    if i==0:
                        bd_min  = 0.
                    else:
                        bd_min  = self.vs_crpert[i-1]

                    bd_step = np.array([vs-self.param_curr[el]['step'], vs+self.param_curr[el]['step']])
                    bd_min   = np.max([self.param_curr[el]['bd'][0], bd_step[0], bd_min])
                    bd_max   = np.min([self.param_curr[el]['bd'][1], bd_step[1]])
                    self.vs_crpert[i]   = np.random.uniform(bd_min, bd_max)
                self.param_pert[el]['val']= self.vs_crpert
                continue

            elif el=='zcrust':
                self.zcrust_pert   = np.copy(self.param_curr[el]['val'])

                thickness_curr  = np.diff(self.zcrust_curr)
                thickness_pert  = np.copy(thickness_curr)
                moho_pert = 10000
                while (moho_pert<self.param_curr[el]['moho_bd'][0]) or (moho_pert>self.param_curr[el]['moho_bd'][1]):
                    for i, layer in enumerate(thickness_curr):
                        bd_step = np.array([layer-self.param_curr[el]['step'], layer+self.param_curr[el]['step']])
                        bd_min   = np.max([self.param_curr[el]['bd'][0], bd_step[0]])
                        bd_max   = np.min([self.param_curr[el]['bd'][1], bd_step[1]])

                        thickness_pert[i]= np.random.uniform(bd_min, bd_max)
                        self.zcrust_pert[i+1]   = self.zcrust_pert[i] + thickness_pert[i]
                    moho_pert   = np.sum(thickness_pert)
                self.zcrust_pert[0] = self.depth_ini
                self.param_pert[el]['val'] = self.zcrust_pert
                continue

            elif el=='distances':
                self.dist_evpert    = np.copy(self.param_curr[el]['val'])
                ndist       = len(self.dist_evpert)
                size_dist   = self.param_curr[el]['npert']
                idx_dist    = []
                j   = np.random.randint(ndist)
                for i in range(size_dist):
                    while j  in idx_dist:
                        j = np.random.randint(ndist)
                    idx_dist.append(j)

                for el_nopert in idx_nopert:
                    if el_nopert in idx_dist:
                        idx_dist.remove(el_nopert)

                for j in idx_dist:
                    dist    = self.param_curr[el]['val'][j]
                    bd_step = np.array([dist-self.param_curr[el]['step'], \
                                        dist+self.param_curr[el]['step']])
                    bd_min  = np.max([self.param_curr[el]['bd'][0], bd_step[0]])
                    bd_max  = np.min([self.param_curr[el]['bd'][1], bd_step[1]])
                    self.dist_evpert[j] = np.random.uniform(bd_min, bd_max)

                self.param_pert[el]['val']  = self.dist_evpert
                continue

            elif el=='depths':
                self.depth_evpert    = np.copy(self.param_curr[el]['val'])
                ndepths     = len(self.idx_depth)
                size_depths = self.param_curr[el]['npert']
                idx_depths  = []
                j   = np.random.randint(ndepths)
                for i in range(size_depths):
                    while j  in idx_depths:
                        j = np.random.randint(ndepths)
                    idx_depths.append(j)

                for el_nopert in idx_nopert:
                    if el_nopert in idx_depths:
                        idx_depths.remove(el_nopert)

                for jj in idx_depths:
                    j = self.idx_depth[jj]      # only depth of events with sS and pP can
                                                # be perturbed
                    print('     Perturb distance of event {}'.format(self.events[j]))

                    depth   = self.param_curr[el]['val'][j]

                    bd_step = np.array([depth-self.param_curr[el]['step'], \
                                        depth+self.param_curr[el]['step']])
                    bd_min  = np.max([self.param_curr[el]['bd'][0], bd_step[0]])
                    bd_max  = np.min([self.param_curr[el]['bd'][1], bd_step[1]])
                    self.depth_evpert[j] = np.random.uniform(bd_min, bd_max)

                self.param_pert[el]['val']  = self.depth_evpert
                continue

            elif el=='temps':
                self.temp_down_pert = np.copy(self.temp_down_curr)
                npert   = self.param_pert[el]['npert']
                ntemps  = len(self.temp_down_curr)
                idx_temps   = []
                j   = np.random.randint(ntemps)
                for i in range(npert):
                    while j  in idx_temps:
                        j = np.random.randint(ntemps)
                    idx_temps.append(j)

                for jj in idx_temps:
                    temp    = self.temp_down_curr[jj]

                    bd_step = np.array([temp-self.param_curr[el]['step'], \
                                        temp+self.param_curr[el]['step']])

                    if jj==0:
                        bd_min0 = 0.
                    else:
                        bd_min0 = np.copy(self.temp_down_curr[jj-1])
                    if jj==ntemps-1:
                        bd_max0 = self.param_curr[el]['bd'][1]
                    else:
                        bd_max0 = np.copy(self.temp_down_curr[jj+1])

                    bd_min  = np.max([self.param_curr[el]['bd'][0],
                                      bd_step[0], bd_min0])
                    bd_max  = np.min([self.param_curr[el]['bd'][1],
                                      bd_step[1], bd_max0])
                    self.temp_down_pert[jj] = np.random.uniform(bd_min, bd_max)


            elif el=='alphatemp':
                bd_step = np.array([self.param_curr[el]['val']-self.param_curr[el]['step'], self.param_curr[el]['val']+self.param_curr[el]['step']])
                bd_max_tlit = (self.tlit_pert-273.15)/(self.zlit_pert**2)
                bd_min  = np.max([self.param_curr[el]['bd'][0], bd_step[0]])
                bd_max  = np.min([self.param_curr[el]['bd'][1], bd_step[1], bd_max_tlit])
                self.param_pert[el]['val']   = np.random.uniform(bd_min, bd_max)

            elif el=='rho1':
                bdmin   = np.copy(self.rho0_pert)
                bd_step = np.array([self.param_curr[el]['val']-self.param_curr[el]['step'], self.param_curr[el]['val']+self.param_curr[el]['step']])
                bd_min  = np.max([self.param_curr[el]['bd'][0], bd_step[0], bdmin])
                bd_max  = np.min([self.param_curr[el]['bd'][1], bd_step[1]])
                self.param_pert[el]['val']   = np.random.uniform(bd_min, bd_max)
                continue
            elif el=='rho0':
                bdmax   = np.copy(self.rho1_pert)
                bdmin   = np.copy(1.e3*self.rho_curr[105])
                bd_step = np.array([self.param_curr[el]['val']-self.param_curr[el]['step'], self.param_curr[el]['val']+self.param_curr[el]['step']])
                bd_min  = np.max([self.param_curr[el]['bd'][0], bd_step[0], bdmin])
                bd_max  = np.min([self.param_curr[el]['bd'][1], bd_step[1], bdmax])
                self.param_pert[el]['val']   = np.random.uniform(bd_min, bd_max)
                continue

            elif el=='k1':
                bdmin   = np.copy(self.k0_pert)
                bd_step = np.array([self.param_curr[el]['val']-self.param_curr[el]['step'], self.param_curr[el]['val']+self.param_curr[el]['step']])
                bd_min  = np.max([self.param_curr[el]['bd'][0], bd_step[0], bdmin])
                bd_max  = np.min([self.param_curr[el]['bd'][1], bd_step[1]])
                self.param_pert[el]['val']   = np.random.uniform(bd_min, bd_max)
                continue
            elif el=='k0':
                bdmax   = np.copy(self.k1_pert)
                bd_step = np.array([self.param_curr[el]['val']-self.param_curr[el]['step'], self.param_curr[el]['val']+self.param_curr[el]['step']])
                bd_min  = np.max([self.param_curr[el]['bd'][0], bd_step[0]])
                bd_max  = np.min([self.param_curr[el]['bd'][1], bd_step[1], bdmax])
                self.param_pert[el]['val']   = np.random.uniform(bd_min, bd_max)
                continue

            else:
                bd_step = np.array([self.param_curr[el]['val']-self.param_curr[el]['step'], self.param_curr[el]['val']+self.param_curr[el]['step']])
                bd_min  = np.max([self.param_curr[el]['bd'][0], bd_step[0]])
                bd_max  = np.min([self.param_curr[el]['bd'][1], bd_step[1]])
                self.param_pert[el]['val']   = np.random.uniform(bd_min, bd_max)


        # asign perturbed values to variables
        self.zlit_pert  = self.param_pert['zlit']['val']
        self.tlit_pert  = self.param_pert['tlit']['val']
        self.zcmb1_pert  = self.param_pert['zcmb1']['val']
        self.alpha_pert = self.param_pert['alpha']['val']
        self.beta_pert  = self.param_pert['beta']['val']
        self.k0_pert    = self.param_pert['k0']['val']
        self.k0p_pert   = self.param_pert['k0p']['val']
        self.rho0_pert  = self.param_pert['rho0']['val']
        self.vs_crpert  = self.param_pert['vs_crust']['val']
        self.zcrust_pert= self.param_pert['zcrust']['val']
        self.dist_evpert= self.param_pert['distances']['val']
        self.alphaT_pert= self.param_pert['alphatemp']['val']

        self.k1_pert    = self.param_pert['k1']['val']
        self.k1p_pert   = self.param_pert['k1p']['val']
        self.rho1_pert  = self.param_pert['rho1']['val']
        self.dlvl_pert  = self.param_pert['dlvl']['val']

        self.vp_crpert  = self.alpha_pert * self.vs_crpert
        self.rho_crpert = self.beta_pert * self.vs_crpert

        return

    def accept_model(self):
        '''
        define if perturbed model is accepted or rejected
        '''

        print('     > Current likelihood: {}'.format(self.loglike_curr))
        print('     > Perturbed likelihood: {}'.format(self.loglike_pert))

        if np.exp(self.loglike_pert - self.loglike_curr) > np.random.random():
            self.accept = True
            self.accept_do()
        else:
            self.accept = False

        if self.count_all > self.max_step:
            # Re read original step size every ite_read iterations
            if self.accept:
                self.accepted_block += 1
            self.count_block+=1
            accept_rate = self.accepted_block / self.count_block
            self.modify_steps(accept_rate)
            print(accept_rate)

            if ((self.count_all - self.max_step) % self.ite_read == 0):
                print(' Re-read original step size')
                self.nblock += 1
                self.accepted_block = 0
                self.count_block    = 0
                self.copy_steps()

            return


    def modify_steps(self, accept_rate):
        '''
        modify steps depending of acceptance rate
        '''

        if accept_rate < 0.3:
            for key in self.param_curr.keys():
                self.param_curr[key]['step'] /= self.step_num
        if accept_rate > 0.6:
            for key in self.param_curr.keys():
                self.param_curr[key]['step'] *= self.step_num

        for key in self.param_curr.keys():
            if self.param_curr[key]['step'] < self.param_curr[key]['stepbd'][0]:
                self.param_curr[key]['step'] = self.param_curr[key]['stepbd'][0]
            if self.param_curr[key]['step'] > self.param_curr[key]['stepbd'][1]:
                self.param_curr[key]['step'] = self.param_curr[key]['stepbd'][1]

        return

    def copy_steps(self, fname='input_parameters_lvl'):
        '''
        copy original step sizes from file
        '''
        prms    = __import__(fname)

        for key in self.param_curr.keys():
            self.param_curr[key]['step'] = prms.param[key]['step']

        return

    def accept_do(self):
        '''
        if model is accepted do
        '''
        self.count_accepted += 1
        print(' >>> ACCEPTED MODEL {} OF {}'.format(self.count_accepted, self.max_ite))
        self.assign_current()
        self.save_model()
        return

    def assign_current(self):
        '''
        assigns current parameters (accepted) of model
        '''
        self.loglike_curr   = self.loglike_pert
        self.param_curr = copy.deepcopy(self.param_pert)

        # asign perturbed values
        self.zlit_curr  = self.param_curr['zlit']['val']
        self.tlit_curr  = self.param_curr['tlit']['val']
        self.zcmb1_curr = self.param_curr['zcmb1']['val']
        self.alpha_curr = self.param_curr['alpha']['val']
        self.beta_curr  = self.param_curr['beta']['val']
        self.k0_curr    = self.param_curr['k0']['val']
        self.k0p_curr   = self.param_curr['k0p']['val']
        self.rho0_curr  = self.param_curr['rho0']['val']
        self.k1_curr    = self.param_curr['k1']['val']
        self.k1p_curr   = self.param_curr['k1p']['val']
        self.rho1_curr  = self.param_curr['rho1']['val']
        self.dlvl_curr  = self.param_curr['dlvl']['val']
        self.vs_crcurr  = self.param_curr['vs_crust']['val']
        self.zcrust_curr= self.param_curr['zcrust']['val']
        self.dist_evcurr= self.param_curr['distances']['val']
        self.depth_evcurr= self.param_curr['depths']['val']

        # Complete profiles to save
        self.depth_curr = np.copy(self.depth_pert)
        self.vp_curr    = np.copy(self.vp_pert)
        self.vs_curr    = np.copy(self.vs_pert)
        self.rho_curr   = np.copy(self.rho_pert)
        self.temp_curr  = np.copy(self.temp_pert)
        self.pres_curr  = np.copy(self.pres_pert)
        self.temp_down_curr = np.copy(self.temp_down_pert)

        return

    def save_model(self):
        '''
        save model and parameters in numpy arrays
        '''

        # Model
        temp = np.zeros_like(self.depth_curr)
        pres = np.zeros_like(self.depth_curr)
        temp[:len(self.temp_curr)] = self.temp_curr
        pres[:len(self.pres_curr)] = self.pres_curr

        if self.run_sp:
            model_accepted = {'likelihood':self.loglike_curr, \
                          'depth':self.depth_curr, 'vp':self.vp_curr, \
                          'vs':self.vs_curr, 'rho':self.rho_curr, \
                          'temp':temp, 'pres':pres, \
                          'zlit':self.zlit_curr, \
                          'tlit':self.tlit_curr,\
                          'zcmb1':self.zcmb1_curr,'alpha':self.alpha_curr,\
                          'beta':self.beta_curr,\
                          'k0': self.k0_curr, 'k0p':self.k0p_curr,\
                          'rho0': self.rho0_curr,\
                          'mean_rho':self.dens_calc, \
                          'mean_moi':self.moi_calc, \
                          'rf_ps':self.data_saveps, \
                          'rf_sp':self.data_savesp, \
                          'time_ps':self.time_saveps, \
                          'time_sp':self.time_savesp, \
                          'distances':self.dist_evcurr,\
                          'depths':self.depth_evcurr,\
                          'arrival_times':self.arrival_pert,
                          'turning_depths':self.zturn_pert}

        else:
            model_accepted = {'likelihood':self.loglike_curr, \
                              'depth':self.depth_curr, 'vp':self.vp_curr, \
                              'vs':self.vs_curr, 'rho':self.rho_curr, \
                              'temp':temp, 'pres':pres, \
                              'zlit':self.zlit_curr, 'tlit':self.tlit_curr,\
                              'zcmb1':self.zcmb1_curr,'alpha':self.alpha_curr,\
                              'beta':self.beta_curr,\
                              'k0': self.k0_curr, 'k0p':self.k0p_curr,\
                              'rho0': self.rho0_curr,\
                              'k1': self.k1_curr, 'k1p':self.k1p_curr,\
                              'rho1': self.rho1_curr,\
                              'dlvl':self.dlvl_curr,\
                              'mean_rho':self.dens_calc, \
                              'mean_moi':self.moi_calc, \
                              'rf_ps':self.data_saveps, \
                              'time_ps':self.time_saveps, \
                              'distances':self.dist_evcurr,\
                              'depths':self.depth_evcurr,\
                              'arrival_times':self.arrival_pert,
                              'turning_depths':self.zturn_pert}

        np.save(os.path.join(self.output_fol, 'Model_{}'.format(self.count_accepted)),
                model_accepted)
        return

#==========================================================================

if __name__=='__main__':

    b = INV()
    if b.ier:
        while b.ier:
            print('  > Looking for initial parameters')
            b = INV()
    b.assign_current()
    obj = b
    while b.count_accepted<b.max_ite:
        print('---------------------------------------------------------')
        print('Iterarion number {}'.format(b.count_all))
        b.inversion()

##========================================================================


