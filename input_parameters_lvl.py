#==========================================================================
# Parameters for Mars model and nodes
#==========================================================================

import numpy as np

#==========================================================================

# Files and folders
input_fol   = '..'
input_picks = 'Picks_SKS.csv'
RF_obsps    = 'PtoS_RF.mseed'
RF_obssp    = 'StoP_RF.mseed'
output_tmp  = 'tmp'
file_tmp    = 'tmp_model{}'
file_tmp_lvl= 'tmp_model_lvl{}'
temp_file   = 'TempAdiab.txt'


input_ref   = 'input_reflectivity'
ref_fol     = 'Reflectivity'
ref_exec    = 'crfl_sac_mars'
adiabatic   = False
output_fol  = '/cluster/scratch/aduran/DW-alpha/output_{}'
#output_fol = 'output_{}'

events_pp   = ['S1000a', 'S0976a']
fixed_loc   = ['S1000a']
# observed parameters
dens_obs    = 3.935	# observed density in g/cm3
dens_sigma  = 0.0012
moi_obs     = 0.3634   # normalized observed MoI
moi_sigma   = 0.00006

#RF_stack    = 'RF_stack'
#cfmasna = np.array([2.66, 13.68, 32.81, 3.49, 46.66, 0.69]) # Khan
#cfmasna = np.array([2.4, 18.7, 30.7, 3.5, 44.1, 0.6]) # Taylor
#cfmasna = np.array([2.0, 17.7, 27.3, 2.5, 47.5, 1.2]) # EH45
#cfmasna = np.array([2.88, 14.7, 31.0, 3.59, 45.5, 0.59]) # YOM
#cfmasna = np.array([2.36, 17.21, 29.71, 2.89, 45.39, 0.98]) #LF
cfmasna = np.array([2.43, 17.89, 30.16, 2.94, 44.47, 0.51]) # DW
#cfmasna = np.array([2.01, 17.81, 27.46, 2.52, 47.79, 1.21]) # SAN

# non times categories in input_picks
exc = ['Dist', 'sig', 'Depth', 'Baz']

# Input nodes
z0      = 0.001		# first depth [km] (cant be 0 for perplex)
zdisc   = 900.		# beginning of discontinuity [km]
nzlit   = 40           # number of layers in lithosphere
nzmant	= 20
nzdisc  = 40
nzlvl   = 10

# core parameters
ncore   = 30            # layers at core

# initial perple values
tsurf   = 273.15	 	# temperature at surface [K]
tstart  = 273.15		# tstop in perplex [K]
ntemp   = 8


# attenuation - not used now
grain_sz    = 1.e-4	# grain size
activ_vol   = 1.e-5	# activation volume
period_at   = 1.		# attenuation period [s]
qkp	        = 10000.
qs_contant	= 600.	# constant for Qs in case we do not use any
			# attenuation model
# Insert custom crust
rho_cr  = np.array([1.71, 2.7, 2.81])
vp_cr   = np.array([3.66, 5.76, 6.0])

# Phases to compute with lvl model
phases_lvl  = ['Pdiff2', 'PcP2']

# RFs parameters
events_ps   = ['S0235b', 'S0183a', 'S0784a', 'S0173a', 'S0864a', 'S0820a', 'S0802a']
events_sp   = ['S0235b', 'S0784a', 'S0173a', 'S0864a', 'S0820a', 'S0802a']

motherps	= 'P'	# mother phase
mothersp 	= 'S'

win_srcps = np.array([10., 50.])
win_resps = np.array([10., 50.])

win_srcsp = np.array([100., 30.])
win_ressp = np.array([150., 50.])

taper_win	= 'hann'
taper_per	= 0.05
filter_rf	= np.array([0.2, 0.5])
filter_order	= 2
#misf_winps	= np.array([-0.45, 8.7])
misf_winps	= np.array([-0.45, 8.7])
misf_winps2	= np.array([8.7, 11.])
misf_winsp	= np.array([0., 4.])

unc_level	= 0.5
unc_level2	= 0.35

# Inversion
run_sp = False
phases_diff = ['P', 'S']
max_ite	    = 12000    # run until it reaches max_ite accepted models
max_step	= 100 # max num of models before changing stepsize
step_read	= 250 # re-read original step size every step-read iteration
step_num	= 1.05 # depending acceptance rate, decrease or increase stepsize by 10%

# ranges for inverted parameters
param   = {'zlit':{'val':430., 'step':1., 'bd':np.array([100,600]),
                   'stepbd':np.array([0.5,20]), 'group':'mantle'},
           'tlit':{'val':1488., 'step':5., 'bd':np.array([900,1800]),
                   'stepbd':np.array([0.5,20]), 'group':'mantle'},
           'alphatemp':{'val':0.005, 'step':0.0005, 'bd':np.array([0,0.15]),
                   'stepbd':np.array([0.00002,0.001]), 'group':'mantle'},
           'temps':{'step':5, 'bd':np.array([1000, 2250]),
                    'stepbd':np.array([2,15]), 'group':'temp', 'npert':2},
           'zcmb1':{'val':1560., 'step':2., 'bd':np.array([1389,1789]),
                    'stepbd':np.array([.5,3.]), 'group':'core'},
           'alpha':{'val':1.66, 'step':0.01, 'bd':np.array([1.65,1.85]),
                    'stepbd':np.array([0.001,0.05]), 'group':'crust'},
           'beta':{'val':0.79, 'step':0.01, 'bd':np.array([0.75,1.]),
                   'stepbd':np.array([0.005,0.05]), 'group':'crust'},
           'vs_crust':{'val':np.array([2.087, 2.961, 3.766]), 'step':0.05,
                       'bd':np.array([1.5,4.2]), 'stepbd':np.array([0.01,0.1]),
                       'group':'crust'},
           'zcrust':{'val':np.array([0.001,10.89,27.8, 43.58]), 'step':1.,
                     'bd':np.array([5.,40.]), 'stepbd':np.array([0.01,3.]),
                     'group':'crust','moho_bd':np.array([20,60]), 'nlayers':3},
           'distances':{'step':1., 'bd':np.array([0,180]),
                        'stepbd':np.array([0.1, 2.]), 'group':'distances',
                        'npert':3},
           'depths':{'step':1., 'bd':np.array([10,50]),
                     'stepbd':np.array([0.1, 2.]),
                     'group':'depths', 'npert':3},
           'k0':{'val':100, 'step':.5, 'bd':np.array([50,150]),
                 'stepbd':np.array([0.1,2.]), 'group':'lvl_k'},
           'k0p':{'val':4.8, 'step':.05, 'bd':np.array([3.,7]),
                  'stepbd':np.array([0.01,.5]), 'group':'lvl_k'},
           'rho0':{'val':4017., 'step':5., 'bd':np.array([3.4e3,4.5e3]),
                   'stepbd':np.array([1.,30]), 'group':'lvl'},
           'dlvl':{'val':160., 'step':2., 'bd':np.array([30,300]),
                   'stepbd':np.array([0.5,5]), 'group':'lvl'},
           'k1':{'val':173., 'step':1., 'bd':np.array([120,220]),
                 'stepbd':np.array([0.5,5.]), 'group':'core_k'},
           'k1p':{'val':5.1, 'step':.1, 'bd':np.array([3.5,7]),
                  'stepbd':np.array([0.01,.25]), 'group':'core_k'},
           'rho1':{'val':6412., 'step':5., 'bd':np.array([5.e3,7.5e3]),
                   'stepbd':np.array([1.,30]), 'group':'core'}}

param_group = ['crust', 'mantle', 'core','distances', 'core_k', 'lvl', 'lvl_k', 'temp']
#==========================================================================
# Planetary settings

r_p     = 3389.5        # martian radius [km]
m_p     = 6.419e23      # martian mass [kg]

