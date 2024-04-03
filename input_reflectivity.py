#==========================================================================
# Parameters for runing reflectivity
#
# For specifications see doc file in reflectivity folder
#=========================================================================

import numpy as np
from pyrocko import moment_tensor as pmt

#==========================================================================

# Line 1
string = 'synthetic input file for reflectivity code'

# Line 2
l1      = ' 0 0 0 0 1   0 0 1 1 1   2 1 0 0 1   0 1 2 0 1   1'

# Line 3
l2      = '    {}    1    0    1    1'

# Line 4 loaded from model

# Line 4a
zr  = 0.
xs  = 0.
ys  = 0.
ts  = 0.
es  = 1.
#mt  = np.array([1.0,0.,0.,1.,0.,1.])  # moment tensor m11,m12,m13,m22,m23,m33

# Moment tensor components in rad, phi, the
# 1,2,3 = up,south,east = r, t, p

src = np.array([280., 79., -79.])

magnitude = 3.0
m0 = pmt.magnitude_to_moment(magnitude)
mt = pmt.MomentTensor(strike=280., dip=79., rake=-79, scalar_moment=m0)

mt = pmt.MomentTensor.m6_up_south_east(mt)
mrr = mt[0]; mtt = mt[1]; mpp = mt[2]
mrt = mt[3]; mrp = mt[4]; mtp = mt[5]
mt1 = np.array([mrr, mtt, mpp, mrt, mrp, mtp])

#mtsp = np.array([mpp, -mtp, -mrp, mtt, mrt, mrr])/1.e13
mtsp = np.array([1,0,0,0,0,-1])
mtps = np.array([1,0,0,1,0,1])
# Line 5
azi = 0.

# Line 6
vred = 12.
tmin = -300.

# Line 7
c2 = 3.
cwil = 3.5
cwir = 23.5
c1 = 25.0


# Line 8
freq    = [.01,.0133,.5,.503]   # float, frequencies to simulate
fr = 0.

# Line 9
dt      = 0.1
na = 0
nextr = 2
Tsec    = 750.     # time series in seconds
perc    = .3










