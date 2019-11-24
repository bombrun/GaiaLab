import numpy as np
from source import Source
from source import Calc_source
from satellite import Satellite



ha = 1
hd = 0
hp = 0
hma = 0
hmd = 0
hga = 0
hgd = 0
hmr = 0
h = np.array([ha, hd, hp, hma, hmd, hga, hgd, hmr])
p=[101.28, -16.7161, 379.21, -546.05, -1223.14, 0, 0, 0]
sirio = Source('sirio', p)
s_plus_h = Source('sirio+',p)
s_minus_h = Source('sirio-', 101.18, -16.7161, 379.21, -546.05, -1223.14, 0, 0)
gaia = Satellite(0, 3*365, 1/24)
t=0
u1=s_plus_h.compute_u(gaia,t)
u0=s_minus_h.compute_u(gaia,t)
der=(u1-u0)/0.2
print(np.round(np.abs(der)))

ha = 0
hd = 1
h = np.array([ha, hd, hp, hma, hmd, hga, hgd])
s_plus_h = Source('sirio+', 101.28, -16.7261, 379.21, -546.05, -1223.14, 0, 0)
s_minus_h = Source('sirio-', 101.28, -16.7061, 379.21, -546.05, -1223.14, 0, 0)
u1=s_plus_h.compute_u(gaia,t)
u0=s_minus_h.compute_u(gaia,t)
der=(u1-u0)/0.02
print(np.round(np.abs(der)))

hd = 0
hp=1
h = np.array([ha, hd, hp, hma, hmd, hga, hgd])
s_plus_h = Source('sirio+', 101.28, -16.7161, 379.31, -546.05, -1223.14, 0, 0)
s_minus_h = Source('sirio-', 101.28, -16.7161, 379.11, -546.05, -1223.14, 0, 0)
u1=s_plus_h.compute_u(gaia,t)
u0=s_minus_h.compute_u(gaia,t)
der=(u1-u0)/0.2
print(np.round(np.abs(der)))

hma=1
hp=0
h = np.array([ha, hd, hp, hma, hmd, hga, hgd])
s_plus_h = Source('sirio+', 101.28, -16.7161, 379.21, -546.15, -1223.14, 0, 0)
s_minus_h = Source('sirio-', 101.28, -16.7161, 379.21, -545.95, -1223.14, 0, 0)
u1=s_plus_h.compute_u(gaia,t)
u0=s_minus_h.compute_u(gaia,t)
der=(u1-u0)/0.2
print(np.round(np.abs(der)))

hma=0
hmd=1
h = np.array([ha, hd, hp, hma, hmd, hga, hgd])
s_plus_h = Source('sirio+', 101.28, -16.7161, 379.21, -546.05, -1224.14, 0, 0)
s_minus_h = Source('sirio-', 101.28, -16.7161, 379.21, -546.05, -1222.14, 0, 0)
u1=s_plus_h.compute_u(gaia,t)
u0=s_minus_h.compute_u(gaia,t)
der=(u1-u0)/2
print(np.round(np.abs(der)))
