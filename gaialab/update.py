"""
Update function implementation in Python
The purpose of this function is to check the derivatives of compute_u,
using the formula for the numerica derivatives;

:Author:MaraBucur
"""
import numpy as np
from source import Source
from source import Calc_source
from satellite import Satellite

def source_update(source, h=[0.01, 0, 0, 0, 0, 0, 0 , 0]):
    param=source.get_parameters(t=0)
    param_plus=param + h
    [a,d,pi,mua,mud, ga, gd, mur]=param_plus
    source_p=Source('source_plus', a,d,pi,mua,mud, ga, gd, mur)
    param_minus=param - h
    [a,d,pi,mua,mud, ga, gd, mur]=param_minus
    source_m=Source('source_minus', a,d,pi,mua,mud, ga, gd, mur)
    gaia = Satellite(0, 3*365, 1/24)
    t=0
    u1=source_p.compute_u(gaia,t)
    u0=source_m.compute_u(gaia,t)
    der=(u1-u0)/0.02
    return np.round(np.abs(der))

sirio=Source('sirio', 101.28, -16.7161, 379.21, -546.05, -1223.14, 0, 0, 0)
der=source_update(source=sirio )
print('derivatives=', der)
