import numpy as np

def count_E(nu, e):
    return 2*np.arctan( np.sqrt((1-e)/(1+e))*np.tan(nu/2) )
