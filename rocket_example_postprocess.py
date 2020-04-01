import numpy as np

def rocket_example_postprocess(t, m, T, pos, vel, acc):
    landing_dist = np.sqrt(pos[-1,1]**2 + pos[-1,2]**2)    
    return landing_dist
