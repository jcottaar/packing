import pandas as pd
import numpy as np
import scipy as sp
import cupy as cp
import kaggle_support as kgs

def place_random(N, inner_size, generator=None):
    '''
    Place trees randomly
    '''
    # IDEA: force all trees inside inner_size, rather than their centers
    if generator is None:
        generator = np.random.default_rng(seed=42)   
    tree_list = kgs.TreeList()
    # use Generator.random (uniform [0,1)) and scale by inner_size
    tree_list.x =  generator.random(N) * inner_size - inner_size/2
    tree_list.y =  generator.random(N) * inner_size - inner_size/2
    tree_list.theta = generator.random(N) * 2 * np.pi
    return tree_list

