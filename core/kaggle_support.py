import pandas as pd
import numpy as np
import scipy as sp
import cupy as cp
import dill # like pickle but more powerful
import itertools
import os
import copy
import json
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import IPython
from dataclasses import dataclass, field, fields
import enum
import typing
import pathlib
import multiprocess
multiprocess.set_start_method('spawn', force=True)
from decorator import decorator
from line_profiler import LineProfiler
import os
import gc
import glob
import h5py
import time
import sklearn
import shutil
import torch
import inspect
from tqdm import tqdm
import hashlib
from contextlib import nullcontext


'''
Determine environment and globals
'''

if os.path.isdir('/mnt/d/packing/'):
    env = 'local'
    d_drive = '/mnt/d/'    
else:
    raise Exception('Unknown environment')
print(env)

profiling = False
debugging_mode = 1
verbosity = 1
disable_any_parallel = False

match env:
    case 'local':
        data_dir = d_drive+'/packing/data/'
        temp_dir = d_drive+'/packing/temp/'             
        code_dir = d_drive+'/packing/core/' 
os.makedirs(data_dir, exist_ok=True)
os.makedirs(temp_dir, exist_ok=True)


if not env=='kaggle':
    import git 
    repo = git.Repo(search_parent_directories=True)
    git_commit_id = repo.head.object.hexsha
else:
    git_commit_id = 'kaggle'


'''
Helper classes and functions
'''

def list_attrs(obj):
    for name, val in inspect.getmembers(obj):
        if name.startswith("_"):
            continue
        # # skip methods, but let descriptors through
        # if callable(val) and not isinstance(val, property):
        #     continue
        print(f"{name} = {val}")

def remove_and_make_dir(path):
    try: shutil.rmtree(path)
    except: pass
    os.makedirs(path)

# Helper class - doesn't allow new properties after construction, and enforces property types. Partially written by ChatGPT.
@dataclass
class BaseClass:
    _is_frozen: bool = field(default=False, init=False, repr=False)
    comment:str = field(init=True, default='')

    def check_constraints(self, debugging_mode_offset = 0):
        global debugging_mode
        debugging_mode = debugging_mode+debugging_mode_offset
        try:
            if debugging_mode > 0:
                self._check_types()
                self._check_constraints()
            return
        finally:
            debugging_mode = debugging_mode - debugging_mode_offset

    def _check_constraints(self):
        pass

    def _check_types(self):
        type_hints = typing.get_type_hints(self.__class__)
        for field_info in fields(self):
            field_name = field_info.name
            expected_type = type_hints.get(field_name)
            actual_value = getattr(self, field_name)
            
            if expected_type and not isinstance(actual_value, expected_type) and not actual_value is None:
                raise TypeError(
                    f"Field '{field_name}' expected type {expected_type}, "
                    f"but got value {actual_value} of type {type(actual_value).__name__}.")

    def __post_init__(self):
        # Mark the object as frozen after initialization
        object.__setattr__(self, '_is_frozen', True)

    def __setattr__(self, key, value):
        # If the object is frozen, prevent setting new attributes
        if self._is_frozen and not hasattr(self, key):
            raise AttributeError(f"Cannot add new attribute '{key}' to frozen instance")
        super().__setattr__(key, value)

# Small wrapper for dill loading
def dill_load(filename):
    filehandler = open(filename, 'rb');
    data = dill.load(filehandler)
    filehandler.close()
    return data

# Small wrapper for dill saving
def dill_save(filename, data):
    filehandler = open(filename, 'wb');
    data = dill.dump(data, filehandler)
    filehandler.close()
    return data

@decorator
def profile_each_line(func, *args, **kwargs):
    if not profiling:
        return func(*args, **kwargs)
    profiler = LineProfiler()
    profiled_func = profiler(func)
    try:
        s=profiled_func(*args, **kwargs)
        profiler.print_stats()
        return s
    except:
        profiler.print_stats()
        raise

def profile_print(string):
    if profiling: print(string)


def add_cursor(sc):

    import mplcursors
    cursor = mplcursors.cursor(sc, hover=True)

    @cursor.connect("add")
    def on_add(sel):
        i = sel.index
        sel.annotation.set_text(f"index={i}")


def to_cpu(array):
    if isinstance(array, cp.ndarray):
        return array.get()
    else:
        return array

def to_gpu(array):
    if isinstance(array, cp.ndarray):
        return array
    else:
        return cp.array(array)
    
def clear_gpu():
    cache = cp.fft.config.get_plan_cache()
    cache.clear()
    cp.get_default_memory_pool().free_all_blocks()
    
'''
The trees!
'''

from decimal import Decimal, getcontext
from matplotlib.patches import Rectangle
from shapely import affinity, touches
from shapely.geometry import Polygon
from shapely.ops import unary_union
from shapely.strtree import STRtree

scale_factor = 1
def create_tree(center_x, center_y, angle):
    """Initializes the Christmas tree with a specific position and rotation."""
    trunk_w = Decimal('0.15')
    trunk_h = Decimal('0.2')
    base_w = Decimal('0.7')
    mid_w = Decimal('0.4')
    top_w = Decimal('0.25')
    tip_y = Decimal('0.8')
    tier_1_y = Decimal('0.5')
    tier_2_y = Decimal('0.25')
    base_y = Decimal('0.0')
    trunk_bottom_y = -trunk_h

    initial_polygon = Polygon(
        [
            # Start at Tip
            (Decimal('0.0') * scale_factor, tip_y * scale_factor),
            # Right side - Top Tier
            (top_w / Decimal('2') * scale_factor, tier_1_y * scale_factor),
            (top_w / Decimal('4') * scale_factor, tier_1_y * scale_factor),
            # Right side - Middle Tier
            (mid_w / Decimal('2') * scale_factor, tier_2_y * scale_factor),
            (mid_w / Decimal('4') * scale_factor, tier_2_y * scale_factor),
            # Right side - Bottom Tier
            (base_w / Decimal('2') * scale_factor, base_y * scale_factor),
            # Right Trunk
            (trunk_w / Decimal('2') * scale_factor, base_y * scale_factor),
            (trunk_w / Decimal('2') * scale_factor, trunk_bottom_y * scale_factor),
            # Left Trunk
            (-(trunk_w / Decimal('2')) * scale_factor, trunk_bottom_y * scale_factor),
            (-(trunk_w / Decimal('2')) * scale_factor, base_y * scale_factor),
            # Left side - Bottom Tier
            (-(base_w / Decimal('2')) * scale_factor, base_y * scale_factor),
            # Left side - Middle Tier
            (-(mid_w / Decimal('4')) * scale_factor, tier_2_y * scale_factor),
            (-(mid_w / Decimal('2')) * scale_factor, tier_2_y * scale_factor),
            # Left side - Top Tier
            (-(top_w / Decimal('4')) * scale_factor, tier_1_y * scale_factor),
            (-(top_w / Decimal('2')) * scale_factor, tier_1_y * scale_factor),
        ]
    )
    rotated = affinity.rotate(initial_polygon, angle, origin=(0, 0))
    polygon = affinity.translate(rotated,
                                      xoff=(center_x * scale_factor),
                                      yoff=(center_y * scale_factor))
    return polygon

@dataclass
class TreeList(BaseClass):
    x: np.ndarray = field(default=None)
    y: np.ndarray = field(default=None)
    theta: np.ndarray = field(default=None)

    # dynamic dependent property N (updates automatically)
    @property
    def N(self) -> int:
        return 0 if self.x is None else len(self.x)

    def _check_constraints(self):

        # If x is None, require y and theta to be None as well
        if self.x is None:
            if (self.y is not None) or (self.theta is not None):
                raise Exception('TreeList: inconsistent lengths (x is None but y/theta not None)')
            return

        # x is not None -> enforce lengths match N
        N = self.N
        if not (len(self.y) == N and len(self.theta) == N):
            raise Exception('TreeList: inconsistent lengths')
    
    def get_trees(self):
        ''' Returns list of shapely Polygons for each tree '''
        trees = []
        for i in range(self.N):
            tree = create_tree(self.x[i], self.y[i], self.theta[i]*360/(2*np.pi))
            trees.append(tree)
        return trees