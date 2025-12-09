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
    env = 'vast'
print(env)

profiling = False
debugging_mode = 1
verbosity = 1
disable_any_parallel = False

match env:
    case 'local':
        data_dir = d_drive+'/packing/data/'
        temp_dir = d_drive+'/packing/temp/'             
        code_dir = d_drive+'/packing/code/core/' 
    case 'vast':
        data_dir = '/packing/data/'
        temp_dir = '/packing/temp/'             
        code_dir = '/packing/code/core/'
os.makedirs(data_dir, exist_ok=True)
os.makedirs(temp_dir, exist_ok=True)


# if not env=='kaggle':
#     import git 
#     repo = git.Repo(search_parent_directories=True)
#     git_commit_id = repo.head.object.hexsha
# else:
#     git_commit_id = 'kaggle'


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

from matplotlib.patches import Rectangle
from shapely import affinity, touches
from shapely.geometry import Polygon, Point
from shapely.ops import unary_union
from shapely.strtree import STRtree
from shapely.prepared import prep
from typeguard import typechecked

scale_factor = 1.
def create_center_tree():
    """Initializes the Christmas tree"""    
    shift=0.
    
    trunk_w = 0.15
    trunk_h = 0.2
    base_w = 0.7
    mid_w = 0.4
    top_w = 0.25
    tip_y = 0.8 + shift
    tier_1_y = 0.5 + shift
    tier_2_y = 0.25 + shift
    base_y = 0.0 + shift
    trunk_bottom_y = -trunk_h + shift
    
    sf = scale_factor
    initial_polygon = Polygon(
        [
            # Start at Tip
            (0.0 * sf, tip_y * sf),
            # Right side - Top Tier
            (top_w / 2 * sf, tier_1_y * sf),
            (top_w / 4 * sf, tier_1_y * sf),
            # Right side - Middle Tier
            (mid_w / 2 * sf, tier_2_y * sf),
            (mid_w / 4 * sf, tier_2_y * sf),
            # Right side - Bottom Tier
            (base_w / 2 * sf, base_y * sf),
            # Right Trunk
            (trunk_w / 2 * sf, base_y * sf),
            (trunk_w / 2 * sf, trunk_bottom_y * sf),
            # Left Trunk
            (-(trunk_w / 2) * sf, trunk_bottom_y * sf),
            (-(trunk_w / 2) * sf, base_y * sf),
            # Left side - Bottom Tier
            (-(base_w / 2) * sf, base_y * sf),
            # Left side - Middle Tier
            (-(mid_w / 4) * sf, tier_2_y * sf),
            (-(mid_w / 2) * sf, tier_2_y * sf),
            # Left side - Top Tier
            (-(top_w / 4) * sf, tier_1_y * sf),
            (-(top_w / 2) * sf, tier_1_y * sf),
        ]
    )
    convex_breakdown = [ Polygon([(0.0 * sf, tip_y * sf), (top_w / 2 * sf, tier_1_y * sf), (-(top_w / 2) * sf, tier_1_y * sf)]),
                        Polygon([(top_w / 4 * sf, tier_1_y * sf), (mid_w / 2 * sf, tier_2_y * sf), (-mid_w / 2 * sf, tier_2_y * sf), (-top_w / 4 * sf, tier_1_y * sf)]),
                        Polygon([(mid_w / 4 * sf, tier_2_y * sf), (base_w / 2 * sf, base_y * sf), (-base_w / 2 * sf, base_y * sf), (-mid_w / 4 * sf, tier_2_y * sf)]),
                        Polygon([(trunk_w / 2 * sf, base_y * sf), (trunk_w / 2 * sf, trunk_bottom_y * sf), (-trunk_w / 2 * sf, trunk_bottom_y * sf), (-trunk_w / 2 * sf, base_y * sf)])  ]


    # Compute the polygon centroid and recenter geometry so centroid is at origin.
    centroid = initial_polygon.centroid
    cx, cy = centroid.x, centroid.y
    if (cx, cy) != (0.0, 0.0):
        initial_polygon = affinity.translate(initial_polygon, xoff=-cx, yoff=-cy)
        convex_breakdown = [affinity.translate(p, xoff=-cx, yoff=-cy) for p in convex_breakdown]

    # Find maximum distance from centroid (now at origin) to any vertex
    max_radius = 0.0
    origin = Point(0, 0)
    for x, y in initial_polygon.exterior.coords[:-1]:  # skip closing vertex
        dist = Point(x, y).distance(origin)
        if dist > max_radius:
            max_radius = dist
    return initial_polygon, convex_breakdown, max_radius, (cx, cy)
center_tree, convex_breakdown, tree_max_radius, tree_centroid_offset = create_center_tree()
center_tree_prepped = prep(center_tree)
tree_area = center_tree.area
tree_vertices = cp.array(np.array(center_tree.exterior.coords[:-1]), dtype=cp.float64)
tree_vertices32 = cp.array(np.array(center_tree.exterior.coords[:-1]), dtype=cp.float32)


@typechecked
def create_tree(center_x:float, center_y:float, angle:float):
    """Initializes the Christmas tree with a specific position and rotation."""
    rotated = affinity.rotate(center_tree, angle, origin=(0, 0))
    polygon = affinity.translate(rotated,
                                 xoff=center_x * scale_factor,
                                 yoff=center_y * scale_factor)
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

    @property
    def xyt(self) -> np.ndarray:
        """Return an (N,3) array with columns [x, y, theta]. Assumes x,y,theta are not None."""
        return np.column_stack((np.asarray(self.x).ravel(),
                                np.asarray(self.y).ravel(),
                                np.asarray(self.theta).ravel()))

    @xyt.setter
    def xyt(self, value: typing.Union[np.ndarray, list]):
        """Accept an (N,3) array-like and set x, y, theta accordingly. Assumes no None."""
        arr = np.asarray(to_cpu(value))
        if arr.ndim != 2 or arr.shape[1] != 3:
            raise ValueError("xyt must be an array with shape (N, 3)")
        self.x = arr[:, 0].astype(float)
        self.y = arr[:, 1].astype(float)
        self.theta = arr[:, 2].astype(float)

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
    
'''Metric'''
@dataclass
class SolutionCollectionSquare(BaseClass):
    xyt: cp.ndarray = field(default=None)  # (N,3) array of tree positions and angles
    h: cp.ndarray = field(default=None)      # (N,3) array; first column is square size, next two are offset

    # add N_solutions and N_trees properties
    @property
    def N_solutions(self) -> int:
        """Number of solution rows in xyt (N_solutions)."""
        if self.xyt is None:
            return 0
        arr = to_cpu(self.xyt)
        return int(np.asarray(arr).shape[0])

    @property
    def N_trees(self) -> int:
        """Number of trees per solution (N_trees)."""
        if self.xyt is None:
            return 0
        arr = to_cpu(self.xyt)
        return int(np.asarray(arr).shape[1])


    def _check_constraints(self):
        if self.xyt.ndim != 3 or self.xyt.shape[2] != 3:
            raise ValueError("Solution: xyt must be an array with shape (N_solutions, N_trees, 3)")
        assert self.h.shape == (self.xyt.shape[0],3)

    def snap(self):
        """Set h such that for each solution it's the smallest possible square containing all trees.
        Vectorized implementation using tree_vertices32.
        Assumes xyt is cp.float32.
        """
        # xyt shape: (n_solutions, n_trees, 3)
        # tree_vertices32 shape: (n_vertices, 2)
        
        n_solutions = self.xyt.shape[0]
        n_trees = self.xyt.shape[1]
        n_vertices = tree_vertices32.shape[0]
        
        # Extract pose components
        x = self.xyt[:, :, 0:1]  # (n_solutions, n_trees, 1)
        y = self.xyt[:, :, 1:2]  # (n_solutions, n_trees, 1)
        theta = self.xyt[:, :, 2:3]  # (n_solutions, n_trees, 1)
        
        # Precompute rotation matrices
        cos_t = cp.cos(theta)  # (n_solutions, n_trees, 1)
        sin_t = cp.sin(theta)  # (n_solutions, n_trees, 1)
        
        # Get local vertices (n_vertices, 2)
        vx_local = tree_vertices32[:, 0]  # (n_vertices,)
        vy_local = tree_vertices32[:, 1]  # (n_vertices,)
        
        # Apply rotation and translation for all trees
        # Broadcast: (n_solutions, n_trees, 1) * (n_vertices,) -> (n_solutions, n_trees, n_vertices)
        vx_rot = cos_t * vx_local - sin_t * vy_local  # (n_solutions, n_trees, n_vertices)
        vy_rot = sin_t * vx_local + cos_t * vy_local  # (n_solutions, n_trees, n_vertices)
        
        # Translate by tree position
        vx_global = vx_rot + x  # (n_solutions, n_trees, n_vertices)
        vy_global = vy_rot + y  # (n_solutions, n_trees, n_vertices)
        
        # Find min/max across all trees and vertices for each solution
        # Reshape to (n_solutions, n_trees * n_vertices) for min/max along all trees+vertices
        vx_flat = vx_global.reshape(n_solutions, -1)  # (n_solutions, n_trees * n_vertices)
        vy_flat = vy_global.reshape(n_solutions, -1)  # (n_solutions, n_trees * n_vertices)
        
        x_min = cp.min(vx_flat, axis=1)  # (n_solutions,)
        x_max = cp.max(vx_flat, axis=1)  # (n_solutions,)
        y_min = cp.min(vy_flat, axis=1)  # (n_solutions,)
        y_max = cp.max(vy_flat, axis=1)  # (n_solutions,)
        
        # Compute center and size of bounding square
        x_center = (x_min + x_max) / 2.0  # (n_solutions,)
        y_center = (y_min + y_max) / 2.0  # (n_solutions,)
        
        # Size is max of width and height
        width = x_max - x_min  # (n_solutions,)
        height = y_max - y_min  # (n_solutions,)
        size = cp.maximum(width, height)  # (n_solutions,)
        
        # Update h: [size, x_offset, y_offset]
        self.xyt[:,:,0] -= x_center[:, cp.newaxis]
        self.xyt[:,:,1] -= y_center[:, cp.newaxis]
        self.h = cp.stack([size, 0*size, 0*size], axis=1)  # (n_solutions, 3)
