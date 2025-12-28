# import os, subprocess
# print("CUDA_MPS_PIPE_DIRECTORY =", os.environ.get("CUDA_MPS_PIPE_DIRECTORY"))
# print("CUDA_MPS_LOG_DIRECTORY  =", os.environ.get("CUDA_MPS_LOG_DIRECTORY"))
# print("mps-server running =", subprocess.run(["bash","-lc","pgrep -x nvidia-cuda-mps-server"], capture_output=True).returncode == 0)

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

def assert_mps():
    import subprocess
    out = subprocess.run(
        ["bash", "-lc", "pgrep -f -x nvidia-cuda-mps-server"],
        stdout=subprocess.PIPE,
    )
    if out.returncode != 0:
        print("WARNING: CUDA MPS not active")
        #if env=='vast':
        #    raise Exception('no MPS')

if os.path.isdir('/mnt/d/packing/'):
    env = 'local'
    d_drive = '/mnt/d/'    
else:
    env = 'vast'
print(env)
assert_mps()

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


'''
Precision control - note set to float32 at the end of the module
'''
USE_FLOAT32, dtype_cp, dtype_np, just_over_one = None, None, None, None
def set_float32(use_float32:bool):
    global USE_FLOAT32, dtype_cp, dtype_np, just_over_one, TREE_EXPANSION
    if use_float32:
        USE_FLOAT32, dtype_cp, dtype_np = True, cp.float32, np.float32
        just_over_one = 1.000001
    else:
        USE_FLOAT32, dtype_cp, dtype_np = False, cp.float64, np.float64
        just_over_one = 1.00000000000001
    TREE_EXPANSION = 10*(1.000001-1)

    # Initialize tree globals after dtype is set (defined later in module)
    # Use late binding to avoid forward reference issues
    initialize_tree_globals()

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

module_profiler = None

def enable_module_profiling(module, include_private=True):
    """Enable profiling for all functions and methods in a module

    Parameters
    ----------
    module : module
        The module to profile
    include_private : bool, default=True
        If True, includes methods starting with single underscore (e.g., _do_move)
        Always excludes dunder methods except __init__ and __call__
    """
    global module_profiler

    module_profiler = LineProfiler()

    for name, obj in inspect.getmembers(module):
        # Profile module-level functions
        if inspect.isfunction(obj) and obj.__module__ == module.__name__:
            print(f'Added function: {name}')
            module_profiler.add_function(obj)

        # Profile class methods
        elif inspect.isclass(obj) and obj.__module__ == module.__name__:
            print(f'Processing class: {name}')
            for method_name, method in inspect.getmembers(obj):
                if inspect.isfunction(method) or inspect.ismethod(method):
                    # Determine if we should include this method
                    should_include = False

                    if method_name.startswith('__'):
                        # Only include specific dunder methods
                        should_include = method_name in ['__init__', '__call__']
                    elif method_name.startswith('_'):
                        # Single underscore - include if include_private is True
                        should_include = include_private
                    else:
                        # Public method - always include
                        should_include = True

                    if should_include:
                        print(f'  Added method: {name}.{method_name}')
                        module_profiler.add_function(method)

    module_profiler.enable()

def print_module_profile():
    """Print accumulated profiling stats"""
    global module_profiler
    if module_profiler:
        module_profiler.disable()
        module_profiler.print_stats()


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
TREE_EXPANSION = 0.0  # Outward expansion distance for trees

def create_center_tree():
    """Initializes the Christmas tree"""

    trunk_w = 0.15 + 2*TREE_EXPANSION
    trunk_h = 0.2 + TREE_EXPANSION
    base_w = 0.7 + 4*TREE_EXPANSION
    mid_w = 0.4 + 6*TREE_EXPANSION
    top_w = 0.25 + 4*TREE_EXPANSION
    tip_y = 0.8 + TREE_EXPANSION
    tier_1_y = 0.5 - TREE_EXPANSION
    tier_2_y = 0.25 - TREE_EXPANSION
    base_y = 0.0 - TREE_EXPANSION
    trunk_bottom_y = -trunk_h
    
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

# Global tree properties - initialized by calling initialize_tree_globals()
center_tree, convex_breakdown, tree_max_radius, tree_centroid_offset = None, None, None, None
center_tree_prepped = None
tree_area = None
tree_vertices = None

def initialize_tree_globals():
    """Initialize global tree properties. Must be called explicitly before using trees."""
    global center_tree, convex_breakdown, tree_max_radius, tree_centroid_offset
    global center_tree_prepped, tree_area, tree_vertices

    center_tree, convex_breakdown, tree_max_radius, tree_centroid_offset = create_center_tree()
    center_tree_prepped = prep(center_tree)
    tree_area = center_tree.area
    tree_vertices = cp.array(np.array(center_tree.exterior.coords[:-1]), dtype=dtype_cp)  # (n_vertices, 2)


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
class SolutionCollection(BaseClass):
    xyt: cp.ndarray = field(default=None)  # (N,3) array of tree positions and angles
    h: cp.ndarray = field(default=None)      # (N,_N_h_DOF) array
    use_fixed_h: bool = field(default=False)
    periodic: bool = field(default=False)  # whether to use periodic boundaries

    _N_h_DOF: int = field(default=None, init=False, repr=False)  # number of h degrees of freedom

    def _check_constraints(self):
        if self.xyt.ndim != 3 or self.xyt.shape[2] != 3:
            raise ValueError("Solution: xyt must be an array with shape (N_solutions, N_trees, 3)")
        assert self.h.shape == (self.xyt.shape[0],self._N_h_DOF)

    # add N_solutions and N_trees properties
    @property
    def N_solutions(self) -> int:
        """Number of solution rows in xyt (N_solutions)."""
        if self.xyt is None:
            return 0
        return self.xyt.shape[0]

    @property
    def N_trees(self) -> int:
        """Number of trees per solution (N_trees)."""
        if self.xyt is None:
            return 0
        return self.xyt.shape[1]
    
    def rotate(self, angle:cp.ndarray):
        # rotate xyt by given angle (radians)
        xyt_cp = self.xyt  # assume already cupy array and non-empty        

        c = cp.cos(angle)[:, None]
        s = cp.sin(angle)[:, None]

        x = xyt_cp[:, :, 0]
        y = xyt_cp[:, :, 1]
        cx = cp.mean(x, axis=1)[:, None]
        cy = cp.mean(y, axis=1)[:, None]

        x0 = x - cx
        y0 = y - cy

        x_rot =  c * x0 + s * y0
        y_rot = -s * x0 + c * y0

        xyt_cp[:, :, 0] = x_rot + cx
        xyt_cp[:, :, 1] = y_rot + cy
        xyt_cp[:, :, 2] = (xyt_cp[:, :, 2] - angle[:, None]) % (2 * np.pi)
    
    def get_crystal_axes_allocate(self):
        """Get crystal axes for each solution. Returns (N_solutions, 2, 2) array."""        
        assert self.periodic
        crystal_axes = cp.array(cp.zeros((self.N_solutions, 2, 2), dtype=dtype_cp))
        self._get_crystal_axes(crystal_axes)
        return crystal_axes
    
    def get_crystal_axes(self, crystal_axes):
        assert self.periodic
        self._get_crystal_axes(crystal_axes)

    def select_ids(self, inds):
        self.xyt = self.xyt[inds]
        self.h = self.h[inds]

    def merge(self, other:'SolutionCollection'):
        self.xyt = cp.concatenate([self.xyt, other.xyt], axis=0)
        self.h = cp.concatenate([self.h, other.h], axis=0)
    
    def create_clone(self, idx: int, other: 'SolutionCollection', parent_id: int):
        self.xyt[idx] = other.xyt[parent_id]
        self.h[idx] = other.h[parent_id]

    def create_clone_batch(self, inds: np.ndarray, other: 'SolutionCollection', parent_ids: np.ndarray):
        """Vectorized batch clone operation."""
        self.xyt[inds] = other.xyt[parent_ids]
        self.h[inds] = other.h[parent_ids]

    def create_empty(self, N_solutions: int, N_trees: int):
        xyt = cp.zeros((N_solutions, N_trees, 3), dtype=dtype_cp)
        h = cp.zeros((N_solutions, self._N_h_DOF), dtype=dtype_cp)        
        return type(self)(xyt=xyt, h=h, use_fixed_h=self.use_fixed_h, periodic=self.periodic)
    # subclasses must implement: snap, compute_cost, compute_cost_single_ref, get_crystal_axes
    

@dataclass
class SolutionCollectionSquare(SolutionCollection):

    def __post_init__(self):
        self._N_h_DOF = 3  # h = [size, x_offset, y_offset]
        return super().__post_init__()
    
    def compute_cost_single_ref(self):
        """Compute area cost and grad_bound for a single reference"""
        h = self.h[0]
        cost = h[0]**2
        # Build grad_bound on the GPU without implicitly converting via NumPy
        grad_bound = cp.empty(3, dtype=h.dtype)
        grad_bound[0] = 2.0 * h[0]
        grad_bound[1] = 0
        grad_bound[2] = 0
        return cost, grad_bound
    
    def compute_cost(self, sol:SolutionCollection, cost:cp.ndarray, grad_bound:cp.ndarray):
        """Compute area cost and grad_bound for multiple solutions"""
        cost[:] = sol.h[:,0]**2
        grad_bound[:,0] = 2.0*sol.h[:,0]
        grad_bound[:,1] = 0
        grad_bound[:,2] = 0

    def _get_crystal_axes(self, crystal_axes):
        crystal_axes[:,0,0] = self.h[:,0]
        crystal_axes[:,0,1] = 0
        crystal_axes[:,1,0] = 0
        crystal_axes[:,1,1] = self.h[:,0]    

    def snap(self):
        """Set h such that for each solution it's the smallest possible square containing all trees.
        Vectorized implementation using tree_vertices.
        Assumes xyt is kgs.dtype_cp.
        """

        if self.use_fixed_h:
            return

        # xyt shape: (n_solutions, n_trees, 3)
        # tree_vertices shape: (n_vertices, 2)
        
        n_solutions = self.xyt.shape[0]
        n_trees = self.xyt.shape[1]
        n_vertices = tree_vertices.shape[0]
        
        # Extract pose components
        x = self.xyt[:, :, 0:1]  # (n_solutions, n_trees, 1)
        y = self.xyt[:, :, 1:2]  # (n_solutions, n_trees, 1)
        theta = self.xyt[:, :, 2:3]  # (n_solutions, n_trees, 1)
        
        # Precompute rotation matrices
        cos_t = cp.cos(theta)  # (n_solutions, n_trees, 1)
        sin_t = cp.sin(theta)  # (n_solutions, n_trees, 1)
        
        # Get local vertices (n_vertices, 2)
        vx_local = tree_vertices[:, 0]  # (n_vertices,)
        vy_local = tree_vertices[:, 1]  # (n_vertices,)
        
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
        if self.h is None:
            self.h = cp.stack([size, 0*size, 0*size], axis=1)  # (n_solutions, 3)
        else:
            self.h = cp.stack([cp.minimum(size, self.h[:,0]), 0*size, 0*size], axis=1)  # (n_solutions, 3)

@dataclass
class SolutionCollectionLattice(SolutionCollection):
    # h[0]: crystal axis 1 length (a_length)
    # h[1]: crystal axis 2 length (b_length)
    # h[2]: angle between axes (radians)

    do_snap: bool = field(init=True, default=True)

    def __post_init__(self):
        self._N_h_DOF = 3
        self.periodic = True
        return super().__post_init__()

    def compute_cost_single_ref(self):
        """Compute area cost and grad_bound for a single solution.

        Area = a_length * b_length * sin(angle)

        Returns:
            cost: scalar area
            grad_bound: (3,) array with derivatives [d/da, d/db, d/dangle]
        """
        h = self.h[0]
        a_length = float(h[0].get().item())
        b_length = float(h[1].get().item())
        angle = float(h[2].get().item())

        sin_angle = cp.sin(h[2])
        cos_angle = cp.cos(h[2])

        # Area = |a × b| = a_length * b_length * sin(angle)
        prod = h[0] * h[1] * sin_angle
        area = cp.abs(prod)
        sgn = cp.sign(prod)
        # Propagate absolute value into gradients (subgradient 0 when prod==0)
        grad_bound = cp.zeros_like(h)
        grad_bound[0] = sgn * (h[1] * sin_angle)          # ∂|prod|/∂a_length
        grad_bound[1] = sgn * (h[0] * sin_angle)          # ∂|prod|/∂b_length
        grad_bound[2] = sgn * (h[0] * h[1] * cos_angle)   # ∂|prod|/∂angle
        return area, grad_bound

    def compute_cost(self, sol:SolutionCollection, cost:cp.ndarray, grad_bound:cp.ndarray):
        """Compute area cost and grad_bound for multiple solutions (vectorized).

        Args:
            sol: This SolutionCollection instance (self)
            cost: (N_solutions,) output array
            grad_bound: (N_solutions, 3) output array
        """
        # Vectorized computation over all solutions
        sin_angle = cp.sin(sol.h[:, 2])
        cos_angle = cp.cos(sol.h[:, 2])

        # Area = a_length * b_length * sin(angle)
        res = sol.h[:, 0] * sol.h[:, 1] * sin_angle
        cost_sign = cp.sign(res)
        cost[:] = cp.abs(res)


        # Gradients
        grad_bound[:, 0] = cost_sign * (sol.h[:, 1] * sin_angle)          # ∂A/∂a_length
        grad_bound[:, 1] = cost_sign * (sol.h[:, 0] * sin_angle)          # ∂A/∂b_length
        grad_bound[:, 2] = cost_sign * (sol.h[:, 0] * sol.h[:, 1] * cos_angle)  # ∂A/∂angle

    def _get_crystal_axes(self, crystal_axes):
        """Fill crystal_axes array with lattice vectors.

        Args:
            crystal_axes: (N_solutions, 2, 2) output array
                crystal_axes[i, 0, :] = first lattice vector (a)
                crystal_axes[i, 1, :] = second lattice vector (b)

        Convention:
            a = (a_length, 0)  - first axis along x
            b = (b_length * cos(angle), b_length * sin(angle))
        """
        # First axis: a = (a_length, 0)
        crystal_axes[:, 0, 0] = self.h[:, 0]  # a_x = a_length
        crystal_axes[:, 0, 1] = 0              # a_y = 0

        # Second axis: b = (b_length * cos(angle), b_length * sin(angle))
        crystal_axes[:, 1, 0] = self.h[:, 1] * cp.cos(self.h[:, 2])  # b_x
        crystal_axes[:, 1, 1] = self.h[:, 1] * cp.sin(self.h[:, 2])  # b_y

    @profile_each_line
    def snap(self):
        if not self.do_snap:
            return
        import pack_cost
        import boolean_line_search
        if self.N_trees > 1 and self.periodic:
            self.periodic = False
            # For each solution, check if overlap exists. If not, scale xy until overlap just occurs.
            overlap_cost, _, _ = pack_cost.CollisionCostOverlappingArea().compute_cost_allocate(self, evaluate_gradient=False)
            needs_scaling = overlap_cost <= 0

            if cp.any(needs_scaling):
                # Store original xy positions for solutions that need scaling
                xy_orig = self.xyt[:, :, :2].copy()  # (N_solutions, N_trees, 2)
                centroids = cp.mean(xy_orig, axis=1, keepdims=True)  # (N_solutions, 1, 2)

                # Create vectorized function for boolean_line_search_vectorized
                # We need to work only on solutions that need scaling
                needs_scaling_indices = cp.where(needs_scaling)[0]

                if len(needs_scaling_indices) > 0:
                    # Extract solutions that need scaling
                    xy_to_scale = xy_orig[needs_scaling_indices]  # (n_needs_scaling, N_trees, 2)
                    centroids_to_scale = centroids[needs_scaling_indices]  # (n_needs_scaling, 1, 2)

                    def f(factors):
                        # factors: (n_needs_scaling,) array of scaling factors
                        # Apply scaling to each solution
                        xy_scaled = (xy_to_scale - centroids_to_scale) * factors[:, cp.newaxis, cp.newaxis] + centroids_to_scale

                        # Temporarily update xyt for overlap check
                        xyt_temp = self.xyt[needs_scaling_indices].copy()
                        xyt_temp[:, :, :2] = xy_scaled

                        # Create temporary solution collection for cost computation
                        sol_tmp = copy.deepcopy(self)
                        sol_tmp.xyt = xyt_temp

                        # Check overlap
                        cost, _, _ = pack_cost.CollisionCostOverlappingArea().compute_cost_allocate(sol_tmp, evaluate_gradient=False)
                        return cost > 0

                    # Find minimal scaling factor s in [lo, hi] where overlap occurs
                    s_lo, s_hi = 1e-2, 10.
                    factors = boolean_line_search.boolean_line_search_vectorized(f, s_lo, s_hi, len(needs_scaling_indices), max_iter=30)

                    # Apply the found scaling to solutions that needed it
                    xy_scaled = (xy_to_scale - centroids_to_scale) * factors[:, cp.newaxis, cp.newaxis] + centroids_to_scale
                    self.xyt[needs_scaling_indices, :, :2] = xy_scaled

            # Final assert: all solutions must now have overlap
            #assert cp.all(pack_cost.CollisionCostOverlappingArea().compute_cost_allocate(self, evaluate_gradient=False)[0] > 0)
            self.xyt[:,:,0] -= cp.mean(self.xyt[:,:,0],axis=1)[:,None]
            self.xyt[:,:,1] -= cp.mean(self.xyt[:,:,1],axis=1)[:,None]                   
            self.periodic = True


@dataclass
class SolutionCollectionLatticeRectangle(SolutionCollectionLattice):
    # h[0]: crystal axis 1 length (a_length)
    # h[1]: crystal axis 2 length (b_length)

    def __post_init__(self):        
        super().__post_init__()
        self._N_h_DOF = 2

    def compute_cost_single_ref(self):
        """Compute area cost and grad_bound for a single solution.

        Area = a_length * b_length * sin(angle)

        Returns:
            cost: scalar area
            grad_bound: (3,) array with derivatives [d/da, d/db, d/dangle]
        """
        h = self.h[0]
        # Area = |a × b| = a_length * b_length * sin(angle)
        prod = h[0] * h[1]
        area = cp.abs(prod)
        sgn = cp.sign(prod)
        # Propagate absolute value into gradients (subgradient 0 when prod==0)
        grad_bound = cp.zeros_like(h)
        grad_bound[0] = sgn * (h[1])          # ∂|prod|/∂a_length
        grad_bound[1] = sgn * (h[0])          # ∂|prod|/∂b_length
        return area, grad_bound

    def compute_cost(self, sol:SolutionCollection, cost:cp.ndarray, grad_bound:cp.ndarray):
        """Compute area cost and grad_bound for multiple solutions (vectorized).

        Args:
            sol: This SolutionCollection instance (self)
            cost: (N_solutions,) output array
            grad_bound: (N_solutions, 3) output array
        """

        # Area = a_length * b_length * sin(angle)
        res = sol.h[:, 0] * sol.h[:, 1]
        cost_sign = cp.sign(res)
        cost[:] = cp.abs(res)


        # Gradients
        grad_bound[:, 0] = cost_sign * (sol.h[:, 1])          # ∂A/∂a_length
        grad_bound[:, 1] = cost_sign * (sol.h[:, 0])          # ∂A/∂b_length

    def _get_crystal_axes(self, crystal_axes):
        """Fill crystal_axes array with lattice vectors.

        Args:
            crystal_axes: (N_solutions, 2, 2) output array
                crystal_axes[i, 0, :] = first lattice vector (a)
                crystal_axes[i, 1, :] = second lattice vector (b)

        Convention:
            a = (a_length, 0)  - first axis along x
            b = (b_length * cos(angle), b_length * sin(angle))
        """
        # First axis: a = (a_length, 0)
        crystal_axes[:, 0, 0] = self.h[:, 0]  # a_x = a_length
        crystal_axes[:, 0, 1] = 0              # a_y = 0

        # Second axis: b = (b_length * cos(angle), b_length * sin(angle))
        crystal_axes[:, 1, 0] = 0  # b_x
        crystal_axes[:, 1, 1] = self.h[:, 1] # b_y


@dataclass
class SolutionCollectionLatticeFixed(SolutionCollectionLattice):
    # h[0]: crystal axis 1 length (a_length)
    
    aspect_ratios: cp.ndarray = field(default=None)  # (N_solutions,) array of aspect ratios (b_length / a_length)

    def __post_init__(self):        
        super().__post_init__()
        self._N_h_DOF = 1

    def _check_constraints(self):
        super()._check_constraints()
        assert self.aspect_ratios.shape == (self.N_solutions,)

    def compute_cost_single_ref(self):
        h = self.h[0]
        prod = h[0] * h[0] * self.aspect_ratios[0]
        area = cp.abs(prod)
        sgn = cp.sign(prod)
        grad_bound = cp.zeros_like(h)
        grad_bound[0] = sgn * (2 * h[0] * self.aspect_ratios[0])          # ∂|prod|/∂a_length
        return area, grad_bound

    def compute_cost(self, sol:SolutionCollection, cost:cp.ndarray, grad_bound:cp.ndarray):
        # Area = a_length * b_length * sin(angle)
        res = sol.h[:, 0]**2 * self.aspect_ratios
        cost_sign = cp.sign(res)
        cost[:] = cp.abs(res)
        grad_bound[:, 0] = cost_sign * (2 * sol.h[:, 0] * self.aspect_ratios)          # ∂A/∂a_length

    def _get_crystal_axes(self, crystal_axes):
        # First axis: a = (a_length, 0)
        crystal_axes[:, 0, 0] = self.h[:, 0]  # a_x = a_length
        crystal_axes[:, 0, 1] = 0              # a_y = 0

        # Second axis: b = (b_length * cos(angle), b_length * sin(angle))
        crystal_axes[:, 1, 0] = 0  # b_x
        crystal_axes[:, 1, 1] = self.h[:, 0] * self.aspect_ratios  # b_y

    def select_ids(self, inds):
        super().select_ids(inds)
        self.aspect_ratios = self.aspect_ratios[inds]

    def merge(self, other:'SolutionCollectionLatticeFixed'):
        super().merge(other)
        self.aspect_ratios = cp.concatenate([self.aspect_ratios, other.aspect_ratios], axis=0)

    def create_clone(self, idx, other, parent_id):
        self.aspect_ratios[idx] = other.aspect_ratios[parent_id]
        super().create_clone(idx, other, parent_id)

    def create_clone_batch(self, inds, other, parent_ids):
        """Vectorized batch clone operation."""
        self.aspect_ratios[inds] = other.aspect_ratios[parent_ids]
        super().create_clone_batch(inds, other, parent_ids)

    def create_empty(self, N_solutions: int, N_trees: int):
        res = super().create_empty(N_solutions, N_trees)
        res.aspect_ratios = cp.array([self.aspect_ratios[0]]*N_solutions, dtype=dtype_cp)
        return res


# ============================================================
# Fitness comparison utilities for tuple-based fitness
# ============================================================

def lexicographic_argmin(fitness_array: np.ndarray) -> int:
    """Find index of minimum fitness value using lexicographic ordering.
    
    Parameters
    ----------
    fitness_array : np.ndarray
        Shape (N_solutions, N_components) - fitness tuples for each solution
        
    Returns
    -------
    int
        Index of the lexicographically smallest fitness tuple
    """
    # Use lexsort which sorts by last column first, so reverse the columns
    if fitness_array.ndim == 1:
        # Handle 1D case (single component)
        return int(np.argmin(fitness_array))
    
    # Sort by all components in order (last column has highest priority in lexsort)
    # We want first column to have highest priority, so reverse
    sort_keys = [fitness_array[:, i] for i in range(fitness_array.shape[1] - 1, -1, -1)]
    sorted_indices = np.lexsort(sort_keys)
    return int(sorted_indices[0])


def lexicographic_argsort(fitness_array: np.ndarray) -> np.ndarray:
    """Sort fitness values using lexicographic ordering.
    
    Parameters
    ----------
    fitness_array : np.ndarray
        Shape (N_solutions, N_components) - fitness tuples for each solution
        
    Returns
    -------
    np.ndarray
        Indices that would sort the array lexicographically
    """
    if fitness_array.ndim == 1:
        # Handle 1D case (single component)
        return np.argsort(fitness_array)
    
    # Sort by all components in order (last column has highest priority in lexsort)
    # We want first column to have highest priority, so reverse
    sort_keys = [fitness_array[:, i] for i in range(fitness_array.shape[1] - 1, -1, -1)]
    return np.lexsort(sort_keys)


def lexicographic_less_than(fitness1: np.ndarray, fitness2: np.ndarray) -> bool:
    """Compare two fitness tuples lexicographically.
    
    Parameters
    ----------
    fitness1, fitness2 : np.ndarray
        Shape (N_components,) - fitness tuples to compare
        
    Returns
    -------
    bool
        True if fitness1 < fitness2 lexicographically
    """
    fitness1 = np.atleast_1d(fitness1)
    fitness2 = np.atleast_1d(fitness2)
    
    for f1, f2 in zip(fitness1, fitness2):
        if f1 < f2:
            return True
        elif f1 > f2:
            return False
    return False  # Equal


def compute_genetic_diversity_matrix_shortcut(
    population_xyt: cp.ndarray,
    reference_xyt: cp.ndarray,
    lap_config=None,
) -> cp.ndarray:
    """
    Memory-lean shortcut diversity metric.

    Same functionality as before, but:
      - avoids stacking all 8 transformed cost tensors at once
      - avoids per-transform .copy() of reference arrays
      - removes unused lap_batch import
    """
    assert lap_config.algorithm in ("min_cost_row", "min_cost_col")

    N_pop, N_trees, _ = population_xyt.shape
    N_ref, N_trees_ref, _ = reference_xyt.shape

    assert N_trees == N_trees_ref, (
        f"Number of trees mismatch: population has {N_trees}, reference has {N_trees_ref}"
    )

    transformations = [
        (0.0, False),
        (np.pi / 2, False),
        (np.pi, False),
        (3 * np.pi / 2, False),
        (0.0, True),
        (np.pi / 2, True),
        (np.pi, True),
        (3 * np.pi / 2, True),
    ]

    # Population coordinates (fixed, not transformed)
    pop_x = population_xyt[:, :, 0]      # (N_pop, N_trees)
    pop_y = population_xyt[:, :, 1]      # (N_pop, N_trees)
    pop_theta = population_xyt[:, :, 2]  # (N_pop, N_trees)

    # Reference base views (do NOT copy; keep immutable)
    ref_x0 = reference_xyt[:, :, 0]      # (N_ref, N_trees)
    ref_y0 = reference_xyt[:, :, 1]      # (N_ref, N_trees)
    ref_theta0 = reference_xyt[:, :, 2]  # (N_ref, N_trees)

    # Running minimum over transformations
    min_distances = cp.full((N_pop, N_ref), cp.inf, dtype=dtype_cp)

    if profiling:
        cp.cuda.Device().synchronize()

    for rot_angle, do_mirror in transformations:
        # ---------------------------------------------------------
        # Step 1: Apply transformation to reference (no .copy())
        # ---------------------------------------------------------
        ref_x = ref_x0
        ref_y = ref_y0
        ref_theta = ref_theta0

        if do_mirror:
            # creates new arrays; does not mutate the base views
            ref_y = -ref_y
            ref_theta = cp.pi - ref_theta

        if rot_angle != 0.0:
            # scalar trig on CPU is fine here (broadcasted over GPU arrays)
            cos_a = np.cos(rot_angle)
            sin_a = np.sin(rot_angle)
            new_x = ref_x * cos_a - ref_y * sin_a
            new_y = ref_x * sin_a + ref_y * cos_a
            ref_x = new_x
            ref_y = new_y
            ref_theta = ref_theta + rot_angle

        # Wrap to [-pi, pi]
        ref_theta = cp.remainder(ref_theta + np.pi, 2 * np.pi) - np.pi

        # ---------------------------------------------------------
        # Step 2: Pairwise cost (GPU)
        # ---------------------------------------------------------
        dx = pop_x[:, :, cp.newaxis, cp.newaxis] - ref_x[cp.newaxis, cp.newaxis, :, :]
        dy = pop_y[:, :, cp.newaxis, cp.newaxis] - ref_y[cp.newaxis, cp.newaxis, :, :]
        dtheta = pop_theta[:, :, cp.newaxis, cp.newaxis] - ref_theta[cp.newaxis, cp.newaxis, :, :]
        dtheta = cp.remainder(dtheta + np.pi, 2 * np.pi) - np.pi

        if profiling:
            cp.cuda.Device().synchronize()

        # (N_pop, N_trees, N_ref, N_trees)
        cost_matrices = dx**2 + dy**2 + dtheta**2

        if profiling:
            cp.cuda.Device().synchronize()

        # ---------------------------------------------------------
        # Step 3: Shortcut assignment reduction for this transform
        # ---------------------------------------------------------
        batched = cost_matrices.transpose(0, 2, 1, 3).reshape(-1, N_trees, N_trees)  # (N_pop*N_ref, N, N)

        if lap_config.algorithm == "min_cost_row":
            min_per_row = cp.amin(batched, axis=2)                 # (N_pop*N_ref, N)
            assignment_costs = cp.sum(cp.sqrt(min_per_row), axis=1) # (N_pop*N_ref,)
        else:  # "min_cost_col"
            min_per_col = cp.amin(batched, axis=1)                 # (N_pop*N_ref, N)
            assignment_costs = cp.sum(cp.sqrt(min_per_col), axis=1) # (N_pop*N_ref,)

        costs_this_transform = assignment_costs.reshape(N_pop, N_ref)  # (N_pop, N_ref)
        min_distances = cp.minimum(min_distances, costs_this_transform)

        if profiling:
            cp.cuda.Device().synchronize()

    return min_distances

def compute_genetic_diversity_matrix(population_xyt: cp.ndarray, reference_xyt: cp.ndarray, lap_config=None, allow_shortcut=True, transform=True) -> cp.ndarray:
    """
    Compute the minimum-cost assignment distance between each pair of individuals
    from two populations, considering all 8 symmetry transformations
    (4 rotations × 2 mirror states).
    
    Uses the Hungarian algorithm to find the optimal tree-to-tree mapping
    that minimizes total distance. The distance metric includes (x, y, theta) with equal weights.
    
    Parameters
    ----------
    population_xyt : cp.ndarray
        Shape (N_pop, N_trees, 3). First population of individuals, where each individual
        has N_trees trees with (x, y, theta) coordinates.
    reference_xyt : cp.ndarray
        Shape (N_ref, N_trees, 3). Second population of individuals to compare against.
        
    Returns
    -------
    cp.ndarray
        Shape (N_pop, N_ref). Minimum assignment distance for each pair, taken over
        all 8 symmetry transformations.
        
    Notes
    -----
    The 8 transformations applied to each population individual are:
    - 0°, 90°, 180°, 270° rotations (about origin)
    - Each rotation with and without x-axis mirroring
    
    For each transformation, we compute the cost matrix between transformed trees
    and reference trees, then solve the linear assignment problem. The minimum
    cost across all 8 transformations is returned.
    
    Distance for each tree pair: sqrt((x1-x2)^2 + (y1-y2)^2 + angular_dist(theta1, theta2)^2)
    where angular_dist wraps to [-pi, pi].
    """
    import lap_batch

    if (lap_config is not None) and (lap_config.algorithm == 'min_cost_row' or lap_config.algorithm == 'min_cost_col') and allow_shortcut \
            and transform:
        return compute_genetic_diversity_matrix_shortcut(population_xyt, reference_xyt, lap_config)
    
    N_pop, N_trees, _ = population_xyt.shape
    N_ref, N_trees_ref, _ = reference_xyt.shape
    
    # Validate shapes
    assert N_trees == N_trees_ref, \
        f"Number of trees mismatch: population has {N_trees}, reference has {N_trees_ref}"
    
    # Pre-compute the 8 transformation parameters
    # Each transformation is (rotation_angle, mirror_x)
    # rotation_angle: angle to rotate coordinates (0, pi/2, pi, 3pi/2)
    # mirror_x: whether to mirror across x-axis before rotation
    if transform:
        transformations = [
            (0.0,        False),  # Identity
            (np.pi/2,    False),  # 90° rotation
            (np.pi,      False),  # 180° rotation
            (3*np.pi/2,  False),  # 270° rotation
            (0.0,        True),   # Mirror only
            (np.pi/2,    True),   # Mirror + 90° rotation
            (np.pi,      True),   # Mirror + 180° rotation
            (3*np.pi/2,  True),   # Mirror + 270° rotation
        ]
    else:
        transformations = [
            (0.0, False),  # Identity only
        ]
    
    # Population coordinates (fixed, not transformed)
    pop_x = population_xyt[:, :, 0]      # (N_pop, N_trees)
    pop_y = population_xyt[:, :, 1]      # (N_pop, N_trees)
    pop_theta = population_xyt[:, :, 2]  # (N_pop, N_trees)
    
    # Compute cost matrices for all 8 transformations on GPU
    all_cost_matrices = []
    if profiling:
        cp.cuda.Device().synchronize()
    for rot_angle, do_mirror in transformations:
        # ---------------------------------------------------------
        # Step 1: Apply transformation to reference individuals (GPU)
        # ---------------------------------------------------------
        ref_x = reference_xyt[:, :, 0].copy()
        ref_y = reference_xyt[:, :, 1].copy()
        ref_theta = reference_xyt[:, :, 2].copy()
        
        if do_mirror:
            ref_y = -ref_y
            ref_theta = cp.pi-ref_theta
        
        if rot_angle != 0.0:
            cos_a = np.cos(rot_angle)
            sin_a = np.sin(rot_angle)
            new_x = ref_x * cos_a - ref_y * sin_a
            new_y = ref_x * sin_a + ref_y * cos_a
            ref_x = new_x
            ref_y = new_y
            ref_theta = ref_theta + rot_angle
        
        ref_theta = cp.remainder(ref_theta + np.pi, 2*np.pi) - np.pi
        
        # ---------------------------------------------------------
        # Step 2: Compute pairwise cost matrix (GPU)
        # ---------------------------------------------------------
        # Shape calculations:
        # pop_x[:, :, None, None]: (N_pop, N_trees, 1, 1)
        # ref_x[None, None, :, :]: (1, 1, N_ref, N_trees)
        # Result: (N_pop, N_trees, N_ref, N_trees)
        dx = pop_x[:, :, cp.newaxis, cp.newaxis] - ref_x[cp.newaxis, cp.newaxis, :, :]
        dy = pop_y[:, :, cp.newaxis, cp.newaxis] - ref_y[cp.newaxis, cp.newaxis, :, :]
        dtheta = pop_theta[:, :, cp.newaxis, cp.newaxis] - ref_theta[cp.newaxis, cp.newaxis, :, :]
        dtheta = cp.remainder(dtheta + np.pi, 2*np.pi) - np.pi
        if profiling:
            cp.cuda.Device().synchronize()
        cost_matrices = cp.sqrt(dx**2 + dy**2 + dtheta**2)
        if profiling:
            cp.cuda.Device().synchronize()
        
        all_cost_matrices.append(cost_matrices)
    
    # ---------------------------------------------------------
    # Step 3: Solve assignment problems on GPU using RAFT
    # ---------------------------------------------------------
    # Stack all cost matrices on GPU: shape (8, N_pop, N_trees, N_ref, N_trees)
    # Then reshape to (8*N_pop*N_ref, N_trees, N_trees) for batched solving
    stacked = cp.stack(all_cost_matrices, axis=0)  # (8, N_pop, N_trees, N_ref, N_trees)
    batched = stacked.transpose(0, 1, 3, 2, 4).reshape(-1, N_trees, N_trees)  # (8*N_pop*N_ref, N_trees, N_trees)

    if profiling:
        cp.cuda.Device().synchronize()
    
    # Solve all LAPs on GPU
    _, all_assignment_costs = lap_batch.solve_lap_batch(batched, config=lap_config)  # (8*N_pop*N_ref,)

    if profiling:
        cp.cuda.Device().synchronize()
    
    # Reshape back and take minimum across transformations
    all_costs_array = all_assignment_costs.reshape(8 if transform else 1, N_pop, N_ref)  # (8, N_pop, N_ref)
    min_distances = all_costs_array.min(axis=0)  # (N_pop, N_ref)
    
    if profiling:
        cp.cuda.Device().synchronize()
    
    return min_distances


def compute_genetic_diversity(population_xyt: cp.ndarray, reference_xyt: cp.ndarray, lap_config=None, transform=True) -> cp.ndarray:
    """
    Compute the minimum-cost assignment distance between each individual in a population
    and a single reference configuration, considering all 8 symmetry transformations
    (4 rotations × 2 mirror states).
    
    This is a wrapper around compute_genetic_diversity_matrix for backward compatibility.
    
    Parameters
    ----------
    population_xyt : cp.ndarray
        Shape (N_pop, N_trees, 3). Population of individuals, where each individual
        has N_trees trees with (x, y, theta) coordinates.
    reference_xyt : cp.ndarray
        Shape (N_trees, 3). Single reference configuration to compare against.
        
    Returns
    -------
    cp.ndarray
        Shape (N_pop,). Minimum assignment distance for each individual, taken over
        all 8 symmetry transformations.
    """
    N_pop, N_trees, _ = population_xyt.shape
    
    # Validate shapes
    assert reference_xyt.shape == (N_trees, 3), \
        f"Reference shape {reference_xyt.shape} doesn't match expected ({N_trees}, 3)"
    
    # Add batch dimension to reference and call the matrix version
    reference_batch = reference_xyt[cp.newaxis, :, :]  # (1, N_trees, 3)
    result_matrix = compute_genetic_diversity_matrix(population_xyt, reference_batch, lap_config=lap_config, transform=transform)  # (N_pop, 1)
    
    return result_matrix[:, 0]  # (N_pop,)


def find_best_transformation(xyt1: cp.ndarray, xyt2: cp.ndarray) -> tuple:
    """
    Find the transformation of xyt2 that minimizes genetic diversity with xyt1.

    Both individuals are centered at their centroids before comparison to make
    the result invariant to initial translation.

    Parameters
    ----------
    xyt1 : cp.ndarray
        Shape (N_trees, 3) - first individual (x, y, theta)
    xyt2 : cp.ndarray
        Shape (N_trees, 3) - second individual to transform

    Returns
    -------
    transformed_xyt2 : cp.ndarray
        Shape (N_trees, 3) - xyt2 after applying best transformation, centered at origin
    rotation_angle : float
        Rotation angle applied in radians (0, π/2, π, 3π/2)
    mirrored : bool
        Whether x-axis mirroring was applied
    """
    import lap_batch

    N_trees = xyt1.shape[0]

    # Validate shapes
    assert xyt1.shape == (N_trees, 3), f"xyt1 shape {xyt1.shape} doesn't match expected ({N_trees}, 3)"
    assert xyt2.shape == (N_trees, 3), f"xyt2 shape {xyt2.shape} doesn't match expected ({N_trees}, 3)"

    # Center both individuals at origin (immune to transforms)
    xyt1_centered = xyt1.copy()
    xyt1_centered[:, 0] -= cp.mean(xyt1[:, 0])
    xyt1_centered[:, 1] -= cp.mean(xyt1[:, 1])

    xyt2_centered = xyt2.copy()
    xyt2_centered[:, 0] -= cp.mean(xyt2[:, 0])
    xyt2_centered[:, 1] -= cp.mean(xyt2[:, 1])

    # Define the 8 transformations (rotation_angle, mirror_x)
    transformations = [
        (0.0,        False),  # Identity
        (np.pi/2,    False),  # 90° rotation
        (np.pi,      False),  # 180° rotation
        (3*np.pi/2,  False),  # 270° rotation
        (0.0,        True),   # Mirror only
        (np.pi/2,    True),   # Mirror + 90° rotation
        (np.pi,      True),   # Mirror + 180° rotation
        (3*np.pi/2,  True),   # Mirror + 270° rotation
    ]

    # Fixed reference (xyt1_centered)
    ref_x = xyt1_centered[:, 0]      # (N_trees,)
    ref_y = xyt1_centered[:, 1]      # (N_trees,)
    ref_theta = xyt1_centered[:, 2]  # (N_trees,)

    # Track best transformation
    best_cost = float('inf')
    best_transform_idx = 0
    best_transformed_xyt2 = None
    best_assignment = None

    # Try all 8 transformations
    for idx, (rot_angle, do_mirror) in enumerate(transformations):
        # Apply transformation to xyt2_centered
        trans_x = xyt2_centered[:, 0].copy()
        trans_y = xyt2_centered[:, 1].copy()
        trans_theta = xyt2_centered[:, 2].copy()

        if do_mirror:
            trans_y = -trans_y
            trans_theta = cp.pi - trans_theta

        if rot_angle != 0.0:
            cos_a = np.cos(rot_angle)
            sin_a = np.sin(rot_angle)
            new_x = trans_x * cos_a - trans_y * sin_a
            new_y = trans_x * sin_a + trans_y * cos_a
            trans_x = new_x
            trans_y = new_y
            trans_theta = trans_theta + rot_angle

        trans_theta = cp.remainder(trans_theta + np.pi, 2*np.pi) - np.pi

        # Compute pairwise cost matrix
        # ref: (N_trees,), trans: (N_trees,) -> cost_matrix: (N_trees, N_trees)
        dx = ref_x[:, cp.newaxis] - trans_x[cp.newaxis, :]
        dy = ref_y[:, cp.newaxis] - trans_y[cp.newaxis, :]
        dtheta = ref_theta[:, cp.newaxis] - trans_theta[cp.newaxis, :]
        dtheta = cp.remainder(dtheta + np.pi, 2*np.pi) - np.pi
        cost_matrix = cp.sqrt(dx**2 + dy**2 + dtheta**2)

        # Solve assignment problem
        cost_matrix_batch = cost_matrix[cp.newaxis, :, :]  # (1, N_trees, N_trees)
        assignments, assignment_costs = lap_batch.solve_lap_batch(cost_matrix_batch)
        assignment_cost = float(assignment_costs[0])

        # Track best
        if assignment_cost < best_cost:
            best_cost = assignment_cost
            best_transform_idx = idx
            best_transformed_xyt2 = cp.stack([trans_x, trans_y, trans_theta], axis=1)
            best_assignment = assignments[0]  # (N_trees,) - col indices for each row

    # Extract best transformation parameters
    best_rotation_angle, best_mirrored = transformations[best_transform_idx]

    # Reorder trees in transformed_xyt2 to match xyt1 ordering
    # best_assignment[i] tells us which tree in transformed_xyt2 corresponds to tree i in xyt1
    best_transformed_xyt2_reordered = best_transformed_xyt2[best_assignment]

    return best_transformed_xyt2_reordered, best_rotation_angle, best_mirrored


set_float32(True)