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


'''
Precision control
'''
USE_FLOAT32, dtype_cp, dtype_np = None, None, None
def set_float32(use_float32:bool):
    global USE_FLOAT32, dtype_cp, dtype_np
    if use_float32:
        USE_FLOAT32, dtype_cp, dtype_np = True, cp.float32, np.float32
    else:
        USE_FLOAT32, dtype_cp, dtype_np = False, cp.float64, np.float64
set_float32(True)

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
tree_vertices64 = cp.array(np.array(center_tree.exterior.coords[:-1]), dtype=cp.float64)
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
class SolutionCollection(BaseClass):
    xyt: cp.ndarray = field(default=None)  # (N,3) array of tree positions and angles
    h: cp.ndarray = field(default=None)      # (N,_N_h_DOF) array
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
        arr = to_cpu(self.xyt)
        return int(np.asarray(arr).shape[0])

    @property
    def N_trees(self) -> int:
        """Number of trees per solution (N_trees)."""
        if self.xyt is None:
            return 0
        arr = to_cpu(self.xyt)
        return int(np.asarray(arr).shape[1])
    
    def get_crystal_axes_allocate(self):
        """Get crystal axes for each solution. Returns (N_solutions, 2, 2) array."""        
        assert self.periodic
        crystal_axes = cp.array(cp.zeros((self.N_solutions, 2, 2), dtype=dtype_cp))
        self._get_crystal_axes(crystal_axes)
        return crystal_axes
    
    def get_crystal_axes(self, crystal_axes):
        assert self.periodic
        self._get_crystal_axes(crystal_axes)
    
    # subclasses must implement: snap, compute_cost, compute_cost_single_ref, get_crystal_axes
    

@dataclass
class SolutionCollectionSquare(SolutionCollection):

    def __post_init__(self):
        self._N_h_DOF = 3  # h = [size, x_offset, y_offset]
        return super().__post_init__()
    
    def compute_cost_single_ref(self, h:cp.ndarray):
        """Compute area cost and grad_bound for a single reference"""
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
        Vectorized implementation using tree_vertices32.
        Assumes xyt is kgs.dtype_cp.
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

@dataclass
class SolutionCollectionLattice(SolutionCollection):
    # h[0]: crystal axis 1 length (a_length)
    # h[1]: crystal axis 2 length (b_length)
    # h[2]: angle between axes (radians)

    def __post_init__(self):
        self._N_h_DOF = 3
        self.periodic = True
        return super().__post_init__()

    def compute_cost_single_ref(self, h:cp.ndarray):
        """Compute area cost and grad_bound for a single solution.

        Area = a_length * b_length * sin(angle)

        Returns:
            cost: scalar area
            grad_bound: (3,) array with derivatives [d/da, d/db, d/dangle]
        """
        a_length = float(h[0].get().item())
        b_length = float(h[1].get().item())
        angle = float(h[2].get().item())

        sin_angle = cp.sin(h[2])
        cos_angle = cp.cos(h[2])

        # Area = |a × b| = a_length * b_length * sin(angle)
        area = h[0] * h[1] * sin_angle

        # Gradients
        grad_bound = cp.zeros_like(h)
        grad_bound[0] = h[1] * sin_angle          # ∂A/∂a_length
        grad_bound[1] = h[0] * sin_angle          # ∂A/∂b_length
        grad_bound[2] = h[0] * h[1] * cos_angle   # ∂A/∂angle

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
        cost[:] = sol.h[:, 0] * sol.h[:, 1] * sin_angle

        # Gradients
        grad_bound[:, 0] = sol.h[:, 1] * sin_angle          # ∂A/∂a_length
        grad_bound[:, 1] = sol.h[:, 0] * sin_angle          # ∂A/∂b_length
        grad_bound[:, 2] = sol.h[:, 0] * sol.h[:, 1] * cos_angle  # ∂A/∂angle

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

    def snap(self):
        pass # skip for now