"""
Minkowski difference computation for tree packing.

The Minkowski difference A ⊖ B = A ⊕ (-B) gives the "configuration space obstacle" -
the set of all translations t such that if tree2's center is at t, then tree1 and 
tree2 overlap (when tree1 is centered at the origin).

For the packing problem, this defines the "forbidden zone" for tree2's center 
relative to tree1.
"""

import numpy as np
from shapely.geometry import Polygon, Point, MultiPolygon, LineString, MultiPoint
from shapely import affinity
from shapely.ops import unary_union

import kaggle_support as kgs


def _polygon_vertices(poly: Polygon) -> list:
    """
    Return unique exterior vertices (without the duplicated closing point).
    """
    coords = list(poly.exterior.coords)
    if len(coords) >= 2 and coords[0] == coords[-1]:
        coords = coords[:-1]
    return coords


def _minkowski_difference_convex(A: Polygon, B: Polygon) -> Polygon:
    """
    Compute the Minkowski difference C = A ⊖ B = A ⊕ (-B).
    
    This gives the set of translations t such that A ∩ (B + t) is non-empty.
    
    For convex polygons, the result is convex and can be computed exactly as 
    the convex hull of all pairwise differences of vertices.
    
    Parameters
    ----------
    A : Polygon
        First convex polygon
    B : Polygon
        Second convex polygon
        
    Returns
    -------
    Polygon
        The Minkowski difference (convex hull)
    """
    A_verts = _polygon_vertices(A)
    B_verts = _polygon_vertices(B)

    # Reflect B through the origin: -B
    B_neg_verts = [(-x, -y) for (x, y) in B_verts]

    # Minkowski sum of convex polygons: convex hull of pairwise vertex sums
    # A ⊕ (-B) = {a + (-b) : a ∈ A, b ∈ B} = {a - b : a ∈ A, b ∈ B}
    pts = [(ax + bx, ay + by) for (ax, ay) in A_verts for (bx, by) in B_neg_verts]
    return MultiPoint(pts).convex_hull


def get_rotated_tree_parts(theta: float) -> list:
    """
    Get the convex decomposition of a tree at a given rotation.
    
    Parameters
    ----------
    theta : float
        Rotation angle in radians
        
    Returns
    -------
    convex_parts : list of Polygon
        Rotated convex decomposition
    """
    convex_parts = [affinity.rotate(p, np.degrees(theta), origin=(0, 0)) 
                    for p in kgs.convex_breakdown]
    return convex_parts


def get_rotated_tree(theta: float) -> Polygon:
    """
    Get the tree polygon at a given rotation.
    
    Parameters
    ----------
    theta : float
        Rotation angle in radians
        
    Returns
    -------
    tree : Polygon
        Rotated tree polygon
    """
    return affinity.rotate(kgs.center_tree, np.degrees(theta), origin=(0, 0))


def compute_forbidden_zone(theta2: float) -> Polygon:
    """
    Compute the Minkowski difference (forbidden zone) for two trees.
    
    Tree1 is fixed at the origin with rotation 0.
    Tree2 has the specified rotation.
    
    The returned polygon is the "forbidden zone" - if tree2's center is inside
    this region, the trees overlap.
    
    This is the Minkowski difference: Tree1 ⊖ Tree2 = Tree1 ⊕ (-Tree2)
    
    Parameters
    ----------
    theta2 : float
        Rotation of tree 2 in radians
        
    Returns
    -------
    forbidden_zone : Polygon or MultiPolygon
        The forbidden zone for tree2's center
    """
    # Get convex decompositions
    # Tree1 is always at rotation 0
    parts1 = kgs.convex_breakdown  # No rotation needed
    parts2 = get_rotated_tree_parts(theta2)
    
    # Compute pairwise Minkowski differences and union them
    polys = []
    for p1 in parts1:
        for p2 in parts2:
            polys.append(_minkowski_difference_convex(p1, p2))
    
    return unary_union(polys)


def compute_forbidden_zone_with_trees(theta2: float) -> tuple:
    """
    Compute the forbidden zone along with the tree polygons for visualization.
    
    Parameters
    ----------
    theta2 : float
        Rotation of tree 2 in radians
        
    Returns
    -------
    forbidden_zone : Polygon or MultiPolygon
        The forbidden zone for tree2's center
    tree1 : Polygon
        Tree1 at origin (rotation 0)
    tree2 : Polygon
        Tree2 at its rotation (centered at origin)
    """
    forbidden_zone = compute_forbidden_zone(theta2)
    tree1 = kgs.center_tree  # No rotation
    tree2 = get_rotated_tree(theta2)
    
    return forbidden_zone, tree1, tree2


def find_boundary_point(forbidden_zone: Polygon, angle: float) -> tuple:
    """
    Find the point on the forbidden zone boundary at a given angle from origin.
    
    Parameters
    ----------
    forbidden_zone : Polygon or MultiPolygon
        The forbidden zone
    angle : float
        Angle from origin in radians
        
    Returns
    -------
    (x, y) : tuple
        Point on boundary, or None if not found
    """
    # Create a ray from origin
    ray_length = 3.0  # Should be longer than max distance to boundary
    ray_end = (ray_length * np.cos(angle), ray_length * np.sin(angle))
    ray = LineString([(0, 0), ray_end])
    
    # Find intersection with boundary
    if isinstance(forbidden_zone, MultiPolygon):
        boundary = unary_union([g.exterior for g in forbidden_zone.geoms])
    else:
        boundary = forbidden_zone.exterior
    
    intersection = ray.intersection(boundary)
    
    if intersection.is_empty:
        return None
    
    # Get the point (might be MultiPoint if ray crosses multiple times)
    if intersection.geom_type == 'Point':
        return (intersection.x, intersection.y)
    elif intersection.geom_type == 'MultiPoint':
        # Take the closest point to origin
        points = list(intersection.geoms)
        closest = min(points, key=lambda p: p.distance(Point(0, 0)))
        return (closest.x, closest.y)
    else:
        # LineString intersection - take first point
        coords = list(intersection.coords)
        return coords[0]


def separation_distance(tree1_pos: tuple, tree2_pos: np.ndarray, theta2: float) -> np.ndarray:
    """
    Compute the signed separation distance between tree1 and multiple tree2 positions.
    
    Tree1 is always at rotation 0.
    
    Parameters
    ----------
    tree1_pos : tuple (x, y)
        Position of tree 1
    tree2_pos : np.ndarray
        Position(s) of tree 2. Can be:
        - (2,) array for a single position
        - (N, 2) array for N positions
    theta2 : float
        Rotation of tree 2 in radians
        
    Returns
    -------
    distance : np.ndarray
        Signed separation distance(s):
        - Positive: penetration depth (trees overlap)
        - Negative: clearance distance (trees don't overlap)
        Shape matches input: scalar-like for single position, (N,) for multiple.
    """
    tree2_pos = np.atleast_2d(tree2_pos)
    N = tree2_pos.shape[0]
    
    # Compute relative positions (tree2 relative to tree1)
    dx = tree2_pos[:, 0] - tree1_pos[0]
    dy = tree2_pos[:, 1] - tree1_pos[1]
    
    # Get forbidden zone for this rotation (computed once)
    forbidden_zone = compute_forbidden_zone(theta2)
    
    # Prepare the boundary for distance computation
    if isinstance(forbidden_zone, MultiPolygon):
        boundary = forbidden_zone.boundary
    else:
        boundary = forbidden_zone.exterior
    
    # Vectorized computation using shapely's array operations
    from shapely import points, contains, distance as shapely_distance
    
    # Create array of points (vectorized)
    point_array = points(dx, dy)
    
    # Vectorized contains check
    inside = contains(forbidden_zone, point_array)
    
    # Vectorized distance computation to boundary
    distances = shapely_distance(point_array, boundary)
    
    # Apply sign: positive if inside (overlap), negative if outside (clearance)
    results = np.where(inside, distances, -distances)
    
    return results if N > 1 else results[0]


def check_overlap(tree1_pos: tuple, tree2_pos: tuple, theta2: float) -> bool:
    """
    Check if two trees overlap.
    
    Tree1 is always at rotation 0.
    
    Parameters
    ----------
    tree1_pos : tuple (x, y)
        Position of tree 1
    tree2_pos : tuple (x, y)
        Position of tree 2
    theta2 : float
        Rotation of tree 2 in radians
        
    Returns
    -------
    bool
        True if trees overlap, False otherwise
    """
    dx = tree2_pos[0] - tree1_pos[0]
    dy = tree2_pos[1] - tree1_pos[1]
    
    forbidden_zone = compute_forbidden_zone(theta2)
    return forbidden_zone.contains(Point(dx, dy))
