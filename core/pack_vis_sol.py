import pandas as pd
import numpy as np
import scipy as sp
import cupy as cp
import kaggle_support as kgs
from shapely import affinity
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union
import colorsys


_TWO_PI = 2.0 * np.pi


def _pastel_color(i, n, s=0.45, l=0.72):
    """
    Generate a pastel-ish RGB color.
    - i: index of the polygon
    - n: total number of polygons
    - s: saturation (lower = more gray/pastel)
    - l: lightness (higher = lighter)
    """
    if n <= 0:
        n = 1
    h = (i / n) % 1.0   # evenly spaced hue
    r, g, b = colorsys.hls_to_rgb(h, l, s)
    return (r, g, b)


def _rotation_color(theta, s=0.85, l=0.55):
    """Map a rotation angle to a vivid, continuous cyclic color.

    The mapping is continuous on [0, 2π) and wraps seamlessly at 2π.
    """
    # Ensure continuity and wrap at 2π (0 and 2π yield identical colors).
    t = float(theta) % _TWO_PI
    h = t / _TWO_PI
    r, g, b = colorsys.hls_to_rgb(h, l, s)
    return (r, g, b)


def _iter_geoms(geom):
    """Yield Polygons from Polygon or MultiPolygon."""
    if isinstance(geom, Polygon):
        yield geom
    elif isinstance(geom, MultiPolygon):
        for g in geom.geoms:
            yield g
    else:
        raise TypeError(f"Expected Polygon or MultiPolygon, got {type(geom)}")


def _plot_polygons(polygons, ax, color_indices=None, thetas=None, alpha=1.0):
    """
    Plot a list of Shapely polygons.
    Overlapping regions are highlighted in black.

    Parameters
    ----------
    polygons : list of shapely.geometry.Polygon or MultiPolygon
    ax       : matplotlib Axes
    color_indices : list of int, optional
        If provided, color_indices[i] determines the color for polygon i.
        Polygons with the same index will have the same color.
    thetas : array-like, optional
        If provided, thetas[i] (radians) determines the color for polygon i using
        a continuous cyclic mapping of (theta mod 2π). Takes precedence over
        color_indices.
    alpha : float, optional
        Transparency level for polygons (0.0 = fully transparent, 1.0 = fully opaque)
        Default: 1.0
    """
    # Flatten any MultiPolygons
    flat_polys = []
    for geom in polygons:
        flat_polys.extend(list(_iter_geoms(geom)))

    n = len(flat_polys)
    if n == 0:
        return

    if thetas is not None and len(thetas) != n:
        raise ValueError(f"Expected thetas of length {n}, got {len(thetas)}")

    # Draw base polygons (pastel, no edges)
    for i, poly in enumerate(flat_polys):
        x, y = poly.exterior.xy
        # Prefer rotation-based coloring if available.
        if thetas is not None:
            facecolor = _rotation_color(thetas[i])
        # Otherwise use color index if provided, else use sequential coloring.
        elif color_indices is not None:
            color_idx = color_indices[i]
            # Determine total number of unique colors
            n_colors = len(set(color_indices))
            facecolor = _pastel_color(color_idx, n_colors)
        else:
            facecolor = _pastel_color(i, n)
        patch = MplPolygon(
            list(zip(x, y)),
            closed=True,
            facecolor=facecolor,
            edgecolor='none',
            alpha=alpha,
            zorder=1,
        )
        ax.add_patch(patch)

    # Compute pairwise intersections and draw them in black
    intersections = []
    eps_area = 1e-9
    for i in range(n):
        for j in range(i + 1, n):
            inter = flat_polys[i].intersection(flat_polys[j])
            if not inter.is_empty and getattr(inter, "area", 0) > eps_area:
                intersections.append(inter)

    if intersections:
        merged = unary_union(intersections)
        # merged may be Polygon, MultiPolygon, or GeometryCollection
        try:
            iter_geoms = list(_iter_geoms(merged))
        except TypeError:
            # if unary_union returned a geometry collection, try to extract polygons
            iter_geoms = []
            for g in getattr(merged, "geoms", []):
                try:
                    iter_geoms.extend(list(_iter_geoms(g)))
                except TypeError:
                    continue

        for geom in iter_geoms:
            x, y = geom.exterior.xy
            red_patch = MplPolygon(
                list(zip(x, y)),
                closed=True,
                facecolor=(0.0, 0.0, 0.0),  # black
                edgecolor=None,
                zorder=3,
                alpha=0.5,
            )
            ax.add_patch(red_patch)


def pack_vis_sol(sol, solution_idx=0, ax=None, margin_factor=0.1, alpha=1.0):
    """
    Visualize a solution from a SolutionCollection.

    Parameters
    ----------
    sol : SolutionCollection
        The solution collection to visualize
    solution_idx : int
        Index of the solution to visualize (default: 0)
    ax : matplotlib Axes, optional
        Existing axes to plot on (if None, creates new figure)
    margin_factor : float
        Fraction of the plot range to add as margin outside the boundary (default: 0.1)
    alpha : float, optional
        Transparency level for tree polygons (0.0 = fully transparent, 1.0 = fully opaque)
        Default: 1.0

    Returns
    -------
    ax : matplotlib Axes
        The axes object with the plot
    """
    if ax is None:
        fig, ax = plt.subplots()

    # Convert solution to TreeList for visualization
    tree_list = kgs.TreeList()
    tree_list.xyt = kgs.to_cpu(sol.xyt[solution_idx])

    thetas = np.asarray(tree_list.xyt)[:, 2]

    # Get the trees as shapely polygons
    trees = tree_list.get_trees()

    # Determine if this is a square or periodic solution
    if isinstance(sol, kgs.SolutionCollectionSquare):
        # Square solution: plot the bounding square
        h = kgs.to_cpu(sol.h[solution_idx])
        half = float(h[0]) / 2.0
        square = Polygon([(-half, -half), (half, -half), (half, half), (-half, half)])
        square = affinity.translate(square, h[1], h[2])

        # Draw square outline
        x, y = square.exterior.xy
        patch = MplPolygon(list(zip(x, y)), closed=True, facecolor='none',
                          edgecolor='blue', linewidth=3.0, zorder=2)
        ax.add_patch(patch)

        # Plot the trees
        _plot_polygons(trees, ax=ax, thetas=thetas, alpha=alpha)

        # Set plot limits based on both square AND trees to ensure all are visible
        square_bounds = square.bounds  # (minx, miny, maxx, maxy)

        # Get bounds from trees
        tree_bounds = [tree.bounds for tree in trees]
        tree_minx = min(b[0] for b in tree_bounds)
        tree_miny = min(b[1] for b in tree_bounds)
        tree_maxx = max(b[2] for b in tree_bounds)
        tree_maxy = max(b[3] for b in tree_bounds)

        # Take the maximum extent to include both square and all trees
        minx = min(square_bounds[0], tree_minx)
        miny = min(square_bounds[1], tree_miny)
        maxx = max(square_bounds[2], tree_maxx)
        maxy = max(square_bounds[3], tree_maxy)

        width = maxx - minx
        height = maxy - miny
        margin_x = width * margin_factor
        margin_y = height * margin_factor

        ax.set_xlim(minx - margin_x, maxx + margin_x)
        ax.set_ylim(miny - margin_y, maxy + margin_y)

    elif sol.periodic:
        # Periodic solution: plot 7x7 tiling and display unit cell

        # Get crystal axes
        crystal_axes = sol.get_crystal_axes_allocate()
        a_vec = kgs.to_cpu(crystal_axes[solution_idx, 0, :])  # (ax, ay)
        b_vec = kgs.to_cpu(crystal_axes[solution_idx, 1, :])  # (bx, by)

        # Create 15x15 tiling (centered on origin)
        all_trees = []
        all_thetas = []
        
        # Trees used for calculating bounds (inner 3x3)
        bounds_trees = []

        for i in range(-7, 8):
            for j in range(-7, 8):
                # Translation vector for this tile
                offset_x = i * a_vec[0] + j * b_vec[0]
                offset_y = i * a_vec[1] + j * b_vec[1]

                # Translate each tree in the base unit cell
                for tree, theta in zip(trees, thetas):
                    translated_tree = affinity.translate(tree, offset_x, offset_y)
                    all_trees.append(translated_tree)

                    # Keep same rotation color for all translated copies.
                    all_thetas.append(theta)
                    
                    # Collect trees for bounds calculation (inner 3x3)
                    if abs(i) <= 1 and abs(j) <= 1:
                        bounds_trees.append(translated_tree)

        # Plot all tiled trees with rotation-based colors
        _plot_polygons(all_trees, ax=ax, thetas=all_thetas, alpha=alpha)

        # Draw the unit cell outline (centered at origin)
        # Unit cell vertices: origin, a, a+b, b
        unit_cell = Polygon([
            (0, 0),
            (a_vec[0], a_vec[1]),
            (a_vec[0] + b_vec[0], a_vec[1] + b_vec[1]),
            (b_vec[0], b_vec[1])
        ])

        x, y = unit_cell.exterior.xy
        patch = MplPolygon(list(zip(x, y)), closed=True, facecolor='none',
                          edgecolor='green', linewidth=3.0, zorder=4, linestyle='--')
        ax.add_patch(patch)

        # Compute bounds of the inner 3x3 tiling
        all_bounds = [tree.bounds for tree in bounds_trees]
        minx = min(b[0] for b in all_bounds)
        miny = min(b[1] for b in all_bounds)
        maxx = max(b[2] for b in all_bounds)
        maxy = max(b[3] for b in all_bounds)

        # Add margin
        width = maxx - minx
        height = maxy - miny
        margin_x = width * margin_factor
        margin_y = height * margin_factor

        ax.set_xlim(minx - margin_x, maxx + margin_x)
        ax.set_ylim(miny - margin_y, maxy + margin_y)

    else:
        # Generic SolutionCollection: just plot trees
        _plot_polygons(trees, ax=ax, thetas=thetas, alpha=alpha)

        # Compute bounds from trees
        all_bounds = [tree.bounds for tree in trees]
        minx = min(b[0] for b in all_bounds)
        miny = min(b[1] for b in all_bounds)
        maxx = max(b[2] for b in all_bounds)
        maxy = max(b[3] for b in all_bounds)

        # Add margin
        width = maxx - minx
        height = maxy - miny
        margin_x = width * margin_factor
        margin_y = height * margin_factor

        ax.set_xlim(minx - margin_x, maxx + margin_x)
        ax.set_ylim(miny - margin_y, maxy + margin_y)

    ax.set_aspect("equal", adjustable="box")
    return ax
