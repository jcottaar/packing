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


def _iter_geoms(geom):
    """Yield Polygons from Polygon or MultiPolygon."""
    if isinstance(geom, Polygon):
        yield geom
    elif isinstance(geom, MultiPolygon):
        for g in geom.geoms:
            yield g
    else:
        raise TypeError(f"Expected Polygon or MultiPolygon, got {type(geom)}")


def _plot_polygons(polygons, ax):
    """
    Plot a list of Shapely polygons with distinct unsaturated colors.
    Overlapping regions are highlighted in bright red.

    Parameters
    ----------
    polygons : list of shapely.geometry.Polygon or MultiPolygon
    ax       : matplotlib Axes
    """
    # Flatten any MultiPolygons
    flat_polys = []
    for geom in polygons:
        flat_polys.extend(list(_iter_geoms(geom)))

    n = len(flat_polys)
    if n == 0:
        return

    # Draw base polygons (pastel, no edges)
    for i, poly in enumerate(flat_polys):
        x, y = poly.exterior.xy
        facecolor = _pastel_color(i, n)
        patch = MplPolygon(
            list(zip(x, y)),
            closed=True,
            facecolor=facecolor,
            edgecolor='none',
            zorder=1,
        )
        ax.add_patch(patch)

    # Compute pairwise intersections and draw them in bright red
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
                facecolor=(1.0, 0.0, 0.0),  # bright red
                edgecolor=None,
                zorder=3,
                alpha=0.5,
            )
            ax.add_patch(red_patch)


def pack_vis_sol(sol, solution_idx=0, ax=None, margin_factor=0.1):
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
        _plot_polygons(trees, ax=ax)

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

    elif isinstance(sol, kgs.SolutionCollectionLattice):
        # Periodic solution: plot 3x3 tiling and display unit cell

        # Get crystal axes
        crystal_axes = sol.get_crystal_axes_allocate()
        a_vec = kgs.to_cpu(crystal_axes[solution_idx, 0, :])  # (ax, ay)
        b_vec = kgs.to_cpu(crystal_axes[solution_idx, 1, :])  # (bx, by)

        # Create 3x3 tiling (centered on origin)
        all_trees = []
        for i in range(-1, 2):
            for j in range(-1, 2):
                # Translation vector for this tile
                offset_x = i * a_vec[0] + j * b_vec[0]
                offset_y = i * a_vec[1] + j * b_vec[1]

                # Translate each tree in the base unit cell
                for tree in trees:
                    translated_tree = affinity.translate(tree, offset_x, offset_y)
                    all_trees.append(translated_tree)

        # Plot all tiled trees
        _plot_polygons(all_trees, ax=ax)

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

        # Compute bounds of the 3x3 tiling
        all_bounds = [tree.bounds for tree in all_trees]
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
        _plot_polygons(trees, ax=ax)

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
