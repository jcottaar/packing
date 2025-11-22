import pandas as pd
import numpy as np
import scipy as sp
import cupy as cp
import kaggle_support as kgs

def visualize_tree_list(tree_list, ax=None):
    trees = tree_list.get_trees()
    return plot_polygons(trees, ax=ax)
    

import matplotlib.pyplot as plt
# Allow large embedded animation output (2 GB) for notebooks
plt.rcParams['animation.embed_limit'] = 2 * 1024 * 1024 * 1024
from matplotlib.patches import Polygon as MplPolygon
from shapely.geometry import Polygon, MultiPolygon
import colorsys
from shapely.ops import unary_union


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


def plot_polygons(polygons, ax=None):
    """
    Plot a list of Shapely polygons with distinct unsaturated colors.
    Overlapping regions are highlighted in bright red.

    Parameters
    ----------
    polygons : list of shapely.geometry.Polygon or MultiPolygon
    ax       : existing matplotlib Axes (optional)
    """
    if ax is None:
        fig, ax = plt.subplots()

    # Flatten any MultiPolygons
    flat_polys = []
    for geom in polygons:
        flat_polys.extend(list(_iter_geoms(geom)))

    n = len(flat_polys)
    if n == 0:
        return ax

    minx = min(poly.bounds[0] for poly in flat_polys)
    miny = min(poly.bounds[1] for poly in flat_polys)
    maxx = max(poly.bounds[2] for poly in flat_polys)
    maxy = max(poly.bounds[3] for poly in flat_polys)

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

    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)
    ax.set_aspect("equal", adjustable="box")
    return ax