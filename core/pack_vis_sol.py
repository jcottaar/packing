"""
Solution Visualization Module

Provides visualization utilities for packing solutions, including color mapping,
polygon rendering, lattice unit cell analysis, and support for both square and
periodic boundary conditions.

This code is released under CC BY-SA 4.0, meaning you can freely use and adapt
it (including commercially), but must give credit to the original author 
(Jeroen Cottaar) and keep it under this license.
"""

import numpy as np
import kaggle_support as kgs
from shapely import affinity
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union
import colorsys
import math


_TWO_PI = 2.0 * np.pi  # Precomputed constant for cyclic color mapping


def _pastel_color(i, n, s=0.45, L=0.72):
    """Generate a pastel RGB color using evenly spaced hue distribution.

    Args:
        i: Index of the polygon (0-based).
        n: Total number of polygons.
        s: Saturation value (0-1), lower values create more pastel colors.
        L: Lightness value (0-1), higher values create lighter colors.

    Returns:
        Tuple[float, float, float]: RGB color tuple with values in [0, 1].
    """
    # Ensure valid denominator
    if n <= 0:
        n = 1

    # Compute evenly spaced hue
    h = (i / n) % 1.0
    r, g, b = colorsys.hls_to_rgb(h, L, s)
    return (r, g, b)


def _rotation_color(theta, s=0.85, L=0.55):
    """Map rotation angle to a vivid cyclic RGB color.

    The mapping is continuous on [0, 2π) and wraps seamlessly at 2π,
    allowing visually distinct colors for different orientations.

    Args:
        theta: Rotation angle in radians.
        s: Saturation value (0-1), higher values create more vivid colors.
        L: Lightness value (0-1).

    Returns:
        Tuple[float, float, float]: RGB color tuple with values in [0, 1].
    """
    # Ensure continuity and wrap at 2π (0 and 2π yield identical colors)
    t = float(theta) % _TWO_PI
    h = t / _TWO_PI
    r, g, b = colorsys.hls_to_rgb(h, L, s)
    return (r, g, b)


def _iter_geoms(geom):
    """Yield individual Polygon objects from Polygon or MultiPolygon.

    Args:
        geom: Shapely Polygon or MultiPolygon object.

    Yields:
        shapely.geometry.Polygon: Individual polygon geometries.

    Raises:
        TypeError: If input is neither Polygon nor MultiPolygon.
    """
    if isinstance(geom, Polygon):
        yield geom
    elif isinstance(geom, MultiPolygon):
        for g in geom.geoms:
            yield g
    else:
        raise TypeError(f"Expected Polygon or MultiPolygon, got {type(geom)}")


def _plot_polygons(polygons, ax, color_indices=None, thetas=None, alpha=1.0):
    """Plot Shapely polygons with color coding and overlap highlighting.

    Overlapping regions between polygons are highlighted in black. Colors can be
    determined by rotation angle (cyclic color map) or by polygon index (pastel).

    Args:
        polygons: List of shapely Polygon or MultiPolygon objects.
        ax: Matplotlib Axes object to draw on.
        color_indices: Optional list of integers. If provided, polygons with the
            same index share the same color. Length must match number of polygons.
        thetas: Optional array of rotation angles (radians). If provided, determines
            colors via continuous cyclic mapping. Takes precedence over color_indices.
            Length must match number of polygons.
        alpha: Transparency level for polygon faces (0.0 = transparent, 1.0 = opaque).

    Returns:
        None (modifies ax in-place).
    """
    # Flatten MultiPolygons into individual Polygon objects
    flat_polys = []
    for geom in polygons:
        flat_polys.extend(list(_iter_geoms(geom)))

    n = len(flat_polys)
    if n == 0:
        return

    # Validate theta array length if provided
    if thetas is not None and len(thetas) != n:
        raise ValueError(f"Expected thetas of length {n}, got {len(thetas)}")

    # Draw base polygons with color coding
    for i, poly in enumerate(flat_polys):
        x, y = poly.exterior.xy

        # Determine facecolor based on available color scheme
        if thetas is not None:
            # Prefer rotation-based coloring if available
            facecolor = _rotation_color(thetas[i])
        elif color_indices is not None:
            # Use color index if provided
            color_idx = color_indices[i]
            n_colors = len(set(color_indices))
            facecolor = _pastel_color(color_idx, n_colors)
        else:
            # Default to sequential coloring
            facecolor = _pastel_color(i, n)

        # Create and add polygon patch
        patch = MplPolygon(
            list(zip(x, y)),
            closed=True,
            facecolor=facecolor,
            edgecolor='none',
            alpha=alpha,
            zorder=1,
        )
        ax.add_patch(patch)

    # Compute and visualize pairwise intersections
    intersections = []
    eps_area = 1e-9

    for i in range(n):
        for j in range(i + 1, n):
            inter = flat_polys[i].intersection(flat_polys[j])
            if not inter.is_empty and getattr(inter, "area", 0) > eps_area:
                intersections.append(inter)

    # Draw overlapping regions in black
    if intersections:
        merged = unary_union(intersections)

        # Extract polygons from merged geometry (may be collection)
        try:
            iter_geoms = list(_iter_geoms(merged))
        except TypeError:
            # Handle GeometryCollection
            iter_geoms = []
            for g in getattr(merged, "geoms", []):
                try:
                    iter_geoms.extend(list(_iter_geoms(g)))
                except TypeError:
                    continue

        # Draw each overlap region
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


def _smallest_angle_deg(vec_a, vec_b):
    """Compute the smaller interior angle between two 2D vectors.

    Args:
        vec_a: First vector as array-like (2,).
        vec_b: Second vector as array-like (2,).

    Returns:
        float: Angle in degrees, always in range [0, 90].
    """
    # Compute vector norms
    na = float(np.linalg.norm(vec_a))
    nb = float(np.linalg.norm(vec_b))

    # Handle degenerate vectors
    if na < 1e-12 or nb < 1e-12:
        return 0.0

    # Compute angle via dot product
    cos_theta = np.clip(np.dot(vec_a, vec_b) / (na * nb), -1.0, 1.0)
    theta = np.degrees(np.arccos(cos_theta))

    # Return smaller of the two supplementary angles
    return min(theta, 180.0 - theta)


def _canonicalize_unit_cell(vec_a, vec_b):
    """Normalize lattice basis vectors to canonical form.

    Removes duplicates related to ordering/sign by enforcing:
    - Positive cross product (right-handed basis)
    - First vector has non-negative x-component

    Args:
        vec_a: First basis vector as array-like (2,).
        vec_b: Second basis vector as array-like (2,).

    Returns:
        Tuple[np.ndarray, np.ndarray]: Canonicalized (a, b) vectors, shape (2,) each.
    """
    a = vec_a.astype(float).copy()
    b = vec_b.astype(float).copy()

    # Ensure positive cross product (right-handed)
    cross = a[0] * b[1] - a[1] * b[0]
    if cross < 0:
        a, b = b, a

    # Ensure first vector points into positive x half-plane
    if a[0] < -1e-9 or (abs(a[0]) <= 1e-9 and a[1] < 0):
        a = -a
        b = -b

    return a, b


def _wedge_contains_positive_x(vec_a, vec_b):
    """Check if the angular wedge from vec_a to vec_b contains the +X axis.

    Args:
        vec_a: First vector defining wedge boundary, array-like (2,).
        vec_b: Second vector defining wedge boundary, array-like (2,).

    Returns:
        bool: True if the counterclockwise angular sweep from vec_a to vec_b
            (with sweep angle ≤ 180°) contains the +X axis direction.
    """
    # Convert vectors to angles in [0, 360) degrees
    theta_a = (math.degrees(math.atan2(vec_a[1], vec_a[0])) + 360.0) % 360.0
    theta_b = (math.degrees(math.atan2(vec_b[1], vec_b[0])) + 360.0) % 360.0

    # Compute counterclockwise angular delta
    delta = (theta_b - theta_a + 360.0) % 360.0

    # If sweep is > 180°, wedge is on the opposite side
    if delta > 180.0:
        return False

    # Check if +X axis (0°) lies within the sweep
    offset = (-theta_a) % 360.0
    return offset <= delta + 1e-9


def _select_preferred_unit_cell(cells):
    """Select the most canonical unit cell from a list of candidates.

    Prefers cells whose wedge contains +X axis; breaks ties by choosing
    the cell with a basis vector closest to +X.

    Args:
        cells: List of tuples (vec_a, vec_b), each representing a unit cell.

    Returns:
        Tuple[np.ndarray, np.ndarray] or None: Preferred (a, b) basis vectors,
            or None if cells is empty.
    """
    if not cells:
        return None

    best = None
    best_score = None

    for a, b in cells:
        # Check if wedge contains +X axis
        contains = _wedge_contains_positive_x(a, b)

        # Compute angular deviations from +X axis
        theta_a = (math.degrees(math.atan2(a[1], a[0])) + 360.0) % 360.0
        theta_b = (math.degrees(math.atan2(b[1], b[0])) + 360.0) % 360.0
        delta_a = min(theta_a, 360.0 - theta_a)
        delta_b = min(theta_b, 360.0 - theta_b)
        deviation = min(delta_a, delta_b)

        # Score: prioritize wedge containment, then minimal angular deviation
        score = (0 if contains else 1, deviation)

        if best_score is None or score < best_score:
            best_score = score
            best = (a, b)

    return best


def _gauss_reduce_basis(vec_a, vec_b, angle_threshold_deg):
    """Apply 2D lattice reduction (Gauss algorithm) to basis vectors.

    Iteratively reduces the basis to guarantee an angle in [60°, 120°] range
    (if achievable), using the 2D Gauss reduction algorithm.

    Args:
        vec_a: First basis vector, array-like (2,).
        vec_b: Second basis vector, array-like (2,).
        angle_threshold_deg: Minimum acceptable angle between basis vectors (degrees).

    Returns:
        Tuple[np.ndarray, np.ndarray] or None: Reduced (a, b) basis vectors if
            successful, None if reduction fails or angle constraint is violated.
    """
    a = vec_a.astype(float).copy()
    b = vec_b.astype(float).copy()

    # Iteratively reduce basis (max 64 iterations)
    for _ in range(64):
        # Ensure |b| >= |a|
        if np.linalg.norm(b) < np.linalg.norm(a):
            a, b = b, a

        # Compute Gram-Schmidt coefficient
        denom = np.dot(a, a)
        if denom < 1e-12:
            return None

        mu = round(np.dot(a, b) / denom)
        b = b - mu * a

        # Check for degenerate basis
        if np.linalg.norm(b) < 1e-12:
            return None

        # Check convergence: |a·b| <= 0.5|a|²
        if abs(np.dot(a, b)) <= 0.5 * denom:
            break

    # Verify angle constraint
    angle = _smallest_angle_deg(a, b)
    if angle + 1e-9 < angle_threshold_deg:
        return None

    return a, b


def _find_alternative_unit_cells(
    vec_a,
    vec_b,
    min_angle_deg,
    coeff_range,
    max_cells,
):
    """Find alternative unit cells via unimodular transformations.

    Searches for equivalent lattice bases by applying integer unimodular
    transformations (det = ±1) that preserve lattice area and satisfy
    angle constraints.

    Args:
        vec_a: First basis vector, array-like (2,).
        vec_b: Second basis vector, array-like (2,).
        min_angle_deg: Minimum acceptable angle between basis vectors (degrees).
        coeff_range: Maximum absolute value for integer transformation coefficients.
        max_cells: Maximum number of alternative cells to return.

    Returns:
        List[Tuple[np.ndarray, np.ndarray]]: List of (a, b) basis vector pairs,
            sorted by decreasing angle and increasing total length. Shape of each
            vector is (2,).
    """
    coeff_range = max(1, int(coeff_range))
    max_cells = max(1, int(max_cells))

    # Build transformation matrix from input basis
    base = np.column_stack((vec_a, vec_b))
    det_base = float(np.linalg.det(base))
    if abs(det_base) < 1e-12:
        return []

    seen = set()  # Track canonicalized cells to avoid duplicates
    alt_cells = []
    values = range(-coeff_range, coeff_range + 1)

    def _add_candidate(candidate_a, candidate_b):
        """Add a candidate cell if not already seen."""
        candidate_a, candidate_b = _canonicalize_unit_cell(
            candidate_a, candidate_b
        )
        key = tuple(np.round(np.concatenate((candidate_a, candidate_b)), 8))
        if key in seen:
            return
        seen.add(key)
        alt_cells.append((candidate_a, candidate_b))

    # Enumerate unimodular transformations (det = 1)
    for m11 in values:
        for m12 in values:
            for m21 in values:
                for m22 in values:
                    # Skip identity transformation
                    if m11 == 1 and m22 == 1 and m12 == 0 and m21 == 0:
                        continue

                    # Check unimodular condition
                    det_m = m11 * m22 - m12 * m21
                    if det_m != 1:
                        continue

                    # Apply transformation
                    transform = np.array(
                        [[m11, m12], [m21, m22]], dtype=float
                    )
                    candidate = base @ transform
                    cand_a = candidate[:, 0]
                    cand_b = candidate[:, 1]

                    # Check angle constraint
                    if _smallest_angle_deg(cand_a, cand_b) + 1e-9 < min_angle_deg:
                        continue

                    # Add candidate and check limit
                    if len(alt_cells) >= max_cells:
                        break
                    _add_candidate(cand_a, cand_b)

                if len(alt_cells) >= max_cells:
                    break
            if len(alt_cells) >= max_cells:
                break
        if len(alt_cells) >= max_cells:
            break

    # Fallback: add Gauss-reduced basis if no candidates found
    if not alt_cells:
        reduced = _gauss_reduce_basis(vec_a, vec_b, min_angle_deg)
        if reduced is not None:
            cand_a, cand_b = reduced
            if _smallest_angle_deg(cand_a, cand_b) + 1e-9 >= min_angle_deg:
                _add_candidate(cand_a, cand_b)

    # Sort by decreasing angle, then increasing total length
    alt_cells.sort(
        key=lambda ab: (
            -_smallest_angle_deg(ab[0], ab[1]),
            np.linalg.norm(ab[0]) + np.linalg.norm(ab[1])
        )
    )

    return alt_cells[:max_cells]


def pack_vis_sol(
    sol,
    solution_idx=0,
    ax=None,
    margin_factor=0.1,
    alpha=1.0,
    plot_alt_unit_cells=False,
    alt_cell_search_range=2,
    max_alt_unit_cells=32,
):
    """Visualize a packing solution with trees and boundary conditions.

    Supports square and periodic boundary conditions. For periodic solutions,
    creates a tiled visualization showing neighboring unit cells. Optionally
    displays alternative unit cell representations.

    Args:
        sol: SolutionCollection object containing the packing configuration.
        solution_idx: Index of the specific solution to visualize (0-based).
        ax: Matplotlib Axes object to draw on. If None, creates new figure.
        margin_factor: Fraction of plot range to add as whitespace margin (0-1).
        alpha: Transparency level for tree polygons (0.0 = transparent, 1.0 = opaque).
        plot_alt_unit_cells: If True, draw alternative unit cells for periodic
            solutions (cells with same area and angle >= 60°).
        alt_cell_search_range: Maximum absolute integer coefficient for unimodular
            transformations when searching for alternative cells.
        max_alt_unit_cells: Maximum number of alternative cells to display.

    Returns:
        matplotlib.axes.Axes: The axes object with the rendered visualization.
    """
    # Create axes if not provided
    if ax is None:
        fig, ax = plt.subplots()

    # Extract tree positions and convert to TreeList
    tree_list = kgs.TreeList()
    tree_list.xyt = kgs.to_cpu(sol.xyt[solution_idx])  # shape: (N_trees, 3)

    thetas = np.asarray(tree_list.xyt)[:, 2]  # rotation angles (N_trees,)

    # Get tree geometries as Shapely polygons
    trees = tree_list.get_trees()

    # Handle different solution types
    if isinstance(sol, kgs.SolutionCollectionSquare):
        # Square boundary visualization
        h = kgs.to_cpu(sol.h[solution_idx])  # shape: (3,) [size, cx, cy]
        half = float(h[0]) / 2.0

        # Create square polygon
        square = Polygon([
            (-half, -half),
            (half, -half),
            (half, half),
            (-half, half)
        ])
        square = affinity.translate(square, h[1], h[2])

        # Draw square outline
        x, y = square.exterior.xy
        patch = MplPolygon(
            list(zip(x, y)),
            closed=True,
            facecolor='none',
            edgecolor='blue',
            linewidth=3.0,
            zorder=2
        )
        ax.add_patch(patch)

        # Plot trees with rotation-based colors
        _plot_polygons(trees, ax=ax, thetas=thetas, alpha=alpha)

        # Compute bounds from both square and trees
        square_bounds = square.bounds  # (minx, miny, maxx, maxy)

        tree_bounds = [tree.bounds for tree in trees]
        tree_minx = min(b[0] for b in tree_bounds)
        tree_miny = min(b[1] for b in tree_bounds)
        tree_maxx = max(b[2] for b in tree_bounds)
        tree_maxy = max(b[3] for b in tree_bounds)

        # Take union of extents
        minx = min(square_bounds[0], tree_minx)
        miny = min(square_bounds[1], tree_miny)
        maxx = max(square_bounds[2], tree_maxx)
        maxy = max(square_bounds[3], tree_maxy)

        # Apply margin
        width = maxx - minx
        height = maxy - miny
        margin_x = width * margin_factor
        margin_y = height * margin_factor

        ax.set_xlim(minx - margin_x, maxx + margin_x)
        ax.set_ylim(miny - margin_y, maxy + margin_y)

    elif sol.periodic:
        # Periodic boundary visualization with tiling
        crystal_axes = sol.get_crystal_axes_allocate()
        a_vec = kgs.to_cpu(crystal_axes[solution_idx, 0, :])  # shape: (2,)
        b_vec = kgs.to_cpu(crystal_axes[solution_idx, 1, :])  # shape: (2,)

        # Create 15x15 tiling centered on origin
        all_trees = []
        all_thetas = []
        bounds_trees = []  # Inner 3x3 for bounds calculation

        for i in range(-7, 8):
            for j in range(-7, 8):
                # Compute translation for this tile
                offset_x = i * a_vec[0] + j * b_vec[0]
                offset_y = i * a_vec[1] + j * b_vec[1]

                # Translate each tree in the base unit cell
                for tree, theta in zip(trees, thetas):
                    translated_tree = affinity.translate(
                        tree, offset_x, offset_y
                    )
                    all_trees.append(translated_tree)
                    all_thetas.append(theta)  # Preserve rotation color

                    # Collect trees for bounds (inner 3x3 only)
                    if abs(i) <= 1 and abs(j) <= 1:
                        bounds_trees.append(translated_tree)

        # Plot all tiled trees
        _plot_polygons(all_trees, ax=ax, thetas=all_thetas, alpha=alpha)

        # Draw primary unit cell outline
        unit_cell = Polygon([
            (0, 0),
            (a_vec[0], a_vec[1]),
            (a_vec[0] + b_vec[0], a_vec[1] + b_vec[1]),
            (b_vec[0], b_vec[1])
        ])

        x, y = unit_cell.exterior.xy
        patch = MplPolygon(
            list(zip(x, y)),
            closed=True,
            facecolor='none',
            edgecolor='green',
            linewidth=3.0,
            zorder=4,
            linestyle='--'
        )
        ax.add_patch(patch)

        # Optionally plot alternative unit cells
        if plot_alt_unit_cells:
            alt_cells = _find_alternative_unit_cells(
                a_vec,
                b_vec,
                min_angle_deg=60.0,
                coeff_range=alt_cell_search_range,
                max_cells=max_alt_unit_cells,
            )

            # Filter to cells with wedge containing +X axis
            if alt_cells:
                positive_x_cells = [
                    cell for cell in alt_cells
                    if _wedge_contains_positive_x(cell[0], cell[1])
                ]
                if positive_x_cells:
                    alt_cells = positive_x_cells
                else:
                    preferred = _select_preferred_unit_cell(alt_cells)
                    alt_cells = [preferred] if preferred is not None else []

                # Draw alternative cells with colormap
                cmap = plt.cm.get_cmap('plasma', len(alt_cells))
                for idx, (alt_a, alt_b) in enumerate(alt_cells):
                    len_a = float(np.linalg.norm(alt_a))
                    len_b = float(np.linalg.norm(alt_b))

                    if len_a < 1e-12 or len_b < 1e-12:
                        continue

                    # Compute cell metrics
                    shorter = min(len_a, len_b)
                    longer = max(len_a, len_b)
                    ratio = shorter / longer
                    angle = _smallest_angle_deg(alt_a, alt_b)

                    print(
                        f"Alternative unit cell {idx+1}: "
                        f"axis ratio={ratio:.4f}, angle={angle:.2f}°"
                    )

                    # Draw alternative cell polygon
                    alt_cell = Polygon([
                        (0.0, 0.0),
                        tuple(alt_a),
                        tuple(alt_a + alt_b),
                        tuple(alt_b),
                    ])

                    alt_x, alt_y = alt_cell.exterior.xy
                    alt_patch = MplPolygon(
                        list(zip(alt_x, alt_y)),
                        closed=True,
                        facecolor='none',
                        edgecolor=cmap(idx),
                        linewidth=2.0,
                        linestyle=':',
                        zorder=4.5,
                    )
                    ax.add_patch(alt_patch)

        # Set plot limits based on inner 3x3 tiling
        all_bounds = [tree.bounds for tree in bounds_trees]
        minx = min(b[0] for b in all_bounds)
        miny = min(b[1] for b in all_bounds)
        maxx = max(b[2] for b in all_bounds)
        maxy = max(b[3] for b in all_bounds)

        # Apply margin
        width = maxx - minx
        height = maxy - miny
        margin_x = width * margin_factor
        margin_y = height * margin_factor

        ax.set_xlim(minx - margin_x, maxx + margin_x)
        ax.set_ylim(miny - margin_y, maxy + margin_y)

    else:
        # Generic solution: just plot trees
        _plot_polygons(trees, ax=ax, thetas=thetas, alpha=alpha)

        # Compute bounds from trees
        all_bounds = [tree.bounds for tree in trees]
        minx = min(b[0] for b in all_bounds)
        miny = min(b[1] for b in all_bounds)
        maxx = max(b[2] for b in all_bounds)
        maxy = max(b[3] for b in all_bounds)

        # Apply margin
        width = maxx - minx
        height = maxy - miny
        margin_x = width * margin_factor
        margin_y = height * margin_factor

        ax.set_xlim(minx - margin_x, maxx + margin_x)
        ax.set_ylim(miny - margin_y, maxy + margin_y)

    # Set equal aspect ratio
    ax.set_aspect("equal", adjustable="box")

    return ax
