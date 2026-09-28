"""
Mesh preprocessing: convexity check and conditional smoothing.

After loading a mesh, this module:
1. Computes the convex hull.
2. Measures the symmetric surface-to-surface Chamfer distance between the
   original mesh and its convex hull (using trimesh.proximity for true
   nearest-surface distances).
3. If the distance is below a threshold the shape is near-convex and we
   replace it with its convex hull.
4. Otherwise, apply edge-preserving Taubin smoothing.
"""

import numpy as np
import trimesh

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


def _iter_with_progress(iterable, desc: str, enable: bool):
    if enable and tqdm is not None:
        return tqdm(iterable, desc=desc, leave=False)
    return iterable


def _mean_surface_distance(target_mesh: trimesh.Trimesh,
                           query_points: np.ndarray,
                           chunk_size: int = 2048,
                           show_progress: bool = False,
                           progress_desc: str = "closest_point") -> float:
    total = 0.0
    count = 0
    starts = range(0, len(query_points), chunk_size)
    starts = _iter_with_progress(starts, desc=progress_desc, enable=show_progress)
    for start in starts:
        pts = query_points[start:start + chunk_size]
        _, dists, _ = trimesh.proximity.closest_point(target_mesh, pts)
        dists = np.abs(np.asarray(dists, dtype=np.float64))
        total += float(np.sum(dists))
        count += int(dists.shape[0])
    return total / max(count, 1)


def surface_chamfer_distance(mesh_a: trimesh.Trimesh,
                             mesh_b: trimesh.Trimesh,
                             n_samples: int = 2000,
                             chunk_size: int = 2048,
                             show_progress: bool = False) -> float:
    """Symmetric surface-to-surface Chamfer distance (L2, averaged).

    For each direction the closest point **on the surface** of the other
    mesh is found via ``trimesh.proximity.closest_point``, so the result
    is a true point-to-surface distance (not vertex-to-vertex).

    CD = 0.5 * ( mean_{p in A_surf} dist(p, B_surf)
               + mean_{q in B_surf} dist(q, A_surf) )

    Args:
        mesh_a: First triangle mesh.
        mesh_b: Second triangle mesh.
        n_samples: Number of points sampled on each mesh surface.

    Returns:
        Symmetric surface Chamfer distance (scalar, same units as mesh).
    """
    # Sample points uniformly on both surfaces
    sample_count = max(256, int(n_samples))
    pts_a = mesh_a.sample(sample_count).astype(np.float64)
    pts_b = mesh_b.sample(sample_count).astype(np.float64)

    dists_a2b = _mean_surface_distance(
        mesh_b,
        pts_a,
        chunk_size=chunk_size,
        show_progress=show_progress,
        progress_desc="Chamfer A→B",
    )
    dists_b2a = _mean_surface_distance(
        mesh_a,
        pts_b,
        chunk_size=chunk_size,
        show_progress=show_progress,
        progress_desc="Chamfer B→A",
    )

    return 0.5 * float(dists_a2b + dists_b2a)


def is_near_convex(mesh: trimesh.Trimesh,
                   threshold: float = 0.0002,
                   n_samples: int = 2000,
                   chunk_size: int = 2048,
                   show_progress: bool = False,
                   verbose: bool = False) -> tuple:
    """Check whether *mesh* is approximately convex.

    Uses surface-to-surface Chamfer distance between the mesh and its
    convex hull.  For objects of size 2–3, the default threshold of 0.02
    corresponds to roughly 1 % of the object extent.

    Args:
        mesh: Input triangle mesh.
        threshold: Absolute surface Chamfer distance threshold
            (same units as mesh coordinates).

    Returns:
        (is_convex, convex_hull, chamfer_dist)
    """
    if verbose:
        print("[mesh_preprocess] Stage 1/3: computing convex hull")
    convex_hull = mesh.convex_hull
    if verbose:
        print("[mesh_preprocess] Stage 2/3: sampling + closest-point Chamfer")
    cd = surface_chamfer_distance(
        mesh,
        convex_hull,
        n_samples=n_samples,
        chunk_size=chunk_size,
        show_progress=show_progress,
    )
    return cd < float(threshold), convex_hull, float(cd)


def taubin_smooth(mesh: trimesh.Trimesh,
                  lamb: float = 0.5,
                  nu: float = 0.53,
                  iterations: int = 10) -> trimesh.Trimesh:
    """Edge-preserving Taubin smoothing.

    Taubin smoothing alternates a positive (shrinking) Laplacian step with
    a negative (inflating) step so that overall volume is roughly preserved
    while high-frequency noise is removed.

    Args:
        mesh: Input triangle mesh (modified **in-place** and returned).
        lamb: Positive smoothing factor (shrink step).
        nu: Negative smoothing factor magnitude (inflate step uses -nu).
        iterations: Number of shrink-inflate cycles.

    Returns:
        The smoothed mesh (same object, modified in-place).
    """
    try:
        trimesh.smoothing.filter_taubin(
            mesh, lamb=lamb, nu=nu, iterations=iterations
        )
    except (AttributeError, TypeError):
        # Fallback: plain Laplacian smoothing if Taubin is unavailable
        trimesh.smoothing.filter_laplacian(
            mesh, lamb=lamb, iterations=iterations
        )
    return mesh


def preprocess_mesh(mesh: trimesh.Trimesh,
                    convexity_threshold: float = 0.02,
                    smooth_iterations: int = 10,
                    convexity_samples: int = 2000,
                    closest_point_chunk_size: int = 2048,
                    show_progress: bool = False,
                    verbose: bool = True) -> trimesh.Trimesh:
    """Pre-process a mesh: replace with convex hull if near-convex, else smooth.

    Args:
        mesh: Loaded triangle mesh.
        convexity_threshold: Absolute Chamfer distance threshold used to decide near-convexity.
        smooth_iterations: Number of Taubin smoothing iterations applied
            to non-convex meshes.
        verbose: Print diagnostics.

    Returns:
        Processed mesh (either the convex hull or smoothed original).
    """
    try:
        is_convex, convex_hull, cd = is_near_convex(
            mesh,
            threshold=convexity_threshold,
            n_samples=convexity_samples,
            chunk_size=closest_point_chunk_size,
            show_progress=show_progress,
            verbose=verbose,
        )
    except Exception as exc:
        if verbose:
            print(f"[mesh_preprocess] Convexity check failed ({exc}) -> fallback to smoothing")
        return taubin_smooth(mesh, iterations=smooth_iterations)

    if verbose:
        print(f"[mesh_preprocess] Chamfer distance to convex hull: {cd:.6f} (threshold={convexity_threshold})")

    if is_convex:
        if verbose:
            print("[mesh_preprocess] Mesh is near-convex → replacing with convex hull "
                  f"({len(mesh.vertices)} → {len(convex_hull.vertices)} verts)")
        return convex_hull
    else:
        if verbose:
            print("[mesh_preprocess] Stage 3/3: smoothing")
            print(f"[mesh_preprocess] Mesh is non-convex → applying Taubin smoothing "
                  f"({smooth_iterations} iterations)")
        return taubin_smooth(mesh, iterations=smooth_iterations)
