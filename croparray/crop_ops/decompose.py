"""
crop_ops.decompose

Decompose clustered spots in CropArray crops using BigFish's gaussian
mixture fitting. Each crop is treated as a dense region and decomposed
into individual molecules using a user-provided reference spot (PSF).

Supports both 2D (y, x) and 3D (z, y, x) decomposition.
"""

from __future__ import annotations

import warnings
from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd
import xarray as xr

__all__ = ["decompose_crops"]


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _fit_gaussian_mixture_2d(
    image: np.ndarray,
    voxel_size_yx: float,
    sigma_yx: float,
    amplitude: float,
    background: float,
    precomputed_gaussian,
    limit_gaussian: int = 1000,
) -> list[list[float]]:
    """
    Fit as many 2D gaussians as possible in a crop image.

    Adapted from BigFish's ``_gaussian_mixture_2d`` but operates on the
    entire crop image directly (no region bounding-box extraction).

    Returns list of [y_nm, x_nm] positions for each fitted gaussian.
    """
    from bigfish.detection.spot_modeling import gaussian_2d, _initialize_grid_2d

    image_flat = image.ravel().astype(np.float64)
    grid = _initialize_grid_2d(image, voxel_size_yx)

    simulation = np.zeros_like(image_flat)
    residual = image_flat - simulation
    ssr = np.sum(residual ** 2)
    diff_ssr = -1
    nb_gaussian = 0
    positions = []

    while diff_ssr < 0 or nb_gaussian == limit_gaussian:
        pos_idx = np.argmax(residual)
        positions.append(list(grid[:, pos_idx]))
        simulation += gaussian_2d(
            grid=grid,
            mu_y=float(positions[-1][0]),
            mu_x=float(positions[-1][1]),
            sigma_yx=sigma_yx,
            voxel_size_yx=voxel_size_yx,
            amplitude=amplitude,
            background=background,
            precomputed=precomputed_gaussian,
        )
        residual = image_flat - simulation
        new_ssr = np.sum(residual ** 2)
        diff_ssr = new_ssr - ssr
        ssr = new_ssr
        nb_gaussian += 1
        background = 0  # only first gaussian gets background

        # no improvement → stop
        if diff_ssr >= 0 and nb_gaussian > 1:
            break

    # remove last gaussian if it didn't improve the fit
    if nb_gaussian > 1 and diff_ssr >= 0:
        positions.pop(-1)

    if nb_gaussian >= limit_gaussian:
        warnings.warn(
            f"Decomposition reached the limit of {limit_gaussian} gaussians. "
            "Consider increasing limit_gaussian or checking for artifacts.",
            UserWarning,
        )

    return positions


def _fit_gaussian_mixture_3d(
    image: np.ndarray,
    voxel_size_z: float,
    voxel_size_yx: float,
    sigma_z: float,
    sigma_yx: float,
    amplitude: float,
    background: float,
    precomputed_gaussian,
    limit_gaussian: int = 1000,
) -> list[list[float]]:
    """
    Fit as many 3D gaussians as possible in a crop image.

    Adapted from BigFish's ``_gaussian_mixture_3d`` but operates on the
    entire crop volume directly.

    Returns list of [z_nm, y_nm, x_nm] positions for each fitted gaussian.
    """
    from bigfish.detection.spot_modeling import gaussian_3d, _initialize_grid_3d

    image_flat = image.ravel().astype(np.float64)
    grid = _initialize_grid_3d(image, voxel_size_z, voxel_size_yx)

    simulation = np.zeros_like(image_flat)
    residual = image_flat - simulation
    ssr = np.sum(residual ** 2)
    diff_ssr = -1
    nb_gaussian = 0
    positions = []

    while diff_ssr < 0 or nb_gaussian == limit_gaussian:
        pos_idx = np.argmax(residual)
        positions.append(list(grid[:, pos_idx]))
        simulation += gaussian_3d(
            grid=grid,
            mu_z=float(positions[-1][0]),
            mu_y=float(positions[-1][1]),
            mu_x=float(positions[-1][2]),
            sigma_z=sigma_z,
            sigma_yx=sigma_yx,
            voxel_size_z=voxel_size_z,
            voxel_size_yx=voxel_size_yx,
            amplitude=amplitude,
            background=background,
            precomputed=precomputed_gaussian,
        )
        residual = image_flat - simulation
        new_ssr = np.sum(residual ** 2)
        diff_ssr = new_ssr - ssr
        ssr = new_ssr
        nb_gaussian += 1
        background = 0

        if diff_ssr >= 0 and nb_gaussian > 1:
            break

    if nb_gaussian > 1 and diff_ssr >= 0:
        positions.pop(-1)

    if nb_gaussian >= limit_gaussian:
        warnings.warn(
            f"Decomposition reached the limit of {limit_gaussian} gaussians. "
            "Consider increasing limit_gaussian or checking for artifacts.",
            UserWarning,
        )

    return positions


def _estimate_background(image: np.ndarray) -> float:
    """
    Estimate background from the edge pixels of a crop image.

    For 2D: uses the 1-pixel border.
    For 3D: uses the 1-pixel border of each z-slice.
    """
    if image.ndim == 2:
        edge = np.concatenate([
            image[0, :], image[-1, :],
            image[1:-1, 0], image[1:-1, -1],
        ])
    else:
        # 3D: collect edges from all z-slices
        edges = []
        for z in range(image.shape[0]):
            s = image[z]
            edges.extend([s[0, :], s[-1, :], s[1:-1, 0], s[1:-1, -1]])
        edge = np.concatenate(edges)
    return float(np.median(edge))


def _max_pairwise_distance(positions: np.ndarray) -> float:
    """
    Compute the maximum pairwise Euclidean distance between positions.

    Parameters
    ----------
    positions : np.ndarray, shape (N, ndim)
        Molecule positions in pixel coordinates.

    Returns
    -------
    float
        Max pairwise distance, or 0.0 if fewer than 2 positions.
    """
    if len(positions) < 2:
        return 0.0
    from scipy.spatial.distance import pdist
    return float(np.max(pdist(positions)))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def decompose_crops(
    ds: xr.Dataset,
    voxel_size: Union[Tuple[float, float], Tuple[float, float, float]],
    *,
    reference_spot: Optional[np.ndarray] = None,
    spot_radius: Optional[Union[Tuple[float, float], Tuple[float, float, float]]] = None,
    sigma_yx: Optional[float] = None,
    sigma_z: Optional[float] = None,
    amplitude: Optional[float] = None,
    background: Optional[Union[float, str]] = None,
    channel: int = 0,
    source: str = "int",
    z_select: Optional[Union[int, str]] = "best",
    denoise: Union[bool, int] = False,
    mask_source: Optional[str] = None,
    mask_dilate: int = 0,
    limit_gaussian: int = 1000,
    add_to_ds: bool = True,
    progress: bool = True,
) -> Tuple[xr.Dataset, pd.DataFrame]:
    """
    Decompose clustered spots in each crop using gaussian mixture fitting.

    Each crop image is treated as a dense region and decomposed into
    individual molecules by iteratively fitting gaussians.

    Gaussian parameters can be provided in two ways:

    1. **From a reference spot**: pass ``reference_spot`` and ``spot_radius``.
       The function will fit a gaussian to the reference spot to extract
       sigma, amplitude, and background automatically.
    2. **Directly**: pass ``sigma_yx``, ``amplitude``, and ``background``
       (and ``sigma_z`` for 3D). This skips the fitting step.

    Parameters
    ----------
    ds : xr.Dataset
        CropArray dataset containing intensity crops.
    voxel_size : tuple of float
        Voxel size in nanometers. (y_nm, x_nm) for 2D or (z_nm, y_nm, x_nm)
        for 3D. Length determines 2D vs 3D decomposition.
    reference_spot : np.ndarray, optional
        A 2D (y, x) or 3D (z, y, x) image of a diffraction-limited
        reference spot. Required if sigma_yx/amplitude/background are not
        provided.
    spot_radius : tuple of float, optional
        Spot radius in nanometers. Required when using ``reference_spot``.
    sigma_yx : float, optional
        Standard deviation of the gaussian in the yx plane, in nanometers.
    sigma_z : float, optional
        Standard deviation of the gaussian along z, in nanometers (3D only).
    amplitude : float, optional
        Amplitude of the gaussian.
    background : float, "auto", or None, optional
        Background value for the gaussian model. If ``"auto"``, the
        background is estimated per crop from the median of the edge
        pixels. If a float, that fixed value is used for all crops.
    channel : int, default 0
        Channel index to decompose.
    source : str, default "int"
        Name of the intensity variable in the dataset.
    z_select : int, "best", or None, default "best"
        How to select the z-slice for 2D decomposition. Ignored for 3D.
        - ``"best"``: use the best-z slice from ``ds['zc']`` per crop
          (requires ``best_z_proj`` to have been run).
        - ``int``: use this fixed z-index for all crops.
        - ``None``: use z=0.
    denoise : bool or int, default False
        Apply a median filter to each crop before decomposition to remove
        hot pixels and noise. If ``True``, uses a 3×3 median filter.
        If an int, uses that as the filter size (must be odd).
    mask_source : str, optional
        Name of a binary mask variable in the dataset (e.g. ``"ch0_mask"``).
        If provided, the crop image is multiplied by this mask before
        decomposition, restricting fitting to the masked region only.
    mask_dilate : int, default 0
        Number of pixels to dilate the mask before applying. Useful to
        add a small buffer around the mask boundary.
    limit_gaussian : int, default 1000
        Maximum number of gaussians to fit per crop.
    add_to_ds : bool, default True
        If True, add ``n_molecules`` and ``spot_length`` variables to ds.
    progress : bool, default True
        If True, show a progress bar.

    Returns
    -------
    ds : xr.Dataset
        Dataset, optionally augmented with ``n_molecules`` and
        ``spot_length`` variables (dims: fov, n, t).
    df_molecules : pd.DataFrame
        DataFrame with columns (fov, n, t, mol_idx, y_px, x_px) for 2D
        or (fov, n, t, mol_idx, z_px, y_px, x_px) for 3D. Contains the
        pixel-coordinate positions of all decomposed molecules.
    """
    from bigfish.detection.spot_modeling import precompute_erf

    # --- Determine dimensionality from voxel_size ---
    ndim = len(voxel_size)
    if ndim not in (2, 3):
        raise ValueError(f"voxel_size must have 2 or 3 elements, got {ndim}.")

    # --- Get gaussian parameters ---
    auto_bg = background == "auto"
    have_direct = sigma_yx is not None and amplitude is not None and (background is not None)
    have_refspot = reference_spot is not None

    if have_direct:
        # Use directly provided parameters
        if ndim == 3:
            if sigma_z is None:
                raise ValueError("sigma_z is required for 3D decomposition.")
            sigma = (sigma_z, sigma_yx, sigma_yx)
        else:
            sigma = (sigma_yx, sigma_yx)
    elif have_refspot:
        # Fit parameters from reference spot
        from bigfish.detection.spot_modeling import modelize_spot

        if spot_radius is None:
            raise ValueError(
                "spot_radius is required when using reference_spot."
            )
        if reference_spot.ndim != ndim:
            raise ValueError(
                f"reference_spot has {reference_spot.ndim}D but voxel_size "
                f"implies {ndim}D."
            )
        if len(spot_radius) != ndim:
            raise ValueError(
                f"spot_radius length ({len(spot_radius)}) must match "
                f"voxel_size length ({ndim})."
            )
        params = modelize_spot(
            reference_spot=reference_spot,
            voxel_size=voxel_size,
            spot_radius=spot_radius,
        )
        if ndim == 3:
            sigma_z, sigma_yx, amplitude, background = params
            sigma = (sigma_z, sigma_yx, sigma_yx)
        else:
            sigma_yx, amplitude, background = params
            sigma = (sigma_yx, sigma_yx)
    else:
        raise ValueError(
            "Provide either (sigma_yx, amplitude, background) directly, "
            "or (reference_spot, spot_radius) to fit them automatically."
        )

    # --- Precompute erf tables ---
    # max_grid based on crop spatial size
    if ndim == 3:
        max_grid = max(ds.sizes.get("z", 1), ds.sizes["y"], ds.sizes["x"]) + 1
    else:
        max_grid = max(ds.sizes["y"], ds.sizes["x"]) + 1

    precomputed = precompute_erf(
        ndim=ndim,
        voxel_size=voxel_size,
        sigma=sigma,
        max_grid=max_grid,
    )

    # --- Iterate over crops ---
    fov_vals = ds.coords["fov"].values
    n_vals = ds.coords["n"].values
    t_vals = ds.coords["t"].values

    ny = ds.sizes["y"]
    nx = ds.sizes["x"]

    n_molecules_arr = np.zeros(
        (len(fov_vals), len(n_vals), len(t_vals)), dtype=np.int32,
    )
    spot_length_arr = np.zeros(
        (len(fov_vals), len(n_vals), len(t_vals)), dtype=np.float64,
    )
    decompose_map = np.zeros(
        (len(fov_vals), len(n_vals), len(t_vals), ny, nx), dtype=np.bool_,
    )
    mol_records = []

    total = len(fov_vals) * len(n_vals) * len(t_vals)

    if progress:
        try:
            from tqdm.auto import tqdm
            iterator = tqdm(total=total, desc="decomposing crops")
        except ImportError:
            iterator = None
            progress = False
    else:
        iterator = None

    for fi, fov in enumerate(fov_vals):
        for ni, n in enumerate(n_vals):
            for ti, t in enumerate(t_vals):
                # Extract crop image
                crop_da = ds[source].sel(fov=fov, n=n, t=t, ch=channel)
                if ndim == 3:
                    img = crop_da.values  # (z, y, x)
                else:
                    if "z" in crop_da.dims:
                        if z_select == "best" and "zc" in ds:
                            zi = int(ds["zc"].sel(fov=fov, n=n, t=t).values)
                            img = crop_da.isel(z=zi).values
                        elif isinstance(z_select, int):
                            img = crop_da.isel(z=z_select).values
                        else:
                            img = crop_da.isel(z=0).values
                    else:
                        img = crop_da.values  # (y, x)

                img = img.astype(np.float64)

                # Denoise if requested
                if denoise:
                    from scipy.ndimage import median_filter
                    filt_size = denoise if isinstance(denoise, int) and not isinstance(denoise, bool) else 3
                    img = median_filter(img, size=filt_size)

                # Apply mask if provided
                if mask_source is not None and mask_source in ds:
                    from scipy.ndimage import binary_dilation
                    mask_da = ds[mask_source].sel(fov=fov, n=n, t=t)
                    if "ch" in mask_da.dims:
                        mask_da = mask_da.isel(ch=0)
                    mask_2d = mask_da.values > 0
                    if mask_dilate > 0:
                        mask_2d = binary_dilation(mask_2d, iterations=mask_dilate)
                    img = img * mask_2d

                # Skip empty crops
                if img.max() <= 0:
                    n_molecules_arr[fi, ni, ti] = 0
                    spot_length_arr[fi, ni, ti] = 0.0
                    if iterator is not None:
                        iterator.update(1)
                    continue

                # Determine background for this crop
                crop_bg = max(0.0, _estimate_background(img)) if auto_bg else background

                # Run decomposition
                if ndim == 3:
                    positions_nm = _fit_gaussian_mixture_3d(
                        image=img,
                        voxel_size_z=voxel_size[0],
                        voxel_size_yx=voxel_size[-1],
                        sigma_z=sigma[0],
                        sigma_yx=sigma[-1],
                        amplitude=amplitude,
                        background=crop_bg,
                        precomputed_gaussian=precomputed,
                        limit_gaussian=limit_gaussian,
                    )
                else:
                    positions_nm = _fit_gaussian_mixture_2d(
                        image=img,
                        voxel_size_yx=voxel_size[-1],
                        sigma_yx=sigma[-1],
                        amplitude=amplitude,
                        background=crop_bg,
                        precomputed_gaussian=precomputed,
                        limit_gaussian=limit_gaussian,
                    )

                positions_nm = np.array(positions_nm)

                if positions_nm.size == 0:
                    n_molecules_arr[fi, ni, ti] = 0
                    spot_length_arr[fi, ni, ti] = 0.0
                    if iterator is not None:
                        iterator.update(1)
                    continue

                # Convert nm positions to pixel coordinates
                positions_px = positions_nm.copy()
                for d in range(ndim):
                    positions_px[:, d] = positions_nm[:, d] / voxel_size[d]

                n_mols = len(positions_px)
                n_molecules_arr[fi, ni, ti] = n_mols
                spot_length_arr[fi, ni, ti] = _max_pairwise_distance(positions_px)

                # Build binary map (2D: y,x positions; 3D: use y,x only)
                if ndim == 3:
                    yx_px = positions_px[:, 1:]  # (N, 2) — y, x
                else:
                    yx_px = positions_px  # (N, 2) — y, x
                for mol_pos in yx_px:
                    # positions_px are in 0-based pixel indices
                    yi = int(np.clip(np.round(mol_pos[0]), 0, ny - 1))
                    xi = int(np.clip(np.round(mol_pos[1]), 0, nx - 1))
                    decompose_map[fi, ni, ti, yi, xi] = True

                # Store molecule positions
                for mi in range(n_mols):
                    if ndim == 3:
                        mol_records.append({
                            "fov": fov,
                            "n": n,
                            "t": t,
                            "mol_idx": mi,
                            "z_px": positions_px[mi, 0],
                            "y_px": positions_px[mi, 1],
                            "x_px": positions_px[mi, 2],
                        })
                    else:
                        mol_records.append({
                            "fov": fov,
                            "n": n,
                            "t": t,
                            "mol_idx": mi,
                            "y_px": positions_px[mi, 0],
                            "x_px": positions_px[mi, 1],
                        })

                if iterator is not None:
                    iterator.update(1)

    if iterator is not None:
        iterator.close()

    # --- Build outputs ---
    n_molecules_da = xr.DataArray(
        n_molecules_arr,
        coords=[fov_vals, n_vals, t_vals],
        dims=["fov", "n", "t"],
        attrs={"long_name": "number of decomposed molecules", "units": "count"},
    )
    spot_length_da = xr.DataArray(
        spot_length_arr,
        coords=[fov_vals, n_vals, t_vals],
        dims=["fov", "n", "t"],
        attrs={
            "long_name": "max pairwise distance between molecules",
            "units": "pixels",
        },
    )

    decompose_map_da = xr.DataArray(
        decompose_map,
        coords=[fov_vals, n_vals, t_vals, ds.coords["y"].values, ds.coords["x"].values],
        dims=["fov", "n", "t", "y", "x"],
        attrs={"long_name": "binary molecule map", "units": "bool"},
    )

    if add_to_ds:
        ds["n_molecules"] = n_molecules_da
        ds["spot_length"] = spot_length_da
        ds["decompose_map"] = decompose_map_da

    df_molecules = pd.DataFrame(mol_records)

    return ds, df_molecules
