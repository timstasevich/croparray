import numpy as np
import xarray as xr
from scipy import ndimage as ndi
from skimage.measure import label, regionprops

__all__ = ["best_z_proj", "measure_signal", "measure_signal_raw", "mask_props", "mask_skeleton_length", "auto_bad"]


def best_z_proj(
    ca,
    ref_ch: int | None = 0,
    disk_r: int = 1,
    roll_n: int = 1,
    use_zc: bool = False,
):
    """
    Return a best-z projection of crop intensities.

    Output is constructed from `ca.int` by:
      1) applying a centered rolling-z max projection of length `roll_n` (min_periods=1)
      2) selecting a single z index per crop.

    Two modes are supported:

    Mode A (default): use_zc=False
      - Compute per-crop best z by maximizing mean intensity within an xy disk of radius `disk_r`.
      - Stores the result as:
          * ca['z_pos_best'] : local z index into the stored `z` dimension (dims like fov,n,t[,ch])
          * ca['zc_best_pix'] : global best z pixel index (if ca['zc_pix'] exists)
          * ca['zc_best'] : global best z coordinate in z-index units (if ca['zc'] exists)
      - Does NOT overwrite ca['zc'] (which is treated as a global float coordinate).

    Mode B: use_zc=True
      - Do not compute best z.
      - Use an existing selector to pick z after rolling-max:
          * prefers ca['z_pos'] if present (local index into stored `z`)
          * otherwise falls back to ca['zc'] (legacy only; must already be local)
      - If selector == -1, returns all zeros for those crops.

    Parameters
    ----------
    ca : Dataset-like
        Must contain `ca.int` with a `z` dimension and coords `x`, `y`.
    ref_ch : int or None, default=0
        Used only when use_zc=False:
          - int: compute z_pos_best from that channel and apply to all channels
          - None: compute independently per channel (produces z_pos_best with ch dim)
    disk_r : int, default=1
        Radius in pixels of xy disk for computing best-z signal.
    roll_n : int, default=1
        Rolling window along z for rolling max projection.
    use_zc : bool, default=False
        If True, use existing z selector (z_pos preferred; else zc).

    Returns
    -------
    xarray.DataArray
        Best-z image with dims like (fov,n,t,y,x,ch).
    """
    res = ca.dx  # used to convert pixel radius to coordinate units (matches your existing convention)
        
    def _rolled_ch(ch_idx):
        out = ca.int.sel(ch=ch_idx).rolling(z=roll_n, center=True, min_periods=1).max()
        return out.expand_dims(ch=[ch_idx])    
    
    def _isel_z_or_zero(da_z, z_index_da):
        """
        Vectorized selection along z with:
        - invalid (z_index == -1 or out of range) -> zeros
        """
        z_len = da_z.sizes["z"]

        z_index = xr.apply_ufunc(
            lambda a: np.asarray(a).astype(np.int64),
            z_index_da,
            dask="allowed",
            output_dtypes=[np.int64],
        )

        # If the indexer carries a scalar/channel coordinate, drop it so it
        # doesn't conflict with da_z, which now has a real length-1 ch dim.
        if "ch" in z_index.coords and "ch" not in z_index.dims:
            z_index = z_index.drop_vars("ch")

        valid = (z_index >= 0) & (z_index < z_len)
        z_safe = z_index.where(valid, 0)

        out = da_z.isel(z=z_safe)
        out = out.where(valid, 0)
        return out

    def _get_existing_selector():
        # New datasets: use local selector
        if "z_pos" in ca:
            return ca["z_pos"]
        # Legacy fallback: allow old datasets that used ca['zc'] as local index
        if "zc" in ca:
            return ca["zc"]
        raise ValueError(
            "use_zc=True requires an existing z selector: ca['z_pos'] (preferred) or ca['zc'] (legacy)."
        )

    def _maybe_write_global_best(z_pos_best):
        """
        If global z center exists, map local z_pos_best -> global best z pixel index.
        Works for both full-z and slab-z datasets if ca['zc_pix'] exists.
        """
        if "zc_pix" not in ca:
            return

        # Determine z offset of stored z coordinate.
        # If your stored z coordinate is slab-relative centered at 0, then:
        #   z_center_local = index where z == 0  (often z_pad)
        # If it's full z starting at 0, z==0 is also at index 0.
        if "z" in ca.coords:
            z_vals = ca["z"].values
            # find index closest to 0 in z coordinate units
            z0_idx = int(np.argmin(np.abs(z_vals)))
        else:
            z0_idx = 0

        # local -> global: zc_best_pix = zc_pix + (z_pos_best - z0_idx)
        # honor invalids
        valid = z_pos_best >= 0
        zc_best_pix = ca["zc_pix"] + (z_pos_best - z0_idx)
        zc_best_pix = zc_best_pix.where(valid, -1).astype(np.int16)

        ca["zc_best_pix"] = zc_best_pix
        ca["zc_best_pix"].attrs["units"] = "index"
        ca["zc_best_pix"].attrs["long_name"] = "best z (global pixel index; -1 invalid)"

        # If you also want a float global z coordinate in z-index units:
        if "zc" in ca:
            # `zc` is float global z center; add same offset
            ca["zc_best"] = (ca["zc"] + (z_pos_best - z0_idx)).where(valid, np.nan)
            ca["zc_best"].attrs["units"] = "z-index"
            ca["zc_best"].attrs["long_name"] = "best z (global movie z-index units; float)"

    # -----------------------
    # Mode B: use existing selector (z_pos preferred)
    # -----------------------
    if use_zc:
        sel_da = _get_existing_selector()

        out_per_ch = []
        for ch_idx in ca.ch.values:
            z_index = sel_da.sel(ch=ch_idx) if "ch" in sel_da.dims else sel_da
            out_per_ch.append(_isel_z_or_zero(_rolled_ch(ch_idx), z_index))

        return xr.concat(out_per_ch, dim="ch")

    # -----------------------------------
    # Mode A: compute best-z (store as z_pos_best; do NOT overwrite global zc)
    # -----------------------------------
    if ref_ch is None:
        # compute independently per channel
        z_pos_best_list = []
        out_list = []
        for ch_idx in ca.ch.values:
            z_sig = (
                ca.sel(ch=ch_idx)
                .int.where(lambda a: a.x**2 + a.y**2 <= (disk_r * res) ** 2)
                .mean(dim=["x", "y"])
                .rolling(z=roll_n, center=True, min_periods=1)
                .max()
            )
#            z_pos_best = z_sig.argmax(dim="z").astype(np.int16)
            valid = z_sig.notnull().any(dim="z")
            z_pos_best = z_sig.fillna(-np.inf).argmax(dim="z").astype(np.int16)
            z_pos_best = z_pos_best.where(valid, -1).astype(np.int16)
            z_pos_best_list.append(z_pos_best)
            out_list.append(_isel_z_or_zero(_rolled_ch(ch_idx), z_pos_best))

        out = xr.concat(out_list, dim="ch")

        ca["z_pos_best"] = xr.concat(
            [zp.expand_dims(ch=[ch_idx]) for zp, ch_idx in zip(z_pos_best_list, ca.ch.values)],
            dim="ch",
        )

        ca["z_pos_best"].attrs["units"] = "index"
        ca["z_pos_best"].attrs["long_name"] = "best z (local index into stored z) per channel"

        _maybe_write_global_best(ca["z_pos_best"])
        return out

    else:
        # compute from reference channel and apply to all channels
        z_sig = (
            ca.sel(ch=ref_ch)
            .int.where(lambda a: a.x**2 + a.y**2 <= (disk_r * res) ** 2)
            .mean(dim=["x", "y"])
            .rolling(z=roll_n, center=True, min_periods=1)
            .max()
        )
#        z_pos_best = z_sig.argmax(dim="z").astype(np.int16)
        valid = z_sig.notnull().any(dim="z")
        z_pos_best = z_sig.fillna(-np.inf).argmax(dim="z").astype(np.int16)
        z_pos_best = z_pos_best.where(valid, -1).astype(np.int16)

        out = xr.concat(
            [_isel_z_or_zero(_rolled_ch(ch_idx), z_pos_best) for ch_idx in ca.ch.values],
            dim="ch",
        )
        ca["z_pos_best"] = z_pos_best
        ca["z_pos_best"].attrs["units"] = "index"
        ca["z_pos_best"].attrs["long_name"] = "best z (local index into stored z)"

        _maybe_write_global_best(ca["z_pos_best"])
        return out

def measure_signal(
    ca,
    ref_ch=None,
    disk_r: int = 1,
    disk_bg=None,
    roll_n: int = 1,
    use_zc: bool = False,
    drop_int: bool = False,
    drop_best_z: bool = False,
):
    """
    Measure background-subtracted crop intensity signal using a best-z projection.

    This function calls `best_z_proj(...)` to obtain a per-crop best-z image, then computes:
      - signal = (mean intensity in an inner disk) - (median intensity in an outer 1-pixel ring)

    Parameters
    ----------
    ca : xarray Dataset-like (CropArray)
        Must contain `ca.int` with dimensions including `z`, and coordinates `x`, `y`.
        If `use_zc=True`, must already contain `ca['zc']`.

    ref_ch : int or None, default=None
        Passed through to `best_z_proj` when `use_zc=False`.
        - If int: compute best-z indices from this channel and apply to all channels.
        - If None: compute best-z indices separately per channel.
        If `use_zc=True`, this parameter is ignored by `best_z_proj`.

    disk_r : int, default=1
        Radius (in pixels) of the inner disk used to compute the signal.

    disk_bg : int, default=None
        Radius (in pixels) of the background ring (1 pixel thick). If None, defaults to `ca.xy_pad`.

    roll_n : int, default=1
        Rolling window length along z used inside `best_z_proj` (rolling-z max projection).

    use_zc : bool, default=False
        If True, `best_z_proj` will not recompute/overwrite `ca['zc']` and will instead use the
        existing `ca['zc']` to pick the z plane (after rolling-z max). Crops with `zc == -1` return
        all-zero best-z images, yielding zero signal after subtraction.

    Returns
    -------
    ca : same type as input
        The input crop array augmented with:
          - `ca['best_z']` : best-z image after background subtraction, dims (fov,n,t,y,x,ch)
          - `ca['signal']` : background-subtracted signal per crop, dims (fov,n,t,ch)
    """
    if disk_bg is None:
        disk_bg = ca.xy_pad

    best_z = best_z_proj(
        ca,
        ref_ch=ref_ch,
        disk_r=disk_r,
        roll_n=roll_n,
        use_zc=use_zc,
    )

    # Inner disk (signal)
    disk_sig = best_z.where(
        lambda a: a.x**2 + a.y**2 <= (disk_r * ca.dx) ** 2
    ).mean(dim=["x", "y"])

    # Outer 1-pixel ring (background)
    donut_sig = best_z.where(
        lambda a: (a.x**2 + a.y**2 >= (disk_bg * ca.dx) ** 2)
        & (a.x**2 + a.y**2 < ((disk_bg + 1) * ca.dx) ** 2)
    ).median(dim=["x", "y"])

    signal = disk_sig - donut_sig

    ca["best_z"] = best_z - donut_sig
    ca["best_z"].attrs["units"] = "intensity (a.u.)"
    ca["best_z"].attrs["long_name"] = "best-z (rolling-z max) image, background-subtracted"

    ca["signal"] = signal
    ca["signal"].attrs["units"] = ca.attrs.get("signal_units", "")
    ca["signal"].attrs["long_name"] = "crop signal (disk mean - ring median)"

    if drop_int and "int" in ca:
        ca = ca.drop_vars("int")
    if drop_best_z and "best_z" in ca:
        ca = ca.drop_vars("best_z")

    return ca

def measure_signal_raw(
    ca,
    ref_ch=None,
    disk_r: int = 1,
    roll_n: int = 1,
    use_zc: bool = False,
    drop_int: bool = False,
    drop_best_z_raw: bool = False
):
    """
    Measure raw (non-background-subtracted) crop intensity signal using a best-z projection.

    This function calls `best_z_proj(...)` to obtain a per-crop best-z image, then computes a raw
    signal as the sum of intensities within an inner disk.

    Parameters
    ----------
    ca : xarray Dataset-like (CropArray)
        Must contain `ca.int` with dimensions including `z`, and coordinates `x`, `y`.
        If `use_zc=True`, must already contain `ca['zc']`.

    ref_ch : int or None, default=None
        Passed through to `best_z_proj` when `use_zc=False` (see its docs).
        If `use_zc=True`, this parameter is ignored by `best_z_proj`.

    disk_r : int, default=1
        Radius (in pixels) of the disk used to compute the signal.

    roll_n : int, default=1
        Rolling window length along z used inside `best_z_proj` (rolling-z max projection).

    use_zc : bool, default=False
        If True, `best_z_proj` will not recompute/overwrite `ca['zc']` and will instead use the
        existing `ca['zc']` to pick the z plane (after rolling-z max). Crops with `zc == -1` return
        all-zero best-z images, yielding zero raw signal.

    Returns
    -------
    ca : same type as input
        The input crop array augmented with:
          - `ca['best_z_raw']` : best-z image (no background subtraction), dims (fov,n,t,y,x,ch)
          - `ca['signal_raw']` : raw signal per crop, dims (fov,n,t,ch)
    """
    best_z = best_z_proj(
        ca,
        ref_ch=ref_ch,
        disk_r=disk_r,
        roll_n=roll_n,
        use_zc=use_zc,
    )

    disk_sig = best_z.where(
        lambda a: a.x**2 + a.y**2 <= (disk_r * ca.dx) ** 2
    ).sum(dim=["x", "y"])

    ca["best_z_raw"] = best_z
    ca["best_z_raw"].attrs["units"] = "intensity (a.u.)"
    ca["best_z_raw"].attrs["long_name"] = "best-z (rolling-z max) image"

    ca["signal_raw"] = disk_sig
    ca["signal_raw"].attrs["units"] = "intensity (a.u.)"
    ca["signal_raw"].attrs["long_name"] = "raw crop signal (disk sum)"

    if drop_int and "int" in ca:
        ca = ca.drop_vars("int")
    if drop_best_z_raw and "best_z" in ca:
        ca = ca.drop_vars("best_z")

    return ca

def mask_props(
    ca,
    *,
    source: str,
    out_prefix: str | None = None,
    props: tuple[str, ...] = ("area_px", "eccentricity", "solidity", "perimeter_px", "centroid_y_px", "centroid_x_px"),
    connectivity: int = 2,
    empty_value: float = np.nan,
):
    """
    Measure morphology from a binary mask layer across the entire crop array and add scalar
    measurement layers back onto `ca`.

    Outputs are named: f"{out_prefix}__{prop}" and have dims equal to the non-(y,x) dims of `source`.
    """        
    if source not in ca:
        raise KeyError(f"source='{source}' not found. Available: {list(ca.data_vars)}")

    out_prefix = out_prefix or source

    da = ca[source]
    if da.ndim < 2:
        raise ValueError(f"Mask layer '{source}' must be at least 2D (y,x). Got dims={da.dims}")

    # Convention: last two dims are spatial
    ydim, xdim = da.dims[-2], da.dims[-1]
    lead_dims = da.dims[:-2]
    lead_shape = da.shape[:-2]
    H, W = da.shape[-2], da.shape[-1]

    arr = np.asarray(da.data).astype(bool).reshape((-1, H, W))
    N = arr.shape[0]

    want = set(props)
    out = {}

    # Area (vectorized)
    if "area_px" in want:
        out["area_px"] = arr.reshape(N, -1).sum(axis=1).astype(float)

    # Centroid (cheap loop)
    if ("centroid_y_px" in want) or ("centroid_x_px" in want):
        cy = np.full(N, empty_value, dtype=float)
        cx = np.full(N, empty_value, dtype=float)
        for i in range(N):
            m = arr[i]
            if m.any():
                yy, xx = ndi.center_of_mass(m.astype(np.uint8))
                cy[i], cx[i] = float(yy), float(xx)
        if "centroid_y_px" in want:
            out["centroid_y_px"] = cy
        if "centroid_x_px" in want:
            out["centroid_x_px"] = cx

    # Pixel-perimeter proxy (fast, robust)
    if "perimeter_px" in want:
        per = np.full(N, empty_value, dtype=float)
        for i in range(N):
            m = arr[i]
            if m.any():
                er = ndi.binary_erosion(m, structure=np.ones((3, 3), dtype=bool))
                boundary = m & (~er)
                per[i] = float(boundary.sum())
        out["perimeter_px"] = per

    # Regionprops-derived (compute only if requested; slower)
    rp_map = {
        "eccentricity": "eccentricity",
        "solidity": "solidity",
        "extent": "extent",
        "major_axis_length_px": "major_axis_length",
        "minor_axis_length_px": "minor_axis_length",
        "orientation_rad": "orientation",
    }
    rp_needed = [(k, v) for k, v in rp_map.items() if k in want]
    if rp_needed:
        for k, _ in rp_needed:
            out[k] = np.full(N, empty_value, dtype=float)

        for i in range(N):
            m = arr[i]
            if not m.any():
                continue
            lab = label(m, connectivity=connectivity)
            regs = regionprops(lab)
            if not regs:
                continue
            r = max(regs, key=lambda rr: rr.area)  # largest component
            for kout, kin in rp_needed:
                out[kout][i] = float(getattr(r, kin))

    # Attach outputs back to ca
    for k, vec in out.items():
        name = f"{out_prefix}__{k}"
        ca[name] = xr.DataArray(vec.reshape(lead_shape), dims=lead_dims)
        ca[name].attrs["source_layer"] = source
        ca[name].attrs["long_name"] = f"{k} from {source}"

    return ca

def mask_skeleton_length(
    ca,
    *,
    source: str,
    out_prefix: str | None = None,
    method: str = "longest_path",  # "longest_path" (head-to-tail) or "total"
    connectivity: int = 2,         # 1=4-neigh, 2=8-neigh
    empty_value: float = np.nan,
):
    """
    Compute a skeleton-based length (in pixels) from a binary mask layer across the entire
    crop array and add a scalar measurement layer back onto `ca`.

    This is designed for "comet-like" objects where you want a robust head-to-tail length.

    Parameters
    ----------
    ca : CropArray-like
        Object containing xarray DataArrays in `ca[data_var_name]` and supporting assignment
        `ca[new_name] = xr.DataArray(...)`.
    source : str
        Name of the binary mask layer. Must have at least 2D (y,x) as its last two dims.
        Nonzero values are treated as True.
    out_prefix : str | None
        Prefix for output layer name(s). If None, uses `source`.
    method : {"longest_path","total"}
        - "longest_path": longest geodesic path along the skeleton graph (recommended for head-to-tail).
        - "total": total skeleton length (sum of unique skeleton edges).
    connectivity : {1,2}
        1 uses 4-neighborhood (up/down/left/right). 2 uses 8-neighborhood (also diagonals).
    empty_value : float
        Value used when the mask is empty (no True pixels), or skeletonization yields no nodes.

    Outputs
    -------
    Adds one scalar layer to `ca`:
      - f"{out_prefix}__skeleton_longest_path_px" if method=="longest_path"
      - f"{out_prefix}__skeleton_total_length_px" if method=="total"

    Notes
    -----
    - "longest_path" is generally the best proxy for head-to-tail length for curved tails.
    - If the skeleton has no endpoints (e.g., a loop), "longest_path" falls back to a
      two-pass Dijkstra diameter estimate on the skeleton graph.
    """
    import heapq
    import numpy as np
    import xarray as xr
    from scipy import ndimage as ndi
    from skimage.morphology import skeletonize

    if source not in ca:
        raise KeyError(f"source='{source}' not found. Available: {list(ca.data_vars)}")

    if method not in ("longest_path", "total"):
        raise ValueError("method must be 'longest_path' or 'total'")

    if connectivity not in (1, 2):
        raise ValueError("connectivity must be 1 (4-neigh) or 2 (8-neigh)")

    out_prefix = out_prefix or source

    da = ca[source]
    if da.ndim < 2:
        raise ValueError(f"Mask layer '{source}' must be at least 2D (y,x). Got dims={da.dims}")

    # Convention: last two dims are spatial
    ydim, xdim = da.dims[-2], da.dims[-1]
    lead_dims = da.dims[:-2]
    lead_shape = da.shape[:-2]
    H, W = da.shape[-2], da.shape[-1]

    arr = np.asarray(da.data).astype(bool).reshape((-1, H, W))
    N = arr.shape[0]

    # Neighbor offsets
    if connectivity == 1:
        nbrs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    else:
        nbrs = [(-1, 0), (1, 0), (0, -1), (0, 1),
                (-1, -1), (-1, 1), (1, -1), (1, 1)]

    def _edge_weight(dy: int, dx: int) -> float:
        return 1.0 if (dy == 0 or dx == 0) else float(np.sqrt(2.0))

    def _build_graph(skel: np.ndarray):
        """Return (coords, index_map, adj) for skeleton pixels."""
        coords = np.column_stack(np.nonzero(skel))  # (M,2) y,x
        M = coords.shape[0]
        if M == 0:
            return coords, None, None

        idx_map = -np.ones((H, W), dtype=int)
        for i, (yy, xx) in enumerate(coords):
            idx_map[yy, xx] = i

        adj = [[] for _ in range(M)]
        for i, (yy, xx) in enumerate(coords):
            for dy, dx in nbrs:
                y2, x2 = yy + dy, xx + dx
                if 0 <= y2 < H and 0 <= x2 < W:
                    j = idx_map[y2, x2]
                    if j >= 0:
                        w = _edge_weight(dy, dx)
                        adj[i].append((j, w))
        return coords, idx_map, adj

    def _dijkstra(adj, start: int):
        """Return dist array from start over weighted graph."""
        M = len(adj)
        dist = np.full(M, np.inf, dtype=float)
        dist[start] = 0.0
        pq = [(0.0, start)]
        while pq:
            d, u = heapq.heappop(pq)
            if d != dist[u]:
                continue
            for v, w in adj[u]:
                nd = d + w
                if nd < dist[v]:
                    dist[v] = nd
                    heapq.heappush(pq, (nd, v))
        return dist

    def _total_length(adj):
        """Sum unique edges once (i<j)."""
        total = 0.0
        seen = set()
        for i, nbr_list in enumerate(adj):
            for j, w in nbr_list:
                if i == j:
                    continue
                a, b = (i, j) if i < j else (j, i)
                if (a, b) not in seen:
                    seen.add((a, b))
                    total += w
        return total

    out = np.full(N, empty_value, dtype=float)

    for i in range(N):
        m = arr[i]
        if not m.any():
            continue

        skel = skeletonize(m)
        coords, idx_map, adj = _build_graph(skel)
        if adj is None:
            continue

        if method == "total":
            out[i] = float(_total_length(adj))
            continue

        # method == "longest_path"
        # Compute degrees and endpoints
        deg = np.array([len(nbrs_i) for nbrs_i in adj], dtype=int)
        endpoints = np.where(deg == 1)[0]

        if endpoints.size >= 2:
            # Exact longest shortest-path over endpoints (small graphs; safe to brute force)
            best = 0.0
            for s in endpoints:
                dist = _dijkstra(adj, int(s))
                # Consider only endpoints
                dmax = np.max(dist[endpoints])
                if np.isfinite(dmax) and dmax > best:
                    best = float(dmax)
            out[i] = best
        else:
            # No endpoints (loop) or single endpoint (degenerate):
            # approximate diameter with 2-pass Dijkstra (exact for trees; good fallback here)
            dist0 = _dijkstra(adj, 0)
            a = int(np.argmax(dist0))
            dista = _dijkstra(adj, a)
            diam = float(np.max(dista))
            out[i] = diam

    # Write back to ca
    if method == "total":
        out_name = f"{out_prefix}__skeleton_total_length_px"
    else:
        out_name = f"{out_prefix}__skeleton_longest_path_px"

    da_out = xr.DataArray(
        out.reshape(lead_shape),
        dims=lead_dims,
        coords={d: da.coords[d] for d in lead_dims},
        name=out_name,
    )
    ca[out_name] = da_out
    return ca


def _auto_bad_sidecar_path(ds, *, filter_name: str, output_dir: str | None = None) -> str | None:
    """
    Determine sidecar path for auto_bad, following the same __manual__ naming convention
    as save_manual_filter_sidecar() in napari_view.py.

    Returns None if no path information is available and output_dir is not given.
    """
    import os

    fn_attr = ds.attrs.get("filename", None) if hasattr(ds, "attrs") else None
    if isinstance(fn_attr, str) and fn_attr.strip():
        base = os.path.basename(fn_attr.strip())
        stem = base[:-3] if base.endswith(".nc") else base
        src = ds.encoding.get("source", None)
        default_dir = os.path.dirname(src) if isinstance(src, str) and src.strip() else os.getcwd()
    else:
        src = ds.encoding.get("source", None)
        if isinstance(src, str) and src.strip():
            base = os.path.basename(src)
            stem = base[:-3] if base.endswith(".nc") else base
            default_dir = os.path.dirname(src)
        else:
            if output_dir is None:
                return None
            stem = "dataset"
            default_dir = output_dir

    out_dir = output_dir if output_dir is not None else default_dir
    return os.path.join(out_dir, f"{stem}__manual__{filter_name}.nc")


def auto_bad(
    ca,
    *,
    source: str,
    ch: int | None = None,
    area_factor: float | None = 2.0,
    max_dist_px: float | None = None,
    exclude_empty: bool = True,
    out_name: str = "auto_bad",
    save_sidecar: bool = True,
    output_dir: str | None = None,
):
    """
    Create an automatic 'bad' filter layer based on mask area and centroid offset.

    A crop is marked as bad (output=0) if any of the following apply:
      - mask area is 0 and exclude_empty=True
      - mask area > area_factor * median(nonzero areas), if area_factor is not None
      - mask centroid distance from crop center > max_dist_px pixels, if max_dist_px is not None

    The output layer (out_name) has value 1 (good/keep) or 0 (bad/reject), matching the
    convention used by manual_filter_montage(). When save_sidecar=True it is written as a
    sidecar .nc file alongside the croparray (using the same {stem}__manual__{filter_name}.nc
    naming), so it is automatically reloaded on next open and can be edited interactively
    with ca.napari.manual_filter_montage().

    Parameters
    ----------
    ca : CropArray-like
    source : str
        Name of the binary mask layer. Last two dims must be (y, x).
    ch : int | None
        If the mask has a 'ch' dimension, select this channel. If None and 'ch' is
        present, defaults to the first channel value.
    area_factor : float | None, default 2.0
        Flag if area > area_factor * median(nonzero areas). None = skip area criterion.
    max_dist_px : float | None, default None
        Flag if centroid distance from crop center > max_dist_px pixels. None = skip.
    exclude_empty : bool, default True
        Also flag crops with an empty mask (area == 0) as bad.
    out_name : str, default "auto_bad"
        Name of the output layer added to ca.
    save_sidecar : bool, default True
        If True, save the filter as a sidecar .nc file alongside the croparray.
    output_dir : str | None
        Directory for the sidecar file. If None, uses the directory of the source .nc file.

    Returns
    -------
    ca : updated with ca[out_name] (uint8, 1=good / 0=bad), dims matching the lead dims
        of the source mask (everything except y, x, and ch if a channel was selected).
    """
    import os
    import warnings

    if source not in ca:
        raise KeyError(f"source='{source}' not found. Available: {list(ca.data_vars)}")

    da = ca[source]

    # Handle ch dimension
    if "ch" in da.dims:
        ch_vals = (
            da.coords["ch"].values if "ch" in da.coords
            else list(range(da.sizes["ch"]))
        )
        if ch is None:
            ch = int(ch_vals[0])
        da = da.sel(ch=ch)

    if da.ndim < 2:
        raise ValueError(
            f"After ch selection, mask must be at least 2D (y,x). Got dims={da.dims}"
        )

    # Last two dims are spatial (y, x)
    lead_dims = da.dims[:-2]
    lead_shape = da.shape[:-2]
    H, W = da.shape[-2], da.shape[-1]
    cy_center = H / 2.0
    cx_center = W / 2.0

    arr = np.asarray(da.data).astype(bool).reshape((-1, H, W))
    N = arr.shape[0]

    area = arr.reshape(N, -1).sum(axis=1).astype(float)

    # Compute centroid distances only when needed
    dist = np.full(N, np.nan)
    if max_dist_px is not None:
        for i in range(N):
            m = arr[i]
            if m.any():
                cy, cx = ndi.center_of_mass(m.astype(np.uint8))
                dist[i] = np.sqrt((cy - cy_center) ** 2 + (cx - cx_center) ** 2)

    # Build bad mask (False = good, True = bad)
    bad = np.zeros(N, dtype=bool)

    if exclude_empty:
        bad |= area == 0

    if area_factor is not None:
        nonzero_areas = area[area > 0]
        if nonzero_areas.size > 0:
            median_area = float(np.median(nonzero_areas))
            bad |= area > float(area_factor) * median_area

    if max_dist_px is not None:
        bad |= np.where(np.isfinite(dist), dist > float(max_dist_px), False)

    # Output: 1=good, 0=bad (matching manual filter convention)
    out = (~bad).astype(np.uint8)

    coords = {d: da.coords[d] for d in lead_dims if d in da.coords}
    da_out = xr.DataArray(
        out.reshape(lead_shape),
        dims=lead_dims,
        coords=coords,
        name=out_name,
    )
    da_out.attrs["source_layer"] = source
    da_out.attrs["long_name"] = f"auto bad filter (1=good, 0=bad) from '{source}'"
    if area_factor is not None:
        da_out.attrs["area_factor"] = float(area_factor)
    if max_dist_px is not None:
        da_out.attrs["max_dist_px"] = float(max_dist_px)
    da_out.attrs["exclude_empty"] = int(exclude_empty)

    ca[out_name] = da_out

    if save_sidecar:
        ds = ca.ds if hasattr(ca, "ds") else ca
        sidecar_path = _auto_bad_sidecar_path(ds, filter_name=out_name, output_dir=output_dir)
        if sidecar_path is None:
            warnings.warn(
                f"Could not determine croparray file path; sidecar for '{out_name}' was NOT saved. "
                "Pass output_dir='...' to specify a save location, or open the croparray from a "
                ".nc file so the path is known.",
                UserWarning,
                stacklevel=2,
            )
        else:
            out_ds = xr.Dataset({out_name: da_out})
            src = ds.encoding.get("source", None)
            if isinstance(src, str):
                out_ds.attrs["croparray_source"] = src
            out_ds.attrs["manual_filter_name"] = out_name
            tmp = sidecar_path + ".tmp"
            out_ds.to_netcdf(tmp, mode="w")
            os.replace(tmp, sidecar_path)
            print(f"[auto_bad] Saved sidecar: {sidecar_path}")

    return ca
