import numpy as np
import xarray as xr
from scipy import ndimage as ndi
from skimage.measure import label, regionprops

__all__ = ["best_z_proj", "measure_signal", "measure_signal_raw", "mask_props", "mask_skeleton_length"]

def best_z_proj(ca, **kwargs):
    '''
    Returns an x-array that holds the best-z projection of intensities of spots in a reference channel and augments ca to include a 'ca.zc' layer that holds the best-z values.

    Parameters
    ----------
    ca: crop array (x-array dataset)
        A crop array.
    ref_ch: int, optional 
        A reference intensity channel for finding the best-z projection. Default: ref_ch = 0. If None, best-z is calculated separately for each channel.
    disk_r: int, optional
        The radius of the disk (in pixels) to make measurements to determine the best-z slice. Default: disk_r = 1
    roll_num: int, optional
        The number of z-slices to use in the rolling-z max projection for determining the best-z slice. min_periods in the da.rolling() function is set to 1 so there will be no Nans at the z-edges of crops. Default: roll_num = 1.

    Returns
    ------- 
    A 'best-z' x-array with dimensions (fov,n,t,y,x,ch). The best-z x-array contains the intensities of each crop in the 'best' z-slice, where 'best' is defined as the slice having the maximal intensity within a centered xy disk of radius disk_r pixels. A rolling-z maximum projection (over roll_n z-slices) can optionally be performed so best-z represents a max-z projection across multiple z-slices. In addition, the inputted crop array ca is augmented to now contain a 'zc' layer that contains the best-z position of each crop. 
    '''
    # Get the optional key word arguments (kwargs):
    ref_ch = kwargs.get('ref_ch', 0)
    disk_r = kwargs.get('disk_r', 1) 
    roll_n = kwargs.get('roll_n', 1)

    res = ca.dx  # resolution for defining disk to make measurements to determine best-z

    # Compute best-z separately for each channel using list comprehension
    if ref_ch is None:
        z_sig = [
            ca.sel(ch=ch_index).int.where(lambda a: a.x**2 + a.y**2 <= (disk_r * res)**2)
            .mean(dim=['x', 'y'])
            .rolling(z=roll_n, center=True, min_periods=1)
            .max()
            for ch_index in ca.ch.values
        ]
        # Choose z-plane
        output = xr.concat([
            ca.int.isel(ch=i).rolling(z=roll_n, center=True, min_periods=1).max().isel(z=z_sig[i].argmax(dim='z'))
            for i in range(len(ca.ch.values))
        ], dim='ch')   
        
        # Add/overwrite the 'zc' layer in the inputted crop-array
        ca['zc'] = xr.concat([z_sig[i].argmax(dim='z') for i in range(len(ca.ch.values))], dim='ch')
        ca.zc.attrs['units'] = 'pixels'
        ca.zc.attrs['long_name'] = 'crop center z for each channel'
    else:
        # Get z-signals in disk within each z-plane and apply rolling z-average of these signals
        z_sig = ca.sel(ch=ref_ch).int.where(lambda a: a.x**2 + a.y**2 <= (disk_r*res)**2).mean(dim=['x','y']).rolling(z=roll_n, center=True, min_periods=1).max()

        # Choose z-plane in ca.int corresponding to max z-signal for each channel, then concatenate x-arrays with coordinate channels
        output = xr.concat([ca.int.sel(ch=i).rolling(z=roll_n,center=True,min_periods=1).max().isel(z_sig.argmax(dim=['z'])) for i in ca.ch], dim='ch') 
        
        # Add/overwrite the 'zc' layer in the inputted crop-array
        ca['zc'] = z_sig.argmax(dim='z')
        ca.zc.attrs['units']='pixels'
        ca.zc.attrs['long_name']='crop center z'        


    return output


def measure_signal(ca, **kwargs):
    '''
    A function to measure and visualize the intensity signal of all crops in the crop array ca.

    Parameters
    ----------
    ca: crop array (x-array dataset)
        A crop array.
    ref_ch: int, optional 
        A reference intensity channel for finding the best-z plane for measurements. Default: None (uses all channels).    
    disk_r: int, optional
        The radius (in pixels) within which the intensity signal for each crop is measured. Default: disk_r = 1
    disk_bg: int, optional
        The radius (in pixels) of an outer ring (of width one pixel) within which the background signal for each crop is measured. Default: disk_bg = ca.xy_pad.
    roll_num: int, optional
        The number of z-slices to use in the rolling-z max projection for determining the best-z slices to perform intensity measurements. Default: roll_num = 1.

    Returns
    ------- 
    An augmented crop array ca with two additional variables: (1) ca.best_z is an x-array with dimensions (fov,n,t,y,x,ch) that contains the best-z-projection after background subtraction; (2) ca.signal is an x-array with dimensions (fov,n,t,ch) that contains the background-subtracted intensity signal of each crop in ca.best_z. 
    '''
    # Get the optional keyword arguments (kwargs):
    my_ref_ch = kwargs.get('ref_ch', None)
    my_disk_r = kwargs.get('disk_r', 1) 
    my_disk_bg = kwargs.get('disk_bg', ca.xy_pad) 
    my_roll_n = kwargs.get('roll_n', 1)

    # Create best-z projection (if not already)
    best_z = best_z_proj(ca, ref_ch=my_ref_ch, disk_r=my_disk_r, roll_n=my_roll_n)
    
    # Make mask for measuring within inner ring (the disk):
    disk_sig = best_z.where(lambda a: a.x**2 + a.y**2 <= (my_disk_r * ca.dx) ** 2).mean(dim=['x', 'y'])
    
    # Make mask for measuring background within outer ring (the donut):
    donut_sig = best_z.where(lambda a: (a.x**2 + a.y**2 >= (my_disk_bg * ca.dx) ** 2) & (a.x**2 + a.y**2 < ((my_disk_bg + 1) * ca.dx) ** 2)).median(dim=['x', 'y'])

    # Measure signal as disk - donut: 
    signal = disk_sig - donut_sig 

    # Add best_z variable to ca
    ca['best_z'] = best_z - donut_sig
    ca['best_z'].attrs['units'] = 'intensity (a.u.)'
    ca['best_z'].attrs['long_name'] = 'max intensity projection into best-z plane(s)'

    # Add best_z_signal variable to ca:
    ca['signal'] = signal
    ca['signal'].attrs['units'] = 'intensity (a.u.)'
    ca['signal'].attrs['long_name'] = 'crop signal'

    return ca

def measure_signal_raw(ca, **kwargs):
    '''
    A function to measure and visualize the intensity signal of all crops in the crop array ca.

    Parameters
    ----------
    ca: crop array (x-array dataset)
        A crop array.
    ref_ch: int, optional 
        A reference intensity channel for finding the best-z plane for measurements. Default: None (uses all channels).    
    disk_r: int, optional
        The radius (in pixels) within which the intensity signal for each crop is measured. Default: disk_r = 1
    disk_bg: int, optional
        The radius (in pixels) of an outer ring (of width one pixel) within which the background signal for each crop is measured. Default: disk_bg = ca.xy_pad.
    roll_num: int, optional
        The number of z-slices to use in the rolling-z max projection for determining the best-z slices to perform intensity measurements. Default: roll_num = 1.

    Returns
    ------- 
    An augmented crop array ca with two additional variables: (1) ca.best_z_raw is an x-array with dimensions (fov,n,t,y,x,ch) that contains the best-z-projection; (2) ca.signal_raw is an x-array with dimensions (fov,n,t,ch) that contains the intensity signal of each crop in ca.best_z_raw. 
    '''
    # Get the optional keyword arguments (kwargs):
    my_ref_ch = kwargs.get('ref_ch', None)
    my_disk_r = kwargs.get('disk_r', 1) 
    my_roll_n = kwargs.get('roll_n', 1)

    # Create best-z projection (if not already)
    best_z = best_z_proj(ca, ref_ch=my_ref_ch, disk_r=my_disk_r, roll_n=my_roll_n)
    
    # Make mask for measuring within inner ring (the disk):
    disk_sig = best_z.where(lambda a: a.x**2 + a.y**2 <= (my_disk_r) ** 2).sum(dim=['x', 'y'])

    # Add best_z variable to ca
    ca['best_z_raw'] = best_z
    ca['best_z_raw'].attrs['units'] = 'intensity (a.u.)'
    ca['best_z_raw'].attrs['long_name'] = 'max intensity projection into best-z plane(s)'

    # Add best_z_signal variable to ca:
    ca['signal_raw'] = disk_sig
    ca['signal_raw'].attrs['units'] = 'intensity (a.u.)'
    ca['signal_raw'].attrs['long_name'] = 'crop signal'

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
