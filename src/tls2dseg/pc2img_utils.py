from typing import Tuple
import numpy as np
from pchandler.geometry.core import PointCloudData
from functools import partial
from pc2img.image_generation import SphericalImageGeneratorFromPCD, OrthographicImageGeneratorFromPCD
from pc2img.util import convert_to_image, OpticalFlowVisualization
from pc2img.core import PCDImageLink
from pathlib import Path
import imageio.v3 as iio
from numpy.typing import NDArray
from sklearn.neighbors import NearestNeighbors
import re
import pyvips
from typing import Union, Literal


def pc2img_run(pcd: PointCloudData, pcd_path: Path, image_generation_parameters: dict,
               image_width: int = 2000, image_height: int = 0) -> [(str, NDArray[np.uint8], Path)]:
    # Unpack necessary variables
    rasterization_method = image_generation_parameters["rasterization_method"]
    features = image_generation_parameters["features"]

    # Create Point Cloud and Spherical Image link:
    # image_width = 2000
    if image_height == 0:
        image_height = int(image_width // pcd.fov.ratio())
    pcd_image_link = PCDImageLink(pcd, [partial(SphericalImageGeneratorFromPCD,
                                                image_resolution=(image_height, image_width),
                                                rasterization_method=rasterization_method, minimum_nb_points=0)])

    # Create image for each selected point cloud feature (default - intensity)
    # Saving results in a list of tuples with: feature_name (str), image (NDArray[np.uint8]), path_to_saved_image (Path)
    images_i = []
    for f in features:
        # Create image (2D np.ndarray)
        spherical_image_data = pcd_image_link.get_image_data(pcd_image_link.available_stacks[0], f,
                                                             normalize=True,
                                                             normilization_percentiles=(5, 95))
        # Replace NaN values with Max
        spherical_image_data = np.nan_to_num(spherical_image_data, nan=np.nanmax(spherical_image_data))

        # Save image in results
        if "output_dir_images" in image_generation_parameters:
            # Convert to RGB image
            spherical_image = convert_to_image(spherical_image_data, "max", normalize=True, colormap='gray')  # gray
            save_image_dir = image_generation_parameters["output_dir_images"]
            save_image_file = save_image_dir / f"{pcd_path.stem}_Spherical_{f}.png"
            iio.imwrite(save_image_file, spherical_image)
        else:
            save_image_file = "no Path - images not saved"
        # Return tuple with (str: feature_name, ndarray: image, Path: path_to_saved_image)
        images_i.append((f, spherical_image_data, save_image_file))
    return images_i


def rotate_pcd_around_z(pcd: PointCloudData, theta: float = 0.0) -> None:
    theta = np.deg2rad(theta)  # Convert degrees to radians

    rotation_matrix = np.array([
        [np.cos(theta), -np.sin(theta), 0, 0],
        [np.sin(theta), np.cos(theta), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    pcd.transform(rotation_matrix)
    return None


def estimate_scanning_resolution(pcd, image_generation_parameters) -> tuple[float, float]:
    """
    Estimate the angular scanning resolution (d_azim, d_elev) of a point cloud.

    Steps:
    1. Randomly subsample `subsample_frac` of the full cloud’s points.
    2. Build a 2D histogram over (elevation, azimuth) with
       bins = floor(1/patch_frac) along each axis.
    3. Pick the bin with the median count of points → that patch’s angular bounds.
    4. Extract the full, ORIGINAL points lying in that patch.
    5. Estimate median 2D nearest-neighbor distance → d_azim = d_elev

    Params
    ------
    pcd : object
        Must have attribute `.spherical_coordinates` as an (N,3) numpy array
        [range, elevation (rad), azimuth (rad)].
    image_generation_parameters: dictionary with parameters for automatic estimation of scanning resolution

    Returns
    -------
    (d_azim, d_elev) : tuple of floats
        Estimated angular resolution in radians along azimuth and elevation.
    """

    # Unpack necessary variables
    subsample_frac = image_generation_parameters["subsampling_fraction"]
    patch_frac = image_generation_parameters["patch_frac"]
    fov = pcd.fov

    # extract angles and the number of points
    sph = pcd.spherical_coordinates  # shape (N,3)
    elev = sph[:, 1]
    azim = sph[:, 2]
    N = elev.shape[0]

    # 1) subsample for patch search
    M = int(N * subsample_frac)
    idx = np.random.choice(N, size=M, replace=False)
    elev_s = elev[idx]
    azim_s = azim[idx]

    # 2) 2D histogram: bins per axis (e - elevation, a - azimuth) = floor(1/patch_frac)
    bins_e = int(1.0 / patch_frac)
    bins_a = bins_e
    H, elev_edges, azim_edges = np.histogram2d(elev_s, azim_s, bins=[bins_e, bins_a],
                                               range=[[fov.elevation_min, fov.elevation_max],
                                                      [fov.horizontal_min, fov.horizontal_max]])

    # 3) find the bin with the median point density (the most representative one)
    # idx_flat = np.argmax(H)
    idx_flat = np.argmin((np.abs(H - np.median(H))))  # Find the bin with the median number of points!
    i_e, i_a = np.unravel_index(idx_flat, H.shape)
    elev_min_p = elev_edges[i_e]
    elev_max_p = elev_edges[i_e + 1]
    azim_min_p = azim_edges[i_a]
    azim_max_p = azim_edges[i_a + 1]

    # 4) extract full original points falling within that patch
    mask = (
            (elev >= elev_min_p)
            & (elev <= elev_max_p)
            & (azim >= azim_min_p)
            & (azim <= azim_max_p)
    )
    elev_p = elev[mask]  # Elevation (extracted points)
    azim_p = azim[mask]  # Azimuth (extracted points)

    # 5) resolution estimation
    pts2d = np.column_stack((elev_p, azim_p))
    nn = NearestNeighbors(n_neighbors=2, algorithm="auto").fit(pts2d)
    dists, _ = nn.kneighbors(pts2d)
    d_azim = d_elev = float(np.median(dists[:, 1]))
    return d_azim, d_elev


def resolve_scanning_resolution_parameter(pcd: PointCloudData,
                                          image_generation_parameters: dict) -> tuple[float, float]:
    """
    Envelope that returns (d_azim, d_elev) in radians according to image_generation_parameters["scan_resolution"]
    Acceptable values:
      - "auto": run estimate_scanning_resolution(...)
      - "<mm>@<m>": parse mm and m, compute d_azim = d_elev = arctan(s/r), where s=mm/1000, r=m
      - numeric or numeric‐string: treat as degrees and convert to radians
    """

    # Unpack necessary variables
    scanning_resolution = image_generation_parameters["scan_resolution"]

    # 1) AUTO
    if isinstance(scanning_resolution, str) and scanning_resolution.strip().lower() == "auto":
        # Set new hyperparameters (used for computational efficiency, should not impact the results)
        #   - subsampling fraction (random subsample percentage of original pcd)
        image_generation_parameters["subsampling_fraction"] = 0.01
        #   - patch fraction (cut-out of a point cloud patch, in percentage of full angular span (spherical coordinates)
        image_generation_parameters["patch_frac"] = 0.01
        # "auto" estimate scan resolution
        d_azim, d_elev = estimate_scanning_resolution(pcd, image_generation_parameters)
        return d_azim, d_elev

    # 2) "mm@<m>" pattern
    if isinstance(scanning_resolution, str) and "mm@" in scanning_resolution.lower():
        m = re.match(
            r"^\s*([\d\.]+)\s*mm\s*@\s*([\d\.]+)\s*m\s*$",
            scanning_resolution,
            re.IGNORECASE
        )
        if not m:
            raise ValueError(f"Invalid mm@ format: {scanning_resolution}")
        mm_val = float(m.group(1))
        m_val = float(m.group(2))
        # Scanning resolution in radians
        d_azim = d_elev = np.arcsin((mm_val / 1000.0) / m_val)
        return d_azim, d_elev

    # 3) numeric or numeric‐string => degrees → radians
    try:
        # Scanning resolution in degrees
        deg = float(scanning_resolution)
        # Scanning resolution in radians
        d_azim = d_elev = np.deg2rad(deg)
        return d_azim, d_elev
    except Exception:
        raise ValueError(f"Unsupported scanning_resolution: {scanning_resolution}")


def compute_image_dimensions(pcd: PointCloudData, image_generation_parameters: dict) -> tuple[int, int, float, float]:
    """
    Compute image width and height (in pixels)
    Input:
      - pcd: point cloud data object with defined fov (field of view) with elevation_min, elevation_max,
             horizontal_min, horizontal_max (in radians)
      - image_generation_parameters: dictionary with all necessary settings parameters for image generation

    Output:
      - image_width, image_height
    """
    # Unpack necessary variables
    image_width = image_generation_parameters["image_width"]

    # Get angular extent of the scan
    elev_span = pcd.fov.elevation_max - pcd.fov.elevation_min
    azim_span = pcd.fov.horizontal_max - pcd.fov.horizontal_min

    # Conditionally resolving image_width and image_height
    d_azim = d_elev = np.NaN
    # parse image_width parameter
    if isinstance(image_width, int):
        # Image width already given in pixels
        width_px = image_width
    elif isinstance(image_width, str):
        # Prepare image_width parameter string for parsing
        s = image_width.strip().lower()
        if s.isdigit():
            # Image width already given in pixels (but str, instead of int)
            width_px = int(s)
        else:
            # Get scan resolution along azimuth and elevation in radians
            d_azim, d_elev = resolve_scanning_resolution_parameter(pcd, image_generation_parameters)
            # Number of pixels for a full resolution image given span and increments in degrees
            full_width = int(np.ceil(azim_span / d_azim))
            if s == "scan_resolution":
                # Image_width matches full scan resolution
                width_px = full_width
            else:
                # Parse string to get a fraction of scan resolution
                m = re.match(r'^([\d\.]+)[\s-]*scan_resolution$', s)
                if m:
                    # Desired fraction of the full scan resolution
                    frac = float(m.group(1))
                    # Image width matching a desired fraction (frac) of full scan resolution
                    width_px = int(np.ceil(full_width * frac))
                else:
                    raise ValueError(f"Invalid image_width format: {image_width!r}")
    else:
        raise TypeError(
            f"image_width must be int or str (of type scan_resolution or frac-scan_resolution, got {type(image_width)}")

    # set image_height to preserve angular aspect ratio
    height_px = int(np.ceil(width_px * (elev_span / azim_span)))

    # Return image_width and image_height in pixels, scan resolution in degrees
    d_azim_deg = np.rad2deg(d_azim)
    d_elev_deg = np.rad2deg(d_elev)
    return width_px, height_px, d_azim_deg, d_elev_deg


def rotate_pcd_to_azimuth_gap(pcd, image_generation_parameters) -> float:
    """
    Rotate the point cloud so that a “gap” (bin/azimuth direction with 0 -or- few points) is centred at 0°.
    Goal: Avoid spherical image edges cutting the object/region of interest.

    Parameters
    ----------
    pcd : PointCloudData
    image_generation_parameters: dict

    Returns
    -------
    theta_deg : float
        Rotation angle around Z-axis in degrees.
    """

    # Unpack necessary variables:
    #   - subsampling fraction (random subsample percentage of original pcd)
    subsample_frac = image_generation_parameters["subsampling_fraction"]
    #   - bin width (in deg) for searching the gap in FoV along horizon
    bin_width_deg = image_generation_parameters["bin_width_for_theta_search_deg"]

    # 1) Sub-sample the cloud
    azim = pcd.spherical_coordinates[:, 2]  # (azim in rad)
    N = azim.shape[0]
    M = int(N * subsample_frac)
    idx = np.random.choice(N, size=M, replace=False)
    azim = azim[idx]  # radians → degrees in (-180,180]

    # 2) 1-D histogram  (-π..π with 1° bins)
    bin_width_rad = np.deg2rad(bin_width_deg)  # bin width by default 1 degree (converted in radians)
    edges = np.arange(-np.pi, np.pi, bin_width_rad)  # Should be 360 edges
    if edges[-1] != np.pi:
        edges = np.concatenate([edges, [np.pi]])  # make sure edges includes +π
    counts, _ = np.histogram(azim, bins=edges)
    centres = edges[:-1] + bin_width_rad * 0.5  # bin centres (radians)

    # 3) If the ±1° bins around 0 have no points → nothing to do
    edge_zone = np.zeros_like(counts, dtype=bool)
    edge_zone[0] = True  # the bin covering [-π, -π+1°)
    edge_zone[-1] = True  # the bin covering [π−1°, π)

    if counts[edge_zone].sum() == 0:
        theta_deg = 0.0
        return theta_deg  # no rotation applied

    # 4) Find best “gap” bin to re-centre
    #    • Prefer bins with 0 points.
    #    • If none has 0 points, take bins with the minimum point count.
    #    • For ties, choose the middle of the longest consecutive run of zero or small count bins.

    # Define a helper function
    def _best_run(mask: np.ndarray) -> int:
        """Return centre-index of the longest consecutive True-run."""
        best_len = best_start = cur_len = cur_start = 0
        for i, v in enumerate(mask):
            if v:
                if cur_len == 0:
                    cur_start = i
                cur_len += 1
            else:
                if cur_len > best_len:
                    best_len, best_start = cur_len, cur_start
                cur_len = 0
        if cur_len > best_len:  # tail run
            best_len, best_start = cur_len, cur_start
        return best_start + best_len // 2  # middle index

    # Try zero-count bins first, then search for bin with minimal point count
    zero_mask = counts == 0
    if np.any(zero_mask):
        chosen_idx = _best_run(zero_mask)
    else:
        min_mask = counts == counts.min()
        chosen_idx = _best_run(min_mask)

    theta_rad = -np.pi - centres[chosen_idx]
    theta_deg = np.rad2deg(theta_rad)

    # -------------------------------------------------
    # 5. Rotate point cloud so that this gap’s centre → 0 °
    #    (i.e. rotate by -gap_center_deg)
    # -------------------------------------------------
    rotate_pcd_around_z(pcd, theta=theta_deg)
    return theta_deg


def resolve_rotate_pcd_parameter(pcd, image_generation_parameters) -> float:
    """
    Parse image_generation_parameters['rotate_pcd'] and rotate the point cloud about +Z, if necessary.

    Parameters
    ----------
    pcd : PointCloudData
    image_generation_parameters: dict, with ['rotate_pcd'] key, with one of the following values:
        'rotate_pcd' : {"auto", float|str-float, False}
            • "auto"  → call rotate_pcd_to_azimuth_gap to find a gap and rotate.
            • float   → rotate by that many degrees (CCW seen from +Z).
            • False   → do nothing.

    Returns
    -------
    theta_deg : float
        Applied rotation in degrees. 0.0 if no rotation.
    """
    # Unpack necessary variables
    rotate_pcd = image_generation_parameters['rotate_pcd']

    # 1) If no rotation wanted
    if rotate_pcd is False:
        theta_deg = 0.0
        return theta_deg

    # 2) If rotate_pcd = "auto" (search for a gap with no/small number of points in FoV along horizon)
    elif isinstance(rotate_pcd, str) and rotate_pcd.strip().lower() == "auto":
        # Set new hyperparameters (used for computational efficiency, should not impact the results)
        #   - subsampling fraction (random subsample percentage of original pcd)
        image_generation_parameters["subsampling_fraction"] = 0.01
        #   - bin width (in deg) for searching the gap in FoV along horizon
        image_generation_parameters["bin_width_for_theta_search_deg"] = 1.0
        # Get the best theta_deg automatically
        theta_deg = rotate_pcd_to_azimuth_gap(pcd, image_generation_parameters)
        return theta_deg

    else:
        # 3) Explicit numeric rotation (accept int/float or numeric str)
        try:
            theta_deg = float(rotate_pcd)
            if theta_deg != 0.0:
                rotate_pcd_around_z(pcd, theta=theta_deg)
        except (TypeError, ValueError):
            raise ValueError(
                "image_generation_parameters['rotate_pcd'] must be 'auto', a numeric value (in deg), or False"
            )

    # Return the rotation angle in degrees (so it can be reversed later)
    return theta_deg


def get_instance_and_semantic_mask(results: dict, text_prompt) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Creates 1 representative instance and 1 semantic segmentation mask from N individual object masks.

    Args:
        results: A dictionary with grounded_sam2 results containing all instance/semantics segmentation info
                'masks' with M x w x h (M = mask number, w = width, h = height),
                'input_boxes' with input bounding boxes (results of object detection),
                'confidences' with confidence scores,
                'class_names', 'class_ids', ...
        text_prompt: a string with text prompts used for object detection with GroundedDINO
                     (each "object" separated by a dot ".")
    Returns:
        instance_mask: A NumPy array of shape (H, W) with unique labels for each instance.
        semantic_mask: A NumPy array of shape (H, W) with labels for each semantic class.
        class_ids: A dictionary with str class_name int class_id value-pairs
    """

    H, W = results["masks"][0].shape  # Mask/image size
    N = len(results["masks"])  # Number of detections

    # Get dictionary mapping "semantic classes" to unique IDs
    keys = text_prompt.split('.')  # → ['house','window','bicycle','door','grass','leaf']
    id_map = {k: i + 1 for i, k in enumerate(keys)}  # → {'house':1, 'window':2, ..., 'leaf':6}
    class_ids = [id_map[q] for q in results["class_names"]]  # a list of corresponding class IDs
    results["class_ids"] = class_ids  # Store real class IDs corresponding to detected classes, not range(#C)

    # Sorting masks from biggest to smallest, so if overlapping, the big ones do not superimpose the small ones
    mask_sizes = np.zeros(N, dtype=int)
    for i in range(N):
        mask_sizes[i] = results["masks"][i].nnz
    # Get indices sorted from biggest to smallest
    sorted_indices = np.argsort(mask_sizes)[::-1]

    # Initialize masks with zeros (background)
    instance_mask = np.zeros((H, W), dtype=np.int32)
    semantic_mask = np.zeros((H, W), dtype=np.int32)

    for i in range(N):
        # Get the instance mask
        mask_i = results["masks"][sorted_indices[i]].toarray().astype(bool)  # Shape: (H, W), dtype: bool
        # Assign a unique label to each instance in the instance mask
        # Labels start from 1 (to have 0 for background)
        instance_label = i + 1
        instance_mask[mask_i] = instance_label

        # Assign the semantic label to the semantic mask
        # Labels are class_id + 1 to avoid using 0
        semantic_mask[mask_i] = class_ids[sorted_indices[i]]

    return instance_mask, semantic_mask, id_map


def get_instance_and_semantic_mask_with_confidence(results: dict, text_prompt, image_hw: tuple)\
        -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """
    Creates 1 representative instance and 1 semantic segmentation mask from N individual object masks.

    Args:
        results: A dictionary with grounded_sam2 results containing all instance/semantics segmentation info
                'masks' with M x w x h (M = mask number, w = width, h = height),
                'input_boxes' with input bounding boxes (results of object detection),
                'confidences' with confidence scores,
                'class_names', 'class_ids', ...
        text_prompt: a string with text prompts used for object detection with GroundedDINO
                     (each "object" separated by a dot ".")
        image_hw: a tuple with width and height of the image
    Returns:
        instance_mask: A NumPy array of shape (H, W) with unique labels for each instance.
        semantic_mask: A NumPy array of shape (H, W) with labels for each semantic class.
        confidence_mask: A NumPy array of shape (H, W) with confidences for each instance (detection).
        class_ids: A dictionary with str class_name int class_id value-pairs
    """

    H, W = image_hw  # Mask/image size
    N = len(results["masks"])  # Number of detections

    # Get dictionary mapping "semantic classes" to unique IDs
    keys = text_prompt.split('.')  # → ['house','window','bicycle','door','grass','leaf']
    id_map = {k: i + 1 for i, k in enumerate(keys)}  # → {'house':1, 'window':2, ..., 'leaf':6}
    class_ids = [id_map[q] for q in results["class_names"]]  # a list of corresponding class IDs
    results["class_ids"] = class_ids  # Store real class IDs corresponding to detected classes, not range(#C)

    # Get confidences
    confidences = results["confidences"]

    # Sorting masks from biggest to smallest, so if overlapping, the big ones do not superimpose the small ones
    mask_sizes = np.zeros(N, dtype=int)
    for i in range(N):
        mask_sizes[i] = results["masks"][i].shape[0]
    # Get indices sorted from biggest to smallest
    sorted_indices = np.argsort(mask_sizes)[::-1]

    # Initialize masks with zeros (background)
    instance_mask = np.zeros((H, W), dtype=np.int32)
    semantic_mask = np.zeros((H, W), dtype=np.int32)
    confidence_mask = np.zeros((H, W), dtype=np.float32)

    for i in range(N):
        # Get the instance mask row and column indices
        mask_i = results["masks"][sorted_indices[i]]
        mask_i = np.unique(mask_i, axis=0)

        # Assign a unique label to each instance in the instance mask
        # Labels start from 1 (to have 0 for background)
        instance_label = i + 1
        instance_mask[mask_i[:, 0], mask_i[:, 1]] = instance_label

        # Assign the semantic label to the semantic mask
        # Labels are class_id + 1 to avoid using 0
        semantic_mask[mask_i[:, 0], mask_i[:, 1]] = class_ids[sorted_indices[i]]

        # Assign confidence score i to confidence mask
        confidence_mask[mask_i[:, 0], mask_i[:, 1]] = confidences[sorted_indices[i]]

    return instance_mask, semantic_mask, confidence_mask, id_map


def project_masks2pcd_as_scalarfields(pcd: PointCloudData, instance_mask, semantic_mask) -> None:
    """
    Assigns segmentation labels from an image to each point in the point cloud based on spherical coordinates.

    Parameters:
    - point_cloud: PointCloudData object.
    - instance/semantic mask: np.ndarray with instance and class labels (dtype= np.int32)

    Modifies:
    - point_cloud.scalar_fields
    """

    # Unpack variables
    image_height, image_width = instance_mask.shape

    # 1: Extract spherical coordinates
    azimuth = pcd.spherical_coordinates[:, 2]
    elevation = pcd.spherical_coordinates[:, 1]

    # 2: Map spherical coordinates to normalized coordinates based on FOV
    # Extract FOV values in radians from image_stack.fov
    azimuth_min, elevation_min, azimuth_max, elevation_max = pcd.fov.as_numpy(unit="rad")

    # Normalize azimuth and elevation to [0, 1]
    normalized_azimuth = (azimuth - azimuth_min) / (azimuth_max - azimuth_min)
    normalized_elevation = (elevation - elevation_min) / (elevation_max - elevation_min)

    # 3: Map normalized coordinates to pixel coordinates
    u = normalized_azimuth * (image_width - 1)
    v = normalized_elevation * (image_height - 1)

    # Round and clip to valid indices
    u_int = np.clip(np.round(u).astype(int), 0, image_width - 1)
    v_int = np.clip(np.round(v).astype(int), 0, image_height - 1)

    # Handle edge cases: points outside FOV
    valid_indices = (
            (normalized_azimuth >= 0) & (normalized_azimuth <= 1) &
            (normalized_elevation >= 0) & (normalized_elevation <= 1)
    )

    # Initialize labels array with a default value (e.g., -1 for invalid points)
    all_labels_semantics = np.full(azimuth.shape, fill_value=-1, dtype=instance_mask.dtype)
    all_labels_instances = np.full(azimuth.shape, fill_value=-1, dtype=instance_mask.dtype)

    # Proceed only with valid points
    if np.any(valid_indices):
        valid_u_int = u_int[valid_indices]
        valid_v_int = v_int[valid_indices]
        labels_semantics = semantic_mask[valid_v_int, valid_u_int]
        labels_instances = instance_mask[valid_v_int, valid_u_int]

        all_labels_semantics[valid_indices] = labels_semantics
        all_labels_instances[valid_indices] = labels_instances

    # Step 5: Store instance and class labels into a scalar field
    pcd.scalar_fields.__setitem__('instances', all_labels_instances)
    pcd.scalar_fields.__setitem__('classes', all_labels_semantics)

    return None


def project_a_mask_2_pcd_as_scalarfield(pcd: PointCloudData, mask: np.ndarray, mask_name: str) -> None:
    """
    Assigns arbitrary values from an image to each point in the point cloud based on spherical coordinates.

    Parameters:
    - point_cloud: PointCloudData object.
    - image: np.ndarray

    Modifies:
    - point_cloud.scalar_fields
    """

    # Unpack variables
    image_height, image_width = mask.shape

    # 1: Extract spherical coordinates
    azimuth = pcd.spherical_coordinates[:, 2]
    elevation = pcd.spherical_coordinates[:, 1]

    # 2: Map spherical coordinates to normalized coordinates based on FOV
    # Extract FOV values in radians from image_stack.fov
    azimuth_min, elevation_min, azimuth_max, elevation_max = pcd.fov.as_numpy(unit="rad")

    # Normalize azimuth and elevation to [0, 1]
    normalized_azimuth = (azimuth - azimuth_min) / (azimuth_max - azimuth_min)
    normalized_elevation = (elevation - elevation_min) / (elevation_max - elevation_min)

    # 3: Map normalized coordinates to pixel coordinates
    u = normalized_azimuth * (image_width - 1)
    v = normalized_elevation * (image_height - 1)

    # Round and clip to valid indices
    u_int = np.clip(np.round(u).astype(int), 0, image_width - 1)
    v_int = np.clip(np.round(v).astype(int), 0, image_height - 1)

    # Handle edge cases: points outside FOV
    valid_indices = (
            (normalized_azimuth >= 0) & (normalized_azimuth <= 1) &
            (normalized_elevation >= 0) & (normalized_elevation <= 1)
    )

    # Initialize labels array with a default value (e.g., -1 for invalid points)
    all_values = np.full(azimuth.shape, fill_value=-1, dtype=mask.dtype)

    # Proceed only with valid points
    if np.any(valid_indices):
        valid_u_int = u_int[valid_indices]
        valid_v_int = v_int[valid_indices]

        values_i = mask[valid_v_int, valid_u_int]
        all_values[valid_indices] = values_i


    # Step 5: Store mask values as scalar field
    pcd.scalar_fields.__setitem__(mask_name, all_values)

    return None


def resolve_necessary_image_resolution(pcd: PointCloudData, pcp_parameters: dict, d_azim_deg: float) -> float:
    """
    Function computes reduction_coefficient for down scaling the spherical image resolution based on the original scan
    resolution, maximum range, and desired output point cloud resolution (in meters)
    """
    # get max range in the point cloud
    range_max = np.max(pcd.spherical_coordinates[:, 0])
    # get desired resolution (euclidean / metric space) for the output point cloud
    output_resolution = pcp_parameters['output_resolution']
    # compute corresponding required angular resolution
    required_ang_res = np.arcsin(output_resolution / range_max)
    # get scanning resolution in rad
    scanning_resolution = np.deg2rad(d_azim_deg)
    # compute image size (image_height and image_width) reduction coefficient
    reduction_coefficient = np.round(scanning_resolution / required_ang_res, 2)
    return reduction_coefficient


def reduce_image_resolution(set_of_images: list, reduction_coefficient: float, image_generation_parameters: dict,
                            pcd_path: Path) -> list:
    # Get image i of currently processed point cloud j
    for i, image_tuple_i in enumerate(set_of_images):
        # Get image feature name
        feature = image_tuple_i[0]
        # Get image data (ndarray) from image tuple and cast to float32 (required by pyvips)
        image_data_i = image_tuple_i[1].astype(np.float32)
        # Remove all nan-values with max value
        image_data_i = np.nan_to_num(image_data_i, nan=np.nanmax(image_data_i))
        # Create pyvips object
        image_data_i = pyvips.Image.new_from_array(image_data_i)
        # Resize image
        image_data_i = image_data_i.resize(reduction_coefficient, kernel='lanczos3')
        # Get new image height and width
        height, width = image_data_i.height, image_data_i.width
        # Convert pyvis object back to numpy
        image_data_i = np.frombuffer(image_data_i.write_to_memory(), dtype=np.float32)
        image_data_i = image_data_i.reshape((height, width))

        # Save image in intermediate results
        if "output_dir_images" in image_generation_parameters:
            # Convert to RGB image
            image_data_rgb_i = convert_to_image(image_data_i, "max", normalize=True, colormap='gray')  # gray
            # Set saving parameters and save image
            save_image_dir = image_generation_parameters["output_dir_images"]
            save_image_file_i = save_image_dir / f"{pcd_path.stem}_Spherical_{feature}_reduced.png"
            iio.imwrite(save_image_file_i, image_data_rgb_i)
        else:
            save_image_file_i = "no Path - images not saved"
        # Return tuple with (str: feature_name, ndarray: image, Path: path_to_saved_image)
        set_of_images[i] = (feature, image_data_i, save_image_file_i)

    return set_of_images


def img_1to3_channels_encoding(img: np.ndarray, output_dtype: Union[str, np.dtype, None] = "float32",
                               replace_nan_with: Union[Literal["max", "min", "random", "zero"], float] = "max",
                               normalize: Union[Literal["0-1", "0-255"], None] = "0-1",
                               broadcast: bool = True) -> np.ndarray:
    """
    Convert a H×W×, channel grayscale array with arbitrary value range and dtype into a H×W×3 channel grayscale array
    of selected dtype with 0-255 value range. Goal: Preparing image data for deep learning frameworks
    (e.g. HuggingFace transformers library, SAM/SAM2 by Facebook/Meta).

    Parameters
    ----------
    img : np.ndarray
        Input image 2-D array of Shape (H, W, ), dtype any, may contain NaNs.
    output_dtype : {'uint8', 'float32', None}
        Desired dtype of the output 3 channel image.
    replace_nan_with : {'max', 'min', 'random', 'zero'} or float
        Strategy for filling NaNs *before* further processing.
    normalize: {'0-1', '0-255'} or None
        If '0-1' or '0-255', normalize data to 0-1 or 0-255 values
        If None - do nothing.
    broadcast : bool
        If True return an O(1) broadcast view
        instead of materialising three copies.

    Returns
    -------
    img : np.ndarray
        Shape (H, W, 3) array of the requested dtype.
    """

    # Early stopping: if image already a 3 channel image with 0-255 value range, do nothing
    if img.ndim == 3 and img.shape[2] == 3:
        img_min, img_max = float(img.min()), float(img.max())
        if img_min >= 0.0 and img_max <= 255.0:
            return img

    # Checks for other cases:
    if img.ndim != 2:
        raise ValueError("Input must be a 2-D array, check if your image is not already"
                         " 3 channel image with 0-255 value range!")

    if output_dtype not in ("uint8", "float32", None):
        raise ValueError("dtype must be 'uint8', 'float32', or None")

    # Make a copy to de-attach the image from the original ndarray
    img = img.copy()

    # ---- 1) handle NaNs ----------------------------------------------------
    nan_mask = np.isnan(img)
    img_min, img_max = np.nanmin(img), np.nanmax(img)
    if nan_mask.any():
        # Get values to replace nan with
        if replace_nan_with == "max":
            fill_val = img_max
        elif replace_nan_with == "min":
            fill_val = img_min
        elif replace_nan_with == "random":
            rng = np.random.default_rng()
            fill_val = rng.uniform(img_min, img_max, size=img.shape[:2])
        elif replace_nan_with == "zero":
            fill_val = 0.0
        elif isinstance(replace_nan_with, (int, float)):
            fill_val = float(replace_nan_with)
            img_max = max(replace_nan_with, img_max)
        else:
            raise ValueError("Invalid replace_nan_with option")
        # Replace values
        if replace_nan_with == "random":
            img[nan_mask] = fill_val[nan_mask]
        else:
            img[nan_mask] = fill_val

    # ---- 2) normalise slice-wise ------------------------------------------
    if normalize is not None and normalize in ("0-1", "0-255"):
        if img_max == img_min:  # constant slice
            img = np.zeros_like(img, dtype=np.float32)
        else:
            img = (img - img_min) / (img_max - img_min)
        if normalize == "0-255":
            img = img * 255.0
    elif normalize is None:
        pass
    else:
        raise ValueError("normalize must have one of the 3 following values: '0-1', '0-255', None")

    # ---- 3) cast to requested dtype ---------------------------------------
    if output_dtype is not None:
        output_dtype_np = np.dtype(output_dtype)
        img = img.astype(output_dtype_np)

    # ---- 4) replicate channels ---------------------------------------
    if broadcast:
        img = np.broadcast_to(img[..., None], (*img.shape, 3))
    else:
        img = np.repeat(img[..., None], 3, axis=2)

    return img
