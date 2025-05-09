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

def pc2img_run(pcd:PointCloudData, pcd_path: Path, out_path:Path, save_results: bool = True,
               rotate_pcd: bool = False, theta: float|int = 0, image_height: int = 1000, image_width: int = 1000,
               rasterization_method: str = 'nanconv',
               features: list = ["intensity"]) -> [(str, NDArray[np.uint8], Path)]:

    # Rotate point cloud (if needed):
    if rotate_pcd is True:
        rotate_pcd_around_z(pcd, theta)


    # Create Point Cloud and Spherical Image link:
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
        # Convert to RGB image
        spherical_image = convert_to_image(spherical_image_data, "max", normalize=True, colormap='gray')
        # Save image in results
        save_image_file = "no Path - images not saved"
        if save_results:
            save_image_dir = out_path / Path("images")
            save_image_dir.mkdir(parents=True, exist_ok=True)
            save_image_file = save_image_dir / f"{pcd_path.stem}_Spherical_{f}.png"
            iio.imwrite(save_image_file, spherical_image)
        images_i.append((f, spherical_image, save_image_file))
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
    3. Pick the bin with the highest count → that patch’s angular bounds.
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

    # 3) find the densest bin
    idx_flat = np.argmax(H)
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


def compute_image_dimensions(pcd: PointCloudData, image_generation_parameters: dict) -> tuple[int, int]:
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
        raise TypeError(f"image_width must be int or str (of type scan_resolution or frac-scan_resolution, got {type(image_width)}")

    # set image_height to preserve angular aspect ratio
    height_px = int(np.ceil(width_px * (elev_span / azim_span)))

    # Return image_width and image_height in pixels
    return width_px, height_px

