import numpy as np
from pchandler.geometry.core import PointCloudData
from functools import partial
from pc2img.image_generation import SphericalImageGeneratorFromPCD, OrthographicImageGeneratorFromPCD
from pc2img.util import convert_to_image, OpticalFlowVisualization
from pc2img.core import PCDImageLink
from pathlib import Path
import imageio.v3 as iio
from numpy.typing import NDArray

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

