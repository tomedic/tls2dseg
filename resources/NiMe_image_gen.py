from pathlib import Path
from functools import partial

import imageio.v3 as iio
import numpy as np

from pchandler.geometry import PointCloudData
from pchandler.data_io import load_e57, load_ply

from pc2img.image_generation import SphericalImageGeneratorFromPCD, OrthographicImageGeneratorFromPCD
from pc2img.util import convert_to_image, OpticalFlowVisualization
from pc2img.core import PCDImageLink


def main():
    pcd_path = Path(
        r"/scratch/00_data/Axpo24/03_3D_Displacement_based/02b_IOF3d/10_improved_implementation/02_Flow_PCD/Axpo24-9_20240903_2115/01_9e-03mgon_900x1600/Axpo24-9_20240903_2115.ply")

    pcd: PointCloudData = load_ply(pcd_path)
    # pcd.transform(np.array([[0.0,-1.0,0.0,0.0],
    #                         [1.0,0.0,0.0,0.0],
    #                         [0.0,0.0,1.0,0.0],
    #                         [0.0,0.0,0.0,1.0]]))

    of_viz = OpticalFlowVisualization()
    uv = np.stack((pcd.scalar_fields["deltaX"], pcd.scalar_fields["deltaY"]), axis=1)
    colors = of_viz.flow_to_image(uv[:, np.newaxis, :], max_quantile=0.95)

    pcd.set_color(colors.squeeze())

    image_width = 20000
    image_height = int(image_width // pcd.fov.ratio())

    pcd_image_link = PCDImageLink(
        pcd,
        [partial(SphericalImageGeneratorFromPCD, image_resolution=(image_height, image_width),
                 rasterization_method='nanconv',
                 minimum_nb_points=0),
         partial(OrthographicImageGeneratorFromPCD, image_resolution=(4000, 4000), rasterization_method='nanconv',
                 minimum_nb_points=0, plane="xy", downsample=False),
         ]
    )

    xy = pcd_image_link.get_image_data(pcd_image_link.available_stacks[0], 'color', True,
                                       normilization_percentiles=(1, 99))
    xy_image = convert_to_image(xy, "max")
    iio.imwrite(Path(f"/mnt/e/99_Temp/{pcd_path.stem}_Top-downOtherDay_xy_norm1-99.png"), xy_image)

    features = ["deltaX", "deltaY", "deltaZ", "deltaR", "deltaTHETA", "deltaPHI"]

    for f in features:
        top_down_image_data = pcd_image_link.get_image_data(pcd_image_link.available_stacks[0], f,
                                                            normalize=True,
                                                            normilization_percentiles=(5, 95), )
        top_down_image = convert_to_image(top_down_image_data, "max", normalize=True, colormap="cividis")
        iio.imwrite(Path(f"/mnt/e/99_Temp/{pcd_path.stem}_Top-down_{f}.png"), top_down_image)

        spherical_image_data = pcd_image_link.get_image_data(pcd_image_link.available_stacks[1], f,
                                                             normalize=True,
                                                             normilization_percentiles=(5, 95))

        spherical_image = convert_to_image(spherical_image_data, "max", normalize=True, colormap='cividis')
        iio.imwrite(Path(f"/mnt/e/99_Temp/{pcd_path.stem}_Spherical_{f}.png"), spherical_image)


if __name__ == "__main__":
    main()
