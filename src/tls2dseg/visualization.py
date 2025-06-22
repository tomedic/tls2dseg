import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata
from matplotlib.patches import Ellipse
from scipy.stats import chi2

#matplotlib.use('qt5agg')

def set_axes_equal(ax):
    """Set equal scaling for a 3D plot."""
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    y_range = abs(y_limits[1] - y_limits[0])
    z_range = abs(z_limits[1] - z_limits[0])

    max_range = max([x_range, y_range, z_range])

    x_middle = np.mean(x_limits)
    y_middle = np.mean(y_limits)
    z_middle = np.mean(z_limits)

    ax.set_xlim3d([x_middle - max_range/2, x_middle + max_range/2])
    ax.set_ylim3d([y_middle - max_range/2, y_middle + max_range/2])
    ax.set_zlim3d([z_middle - max_range/2, z_middle + max_range/2])


def plot_3d_scatter(points):
    """
    Visualizes a 3D scatter plot of 3D points.
    Parameters:
    points (numpy.ndarray): Nx3 numpy array where each row represents a 3D point (x, y, z).
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Extract x, y, z coordinates from points
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    # Scatter plot
    ax.scatter(x, y, z)

    # Labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Use the custom function to set the axes to equal
    set_axes_equal(ax)
    # Show the plot
    plt.show()
    return

def plot_2_point_clouds(source_points, target_points, source_color='r', target_color='b', title='Point Cloud Comparison'):
    """
    Plots two point clouds using Matplotlib.

    Parameters:
        source_points (numpy.ndarray): The source point cloud as a NumPy array of shape (N, 3).
        target_points (numpy.ndarray): The target point cloud as a NumPy array of shape (M, 3).
        source_color (str): Color for the source point cloud (default is 'r' for red).
        target_color (str): Color for the target point cloud (default is 'b' for blue).
        title (str): Title of the plot (default is 'Point Cloud Comparison').
    """
    # Check if both point clouds have three dimensions
    assert source_points.shape[1] == 3, "Source point cloud must have shape (N, 3)"
    assert target_points.shape[1] == 3, "Target point cloud must have shape (M, 3)"

    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the source point cloud
    ax.scatter(source_points[:, 0], source_points[:, 1], source_points[:, 2],
               c=source_color, label='Source', alpha=0.5)

    #target_points = target_points + 0.05
    # Plot the target point cloud
    ax.scatter(target_points[:, 0], target_points[:, 1], target_points[:, 2],
               c=target_color, label='Target', alpha=0.5)

    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)

    # Add legend
    ax.legend()
    # Use the custom function to set the axes to equal
    set_axes_equal(ax)
    # Show the plot
    plt.show()
    return


def display_histogram(d, bins=10, title="Histogram", xlabel="Values", ylabel="Frequency"):
    """
    Displays a histogram of the values stored in vector d.

    Parameters:
    - d: Input data (array-like)
    - bins: Number of bins for the histogram (default is 10)
    - title: Title of the histogram plot (default is "Histogram")
    - xlabel: Label for the x-axis (default is "Values")
    - ylabel: Label for the y-axis (default is "Frequency")
    """
    plt.figure(figsize=(8, 6))
    plt.hist(d, bins=bins, edgecolor='black')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.show()
    return