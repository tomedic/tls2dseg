import numpy as np
from joblib import Parallel, delayed
from sklearn.covariance import MinCovDet
from scipy.stats import chi2
from concurrent.futures import ProcessPoolExecutor, as_completed


def merge_list_of_2d_detections(dict_list: list[dict]) -> dict:
    dict_keys = dict_list[0].keys()
    merged_dict = {}

    for key_i in dict_keys:
        vals_i = [dict_i[key_i] for dict_i in dict_list]

        # concatenate numpy arrays along axis 0
        if all(isinstance(val, np.ndarray) for val in vals_i):
            merged_dict[key_i] = np.concatenate(vals_i, axis=0)

        # flatten lists of lists using a list comprehension
        elif all(isinstance(val, list) for val in vals_i):
            merged_dict[key_i] = [item for sublist in vals_i for item in sublist]
        # rais an error if dtype under same key not consistent or not supported
        else:
            raise TypeError("not all dictionaries within a list of dictionary are consistently"
                            " lists or np.ndarrays")

    return merged_dict


def filter_out_samples_in_2d_detections(detections_2d: dict, remove_mask: np.ndarray) -> dict:
    keep_mask = ~remove_mask
    N = len(remove_mask)
    detections_2d_filtered = {}

    for key, vals in detections_2d.items():
        if len(vals) != N:
            raise ValueError(f"Key '{key}' has length {len(vals)}, but remove_mask has {N}")

        if isinstance(vals, np.ndarray):
            detections_2d_filtered[key] = vals[keep_mask]
        elif isinstance(vals, list):
            detections_2d_filtered[key] = [v for v, keep in zip(vals, keep_mask) if keep]
        else:
            raise TypeError(f"Filtering not supported for type {type(vals)} in key: '{key}'")

    return detections_2d_filtered


def _compute_2d_mask_features(mask_xy: np.ndarray, rng: float) -> np.ndarray:
    """
    Compute 5-D feature vector for a single 2-D mask.
    Returns: [log(major*range), log(minor*range),
              sin(2θ), cos(2θ),
              log(pixel_count*range²)]
    """
    # ----- PCA on 2-D pixel coordinates -----
    if mask_xy.shape[0] > 256:
        mask_xy = mask_xy[np.random.choice(mask_xy.shape[0], 256, replace=False)]

    # centre
    centred = mask_xy.astype(np.float64) - mask_xy.mean(axis=0)
    # covariance and eigen-decomposition
    cov = np.cov(centred, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    # sort by descending eigenvalue
    order = np.argsort(eigvals)[::-1]
    axes = eigvecs[:, order]  # defining PCA-frame (ordered PCA eigenvectors)

    # get pc1 and project to pc1 to obtain extents
    pc1 = axes[:, 0]
    proj1 = centred @ pc1
    major = proj1.max() - proj1.min()

    # get pc2
    pc2 = axes[:, 1]
    proj2 = centred @ pc2
    minor = proj2.max() - proj2.min()

    # orientation (radians) and 2θ encoding
    theta = np.arctan2(pc1[1], pc1[0])
    sin2, cos2 = np.sin(2 * theta), np.cos(2 * theta)

    # pixel (point) count
    pix_cnt = mask_xy.shape[0]

    # ----- range normalisation & log transforms -----
    # range normalisation to account for different object distances
    major_phys = major * rng          # ∝ true length
    minor_phys = minor * rng          # ∝ true width
    area_phys  = pix_cnt * rng**2     # ∝ true area
    # log transform to get closer to normal distribution
    major_phys = np.log1p(major_phys)
    minor_phys = np.log1p(minor_phys)
    area_phys = np.log1p(area_phys)

    # Final feature set
    features = np.array([major_phys, minor_phys, sin2, cos2, area_phys], dtype=np.float32)

    return features


def robust_fit_multivariate_normal(data) -> tuple[np.ndarray, np.ndarray]:
    """
    Robust fit of Gaussian/Normal distribution using the Minimum Covariance Determinant estimator.

    Parameters:
      data: numpy array of shape (n_samples, n_dimensions)

    Returns:
      mean: Estimated nD mean.
      cov:  Robust covariance matrix.
    """
    mcd = MinCovDet().fit(data)
    mean = mcd.location_
    cov = mcd.covariance_
    return mean, cov


def get_mahalanobis_distance(data, mean, cov) -> np.ndarray:
    """
    Compute the Mahalanobis distance for each data point.

    Parameters:
      data: numpy array of shape (n_samples, n_dimensions)
      mean: Mean vector from robust_gaussian_fit.
      cov: Covariance matrix from robust_gaussian_fit.

    Returns:
      m_dist: 1D array of Mahalanobis distances.
    """
    diff = data - mean
    cov += 1e-6 * np.eye(cov.shape[0])  # for numerical stability
    inv_cov = np.linalg.inv(cov)
    left_term = np.dot(diff, inv_cov)
    mh_dist = np.sqrt(np.sum(left_term * diff, axis=1))
    return mh_dist


def multivariate_normal_outlier_removal(data: np.ndarray, quantile: float = 0.99) -> np.ndarray:

    # Robustly fit multivariate normal distribution to data
    mean, cov = robust_fit_multivariate_normal(data)
    # Get mahalanobis distance to each point
    mh_dist = get_mahalanobis_distance(data, mean, cov)
    # Get degrees of freedom
    df = data.shape[1]
    # Get chi2 test value
    chi2_val = chi2.ppf(q=quantile, df=df)
    # Detect outliers
    is_outlier = mh_dist > np.sqrt(chi2_val)

    return is_outlier


def d2d_outlier_removal(d2d_collection: dict, task_parameters: dict, per_class_separation: bool = False,
                        confidence_interval: float = 0.99) -> tuple[dict, np.ndarray]:

    # Extract relevant values:
    masks = d2d_collection["masks"]
    ranges = d2d_collection["object_distances"]
    class_ids = d2d_collection["class_ids"]
    n_jobs = task_parameters['n_workers']

    # Compute features for outlier removal using parallel computing:
    #   prep arguments for each worker (mask, range_image)
    work_items = list(zip(masks, ranges))
    #   initialize empty results
    features_or = np.empty((len(masks), 5), dtype=np.float32)
    #   run computing PCA-based features in parallel
    with ProcessPoolExecutor(max_workers=n_jobs) as exe:
        for i, feat in enumerate(exe.map(lambda t: _compute_2d_mask_features(*t), work_items)):
            features_or[i] = feat

    # Collapse different classes (optional):
    if per_class_separation is False:
        class_ids = np.ones_like(class_ids)

    # Get unique classes:
    unique_cls = np.unique(class_ids)

    # Compute statistics and outliers per class:
    is_outlier = np.zeros_like(class_ids, dtype=np.bool_)
    nd = features_or.shape[1]  # get number of dimensions
    for cls in unique_cls:
        class_mask = (class_ids == cls)
        nr_samples = np.sum(class_mask)
        if nr_samples > nd * 10:
            features_i = features_or[class_mask]
            is_outlier_i = multivariate_normal_outlier_removal(features_i, confidence_interval)
            is_outlier[class_mask] = is_outlier_i
        else:
            print(f"Warning: 2d statistical outlier removal skipped for class {cls} due to too few samples")

    # Filter out 2d detections
    d2d_collection = filter_out_samples_in_2d_detections(d2d_collection, is_outlier)

    # Return all outliers:
    return d2d_collection, is_outlier
