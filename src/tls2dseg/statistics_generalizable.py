import numpy as np
from scipy.stats import chi2
from sklearn.covariance import MinCovDet


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
