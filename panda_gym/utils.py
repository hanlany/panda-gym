import numpy as np


def distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute the distance between two array. This function is vectorized.

    Args:
        a (np.ndarray): First array.
        b (np.ndarray): Second array.

    Returns:
        np.ndarray: The distance between the arrays.
    """
    assert a.shape == b.shape
    dist = np.linalg.norm(a - b, axis=-1)
    # round at 1e-6 (ensure determinism and avoid numerical noise)
    return np.round(dist, 6)


def angle_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute the geodesic distance between two array of angles. This function is vectorized.

    Args:
        a (np.ndarray): First array.
        b (np.ndarray): Second array.

    Returns:
        np.ndarray: The geodesic distance between the angles.
    """
    assert a.shape == b.shape
    if a.ndim == 1:
        dist = 1 - np.inner(a, b) ** 2
    else:
        dist = 1 - np.einsum("ij,ij->i", a, b) ** 2
    return dist

def euler_to_quaternion(euler: np.ndarray) -> np.ndarray:
    """Convert euler angles to quaternion.

    Args:
        euler (np.ndarray): Euler angles.

    Returns:
        np.ndarray: Quaternion.
    """
    assert euler.shape == (3,)
    roll, pitch, yaw = euler
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return np.array([x, y, z, w])
