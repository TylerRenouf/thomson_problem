import numpy as np
from typing import List, Optional
from scipy.spatial.distance import cdist


def chord_distance(p1: np.ndarray, p2: Optional[np.ndarray] = None) -> float:
    """ Compute Euclidean distance between two points or two arrays of points. """
    if not p2: return cdist(p1 , p1)
    return np.linalg.norm(p1 - p2)


def get_random_points(n: int, radius: int) -> np.ndarray:
    """ 
    Generate n random points on a sphere of given radius. 
    
    x = Rsin(φ)cos(θ)
    y = Rsin(φ)sin(θ)
    z = cos(θ)
    
    For a random 0<=θ<=2pi and 0<=φ<=pi
    
    """
    theta = np.random.uniform(0, 2*np.pi, n)
    phi = np.random.uniform(0, np.pi, n)
    return np.array([np.array([radius * np.sin(phi[i]) * np.cos(theta[i]),
                  radius * np.sin(phi[i]) * np.sin(theta[i]),
                  radius * np.cos(phi[i])]) for i in range(n)])