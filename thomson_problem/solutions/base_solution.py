from abc import ABC, abstractmethod

import numpy as np
from typing import List, Callable, Optional



class BaseSolution(ABC):
    def __init__(self, points: np.ndarray, radius: int, distance_metric: Callable[[np.ndarray], np.ndarray]):
        """
        Initializes the base solution class.
        
        :param points: Array of points representing the solution.
        :param radius: Constraint radius for normalization.
        :param distance_metric: Function to compute distance between points.
        """
        self.points = points
        self.n=len(points)
        self.radius = radius
        self.distance_metric = distance_metric
        
    @abstractmethod
    def move(self, **kwargs):
        """Abstract method to perform a move operation on the solution."""
        pass
    
    @abstractmethod
    def get_cost(self):
        """Abstract method to compute the cost of the solution."""
        pass