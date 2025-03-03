from .base_solution import BaseSolution

import numpy as np
from typing import Optional, Callable


class RandomShift(BaseSolution):
    def __init__(self, points: np.ndarray, radius: int, distance_metric: Callable[[np.ndarray], np.ndarray], distances: Optional[np.ndarray] = None, best_cost: Optional[float] = None):
        """
        Initializes the Random Shift solution.
        
        :param points: Array of points representing the solution.
        :param radius: Constraint radius for normalization.
        :param distance_metric: Function to compute distance between points.
        :param distances: Precomputed distance matrix (optional).
        :param best_cost: Best cost encountered (optional).
        """
        super().__init__(points, radius, distance_metric)
        if distances is not None:
            self.distances = distances
        else:
            self.calculate_distances() # Initialise the distances
        
        self.best_cost = best_cost if best_cost is not None else np.inf

        
    def move(self, step_size: float = 0.05) -> ("RandomShift", int):
        """
        Performs a random shift operation on a randomly selected point.
        
        :param step_size: Step size for the random shift.
        :return: A new RandomShift instance with updated points.
        """
        idx, new_point = self.shift_point(step_size)
        self.update_cost()
        
        new_solution = RandomShift(self.points.copy(), self.radius, self.distance_metric, self.distances.copy(), self.best_cost)
        new_solution.points[idx] = new_point
        new_solution.update_distances(idx)
        
        return new_solution, idx
    
    
    def update_cost(self) -> None:
        """Updates the best cost if the current cost is lower."""
        curr_cost = self.get_cost()
        if curr_cost < self.best_cost:
            self.best_cost = curr_cost
            
        
    def get_cost(self) -> float:
        """Computes the cost of the current solution based on pairwise distances."""
        upper_triangular_indices = np.triu_indices(self.n, k=1)
        distances = self.distances[upper_triangular_indices]
        return np.sum(1 / distances)
    
    
    def shift_point(self, step_size: float) -> (int, np.ndarray):
        """Randomly shifts a point within the solution space."""
        idx = np.random.randint(0, self.n)
        old_point = self.points[idx]
        
        difference = np.random.normal(0, step_size, 3)
        updated = old_point + difference
        norm  = np.linalg.norm(updated)
        
        new_point = updated/norm # Normalise to sphere
        
        return idx, new_point
    
    
    def calculate_distances(self) -> None:
        """Computes and stores pairwise distances between points."""
        self.distances = self.distance_metric(self.points)
        np.fill_diagonal(self.distances, np.inf)
        
    def update_distances(self, idx: int) -> None:
        """Updates the distance matrix after a point shift."""
        p_idx = self.points[idx]
        updated = np.linalg.norm(self.points-p_idx, axis=1)
        self.distances[idx, :] = updated
        self.distances[:, idx] = updated