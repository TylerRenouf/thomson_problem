from .base_solution import BaseSolution

from typing import Callable, Optional
import numpy as np

class GradientDescentSolution(BaseSolution):
    def __init__(self, points: np.ndarray, radius: int, distance_metric: Callable[[np.ndarray], np.ndarray], distances: Optional[np.ndarray] = None, best_cost: Optional[float] = None):
        """
        Initializes the Gradient Descent solution.
        
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
    
    def move(self, learning_rate: float = 0.01) -> "GradientDescentSolution":
        """
        Performs a gradient-descent move.
        
        :param learning_rate: Step size for updating the points.
        :return: A new GradientDescent instance with updated points.
        """
        points = self.points
        gradients = self.compute_gradient(points)

        # Update points using computed gradients
        points -= learning_rate * gradients
        
        # Normalize points to unit length for a unit sphere
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        points /= norms
        
        # Create updated solution
        new_solution = GradientDescentSolution(points, self.radius, self.distance_metric, self.distances, self.best_cost)
        new_solution.calculate_distances() # get new distances
        
        return new_solution
    
    def compute_gradient(self, points: np.ndarray) -> np.ndarray:
        """
        Computes the gradient of the cost function with respect to the points.
        
        :param points: Current solution points.
        :return: Gradient vector for each point.
        """
        gradients = np.zeros_like(points)  # nx3
        epsilon = 1e-8
        
        # Compute pairwise distances
        diff = points - points[:, np.newaxis] # Shape (n, n, 3)
        distances = np.linalg.norm(diff, axis=2)  # Shape (n, n)
        distances = np.maximum(distances, epsilon)  # Avoid division by zero
        
        # Compute gradient for each point
        for i in range(self.n):
            gradients[i] = np.sum(diff[i] / distances[i, :, np.newaxis], axis=0)

        return gradients
                
    
    def update_cost(self, curr_cost: float) -> None: 
        """Updates the best cost if the current cost is lower."""
        if curr_cost < self.best_cost:
            self.best_cost = curr_cost
            
        
    def get_cost(self) -> float:
        """Computes the cost of the current solution based on pairwise distances."""
        upper_triangular_indices = np.triu_indices(self.n, k=1)
        distances = self.distances[upper_triangular_indices]
        cost = np.sum(1 / distances)
        self.update_cost(cost)
        return cost
    
    
    def calculate_distances(self) -> None:
        """Computes and stores pairwise distances between points."""
        self.distances = self.distance_metric(self.points)
        np.fill_diagonal(self.distances, np.inf)