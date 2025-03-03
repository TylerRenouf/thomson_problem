from abc import ABC, abstractmethod

import numpy as np


class BaseOptimizer(ABC):
    def __init__(self, n_iter, **kwargs):
        """
        Initializes the base optimizer class.
        
        :param n_iter: Number of iterations for the optimization process.
        :param kwargs: Additional keyword arguments for future extensions.
        """
        self.n_iter = n_iter
        self.history = []
        self.kwargs = kwargs
        self.best_solution = None
        self.best_cost = np.inf
    
    @abstractmethod
    def optimize(self, solution):
        """
        Abstract method to be implemented by specific optimization strategies.
        
        :param solution: Initial solution to optimize.
        """
        pass
      
        