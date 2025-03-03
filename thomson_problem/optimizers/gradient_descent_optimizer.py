from .base_optimizer import BaseOptimizer

import numpy as np
from typing import Optional


# Gradient Descent Optimizer with Cosine Annealing Learning Rate
class GradientDescentOptimizer(BaseOptimizer):
    def __init__(self, n_iter: int, initial_lr: float, decay_factor: float = 0.5, restart_interval: float = 500):
        """
        Initializes the Gradient Descent optimizer with cosine annealing and restart mechanism.
        
        :param n_iter: Number of iterations for optimization.
        :param initial_lr: Initial learning rate.
        :param decay_factor: Factor by which learning rate is reduced at restarts.
        :param restart_interval: Number of iterations before a learning rate restart.
        """
        super().__init__(n_iter)
        
        self.initial_lr = initial_lr
        self.decay_factor = decay_factor
        self.restart_interval = restart_interval


        
    def optimize(self, solution: "BaseSolution", renderer: Optional["BaseRenderer"] = None, render_interval: int = 100) -> "BaseSolution":
        """
        Performs optimization using gradient descent with cosine annealing.
        
        :param solution: Initial solution to optimize.
        :param renderer: Optional renderer for visualization.
        :param render_interval: Number of iterations between rendering updates.
        :return: The optimized solution.
        """
        min_lr = self.initial_lr * 0.01
        step = 0
        prev_cost = np.inf

        if renderer:
            renderer.initialise(solution)

        print(f"{'Step':<6}{'LR':<10}{'Energy':<10}")
        
        while step < self.n_iter:
            step += 1
            
            # Compute learning rate and calculate cost change of the solution
            lr = self.cosine_annealing(step, self.n_iter, self.initial_lr, min_lr)
            new_solution = solution.move(learning_rate=lr)
            new_cost = new_solution.get_cost()
            delta_energy = new_cost - prev_cost
            
            # Accept the solution
            if delta_energy < 0: 
                solution = new_solution

            # Update best solution
            if self.best_solution is None or new_cost < self.best_cost:
                self.best_solution = solution
                self.best_cost = new_cost

            
            prev_cost = new_cost
            
            if step % render_interval == 0 and renderer:
                renderer.update(solution, self.history)

            print(f"{step:<6}{lr:<10.6f}{new_cost:<10.5f}", end='\r')

            self.history.append(new_cost)
            

            if step % self.restart_interval == 0:
                lr = max(lr * self.decay_factor, min_lr)
        print('\n')
        if renderer:
            renderer.finalize(solution, self.history)

        return solution
    
    
    def cosine_annealing(self, step: int, n_iter: int, initial_lr: float, min_lr: float) -> float:
        """
        Computes the learning rate using cosine annealing.
        
        :param step: Current iteration step.
        :param n_iter: Total number of iterations.
        :param initial_lr: Initial learning rate.
        :param min_lr: Minimum learning rate.
        :return: Adjusted learning rate.
        """
        return min_lr + 0.5 * (initial_lr - min_lr) * (1 + np.cos(np.pi * step / n_iter))
    