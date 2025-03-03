from .base_optimizer import BaseOptimizer

from typing import Optional
import numpy as np

class SimulatedAnnealing(BaseOptimizer):
    def __init__(self, n_iter: int, temperature: int):
        """
        Initializes the Simulated Annealing optimizer.
        
        :param n_iter: Number of iterations.
        :param temperature: Initial temperature for annealing.
        """
        super().__init__(n_iter)
        self.temperature = temperature
        
        
    def optimize(self, solution: "BaseSolution", renderer: Optional["BaseRenderer"] = None, render_interval: int = 100) -> "BaseSolution":
        """
        Performs optimization using simulated annealing.
        
        :param solution: Initial solution to optimize.
        :param renderer: Optional renderer for visualization.
        :param render_interval: Number of iterations between rendering updates.
        :return: The optimized solution.
        """
        temperature  = self.temperature
        initial_temperature = temperature
        Tfactor = -np.log(self.temperature / 0.05)

        step = 0
        prev_cost = np.inf
        
        
        
        if renderer:
            renderer.initialise(solution)
            
        print(f"{'Temp':<8}{'Energy':<10}")
        
        while step<self.n_iter:
            step+=1

            # Take a step and get cost
            new_solution, _ = solution.move(step_size=0.05)
            new_cost = new_solution.get_cost()
            delta_energy = new_cost - prev_cost
            
            # Accept step stochastically
            if delta_energy < 0 or np.random.rand() < np.exp(-delta_energy / temperature):
                solution = new_solution 
            
            if step % render_interval == 0 and renderer:
                renderer.update(solution, self.history)
            
            prev_cost = new_cost
            
            if self.best_solution is None or new_cost < self.best_cost:
                self.best_solution = solution
                self.best_cost = new_cost
                
            self.history.append(new_cost)
            
            print(f"{temperature:<8.2f}{prev_cost:<10.5f}", end='\r')
            
            # Update temperature using exponential decay
            temperature = self.temperature * np.exp(Tfactor * step / self.n_iter)
            
        print('\n')
        if renderer:
            renderer.finalize(solution, self.history)
            
        return solution