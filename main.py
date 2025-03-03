from thomson_problem.utils import get_random_points, chord_distance
from thomson_problem.optimizers.gradient_descent_optimizer import GradientDescentOptimizer
from thomson_problem.solutions.gradient_descent_solution import GradientDescentSolution
from thomson_problem.renderers.convex_hull_renderer import ConvexHullRenderer
import time



if __name__ == "__main__":
    start = time.time()
    radius = 1
    n = 30 # Number of electrons
    iters = 100000 # optimizaion iterations
    learning_rate = 0.03
    
    renderer = ConvexHullRenderer(n, radius)
    
    instance = GradientDescentSolution(get_random_points(n, radius), radius, chord_distance)
    
    simulator = GradientDescentOptimizer(iters, learning_rate)
    simulator.optimize(instance, renderer)
    print(f"Best cost: {simulator.best_cost}, Elaspsed: {time.time()-start}")
    
    