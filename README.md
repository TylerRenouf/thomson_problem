# Thomson Problem Optimization
![TPSC](https://github.com/user-attachments/assets/e81d26f2-3c76-45eb-9803-1e1ea320adf1)

## Overview
This project implements optimization techniques to solve the **Thomson Problem**, which involves distributing electrons (points) on a sphere to minimize electrostatic potential energy. The solution employs **Gradient Descent** and **Simulated Annealing** optimizers with real-time visualization using **PyVista**.

## Features
- **Gradient Descent Optimization** for energy minimization
- **Simulated Annealing** for global optimization
- **Convex Hull Rendering** for 3D visualization of point distribution
- **Euclidian/Chord distance metric & random initialization**

## Installation
### Prerequisites
Ensure you have **Python 3.8+** installed along with the required dependencies.

### Install Dependencies
```sh
pip install numpy scipy pyvista
```

## Usage
### Running the Optimization
The following script initializes a **30-electron system** and optimizes their positions using **Gradient Descent** with real-time rendering.

```python
from thomson_problem.utils import get_random_points, chord_distance
from thomson_problem.optimizers.gradient_descent_optimizer import GradientDescentOptimizer
from thomson_problem.solutions.gradient_descent_solution import GradientDescentSolution
from thomson_problem.renderers.convex_hull_renderer import ConvexHullRenderer
import time

if __name__ == "__main__":
    start = time.time()
    radius = 1
    n = 30  # Number of electrons
    iters = 100000  # Optimization iterations
    learning_rate = 0.03  # Step size
    
    renderer = ConvexHullRenderer(n, radius)
    instance = GradientDescentSolution(get_random_points(n, radius), radius, chord_distance)
    
    simulator = GradientDescentOptimizer(iters, learning_rate)
    simulator.optimize(instance, renderer)
    
    print(f"Best cost: {simulator.best_cost}, Elapsed: {time.time()-start}")
```

### Simulated Annealing
To use Simulated Annealing, replace the optimizer:

```python
from thomson_problem.optimizers.simulated_annealing import SimulatedAnnealing
simulator = SimulatedAnnealing(iters=100000, temperature=25000, step_size=0.05)
```

## Components
### **1. Solutions**
- **GradientDescentSolution**: Implements a **gradient-based approach** to optimize point positions moving each point in the negative gradient direction each iteration.
- **RandomShift**: Randomly perturbs points to explore the solution space.

### **2. Optimizers**
- **GradientDescentOptimizer**: Controls the learning rate schedule and number of iterations for gradient descent.
- **SimulatedAnnealing**: Uses probabilistic exploration to avoid local minima, typical simulated annealing implementation.

### **3. Renderer**
- **ConvexHullRenderer**: Uses **PyVista** to visualize the points and their convex hull in 3D.

## Performance Tuning
- **Increase Iterations (`iters`)**: Improves convergence.
- **Adjust Learning Rate (`learning_rate`)**: Fine-tune step sizes for better optimization.
- **Experiment with Temperature (`temperature`)**: Controls randomness in **Simulated Annealing**.

## Impovements
- Add GPU support
- Add more customizability in current classes (different schedules to be passed in or different cost functions)
- Possible hybrid approaches or novel algorithms could be explored

## License
MIT License. Free to use and modify.

