# Thomson Problem Optimization

[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

![TPSC](https://github.com/user-attachments/assets/e81d26f2-3c76-45eb-9803-1e1ea320adf1)

## Overview
An optimization-based solver for the **Thomson Problem**, focused on minimizing electrostatic potential energy by distributing electrons on a sphere using **Gradient Descent** and **Simulated Annealing**, with real-time 3D visualization.

> The Thomson Problem seeks the minimum-energy configuration of N electrons confined to the surface of a unit sphere, where electrons repel each other according to Coulomb's law. This project implements numerical optimization techniques to approximate these configurations efficiently.

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

### Gradient Descent
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
    
    print(f"Best cost: {optimizer.best_cost}, Elapsed: {time.time() - start:.2f} seconds")
```

### Simulated Annealing

```python
from thomson_problem.optimizers.simulated_annealing import SimulatedAnnealing
optimizer = SimulatedAnnealing(iters=100000, temperature=25000, step_size=0.05)
```

## Architecture
### **1. Solutions**
- `GradientDescentSolution`: Encapsulates the energy function and gradient computation. Supports updates based on gradient flow.
- `RandomShift`: Random perturbations for escaping local minima (used in annealing).

### **2. Optimizers**
- `GradientDescentOptimizer`: Controls the learning rate schedule and number of iterations for gradient descent.
- `SimulatedAnnealing`: Probabilistic optimizer with temperature-based acceptance criteria.

### **3. Renderer**
- `ConvexHullRenderer`: Real-time rendering of the spherical distribution and convex hull using **PyVista**.

## 📈 Performance Tuning
- **Increase Iterations (`iters`)**: Higher values may yield better convergence but increase runtime.
- **Adjust Learning Rate (`learning_rate`)**: Too large may overshoot minima; too small may slow convergence.
- **Temperature (`temperature`)**: Higher initial temperatures allow better exploration in simulated annealing.

## 🔧 Impovements
- Add GPU support
- Support for custom cost functions and learning schedules
- Implement hybrid or evolutionary algorithms

## 📜 License
This project is licensed under the [MIT License](LICENSE).

