import pyvista as pv
import numpy as np
from scipy.spatial import ConvexHull
from .base_renderer import BaseRenderer

from typing import List, Tuple

class ConvexHullRenderer(BaseRenderer):
    """A 3D renderer that visualizes the Thomson Problem using PyVista.
    
    This renderer displays:
    - A sphere with charged points positioned on it.
    - The convex hull of the points (if `n_points > 3`).
    - A real-time energy history chart tracking optimization progress.
    """
    def __init__(self, n_points: int, radius: int):
        """
        Initializes the ConvexHullRenderer with visualization settings.
        
        :param n_points: Number of points (electrons) in the solution.
        :param radius: Radius of the sphere where the points are distributed.
        """
        super().__init__()
        
        self.n_points = n_points
        
        # Create a 1-row, 2-column plotter layout
        self.plotter = pv.Plotter(shape=(1, 2))
        self.plotter.add_title("Thomson Problem")
        
        # Left subplot: Sphere visualization
        self.plotter.subplot(0, 0)
        sphere = pv.Sphere(radius=1)
        self.plotter.add_mesh(sphere, style='wireframe', opacity=0.2, color='lightblue')
        self.plotter.add_text(text=f"Number of electrons: {self.n_points}" , position='lower_left')
        
        # Right subplot: Energy history chart
        self.plotter.subplot(0, 1)
        self.plotter.add_title("Energy History", font_size=12)
        self.plotter.add_text(text="" , position='lower_left', name='score_text',font_size=10)
        
        # Create an empty 2D chart for energy history
        self.energy_chart = pv.Chart2D(size=(0.75,0.75),loc=(0.125,0.125),x_label="Iterations", y_label="Energy")
        self.plotter.add_chart(self.energy_chart)
        
        # Variables to store point cloud and convex hull
        self.cloud_points = None
        self.hull = None
    
    
    def initialise(self, solution: "BaseSolution") -> None:
        """Initializes the visualization with the given solution.
        
        - Creates an empty energy history chart.
        - Renders the initial distribution of points.
        - Computes and displays the convex hull if applicable.
        """
        
        # Initialise the history chart
        history = [] 
        x, y = self.create_energy_line(history)
        self.energy_line = self.energy_chart.line(x,y)
        self.update_score(solution.best_cost)
       
       
        # Initialise the sphere plot
        self.plotter.subplot(0, 0)
        points = solution.points
        self.cloud_points = pv.PolyData(points)
        self.plotter.add_points(self.cloud_points, color='blue')
        
        # Add convex hull if there are enough points (must be > 3)
        if len(solution.points)>3:
            self.hull_vertices, self.hull_faces = self.get_hull(points)
            self.hull = pv.PolyData(self.hull_vertices, faces=self.hull_faces)
            self.plotter.add_mesh(self.hull, color='red', opacity=0.5)
            
        # Start interactive plotting (for dynamic updates)
        self.plotter.show(interactive_update=True)
        
    def update(self, solution: "BaseSolution", history: List[float]) -> None:
        """Updates the visualization as the solution evolves.
        
        - Moves the points based on the new solution.
        - Updates the convex hull if necessary.
        - Refreshes the energy history chart.
        """
        points = solution.points
        
        # Update the plotter left subplot
        self.plotter.subplot(0, 0)
        self.cloud_points.points = points # Add points
        
        if self.hull is not None:
            hull_vertices, hull_faces = self.get_hull(points)
            self.hull.points = hull_vertices
            self.hull.faces = hull_faces

        # Update the energy chart
        self.plotter.subplot(0, 1)
        
        self.update_score(solution.best_cost)
        
        x, y = self.create_energy_line(history)
        self.energy_line.update(x, y)

        # Render out results
        self.plotter.update(force_redraw=False)
    
    def finalize(self, solution: "BaseSolution", history: List[float]) -> None:
        """Finalizes the visualization, ensuring all updates are displayed."""
        self.update(solution, history)
        self.plotter.show()
    
    def create_energy_line(self, history: List) -> Tuple[np.ndarray, np.ndarray]:
        """Generates x and y data points for the energy history chart.
        
        :param history: A list of recorded energy values over iterations.
        :return: Tuple containing x-coordinates (iterations) and y-coordinates (energy values).
        """
        x = np.arange(len(history))
        y = np.array(history)
        return x, y
    
    def get_hull(self, points_array: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Computes the convex hull of a given set of points.
        
        :param points_array: A (n,3) NumPy array of points in 3D space.
        :return: Tuple containing the hull vertices and face connectivity.
        """
        hull = ConvexHull(points_array)
        hull_vertices = points_array
        faces = np.column_stack((3*np.ones((len(hull.simplices), 1), dtype=int), hull.simplices)).flatten()
        return hull_vertices, faces
    
    def update_score(self, score: float) -> None:
        """Updates the displayed best score in the visualization.
        
        :param score: The best cost (energy) found so far.
        """
        self.plotter.actors['score_text'].set_text(position='lower_left',text=f'Best Score: {score:0.3f}')