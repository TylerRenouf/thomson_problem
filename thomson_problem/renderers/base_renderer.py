from abc import ABC, abstractmethod
from typing import List

class BaseRenderer(ABC):
    """Abstract base class for rendering solutions to the Thomson Problem."""
    
    def __init__(self) -> None:
        """Base constructor (does nothing, but ensures subclass consistency)."""
        pass
    @abstractmethod
    def initialise(self, solution: "BaseSolution") -> None:
        """Initializes the rendering process with a given solution."""
        pass
    @abstractmethod
    def update(self, solution: "BaseSolution", history: List[float] ) -> None:
        """Updates the visualization based on the current state of the solution."""
        pass
    @abstractmethod
    def finalize(self, solution: "BaseSolution", history: List[float]) -> None:
        """Finalizes and displays the visualization after completion."""
        pass