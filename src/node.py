from abc import ABC, abstractmethod
from typing import Optional


class Node(ABC):
    """
    Abstract class to facilitate the utilization of nodes, i.e. storing names, adding new nodes, etc.
    """

    def __init__(self):
        self.label: str | None = None

    @abstractmethod
    def apply(self, x: float, y: Optional[float]) -> float:
        pass


class Variable(Node):
    def __init__(self, label):
        super().__init__()
        self.label = label

    def apply(self, x: float, y: Optional[float]) -> float:
        return x


class Addition(Node):
    def __init__(self):
        super().__init__()
        self.label = "add"

    def apply(self, x: float, y: float) -> float:
        return x + y


class Subtraction(Node):
    def __init__(self):
        super().__init__()
        self.label = "sub"

    def apply(self, x: float, y: float) -> float:
        return x - y


class Multiplication(Node):
    def __init__(self):
        super().__init__()
        self.label = "mul"

    def apply(self, x: float, y: float) -> float:
        return x * y


class Division(Node):
    def __init__(self):
        super().__init__()
        self.label = "div"

    def apply(self, x: float, y: float) -> float:
        assert y != 0, "y must be =/= 0"  # TODO: w sumie to może wywalać w kodzie, może trzeba jakieś 0.00001 dodawać xd
        return x / y
