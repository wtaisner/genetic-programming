from src.node import *
from abc import ABC


class GenericTree(ABC):
    def __init__(self, node: Node = None, left=None, right=None):
        self.node = node
        self.left = left
        self.right = right

    def print_tree(self, spacer: str = ""):
        print(f"{spacer}{self.node.label}")
        if self.left:
            self.left.print_tree(spacer + "--- ")
        if self.right:
            self.right.print_tree(spacer + "--- ")

    def apply(self):
        """
        compute the tree or sth
        :return:
        """
        pass
        # TODO:


class Tree(GenericTree):
    def __init__(self, node: Node = None, left: GenericTree = None, right: GenericTree = None):
        super().__init__(node, left, right)
        self.node = node
        self.left = left
        self.right = right


if __name__ == "__main__":
    tree = Tree()
    tree.node = Multiplication()
    tree.left = Tree(node=Addition())
    tree.left.right = Tree(node=Variable("y"))
    tree.left.left = Tree(node=Variable("x"))
    tree.right = Tree(node=Variable("y"))
    tree.print_tree()
