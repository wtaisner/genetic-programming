import operator
from typing import Tuple
import matplotlib.pyplot as plt
import networkx as nx

from deap import (
    base,
    gp,
    creator,
    tools
)


class Solution:  # TODO: nie wiedziałem jak nazwać tę klasę też xd

    def __init__(self, num_variables: int):
        """

        :param num_variables:
        """
        self.pset = gp.PrimitiveSet("main", num_variables)
        self._add_operators()

        # można podmienić nazwy, ale chyba worthless
        # self.pset.renameArguments(ARG0="x")
        # self.pset.renameArguments(ARG1="y")

        # fintess jest minimalizowny/maksymalizowany w zależności od wag (ujemne = min), wagi muszą być tuplem, może być
        # wiele kryteriów
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))

        # jak będzie wyglądało pojedyncze rozwiązanie, typ, itp
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin, pset=self.pset)

        # ogólnie każda metoda create/register dodaje jakby nową metodę do obiektu, więc każdy sprawdzacz kodu będzie płakał
        self.toolbox = base.Toolbox()
        self.toolbox.register("expr", gp.genFull, pset=self.pset, min_=1, max_=4)

        self.toolbox.register("individual", tools.initIterate, creator.Individual, self.toolbox.expr)

        # TEST
        ind = self.toolbox.individual()
        print(str(ind))
        function = gp.compile(ind, self.pset)
        print(function(2, 1))
        self.print_tree(ind)

    @staticmethod
    def print_tree(individual: gp.PrimitiveTree) -> None:
        """
        Create a plot of an individual
        :param individual: instance of gp.PrimitiveTree created with self.toolbox.individual()
        :return: None
        """
        nodes, edges, labels = gp.graph(individual)
        g = nx.Graph()
        g.add_nodes_from(nodes)
        g.add_edges_from(edges)
        pos = nx.nx_agraph.graphviz_layout(g, prog="dot")

        nx.draw_networkx_nodes(g, pos)
        nx.draw_networkx_edges(g, pos)
        nx.draw_networkx_labels(g, pos, labels)
        plt.show()
        # TODO: może dodać zapis do pliku czy coś

    def _add_operators(self):
        self.pset.addPrimitive(operator.add, 2)
        self.pset.addPrimitive(operator.mul, 2)
        self.pset.addPrimitive(operator.sub, 2)
        # TODO: można rozwinąć / dodać np. listę operatorów do zaincjalizowania, nie wiem

    @staticmethod
    def evaluate(individual: gp.PrimitiveTree) -> Tuple:
        """
        # TODO: ogólnie to potem będzie określanie fitnessu -> pewnie trzeba będzie zrobić to problem-specific
        https://deap.readthedocs.io/en/master/tutorials/basic/part2.html#evaluation
        :param individual:
        :return:
        """
        print(individual)
        return tuple(sum(individual))


if __name__ == "__main__":
    adam = Solution(2)
