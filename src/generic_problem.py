import math
import operator
import random
from abc import ABC, abstractmethod
from typing import Callable, List, Tuple
from copy import deepcopy
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import numpy as np
import pandas as pd
from deap import (
    base,
    creator,
    tools,
    algorithms,
    gp
)

from custom_algorithms import eaSimple_modified
from custom_operators import *


class Problem(ABC):

    def __init__(
            self,
            num_variables: int,
            additional_operators: List[Tuple[Callable, int]] = None,
            additional_statistics: List[Callable] = None
    ):
        """

        :param num_variables:
        """
        self.pset = gp.PrimitiveSet("main", num_variables)
        self._init_operators()

        stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
        stats_size = tools.Statistics(len)
        self.mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
        self._init_statistics()

        if additional_statistics is not None:
            for func, arity in additional_operators:
                self.add_operator(func, arity)
        if additional_statistics is not None:
            for name, func in additional_statistics:
                self.add_statistics(name, func)

        # można podmienić nazwy, ale chyba worthless
        # self.pset.renameArguments(ARG0="x")
        # self.pset.renameArguments(ARG1="y")

        # fitness jest minimalizowany/maksymalizowany w zależności od wag (ujemne = min), wagi muszą być tuplem, może być wiele kryteriów
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))

        # jak będzie wyglądało pojedyncze rozwiązanie, typ, itp
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin, pset=self.pset)

        # ogólnie każda metoda create/register dodaje jakby nową metodę do obiektu, więc każdy sprawdzacz kodu będzie płakał
        self.toolbox = base.Toolbox()
        self.toolbox.register("expr", gp.genFull, pset=self.pset, min_=1, max_=4)
        self.toolbox.register("individual", tools.initIterate, creator.Individual, self.toolbox.expr)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("compile", gp.compile, pset=self.pset)
        self.toolbox.register("select", tools.selTournament, tournsize=3)
        self.toolbox.register("mate", gp.cxOnePoint)
        self.toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
        self.toolbox.register("mutate", gp.mutUniform, expr=self.toolbox.expr_mut, pset=self.pset)
        self.toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
        self.toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))

    def calculate_epistasis(self, individual: gp.PrimitiveTree):
        """

        :param individual:
        :return:
        """
        initial_fitness = self.evaluate(individual)
        print(f"Initial fitness: {initial_fitness}")
        # TODO: determine how to represent a matrix and calculate and fill it accordingly
        all_operators = list(self.pset.primitives.values())[0]
        data = []
        for idx, op in enumerate(individual):
            if isinstance(op, gp.Primitive):
                for op2 in all_operators:
                    tmp_individual = deepcopy(individual)
                    tmp_individual[idx] = op2
                    fitness_change = self.evaluate(tmp_individual)[0] - initial_fitness[0]
                    print(
                        f"Idx {idx} Original operator: {op.name}, switched to: {op2.name}, fitness change: {fitness_change}")
                    data.append([f"{idx}_{op.name}", f"{idx}_{op2.name}", fitness_change])
        df = pd.DataFrame(data, columns=["original_operator", "changed_operator", "fitness_change"])
        df = df.pivot(index="original_operator", columns="changed_operator", values="fitness_change").fillna(0)

        self.print_tree(individual)
        sns.heatmap(df, annot=True, cmap="seismic")
        plt.show()

    @abstractmethod
    def evaluate(self, individual: gp.PrimitiveTree):
        pass

    def run_evolution(
            self,
            population_size: int = 300,
            cxpb: float = 0.5,
            mutpb: float = 0.1,
            num_generations: int = 40
    ) -> Tuple:
        pop = self.toolbox.population(n=population_size)
        hof = tools.HallOfFame(1)
        pop, log = eaSimple_modified(
            pop,
            self.toolbox,
            cxpb=cxpb,
            mutpb=mutpb,
            ngen_stop=num_generations,
            stats=self.mstats,
            halloffame=hof,
            verbose=True
        )
        return hof, pop, log

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

    def _init_operators(self):
        self.pset.addPrimitive(operator.add, 2)
        self.pset.addPrimitive(operator.mul, 2)
        self.pset.addPrimitive(operator.sub, 2)
        # self.pset.addPrimitive(protected_div, 2)
        self.pset.addPrimitive(operator.neg, 1)
        self.pset.addPrimitive(math.cos, 1)
        self.pset.addPrimitive(math.sin, 1)

        # na razie zostawiam, potem wyrzuci się najwyżej albo zmieni
        self.pset.addEphemeralConstant("rand101", lambda: random.randint(-1, 1))

    def add_operator(self, func: Callable, arity: int):
        """

        :param func:
        :param arity:
        :return:
        """
        self.pset.addPrimitive(func, arity)

    def _init_statistics(self) -> None:
        """

        """
        self.mstats.register("avg", np.mean)
        self.mstats.register("std", np.std)
        self.mstats.register("min", np.min)
        self.mstats.register("max", np.max)

    def add_statistics(self, name: str, func: Callable) -> None:
        """

        :param name:
        :param func:
        :return:
        """
        self.mstats.register(name, func)
