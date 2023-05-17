import math
import operator
import random
from abc import ABC, abstractmethod
from copy import deepcopy
from itertools import product
from typing import Callable, List, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from deap import (
    base,
    creator,
    tools,
    gp
)

from custom_algorithms import eaSimple_modified


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

        # fitness jest minimalizowany/maksymalizowany w zależności od wag (ujemne = min),
        # wagi muszą być tuplem, może być wiele kryteriów
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))

        # jak będzie wyglądało pojedyncze rozwiązanie, typ, itp
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin, pset=self.pset)

        # ogólnie każda metoda create/register dodaje jakby nową metodę do obiektu,
        # więc każdy sprawdzacz kodu będzie płakał
        self.toolbox = base.Toolbox()
        self.toolbox.register("expr", gp.genFull, pset=self.pset, min_=1, max_=3)
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
        all_operators = list(self.pset.primitives.values())[0]  # get all available operators
        self.print_tree(individual)

        # pairs to dict, w którym kluczem jest indeks operatora,
        # a wartością lista par (indeks, operator_oryginalny, operator_zmieniony)
        pairs = dict()
        for idx, op in enumerate(individual):
            if isinstance(op, gp.Primitive):  # exclude terminals
                pairs[idx] = []
                for op2 in all_operators:
                    if op.name != op2.name:
                        pairs[idx].append(
                            [idx, op, op2])  # TODO: jeśli chcesz łatwo debugować, dopisz .name do każdego op*, wywali tylko liczenie fitnessu

        baby_matrices_epistasis = []
        # robimy produkt kartezjański, żeby uzyskać wszystkie możliwe kombinacje dla wartości w słowniku pairs
        for comb in product(*pairs.values()):
            data = []
            initial_fitness = self.evaluate(individual)[0]
            # znowu robimy produkt, tym razem dla każdej pary z comb, żeby porównać każdy z każdym
            for comb_1, comb_2 in product(comb, repeat=2):
                tmp_individual = deepcopy(individual)
                idx_1, op_1_original, op_1_changed = comb_1
                idx_2, op_2_original, op_2_changed = comb_2
                if idx_1 == idx_2:  # jeśli to ta sama para, to zmieniamy tylko jeden operator
                    tmp_individual[idx_1] = op_1_changed
                else:  # jeśli to różne pary, to zmieniamy oba operatory
                    tmp_individual[idx_1] = op_1_changed
                    tmp_individual[idx_2] = op_2_changed
                delta_fitness = initial_fitness - self.evaluate(tmp_individual)[0]

                # trochę z braku pomysłu, ale indeks daję na pałę string, działa to nie ruszajmy niczego :)
                data.append(
                    [f"({op_1_original.name}, {op_1_changed.name})", f"({op_2_original.name}, {op_2_changed.name})",
                     delta_fitness])

            df = pd.DataFrame(data, columns=["comb_1", "comb_2", "fitness_change"])
            # print(df)
            epistasis = 0
            for idx, (comb_1, comb_2, fitness_change) in df.iterrows():
                if comb_1 == comb_2:  # tutaj pomijamy te same pary
                    continue
                else:
                    # jak sobie odkomentujesz printy to możesz sprawdzić, czy dobre wartości odejmuje (ale musisz też df wyprintować)
                    # print(comb_1, comb_2)
                    tmp_epistasis = fitness_change - df.loc[
                        (df["comb_1"] == comb_1) & (df["comb_2"] == comb_1), "fitness_change"].values[0] \
                                    - df.loc[
                                        (df["comb_1"] == comb_2) & (df["comb_2"] == comb_2), "fitness_change"].values[0]
                    # print(df.loc[(df["comb_1"] == comb_1) & (df["comb_2"] == comb_1), "fitness_change"].values[0])
                    # print(df.loc[(df["comb_1"] == comb_2) & (df["comb_2"] == comb_2), "fitness_change"].values[0])
                    # print("-------------------------------")
                epistasis += tmp_epistasis
            # w tym momencie epistasis to epistaza małej macierzy
            baby_matrices_epistasis.append(epistasis)
        return np.array(baby_matrices_epistasis)

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
        # plt.savefig("tmp_tree.png", dpi=300)
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
