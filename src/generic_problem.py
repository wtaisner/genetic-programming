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
            height: int = 17,
            init_default: bool = True,
            additional_operators: List[Tuple[Callable, int]] = None,
            additional_statistics: List[Callable] = None,
    ):
        """

        :param num_variables:
        """
        self.pset = gp.PrimitiveSet("main", num_variables)
        if init_default:
            self._init_operators()

        stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
        stats_size = tools.Statistics(len)
        self.mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
        self._init_statistics()

        if additional_operators is not None:
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
        self.toolbox.register("expr", gp.genGrow, pset=self.pset, min_=1, max_=10)
        self.toolbox.register("individual", tools.initIterate, creator.Individual, self.toolbox.expr)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("compile", gp.compile, pset=self.pset)
        self.toolbox.register("select", tools.selTournament, tournsize=3)
        self.toolbox.register("mate", gp.cxOnePoint)
        self.toolbox.register("expr_mut", gp.genFull, min_=0, max_=4)
        self.toolbox.register("mutate", gp.mutUniform, expr=self.toolbox.expr_mut, pset=self.pset)
        self.toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=height))
        self.toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=height))

    def _aggregate(self, baby_matrices_epistasis: np.ndarray, ids, aggr: str):
        if aggr == 'absolute' or aggr == 'voting' or aggr == 'voting_mean':
            baby_matrices_epistasis = np.around(baby_matrices_epistasis, decimals=5)
            aggr_matrix = np.sign(baby_matrices_epistasis)
            ones = np.count_nonzero(aggr_matrix == 1, axis=0)
            m_ones = np.count_nonzero(aggr_matrix == -1, axis=0)
            zeros = np.count_nonzero(aggr_matrix == 0, axis=0)
            aggr_matrix = np.zeros(ones.shape)
            if aggr == 'absolute':
                for (x, y), value in np.ndenumerate(aggr_matrix):
                    if ones[x, y] == len(baby_matrices_epistasis):
                        aggr_matrix[x, y] = 1
                    elif m_ones[x, y] == len(baby_matrices_epistasis):
                        aggr_matrix[x, y] = -1
                    elif zeros[x, y] == len(baby_matrices_epistasis):
                        aggr_matrix[x, y] = 0
                    else:
                        aggr_matrix[x, y] = np.nan
            elif aggr == 'voting':
                for (x, y), value in np.ndenumerate(aggr_matrix):
                    if ones[x, y] > m_ones[x, y] and ones[x, y] > zeros[x, y]:
                        aggr_matrix[x, y] = 1
                    elif zeros[x, y] > m_ones[x, y] and zeros[x, y] > ones[x, y]:
                        aggr_matrix[x, y] = 0
                    elif m_ones[x, y] > ones[x, y] and m_ones[x, y] > zeros[x, y]:
                        aggr_matrix[x, y] = -1
                    else:
                        aggr_matrix[x, y] = np.nan
            elif aggr == 'voting_mean':
                for (x, y), value in np.ndenumerate(aggr_matrix):
                    if ones[x, y] > m_ones[x, y] and ones[x, y] > zeros[x, y]:
                        mean_positive, divide = 0, 0
                        for bm in baby_matrices_epistasis:
                            if bm[x, y] > 0:
                                mean_positive += bm[x, y]
                                divide += 1
                        aggr_matrix[x, y] = mean_positive / divide
                    elif zeros[x, y] > m_ones[x, y] and zeros[x, y] > ones[x, y]:
                        aggr_matrix[x, y] = 0
                    elif m_ones[x, y] > ones[x, y] and m_ones[x, y] > zeros[x, y]:
                        mean_negative, divide = 0, 0
                        for bm in baby_matrices_epistasis:
                            if bm[x, y] < 0:
                                mean_negative += bm[x, y]
                                divide += 1
                        aggr_matrix[x, y] = mean_negative / divide
                    else:
                        aggr_matrix[x, y] = np.nan
        elif aggr == 'mean':
            aggr_matrix = np.mean(baby_matrices_epistasis, axis=0)
        #aggr_matrix /= np.max(np.abs(aggr_matrix))
        aggr_data = pd.DataFrame(aggr_matrix, index=ids, columns=ids)
        return aggr_data

    def calculate_epistasis(self, individual: gp.PrimitiveTree, aggr: str = 'absolute', return_all: bool = False):
        """

        :param return_all:
        :param aggr:
        :param individual:
        :return:
        """
        all_operators = list(self.pset.primitives.values())[0]  # get all available operators
        # self.print_tree(individual)

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
        ids, ind = np.unique([i[0] for iis in list(pairs.values()) for i in iis], return_index=True)
        ids = ids[np.argsort(ind)]
        baby_matrices_epistasis = []
        # robimy produkt kartezjański, żeby uzyskać wszystkie możliwe kombinacje dla wartości w słowniku pairs
        for comb in product(*pairs.values()):
            zero_data = np.zeros((len(comb), len(comb)))
            data = pd.DataFrame(zero_data, index=ids, columns=ids)
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

                data.loc[idx_1, idx_2] = delta_fitness
                data.loc[idx_2, idx_1] = delta_fitness
            epistasis = data.to_numpy()
            diag_epistasis = np.array([epistasis.diagonal()])
            epistasis = epistasis - diag_epistasis
            epistasis = epistasis - diag_epistasis.T
            np.fill_diagonal(epistasis, 0)

            baby_matrices_epistasis.append(epistasis)
        baby_matrices_epistasis = np.array(baby_matrices_epistasis)

        if aggr == 'all':
            a1 = self._aggregate(baby_matrices_epistasis, ids, 'absolute')
            a2 = self._aggregate(baby_matrices_epistasis, ids, 'voting')
            a3 = self._aggregate(baby_matrices_epistasis, ids, 'mean')
            a4 = self._aggregate(baby_matrices_epistasis, ids, 'voting_mean')
            if return_all:
                aggr_matrix = np.sign(baby_matrices_epistasis)
                ones = np.count_nonzero(aggr_matrix == 1, axis=0)
                m_ones = np.count_nonzero(aggr_matrix == -1, axis=0)
                zeros = np.count_nonzero(aggr_matrix == 0, axis=0)
                ones = pd.DataFrame(ones, index=ids, columns=ids)
                m_ones = pd.DataFrame(m_ones, index=ids, columns=ids)
                zeros = pd.DataFrame(zeros, index=ids, columns=ids)
                return baby_matrices_epistasis, a1, a2, a3, a4, ones, zeros, m_ones
            else:
                return baby_matrices_epistasis, a1, a2, a3, a4

        a1 = self._aggregate(baby_matrices_epistasis, ids, aggr)
        if return_all:
            aggr_matrix = np.sign(baby_matrices_epistasis)
            ones = np.count_nonzero(aggr_matrix == 1, axis=0)
            m_ones = np.count_nonzero(aggr_matrix == -1, axis=0)
            zeros = np.count_nonzero(aggr_matrix == 0, axis=0)
            ones = pd.DataFrame(ones, index=ids, columns=ids)
            m_ones = pd.DataFrame(m_ones, index=ids, columns=ids)
            zeros = pd.DataFrame(zeros, index=ids, columns=ids)
            return baby_matrices_epistasis, a1, ones, zeros, m_ones
        else:
            return baby_matrices_epistasis, a1

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
        pop, log, best_individuals = eaSimple_modified(
            pop,
            self.toolbox,
            cxpb=cxpb,
            mutpb=mutpb,
            ngen_stop=num_generations,
            stats=self.mstats,
            halloffame=hof,
            verbose=True
        )
        return hof, pop, log, best_individuals

    @staticmethod
    def print_tree(individual: gp.PrimitiveTree, save_path: str = None) -> None:
        """
        Create a plot of an individual
        :param individual: instance of gp.PrimitiveTree created with self.toolbox.individual()
        :param save_path
        :return: None
        """
        fig, ax = plt.subplots()
        nodes, edges, labels = gp.graph(individual)
        g = nx.Graph()
        g.add_nodes_from(nodes)
        g.add_edges_from(edges)
        pos = nx.nx_agraph.graphviz_layout(g, prog="dot")

        nx.draw_networkx_nodes(g, pos)
        nx.draw_networkx_edges(g, pos)
        nx.draw_networkx_labels(g, pos, labels)
        if save_path is not None:
            plt.savefig(save_path, dpi=300)
        plt.show()

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
