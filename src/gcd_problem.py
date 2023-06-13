import math
import operator
from typing import Tuple

import pandas as pd
from deap import gp

from src.custom_operators import protected_div
from src.generic_problem import Problem


class GCDProblem(Problem):
    def __init__(self, num_variables: int, csv_path: str, height: int = 17, length: int = 150, init_default: bool = True):
        super().__init__(num_variables, height, length, init_default)
        csv_data = pd.read_csv(csv_path, sep=',')
        self.inputs = [[i1, i2] for i1, i2 in zip(csv_data['input1'], csv_data['input2'])]
        self.outputs = list(csv_data['output1'])

        self.toolbox.register("evaluate", self.evaluate)

    def _init_operators(self):
        self.pset.addPrimitive(operator.add, 2)
        self.pset.addPrimitive(operator.mul, 2)
        self.pset.addPrimitive(operator.sub, 2)
        self.pset.addPrimitive(protected_div, 2)
        # self.pset.addPrimitive(int_gcd, 2)

    def evaluate(self, individual: gp.PrimitiveTree) -> Tuple:
        func = self.toolbox.compile(expr=individual)
        sqerrors = (abs(func(*in_) - out) for in_, out in zip(self.inputs, self.outputs))
        # sqerrors = ((func(*in_) - out) ** 2 for in_, out in zip(self.inputs, self.outputs))
        return math.fsum(sqerrors) / len(self.outputs),
