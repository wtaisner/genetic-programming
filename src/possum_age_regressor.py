from typing import Tuple

import pandas as pd
from deap import gp
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from src.generic_problem import Problem


class PossumRegressionProblem(Problem):
    def __init__(self, num_variables: int, csv_path: str, height: int = 17, init_default: bool = True):
        super().__init__(num_variables, height, init_default)
        csv_data = pd.read_csv(csv_path, sep=',')
        csv_data = csv_data.drop(columns=["case", "site", "Pop", "sex"], axis=1)
        csv_data.dropna(inplace=True)
        train, test = train_test_split(csv_data, train_size=0.9)
        self.inputs = train.drop("age", axis=1)
        self.inputs = [x.tolist() for idx, x in self.inputs.iterrows()]
        self.outputs = list(train['age'])

        self.toolbox.register("evaluate", self.evaluate)

    def evaluate(self, individual: gp.PrimitiveTree) -> Tuple:
        func = self.toolbox.compile(expr=individual)
        y_pred = [func(*x) for x in self.inputs]
        return mean_squared_error(self.outputs, y_pred),


if __name__ == "__main__":
    adam = PossumRegressionProblem(9, "../possum/possum.csv")
    individual = adam.toolbox.individual()
    adam.print_tree(individual)
    print(adam.evaluate(individual))
