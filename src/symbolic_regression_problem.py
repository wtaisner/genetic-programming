from generic_problem import *


class SymbolicRegressionProblem(Problem):

    def __init__(self, num_variables: int):
        super().__init__(num_variables)

        self.toolbox.register("evaluate", self.evaluate, points=[x / 10. for x in range(-10, 10)])

    def evaluate(self, individual: gp.PrimitiveTree, points: List) -> Tuple:
        """
        Przykład dla symbolic regression
        :param points:
        :param individual:
        :return:
        """
        # Transform the tree expression in a callable function
        func = self.toolbox.compile(expr=individual)
        # Evaluate the mean squared error between the expression
        # and the real function : x**4 + x**3 + x**2 + x
        sqerrors = ((func(x) - x ** 4 - x ** 3 - x ** 2 - x) ** 2 for x in points)
        return math.fsum(sqerrors) / len(points),


if __name__ == "__main__":
    problem = SymbolicRegressionProblem(1)
    problem.run_evolution()
