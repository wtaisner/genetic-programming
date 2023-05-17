from gcd_problem import GCDProblem
from src.dice_game_problem import DiceGameProblem

if __name__ == "__main__":

    problem = GCDProblem(2, '../gcd/gcd-random.csv')
    print(problem.calculate_epistasis(problem.toolbox.individual()))
    # population = problem.toolbox.population(n=1)
    # total_fitness = 0
    # for ind in population:
    #     total_fitness += problem.toolbox.evaluate(ind)[0]
    # avg_fitness = total_fitness / len(population)
    #
    # for ind in population:
    #     avg_fitness += problem.calculate_epistasis(ind)
    #
    # print(avg_fitness)
    #
    # problem = DiceGameProblem(2, '../PSB2/datasets/dice-game/dice-game-random.csv')
    # population = problem.toolbox.population(n=10)
    # total_fitness = 0
    # for ind in population:
    #     total_fitness += problem.toolbox.evaluate(ind)[0]
    # avg_fitness = total_fitness / len(population)
    #
    # for ind in population:
    #     avg_fitness += problem.calculate_epistasis(ind)
    #
    # print(avg_fitness)
