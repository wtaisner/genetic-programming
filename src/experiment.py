import os
from typing import List

import operator
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from gcd_problem import GCDProblem
from dice_game_problem import DiceGameProblem
from custom_operators import protected_div, minimal, maximal


#TODO: save trees
def experiment_one_problem_different_height(problem_name: str, heights: List[int], rep: int, folder_path: str,
                                            path_csv: str):
    results = {'height': [], 'rep': [], 'path_baby': [], 'path_img': [], 'coeffs': []}
    cmap = sns.color_palette("coolwarm", as_cmap=True)
    cmap.set_bad("black")
    for height in heights:
        for i in range(rep):
            if problem_name == 'gcd':
                problem = GCDProblem(2, '../gcd/gcd-custom.csv', height=height)  # ścieżka do ewentualnej zmiany
            elif problem_name == 'dice':
                problem = DiceGameProblem(2, '', height=height)  # TODO: napisać ścieżkę
            hof, _, _ = problem.run_evolution(num_generations=10) #TODO: change num_generations
            individual = hof[0]
            baby_matrices, mean, coeff = problem.calculate_epistasis(
                individual, aggr='voting_mean', return_all=False)
            path_baby = os.path.join(folder_path, f'{problem_name}_{height}_{i}.npy')
            path_img = os.path.join(folder_path, f'{problem_name}_{height}_{i}.png')
            np.save(path_baby, baby_matrices)
            max_val = np.max(np.absolute(mean))[0]
            sns.heatmap(mean, vmin=-max_val, vmax=max_val, annot=True, fmt=".1f", cmap=cmap).set(
                title='Aggr=voting_mean')
            plt.savefig(path_img, dpi=300)
            results['height'].append(height)
            #results['height_real'].append(individual.) - nie wiem, jak się sprawdz w deapie to, nie mogę znaleźć
            results['rep'].append(i)
            results['path_baby'].append(path_baby)
            results['path_img'].append(path_img)
            results['coeffs'].append(coeff)
    results = pd.DataFrame(results)
    results.to_csv(os.path.join(folder_path, path_csv))


def experiment_one_problem_different_operators(problem_name: str, operators: List, rep: int, folder_path: str,
                                               path_csv: str):
    results = {'op': [], 'rep': [], 'path_baby': [], 'path_img': [], 'coeffs': []}
    cmap = sns.color_palette("coolwarm", as_cmap=True)
    cmap.set_bad("black")
    for op in operators:
        for i in range(rep):
            if problem_name == 'gcd':
                problem = GCDProblem(2, '../gcd/gcd-custom.csv', init_default=False)  # ścieżka do ewentualnej zmiany
            elif problem_name == 'dice':
                problem = DiceGameProblem(2, '', init_default=False)  # TODO: napisać ścieżkę
            for o in op:
                problem.pset.addPrimitive(o, 2)
            hof, _, _ = problem.run_evolution(num_generations=10)  # TODO: change num_generations
            individual = hof[0]
            baby_matrices, mean, coeff = problem.calculate_epistasis(
                individual, aggr='voting_mean', return_all=False)
            path_baby = os.path.join(folder_path, f'{problem_name}_{len(op)}_{i}.npy')
            path_img = os.path.join(folder_path, f'{problem_name}_{len(op)}_{i}.png')
            np.save(path_baby, baby_matrices)
            max_val = np.max(np.absolute(mean))[0]
            sns.heatmap(mean, vmin=-max_val, vmax=max_val, annot=True, fmt=".1f", cmap=cmap).set(
                title='Aggr=voting_mean')
            plt.savefig(path_img, dpi=300)
            results['op'].append(len(op))
            results['rep'].append(i)
            results['path_baby'].append(path_baby)
            results['path_img'].append(path_img)
            results['coeffs'].append(coeff)
    results = pd.DataFrame(results)
    results.to_csv(os.path.join(folder_path, path_csv))


def experiment_many_problems(problem_names: List[str], rep: int, folder_path: str, path_csv: str):
    results = {'problem': [], 'rep': [], 'path_baby': [], 'path_img': [], 'coeffs': []}
    cmap = sns.color_palette("coolwarm", as_cmap=True)
    cmap.set_bad("black")
    for problem_name in problem_names:
        if problem_name == 'gcd':
            problem = GCDProblem(2, '../gcd/gcd-custom.csv')  # ścieżka do ewentualnej zmiany
        elif problem_name == 'dice':
            problem = DiceGameProblem(2, '')  # TODO: napisać ścieżkę
        for i in range(rep):
            hof, _, _ = problem.run_evolution(num_generations=10)  # TODO: change num_generations
            individual = hof[0]
            baby_matrices, mean, coeff = problem.calculate_epistasis(
                individual, aggr='voting_mean', return_all=False)
            path_baby = os.path.join(folder_path, f'many_{problem_name}_{i}.npy')
            path_img = os.path.join(folder_path, f'many_{problem_name}_{i}.png')
            np.save(path_baby, baby_matrices)
            max_val = np.max(np.absolute(mean))[0]
            sns.heatmap(mean, vmin=-max_val, vmax=max_val, annot=True, fmt=".1f", cmap=cmap).set(
                title='Aggr=voting_mean')
            plt.savefig(path_img, dpi=300)
            results['problem'].append(problem_name)
            results['rep'].append(i)
            results['path_baby'].append(path_baby)
            results['path_img'].append(path_img)
            results['coeffs'].append(coeff)
    results = pd.DataFrame(results)
    results.to_csv(os.path.join(folder_path, path_csv))


if __name__ == '__main__':
    rep = 2
    heights = [2, 3, 4]
    operators = [operator.add, operator.mul, operator.sub, protected_div, minimal, maximal]

    operators_subgroups = [operators[:i] for i in range(2, len(operators))]

    experiment_one_problem_different_height('gcd', heights, rep=rep, folder_path='../experiments', path_csv='gcd_heights.csv')
    #experiment_one_problem_different_operators('gcd', operators_subgroups, rep=rep, folder_path='../experiments', path_csv='gcd_operators.csv')
    #experiment_many_problems(['gcd', 'dice'], rep=rep, folder_path='../experiments', path_csv='many_gcd_dice.csv')
