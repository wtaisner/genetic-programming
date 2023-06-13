import os
import multiprocessing as mp
from typing import List

import operator
import numpy as np
import pandas as pd
import seaborn as sns
from deap import gp
from matplotlib import pyplot as plt

from gcd_problem import GCDProblem
from dice_game_problem import DiceGameProblem
from possum_age_regressor import PossumRegressionProblem
from custom_operators import protected_div, minimal, maximal

import warnings
warnings.filterwarnings("ignore")


# TODO: save trees
def experiment_one_problem(problem_name: str, max_height: int, max_length: int, max_length_nont, operators: List, rep: int, folder_path: str,
                           path_csv: str):
    results = {'problem': [], 'op': [], 'height': [], 'initial_fitness': [], 'nonterminal_nodes': [], 'rep': [], 'path_baby': [], 'path_img': [], 'path_tree': [], 'mean': [],
               'median': [], 'std': [], 'min': [], 'max': []}
    cmap = sns.color_palette("coolwarm", as_cmap=True)
    cmap.set_bad("black")
    for op in operators:
        # print(len(op))
        if problem_name == 'gcd':
            problem = GCDProblem(2, '../gcd/gcd-edge.csv', height=max_height, length=max_length, init_default=False)  # TODO change path (or not)
        elif problem_name == 'dice':
            problem = DiceGameProblem(2, '../PSB2/datasets/dice-game/dice-game-edge.csv', length=max_length, init_default=False)  # TODO: change path
        elif problem_name == 'automl':
            problem = PossumRegressionProblem(9, "../possum/possum.csv", length=max_length, init_default=False)
        for o in op:
            problem.pset.addPrimitive(o, 2)
        hof, _, _, best_individuals = problem.run_evolution(num_generations=10)  # TODO: change num_generations
        real_heights = [(i, i.height, len([j for j in i if isinstance(j, gp.Primitive)])) for i in best_individuals if i.height > 2]
        unique_len = np.unique([i[2] for i in real_heights if i[2] < max_length_nont])
        mini, maxi = np.min(unique_len), np.max(unique_len)
        # 4 buckets
        buckets = [[mini + (i - 1) * (maxi - mini) / 4, mini + i * (maxi - mini) / 4] for i in range(1, 5)]
        buckets[0][0] -= 0.1
        buckets[3][1] += 0.1
        subgroups = [[], [], [], []]
        for i, bucket in enumerate(buckets):
            for ind in real_heights:
                if bucket[0] <= ind[2] < bucket[1]:
                    subgroups[i].append(ind)
        selected_individuals = []
        for subgroup in subgroups:
            selected = np.random.choice(range(len(subgroup)), size=min(len(subgroup), rep), replace=False)
            for s in selected:
                selected_individuals.append(subgroup[s])
        for i, selected in enumerate(selected_individuals):
            print(f"{problem_name} Len operators: {len(op)}, i: {i}, all: {len(selected_individuals)}")
            individual, height, nont_nodes = selected
            fitness = individual.fitness[0]
            path_tree = os.path.join(folder_path, f'tree_{problem_name}_{len(op)}_{nont_nodes}_{i}.png')
            problem.print_tree(individual, save_path=path_tree)
            baby_matrices, mean, ones, zeros, m_ones = problem.calculate_epistasis(
                individual, aggr='voting_mean', return_all=True)
            path_baby = os.path.join(folder_path, f'babies_{problem_name}_{len(op)}_{height}_{nont_nodes}_{i}.npy')
            path_img = os.path.join(folder_path, f'plots_{problem_name}_{len(op)}_{height}_{nont_nodes}_{i}.png')
            np.save(path_baby, baby_matrices)
            mean_norm = mean / np.max(np.abs(mean))
            max_val = np.max(np.absolute(mean_norm))[0]
            figsize = (2*len(mean_norm.index), len(mean_norm.index))
            fig, axes = plt.subplots(1, 2, figsize=figsize)
            sns.heatmap(mean_norm, ax=axes[0], vmin=-max_val, vmax=max_val, annot=True, fmt=".2f", cmap=cmap, mask=mean.isnull())
            axes[0].set(title='Aggr=voting_mean')
            cmap2 = plt.cm.get_cmap('OrRd')
            v_max = len(baby_matrices)
            labels = (np.asarray([f"p:{one}\n z:{zero}\n n:{mone}\n"
                                  for one, zero, mone in zip(ones.to_numpy().flatten(), zeros.to_numpy().flatten(),
                                                             m_ones.to_numpy().flatten())])
                      ).reshape(zeros.shape)
            labels = pd.DataFrame(labels, index=zeros.index, columns=zeros.columns)
            sns.heatmap(zeros, ax=axes[1], vmin=0, vmax=v_max, annot=labels, cmap=cmap2, fmt='')
            axes[1].set(title='Baby matrices sum-up')
            plt.savefig(path_img, dpi=300)
            results['problem'].append(problem_name)
            results['op'].append(len(op))
            results['height'].append(height)
            results['initial_fitness'].append(fitness)
            results['nonterminal_nodes'].append(nont_nodes)
            results['rep'].append(i)
            results['path_baby'].append(path_baby)
            results['path_img'].append(path_img)
            results['mean'].append(np.mean(mean.to_numpy()))
            results['median'].append(np.median(mean.to_numpy()))
            results['std'].append(np.std(mean.to_numpy()))
            results['max'].append(np.max(mean.to_numpy()))
            results['min'].append(np.min(mean.to_numpy()))
            results['path_tree'].append(path_tree)
    results = pd.DataFrame(results)
    results.to_csv(os.path.join(folder_path, path_csv))


if __name__ == '__main__':
    operators = [operator.add, operator.sub, operator.mul, protected_div, minimal, maximal]
    operators_subgroups = [operators[:i] for i in range(2, len(operators)+1)]
    operators_small = [operator.add, operator.sub, operator.mul, protected_div]
    operators_subgroups_small = [operators[:i] for i in range(2, len(operators)+1)]
    paths = ['../experiments/gcd', '../experiments/dice', '../experiments/automl', '../experiments/gcd_small', '../experiments/dice_small', '../experiments/automl_small']
    for path in paths:
        if not os.path.exists(path):
            os.mkdir(path)
    args = [
        ("gcd", 12, 150, 20, operators_subgroups, 4, '../experiments/gcd', 'gcd.csv'),
        ("dice", 12, 150, 20, operators_subgroups, 4, '../experiments/dice', 'dice.csv'),
        ("automl", 12, 150, 20, operators_subgroups, 4, '../experiments/automl', 'automl.csv'),
        ("gcd", 12, 100, 15, operators_subgroups_small, 4, '../experiments/gcd_small', 'gcd.csv'),
        ("dice", 12, 100, 15, operators_subgroups_small, 4, '../experiments/dice_small', 'dice.csv'),
        ("automl", 12, 100, 15, operators_subgroups_small, 4, '../experiments/automl_small', 'automl.csv'),
    ]
    with mp.Pool(mp.cpu_count()//2) as pool:
        pool.starmap(experiment_one_problem, args)
    # problem_name = 'gcd'  # 'gcd', 'dice' or 'automl'
    # rep = 1  # how many tree we select from one bucket (subgroup)
    # folder_path = f'../experiments/{problem_name}'
    # path_csv = f'{problem_name}.csv'  # path to csv
    # max_height = 12  # maximal height of the tree
    # max_length_overall = 50
    # max_length_nont = 10
    # experiment_one_problem(problem_name, max_height, max_length_overall, max_length_nont, operators_subgroups, rep=rep, folder_path=folder_path,
    #                        path_csv=path_csv)
