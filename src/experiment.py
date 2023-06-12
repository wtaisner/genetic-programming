import os
from typing import List

import operator
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from gcd_problem import GCDProblem
from dice_game_problem import DiceGameProblem
from possum_age_regressor import PossumRegressionProblem
from custom_operators import protected_div, minimal, maximal

import warnings
warnings.filterwarnings("ignore")


# TODO: save trees
def experiment_one_problem(problem_name: str, max_height: int, operators: List, rep: int, folder_path: str,
                           path_csv: str):
    results = {'problem': [], 'op': [], 'height': [], 'rep': [], 'path_baby': [], 'path_img': [], 'path_tree': [], 'mean': [],
               'median': [], 'std': [], 'min': [], 'max': []}
    cmap = sns.color_palette("coolwarm", as_cmap=True)
    cmap.set_bad("black")
    for op in operators:
        print(len(op))
        if problem_name == 'gcd':
            problem = GCDProblem(2, '../gcd/gcd-edge.csv', height=max_height, init_default=False)  # TODO change path (or not)
        elif problem_name == 'dice':
            problem = DiceGameProblem(2, '', height=max_height, init_default=False)  # TODO: change path
        elif problem_name == 'automl':
            problem = PossumRegressionProblem(9, "../possum/possum.csv", height=max_height, init_default=False)
        for o in op:
            problem.pset.addPrimitive(o, 2)
        hof, _, _, best_individuals = problem.run_evolution(num_generations=10)  # TODO: change num_generations
        real_heights = [(i, i.height) for i in best_individuals if i.height > 2]
        unique_heights = np.unique([i[1] for i in real_heights])
        mini, maxi = np.min(unique_heights), np.max(unique_heights)
        # 4 buckets
        buckets = [[mini + (i - 1) * (maxi - mini) / 4, mini + i * (maxi - mini) / 4] for i in range(1, 5)]
        buckets[0][0] -= 0.1
        buckets[3][1] += 0.1
        subgroups = [[], [], [], []]
        for i, bucket in enumerate(buckets):
            for ind in real_heights:
                if bucket[0] <= ind[1] < bucket[1]:
                    subgroups[i].append(ind)
        selected_individuals = []
        for subgroup in subgroups:
            selected = np.random.choice(range(len(subgroup)), size=min(len(subgroup), rep), replace=False)
            for s in selected:
                selected_individuals.append(subgroup[s])
        for i, selected in enumerate(selected_individuals):
            individual, height = selected
            path_tree = os.path.join(folder_path, f'tree_{problem_name}_{len(op)}_{height}_{i}.png')
            problem.print_tree(individual, save_path=path_tree)
            baby_matrices, mean, ones, zeros, m_ones = problem.calculate_epistasis(
                individual, aggr='voting_mean', return_all=True)
            path_baby = os.path.join(folder_path, f'babies_{problem_name}_{len(op)}_{height}_{i}.npy')
            path_img = os.path.join(folder_path, f'plots_{problem_name}_{len(op)}_{height}_{i}.png')
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
    rep = 4  # how many tree we select from one bucket (subgroup)
    folder_path = '../experiments/gcd'
    path_csv = 'gcd.csv'  # path to csv
    max_height = 12  # maximal height of the tree
    problem_name = 'gcd'  # 'gcd', 'dice' or 'automl'
    operators = [operator.add, operator.sub, operator.mul, protected_div, minimal, maximal]
    operators_subgroups = [operators[:i] for i in range(2, len(operators))]
    experiment_one_problem('gcd', max_height, operators_subgroups, rep=rep, folder_path=folder_path,
                           path_csv=path_csv)
