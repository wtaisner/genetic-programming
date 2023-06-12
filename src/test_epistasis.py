import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from gcd_problem import GCDProblem
from src.dice_game_problem import DiceGameProblem

if __name__ == "__main__":
    problem = GCDProblem(2, '../gcd/gcd-test.csv')
    baby_matrices, aggr_data_abs, aggr_data_v, aggr_data_mean, aggr_data_vm, ones, zeros, m_ones = problem.calculate_epistasis(problem.toolbox.individual(), aggr='all', return_all=True)
    print('Aggr=absolute')
    print(aggr_data_abs)
    cmap = sns.color_palette("coolwarm", as_cmap=True)
    cmap.set_bad("black")
    sns.heatmap(aggr_data_abs, vmin=-1, vmax=1, annot=True, cmap=cmap, mask=aggr_data_abs.isnull()).set(title='Aggr=absolute')
    #plt.savefig('../static/absolute.png', dpi=300)
    plt.show()
    print('Aggr=voting')
    print(aggr_data_v)
    sns.heatmap(aggr_data_v, vmin=-1, vmax=1, annot=True, cmap=cmap, mask=aggr_data_v.isnull()).set(title='Aggr=voting')
    #plt.savefig('../static/voting.png', dpi=300)
    plt.show()
    print('Aggr=mean')
    print(aggr_data_mean)
    max_val = np.max(np.absolute(aggr_data_mean))[0]
    sns.heatmap(aggr_data_mean, vmin=-max_val, vmax=max_val, annot=True, fmt=".1f", cmap=cmap).set(title='Aggr=mean')
    #plt.savefig('../static/mean.png', dpi=300)
    plt.show()
    print('Aggr=voting_mean')
    print(aggr_data_mean)
    max_val = np.max(np.absolute(aggr_data_vm))[0]
    sns.heatmap(aggr_data_vm, vmin=-max_val, vmax=max_val, annot=True, fmt=".1f", cmap=cmap, mask=aggr_data_vm.isnull()).set(title='Aggr=voting_mean')
    plt.show()

    cmap = plt.cm.get_cmap('OrRd')
    v_max = len(baby_matrices)
    print('Positive epistasis:')
    print(ones)
    sns.heatmap(ones, vmin=0, vmax=v_max, annot=True, cmap=cmap, fmt='d').set(title='Positive epistasis')
    #plt.savefig('../static/positive.png', dpi=300)
    plt.show()
    print('No epistasis:')
    print(zeros)
    sns.heatmap(zeros, vmin=0, vmax=v_max, annot=True, cmap=cmap, fmt='d').set(title='No epistasis')
    #plt.savefig('../static/no_ep.png', dpi=300)
    plt.show()
    print('Negative epistasis:')
    print(m_ones)
    sns.heatmap(m_ones, vmin=0, vmax=v_max, annot=True, cmap=cmap, fmt='d').set(title='Negative epistasis')
    #plt.savefig('../static/negative.png', dpi=300)
    plt.show()

    labels = (np.asarray([f"p:{one}\n z:{zero}\n n:{mone}\n"
                          for one, zero, mone in zip(ones.to_numpy().flatten(), zeros.to_numpy().flatten(), m_ones.to_numpy().flatten())])
              ).reshape(zeros.shape)
    labels = pd.DataFrame(labels, index=zeros.index, columns=zeros.columns)
    sns.heatmap(zeros, vmin=0, vmax=v_max, annot=labels, cmap=cmap, fmt='')
    plt.show()
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
