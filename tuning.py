from typing import List

import numpy as np
# you need to install this package `ioh`. Please see documentations here: 
# https://iohprofiler.github.io/IOHexp/ and
# https://pypi.org/project/ioh/
from ioh import get_problem, logger, ProblemClass
from GA import s4402146_s4436385, create_problem
import matplotlib.pyplot as plt

budget = 5000000

# To make your results reproducible (not required by the assignment), you could set the random seed by
# `np.random.seed(some integer, e.g., 42)`
import random
from dataclasses import dataclass

from dataclasses import dataclass
import numpy as np

from skopt import gp_minimize
from skopt.space import Integer, Real
from skopt.utils import use_named_args

@dataclass
class HyperParams:
    pop_size: int = 100  # Population size (10, 300)
    num_offsprings: int = 50  # Derived: pop_size - num_elite_parents
    mutation_rate: float = 0.05  # Mutation rate (0.001, 0.1)
    crossover_points: int = 2  # Number of crossover points (1, 5)
    num_elite_parents: float = 0.5  # Ratio of elite parents to population size (0.25, 0.75)
    boltzman_temp: int = 100  # Boltzmann temperature (50, 200)

# Define the search space
space = [
    Integer(10, 300, name="pop_size"),                  # Population size
    Real(0.001, 0.1, name="mutation_rate"),             # Mutation rate
    Integer(1, 5, name="crossover_points"),             # Crossover points
    Real(50, 200, name="boltzman_temp"),                # Boltzmann temperature
    Real(0.25, 0.75, name="elite_ratio")                # Elite ratio
]

fitness_history = {
    "labs": [],
    "nqueens": [],
    "aggregated": []
}

curr_iteration= 1
max_iteration= 50

def evaluate_fitness(hyperparams):
    global curr_iteration
    print(f"Running iteration {curr_iteration} / {max_iteration}")
    curr_iteration += 1

    pop_size = hyperparams[0]
    mutation_rate = hyperparams[1]
    crossover_points = hyperparams[2]
    boltzman_temp = hyperparams[3]
    elite_ratio = hyperparams[4]

    # Derived parameters
    num_elite_parents = int(pop_size * elite_ratio)
    num_offsprings = pop_size - num_elite_parents

    hyper_params= HyperParams(pop_size, num_offsprings, mutation_rate, crossover_points, num_elite_parents, boltzman_temp)

    # LABS Problem
    F18, _logger18 = create_problem(dimension=50, fid=18)
    labs_fitness_values = []
    for run in range(20):
        best_fitness = s4402146_s4436385(F18, hyper_params)
        labs_fitness_values.append(best_fitness)
        F18.reset()
    _logger18.close()
    labs_mean_fitness = np.mean(labs_fitness_values)

    # N-Queens Problem
    F23, _logger23 = create_problem(dimension=49, fid=23)
    nqueens_fitness_values = []
    for run in range(20):
        best_fitness = s4402146_s4436385(F23, hyper_params)
        nqueens_fitness_values.append(best_fitness)
        F23.reset()
    _logger23.close()
    nqueens_mean_fitness = np.mean(nqueens_fitness_values)

    # Aggregate fitness
    aggregated_fitness = labs_mean_fitness + nqueens_mean_fitness

    # Store for plotting
    fitness_history["labs"].append(labs_mean_fitness)
    fitness_history["nqueens"].append(nqueens_mean_fitness)
    fitness_history["aggregated"].append(aggregated_fitness)

    return -aggregated_fitness  # Negative for minimization

# Map search space to the function
@use_named_args(space)
def objective(**params):
    hyperparams = [
        params["pop_size"],
        params["mutation_rate"],
        params["crossover_points"],
        params["boltzman_temp"],
        params["elite_ratio"]
    ]
    return evaluate_fitness(hyperparams)


# Hyperparameter tuning function
def tune_hyperparameters() -> HyperParams:
    # You should decide/engineer the `score` youself, which is the tuning objective
    best_score = float('inf')
    best_params = None


    # Bayesian Optimization
    result = gp_minimize(
        func=objective,
        dimensions=space,
        n_calls=max_iteration,
        n_random_starts=1,
        random_state=1342
    )

    # Output results
    best_hyperparams = result.x
    best_fitness = -result.fun  # Convert back to positive fitness
    # Calculate derived field `num_elite_parents`
    num_elite_parents = int(best_hyperparams[0] * best_hyperparams[4])  # pop_size * elite_ratio

    # Create the HyperParams object
    best_hyperparams_obj = HyperParams(
        pop_size=best_hyperparams[0],
        num_offsprings=best_hyperparams[0] - num_elite_parents,
        mutation_rate=best_hyperparams[1],
        crossover_points=best_hyperparams[2],
        num_elite_parents=num_elite_parents,
        boltzman_temp=int(best_hyperparams[3])
    )

    # Print the HyperParams object
    print(f"Best HyperParams Object: {best_hyperparams_obj} with aggregated fitness {best_fitness}")


    # Plot individual and aggregated fitness values
    plt.figure(figsize=(10, 6))
    plt.plot(fitness_history["labs"], label="LABS Fitness", marker="o")
    plt.plot(fitness_history["nqueens"], label="N-Queens Fitness", marker="s")
    plt.plot(fitness_history["aggregated"], label="Aggregated Fitness", marker="^")
    plt.xlabel("Iteration")
    plt.ylabel("Fitness Value")
    plt.title("Fitness Values Over Iterations")
    plt.legend()
    plt.grid(True)
    plt.show()

    return best_hyperparams




if __name__ == "__main__":
    # Hyperparameter tuning to determine the best parameters for both problems
    best_config = tune_hyperparameters()
