from typing import Tuple 
import numpy as np
# you need to install this package `ioh`. Please see documentations here: 
# https://iohprofiler.github.io/IOHexp/ and
# https://pypi.org/project/ioh/
import ioh
from ioh import get_problem, logger, ProblemClass
from dataclasses import dataclass
import matplotlib.pyplot as plt

budget = 5000

# To make your results reproducible (not required by the assignment), you could set the random seed by
# `np.random.seed(some integer, e.g., 42)`

@dataclass
class HyperParams:
    pop_size: int = 100
    n_selections: int = 200
    mutation_rate: float = 0.05
    crossover_probability: float = 0.1
    crossover_points: int= 2

hyper_params = HyperParams()



def crossover(p1, p2, num_crossover_points=1):
    """
    Perform crossover with a specified number of crossover points.

    Parameters:
    - p1: Parent 1 (list or array)
    - p2: Parent 2 (list or array)
    - num_crossover_points: Number of crossover points (int)

    Returns:
    - child1: The offspring generated from crossover
    """
    if num_crossover_points < 1 or num_crossover_points >= len(p1):
        raise ValueError("Number of crossover points must be between 1 and len(p1) - 1.")
    
    # Generate unique random crossover points and sort them
    crossover_points = sorted(np.random.choice(range(1, len(p1)), num_crossover_points, replace=False))
    
    # Perform crossover
    child1 = []
    toggle = True  # Toggles between parents
    last_point = 0
    for point in crossover_points:
        if toggle:
            child1.extend(p1[last_point:point])
        else:
            child1.extend(p2[last_point:point])
        toggle = not toggle
        last_point = point

    # Add the remaining segment from the toggled parent
    child1.extend(p1[last_point:] if toggle else p2[last_point:])
    
    return np.array(child1).astype(int).tolist()



# Standard bit mutation using mutation rate p
def mutation(p, mutation_rate):
    p_mutated = [b ^ 1 if np.random.rand() <= mutation_rate else b for b in p]
    return p_mutated

# Using the Fitness proportional selection
# TODO if i'm not learning chnage me to boltzman or smth
def mating_selection(parents, parents_f, n_selections) :   
    parents_f_np= np.array(parents_f)
    fitness_sum= sum(parents_f)
    parents_pdf = parents_f_np / fitness_sum

    selected_parents= []
    for _ in range(n_selections):
        parent_indices = np.arange(len(parents))
        p1_idx = np.random.choice(parent_indices, p=parents_pdf)
        p2_idx = np.random.choice(parent_indices, p=parents_pdf)

        # Select the parents based on the sampled indices
        p1 = parents[p1_idx]
        p2 = parents[p2_idx]
        selected_parents.append((p1,p2))

    return selected_parents



def studentnumber1_studentnumber2_GA(problem: ioh.problem.PBO) -> None:
    # Store fitness metrics
    best_fitness_per_generation = []
    avg_fitness_per_generation = []

    # Initial population
    parents = [np.random.randint(2, size=problem.meta_data.n_variables) for _ in range(hyper_params.pop_size)]
    parents_fitness = [problem(solution) for solution in parents]

    while problem.state.evaluations < budget:
        # Log fitness metrics
        best_fitness_per_generation.append(max(parents_fitness))
        avg_fitness_per_generation.append(np.mean(parents_fitness))

        # Selection, crossover, and mutation
        selected_parents = mating_selection(parents, parents_fitness, hyper_params.n_selections)
        offsprings = [crossover(p1, p2, hyper_params.crossover_points) for p1, p2 in selected_parents]
        offsprings = [mutation(offspring, hyper_params.mutation_rate) for offspring in offsprings]

        parents = offsprings + parents
        parents_fitness = [problem(solution) for solution in parents]

        # ELITIST SELECTION
        top_indices = np.argsort(parents_fitness)[-hyper_params.pop_size:]  # Keep the best individuals
        parents = [parents[i] for i in top_indices]
        parents_fitness = [parents_fitness[i] for i in top_indices]

    # Plot results
    plt.plot(best_fitness_per_generation, label="Best Fitness")
    plt.plot(avg_fitness_per_generation, label="Average Fitness")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.legend()
    plt.title("Fitness Progression")
    plt.show()


def create_problem(dimension: int, fid: int) -> Tuple[ioh.problem.PBO, ioh.logger.Analyzer]:
    # Declaration of problems to be tested.
    problem = get_problem(fid, dimension=dimension, instance=1, problem_class=ProblemClass.PBO)

    # Create default logger compatible with IOHanalyzer
    # `root` indicates where the output files are stored.
    # `folder_name` is the name of the folder containing all output. You should compress the folder 'run' and upload it to IOHanalyzer.
    l = logger.Analyzer(
        root="data",  # the working directory in which a folder named `folder_name` (the next argument) will be created to store data
        folder_name="run",  # the folder name to which the raw performance data will be stored
        algorithm_name="genetic_algorithm",  # name of your algorithm
        algorithm_info="Practical assignment of the EA course",
    )
    # attach the logger to the problem
    problem.attach_logger(l)
    return problem, l


if __name__ == "__main__":
    # this how you run your algorithm with 20 repetitions/independent run
    # create the LABS problem and the data logger

    ## CHAT GPT SETUP A COUNTING ONES PROBLEM TO SEE IF MU GA IMPLEMENTATION IS WORKING JUST GIVE ME THIS PART
    # Create a Counting Ones problem with dimension 50 and attach a logger



    F18, _logger = create_problem(dimension=50, fid=18)
    for run in range(20): 
        studentnumber1_studentnumber2_GA(F18)
        F18.reset() # it is necessary to reset the problem after each independent run
    _logger.close() # after all runs, it is necessary to close the logger to make sure all data are written to the folder

    exit()

    # create the N-Queens problem and the data logger
    F23, _logger = create_problem(dimension=49, fid=23)
    for run in range(20): 
        studentnumber1_studentnumber2_GA(F23)
        F23.reset()
    _logger.close()