from typing import Tuple 
import numpy as np
# you need to install this package `ioh`. Please see documentations here: 
# https://iohprofiler.github.io/IOHexp/ and
# https://pypi.org/project/ioh/
import ioh
from ioh import get_problem, logger, ProblemClass
from dataclasses import dataclass

budget = 5000

# To make your results reproducible (not required by the assignment), you could set the random seed by
# `np.random.seed(some integer, e.g., 42)`

@dataclass
class HyperParams:
    pop_size: int = 1000
    n_selections: int = 1000
    mutation_rate: float = 0.01
    crossover_probability: float = 0.1

hyper_params = HyperParams()

# Uniform Crossover
def crossover(p1, p2):
    return [np.random.choice([b1,b2]) for b1,b2 in zip(p1,p2)]

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
    # initial_pop = ... make sure you randomly create the first population
    parents= [np.random.randint(2, size = problem.meta_data.n_variables) for _ in range(hyper_params.pop_size)]

    ## TODO i might break
    parents_fitness= [problem(solution) for solution in parents]


    # `problem.state.evaluations` counts the number of function evaluation automatically,
    # which is incremented by 1 whenever you call `problem(x)`.
    # You could also maintain a counter of function evaluations if you prefer.

    while problem.state.evaluations < budget:
        # please implement the mutation, crossover, selection here
        selected_parents= mating_selection(parents, parents_fitness, hyper_params.n_selections)
        offsprings= [crossover(selected_parent[0], selected_parent[1]) for selected_parent in selected_parents]
        offsprings= [mutation(offspring, hyper_params.mutation_rate) for offspring in offsprings]

        parents= offsprings
        parents_fitness= [problem(solution) for solution in parents]

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
    F18, _logger = create_problem(dimension=50, fid=18)
    for run in range(20): 
        studentnumber1_studentnumber2_GA(F18)
        F18.reset() # it is necessary to reset the problem after each independent run
    _logger.close() # after all runs, it is necessary to close the logger to make sure all data are written to the folder

    # create the N-Queens problem and the data logger
    F23, _logger = create_problem(dimension=49, fid=23)
    for run in range(20): 
        studentnumber1_studentnumber2_GA(F23)
        F23.reset()
    _logger.close()