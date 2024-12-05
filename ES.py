import numpy as np
# you need to install this package `ioh`. Please see documentations here: 
# https://iohprofiler.github.io/IOHexp/ and
# https://pypi.org/project/ioh/
from ioh import get_problem, logger, ProblemClass
from GA import s4402146_s4436385
import matplotlib.pyplot as plt

budget = 50000
dimension = 10

num_offsprings= 7

# To make your results reproducible (not required by the assignment), you could set the random seed by
# `np.random.seed(some integer, e.g., 42)`

def gaussian():
    return np.random.normal(loc=0, scale=1)

def mutate_parent(parent:np.array ,num_offsprings:int) -> list[np.array]:
    offsprings= []
    tao= 1/ len(parent)
    parent_sigma= parent[-1]
    for _ in range(num_offsprings): 
        offspring_sigma = parent_sigma * np.exp(tao * gaussian())
        offspring= [xi + offspring_sigma * gaussian() for xi in parent[:-1]]
        offspring.append(offspring_sigma)
        offsprings.append(np.array(offspring))

    return offsprings

def studentnumber1_studentnumber2_ES(problem):
    # Initialize parent
    parent = np.random.random(size=problem.meta_data.n_variables) * 10 - 5
    parent_fitness = problem(parent)
    parent_sigma= 2
    parent= np.append(parent, parent_sigma)

    sliding_window= []

    # Store mutation rates and fitness values for plotting
    mutation_rates = []
    fitness_values = []

    # Evolutionary strategy loop
    while problem.state.evaluations < budget:
        mutation_rate = parent[-1]  # Assuming the mutation rate is the last variable


        # sliding_window.append(mutation_rate)
        # if len(sliding_window) >= 100:
        #     sliding_window.pop(0)
        #     min_sigma= min(sliding_window)
        #     max_sigma= max(sliding_window)
        #     range_sigma= max_sigma - min_sigma
        #     if range_sigma <= 0.1:
        #         mutation_rate += gaussian()* 10

        mutation_rates.append(mutation_rate)
        fitness_values.append(parent_fitness)

        # Generate offspring and evaluate fitness
        offsprings = mutate_parent(parent, num_offsprings)
        offsprings= [np.clip(offspring, -5, 5) for offspring in offsprings]
        
        offsprings_fitness = [problem(offspring[:-1].tolist()) for offspring in offsprings]
        #print(offsprings_fitness)

        # Select the best offspring
        max_offspring_index = np.argmin(offsprings_fitness)
        parent = offsprings[max_offspring_index]
        parent_fitness = offsprings_fitness[max_offspring_index]


    # Final plotting
    plt.figure(figsize=(10, 5))

    # Plot objective value over evaluations
    plt.subplot(1, 2, 1)
    plt.plot(fitness_values, label="Objective Value")
    plt.xlabel("Evaluations")
    plt.ylabel("Objective Value")
    plt.title("Objective Value Progression")
    plt.legend()

    # Plot mutation rate over evaluations
    plt.subplot(1, 2, 2)
    plt.plot(mutation_rates, label="Mutation Rate", color='orange')
    plt.xlabel("Evaluations")
    plt.ylabel("Mutation Rate")
    plt.title("Mutation Rate Progression")
    plt.legend()

    plt.tight_layout()
    plt.show()



def create_problem(fid: int):
    # Declaration of problems to be tested.
    problem = get_problem(fid, dimension=dimension, instance=1, problem_class=ProblemClass.BBOB)

    # Create default logger compatible with IOHanalyzer
    # `root` indicates where the output files are stored.
    # `folder_name` is the name of the folder containing all output. You should compress the folder 'run' and upload it to IOHanalyzer.
    l = logger.Analyzer(
        root="data",  # the working directory in which a folder named `folder_name` (the next argument) will be created to store data
        folder_name="run",  # the folder name to which the raw performance data will be stored
        algorithm_name="evolution strategy",  # name of your algorithm
        algorithm_info="Practical assignment part2 of the EA course",
    )
    # attach the logger to the problem
    problem.attach_logger(l)
    return problem, l


if __name__ == "__main__":
    # this how you run your algorithm with 20 repetitçions/independent run
    F23, _logger = create_problem(23)
    print(F23) 
    for run in range(20): 
        print(f"Running run {run}")
        studentnumber1_studentnumber2_ES(F23)
        F23.reset() # it is necessary to reset the problem after each independent run
    _logger.close() # after all runs, it is necessary to close the logger to make sure all data are written to the folder


