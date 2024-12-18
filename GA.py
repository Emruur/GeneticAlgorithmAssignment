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
    pop_size: int = 55 #10,300
    num_offsprings: int = 14 # = pop_size - num_elite_parents
    mutation_rate: float = 0.0211 # 0.001 , 0.1
    crossover_points: int= 5# 1-5
    num_elite_parents: int = 41 # <= 3*pop_size/4 , >= pop_size/4  
    boltzman_temp: int= 200 # u figure it out chat gpt

#hyper_params_default = HyperParams(pop_size=10, num_offsprings=5, mutation_rate=0.030757723064132023, crossover_points=1, num_elite_parents=5, boltzman_temp=200)
#hyper_params_default = HyperParams(pop_size=46, num_offsprings=12, mutation_rate=0.020193969567736114, crossover_points=5, num_elite_parents=34, boltzman_temp=200)
hyper_params_default= HyperParams()

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

# Using Boltzmann selection
def mating_selection(parents, parents_f, n_selections, temperature):
    parents_f_np = np.array(parents_f)
    scaled_fitness = np.exp(parents_f_np / temperature)
    fitness_sum = np.sum(scaled_fitness)
    parents_pdf = scaled_fitness / fitness_sum  # Probability distribution for selection

    selected_parents = []
    for _ in range(n_selections):
        parent_indices = np.arange(len(parents))
        # Select parents based on the Boltzmann-scaled probabilities
        p1_idx = np.random.choice(parent_indices, p=parents_pdf)
        p2_idx = np.random.choice(parent_indices, p=parents_pdf)

        # Select the parents based on the sampled indices
        p1 = parents[p1_idx]
        p2 = parents[p2_idx]
        selected_parents.append((p1, p2))

    return selected_parents


def s4402146_s4436385(problem: ioh.problem.PBO, hyper_params: HyperParams= hyper_params_default, verbose= False, fid= None) -> float:

    best_fitness_per_generation = []
    avg_fitness_per_generation = []
    min_fitness_per_generation = []  # To track the minimum fitness
    unique_individuals_per_generation = []  # To track the number of unique individuals

    # Initial population
    parents = [np.random.randint(2, size=problem.meta_data.n_variables) for _ in range(hyper_params.pop_size)]
    parents_fitness = [problem(solution) for solution in parents]

    while problem.state.evaluations < budget:
        # Log fitness metrics
        best_fitness_per_generation.append(max(parents_fitness))
        avg_fitness_per_generation.append(np.mean(parents_fitness))
        min_fitness_per_generation.append(min(parents_fitness))  # Log the minimum fitness
        
        # Count unique individuals
        unique_individuals = len(np.unique(parents, axis=0))
        unique_individuals_per_generation.append(unique_individuals)

        # Selection, crossover, and mutation
        selected_parents = mating_selection(parents, parents_fitness, hyper_params.num_offsprings, hyper_params.boltzman_temp)
        offsprings = [crossover(p1, p2, hyper_params.crossover_points) for p1, p2 in selected_parents]
        offsprings = [mutation(offspring, hyper_params.mutation_rate) for offspring in offsprings]

        # Second approach to elitism: first select best parents than add babies
        top_indices = np.argsort(parents_fitness)[-hyper_params.num_elite_parents:]  # Keep the best individuals
        elite_parents = [parents[i] for i in top_indices]
        
        # New generation 
        num_offsprings_to_next_gen = hyper_params.pop_size - hyper_params.num_elite_parents
        selected_offsprings= offsprings[0:num_offsprings_to_next_gen]
        parents = elite_parents + selected_offsprings

        selected_parents_fitness= [parents_fitness[i] for i in top_indices]
        selected_offsprings_fitness= [problem(solution) for solution in selected_offsprings]
        parents_fitness =  selected_parents_fitness+ selected_offsprings_fitness


    best_solution_idx = np.argmax(parents_fitness)
    best_fitness = parents_fitness[best_solution_idx]
    if verbose:
        # Plot fitness metrics
        plt.figure(figsize=(10, 6))
        plt.plot(best_fitness_per_generation, label="Best Fitness")
        plt.plot(avg_fitness_per_generation, label="Average Fitness")
        plt.plot(min_fitness_per_generation, label="Min Fitness", linestyle="--")
        plt.xlabel("Generation")
        plt.ylabel("Fitness")
        plt.legend()
        plt.title("Fitness Progression")
        plt.show()

        # Plot unique individuals
        plt.figure(figsize=(10, 6))
        plt.plot(unique_individuals_per_generation, label="Unique Individuals", color="purple", linestyle="-.")
        plt.xlabel("Generation")
        plt.ylabel("Number of Unique Individuals")
        plt.legend()
        plt.title("Population Diversity Over Generations")
        plt.show()

        def plot_chessboard(solution, fitness):
            """
            Plots the N-Queens chessboard for a binary vector solution (0 indicates queen's position).
            
            :param solution: Binary vector representing the chessboard.
            :param fitness: Fitness score of the solution.
            """
            n = int(np.sqrt(len(solution)))  # Determine the size of the board (7 for a 7x7 board)
            board = np.ones((n, n))  # Create a white board (all cells initialized to 1)
            
            # Place queens where solution has 0s
            for idx, value in enumerate(solution):
                if value == 0:
                    row, col = divmod(idx, n)
                    board[row, col] = 0  # Indicate the queen's position with 0

            # Plot the chessboard
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.imshow(board, cmap="gray", extent=(0, n, 0, n))  # Chessboard grid
            
            # Draw the grid lines
            ax.set_xticks(np.arange(0, n + 1, 1))
            ax.set_yticks(np.arange(0, n + 1, 1))
            ax.grid(color='white', linestyle='-', linewidth=3)
            ax.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)
            
            # Add queens as red 'Q's
            for idx, value in enumerate(solution):
                if value == 0:
                    row, col = divmod(idx, n)
                    #ax.text(col + 0.5, n - row - 0.5, 'Q', ha='center', va='center', fontsize=20, color='red')

            plt.title(f"Best Solution Chessboard (Fitness: {fitness})", fontsize=16)
            plt.show()

        if fid == 23:
            best_solution = parents[best_solution_idx]
            best_fitness = parents_fitness[best_solution_idx]

            plot_chessboard(best_solution, best_fitness)

    return best_fitness

    



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

    # Create a Counting Ones problem with dimension 50 and attach a logger

    F18, _logger = create_problem(dimension=50, fid=18)
    for run in range(20): 
        best_fitness= s4402146_s4436385(F18, verbose=False)
        print(f"LABS run {run} best fitness {best_fitness}")
        F18.reset() # it is necessary to reset the problem after each independent run
    _logger.close() # after all runs, it is necessary to close the logger to make sure all data are written to the folder

    # create the N-Queens problem and the data logger
    F23, _logger = create_problem(dimension=49, fid=23)
    for run in range(20): 
        best_fitness= s4402146_s4436385(F23,verbose=False, fid=23)
        print(f"N-QUEENS run {run} best fitness {best_fitness}")
        F23.reset()
    _logger.close()