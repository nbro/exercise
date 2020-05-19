# https://www.rennard.org/alife/english/gavintrgb.html
# The Genetic Algorithm for finding the maxima of single-variable functions
#   (http://www.researchinventy.com/papers/v4i3/F043046054.pdf)
import os
import time
from os import path

import matplotlib.pyplot as plt
import numpy as np

np.set_printoptions(suppress=True)

GENE_MIN_VALUE = 0
GENE_MAX_VALUE = 100
GENE_MAX_VALUE_EPSILON = 0.1
GENE_MUTATION_RATE = 0.5
GENE_CROSSOVER_RATE = 0.8

MIN_POPULATION_SIZE = 2
MAX_POPULATION_SIZE = 100
POPULATION_REPRODUCTION_RATE = 1.0
NUM_GENERATIONS = 10000
REPLACEMENT_RATE = 0.99

INDIVIDUAL_SIZE = 3
WORST_INDIVIDUAL_FITNESS = 0


def is_valid_gene(gene):
    if GENE_MIN_VALUE <= gene <= GENE_MAX_VALUE:
        return True
    else:
        return False


def generate_initial_population(population_size=MAX_POPULATION_SIZE):
    return np.random.uniform(GENE_MIN_VALUE, GENE_MAX_VALUE + GENE_MAX_VALUE_EPSILON,
                             size=(population_size, INDIVIDUAL_SIZE))


def mutate(individual, gene_mutation_rate=GENE_MUTATION_RATE):
    assert individual.shape == (3,)

    def randomly_mutate(gene):
        if np.random.random() <= gene_mutation_rate:
            mutation = np.random.uniform(-GENE_MAX_VALUE, GENE_MAX_VALUE + 1)
            while not is_valid_gene(gene + mutation):
                mutation = np.random.uniform(-GENE_MAX_VALUE, GENE_MAX_VALUE + 1)
            return gene + mutation
        else:
            return gene

    for i, gene in enumerate(individual):
        individual[i] = randomly_mutate(gene)


def crossover(parent1, parent2, gene_crossover_rate=GENE_CROSSOVER_RATE):
    assert parent1.shape == parent2.shape == (3,)
    child1 = np.copy(parent1)
    child2 = np.copy(parent2)
    for i, (gene1, gene2) in enumerate(zip(child1, child2)):
        if np.random.random() <= gene_crossover_rate:
            child1[i] = gene2
            child2[i] = gene1
    return child1, child2


def get_local_maximum():
    return np.array([1, 1 / 3, np.sqrt(2 / np.e) / 3])


def evaluate_individual(individual):
    """Compute the value of the function f(x, y, z) = 2*x*z * exp(-x) - 2*y^3 + y^2 - 3*z^3 given a point of the
    domain, i.e. the individual.

    :param individual: a tuple of 3 elements (x, y, z).

    :return: the value of f at individual
    """
    assert individual.shape == (3,)
    x, y, z = individual
    return 2 * x * z * np.exp(-x) - 2 * (y ** 3) + (y ** 2) - 3 * (z ** 3)


def get_fitnesses(population, worst_individual_fitness=WORST_INDIVIDUAL_FITNESS):
    # This function performs "windowing" of the fitnesses values.
    assert population.shape[0] >= MIN_POPULATION_SIZE and population.shape[1] == INDIVIDUAL_SIZE
    assert worst_individual_fitness >= 0
    fitnesses = np.zeros(population.shape[0])
    for i, individual in enumerate(population):
        fitnesses[i] = evaluate_individual(individual)
    fitnesses = fitnesses + (np.abs(np.min(fitnesses)) + worst_individual_fitness)  # Windowing.
    assert all(i >= 0 for i in fitnesses)
    return fitnesses


def get_selection_probabilities(fitnesses):
    # Calculate the selection probabilities based on the "fitness proportionate selection" (aka roulette wheel
    # selection) strategy.
    assert all(x >= 0 for x in fitnesses)
    s = np.sum(fitnesses)
    selection_probabilities = fitnesses / s
    assert np.isclose(sum(selection_probabilities), 1)
    return selection_probabilities


def evaluate_gradient(individual):
    # An individual is fitter the closer the x returned by this function is to zero.
    assert individual.shape == (3,)
    x, y, z = individual
    # individual is a critical point of the function when dfdx, dfdy and dfdz are zero.
    dfdx = 2 * z * (np.exp(-x) * (1 - x))
    dfdy = 2 * y - 6 * (y ** 2)
    dfdz = 2 * x * np.exp(-x) - 9 * (z ** 2)
    return np.abs(dfdx) + np.abs(dfdy) + np.abs(dfdz)


def ga(population_size=MAX_POPULATION_SIZE,
       num_generations=NUM_GENERATIONS,
       population_reproduction_rate=POPULATION_REPRODUCTION_RATE,
       gene_crossover_rate=GENE_CROSSOVER_RATE,
       gene_mutation_rate=GENE_MUTATION_RATE,
       replacement_rate=REPLACEMENT_RATE,
       use_elitism=True):
    population = generate_initial_population(population_size)

    fitnesses = get_fitnesses(population)

    # We keep track of the best fitness at each generation.
    generations_best_fitness = np.zeros(num_generations)

    # We keep track of the best individual of the population across all generations.
    generations_best_individual = np.zeros((num_generations, INDIVIDUAL_SIZE))

    for generation in range(num_generations):

        if np.random.random() <= population_reproduction_rate:
            selection_probabilities = get_selection_probabilities(fitnesses)

            # Select two parents for mating according to their fitness.
            # indices contains the indices of the two chosen parent chromosomes in the population.
            # This is the "roulette wheel selection" approach.
            indices = np.random.choice(population_size, 2, replace=False, p=selection_probabilities)

            parent1 = population[indices[0]]
            parent2 = population[indices[1]]

            # Cross-over the 2 parents to produce 2 new children.
            child1, child2 = crossover(parent1, parent2, gene_crossover_rate=gene_crossover_rate)

            # Mutate the two children.
            mutate(child1, gene_mutation_rate=gene_mutation_rate)
            mutate(child2, gene_mutation_rate=gene_mutation_rate)

            # Replace the chromosomes in the population with the lowest fitness.
            if np.random.random() < replacement_rate:
                population[np.argmin(fitnesses)] = child1
                population[np.argmin(fitnesses)] = child2

            fitnesses = get_fitnesses(population)

        generations_best_individual[generation] = population[np.argmax(fitnesses)]
        generations_best_fitness[generation] = np.max(fitnesses)

    return generations_best_individual, generations_best_fitness


def array_map(x, f):
    return np.array(list(map(f, x)))


def plot_evolution(ys, xs=None, x_label="Generations", y_label="Best fitness", title=None):
    plt.figure()
    if xs is None:
        plt.plot(ys)
    else:
        plt.plot(xs, ys)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.ticklabel_format(useOffset=False, style='plain')
    plt.show()


def create_experiments_folder(experiments_folder_name="experiments", experiment_folder_name=None):
    if not path.exists(experiments_folder_name):
        os.makedirs(experiments_folder_name)
    if experiment_folder_name is None:
        experiment_folder_name = time.strftime("%Y-%m-%d-%H-%M-%S")
    experiment_folder_path = path.join(experiments_folder_name, experiment_folder_name)
    if not path.exists(experiment_folder_path):
        os.makedirs(experiment_folder_path)
    return experiment_folder_path


def experiment():
    experiment_folder_path = create_experiments_folder()

    generations_best_individual, generations_best_fitness = ga()

    generations_best_value = array_map(generations_best_individual, evaluate_individual)
    generations_best_gradient = array_map(generations_best_individual, evaluate_gradient)

    idx = np.argmax(generations_best_value)
    best_individual = generations_best_individual[idx]
    best_function_value = generations_best_value[idx]
    best_gradient = generations_best_gradient[idx]

    print("Best individual =", best_individual)
    print("Local maximum =", get_local_maximum())
    print("Best function value =", best_function_value)
    print("Best gradient =", best_gradient)

    np.savetxt(path.join(experiment_folder_path, "generations_best_individual.csv"), generations_best_individual)
    np.savetxt(path.join(experiment_folder_path, "generations_best_fitness.csv"), generations_best_fitness)
    np.savetxt(path.join(experiment_folder_path, "generations_best_value.csv"), generations_best_value)
    np.savetxt(path.join(experiment_folder_path, "generations_best_gradient.csv"), generations_best_gradient)
    np.savetxt(path.join(experiment_folder_path, "best_individual.csv"), best_individual)
    np.savetxt(path.join(experiment_folder_path, "best_function_value.csv"), np.array([best_function_value]))
    np.savetxt(path.join(experiment_folder_path, "best_gradient.csv"), np.array([best_gradient]))

    plot_evolution(generations_best_fitness,
                   title="Fitness of the best individual through the generations")
    plot_evolution(generations_best_value, y_label="f(x)",
                   title="Value of the best individual through the generations")
    plot_evolution(generations_best_gradient, y_label=r"$\nabla$ f(x)",
                   title="Gradient of the best individual through the generations")


if __name__ == '__main__':
    experiment()
