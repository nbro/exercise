import matplotlib.pyplot as plt
import numpy as np

GENE_MIN_VALUE = 0
GENE_MAX_VALUE = 100
GENE_MUTATION_RATE = 0.5
GENE_CROSSOVER_RATE = 0.1

MIN_POPULATION_SIZE = 2
MAX_POPULATION_SIZE = 100
POPULATION_REPRODUCTION_RATE = 1.0
INDIVIDUAL_SIZE = 3
NUM_GENERATIONS = 2000
SURVIVAL_RATE = 0.95


def is_valid_gene(gene):
    if GENE_MIN_VALUE <= gene <= GENE_MAX_VALUE:
        return True
    else:
        return False


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


def get_loss(individual):
    # An individual is fitter the closer the value returned by this function is to zero.
    assert individual.shape == (3,)
    x, y, z = individual
    # individual is a critical point of the function when dfdx, dfdy and dfdz are zero.
    dfdx = 2 * z * (np.exp(-x) * (1 - x))
    dfdy = 2 * y - 6 * (y ** 2)
    dfdz = 2 * x * np.exp(-x) - 9 * (z ** 2)
    return np.abs(dfdx) + np.abs(dfdy) + np.abs(dfdz)


def generate_initial_population(population_size=MAX_POPULATION_SIZE):
    return np.random.randint(GENE_MIN_VALUE, GENE_MAX_VALUE + 1, size=(population_size, INDIVIDUAL_SIZE))


def loss_to_probability(loss):
    # When the loss is zero, the probability of selection is 1.
    return 1 / (1 + loss ** 2)


def calculate_fitnesses(population):
    fitnesses = np.zeros(population.shape[0])
    for i, individual in enumerate(population):
        fitnesses[i] = loss_to_probability(get_loss(individual))
    return fitnesses


def calculate_selection_probabilities(fitnesses):
    # Calculate the selection probabilities based on the "fitness proportionate selection" (aka roulette wheel
    # selection) strategy.
    s = np.sum(fitnesses)
    selection_probabilities = fitnesses / s
    assert np.isclose(sum(selection_probabilities), 1)
    return selection_probabilities


def func(individual):
    assert individual.shape == (3,)
    x, y, z = individual
    return 2 * x * z * np.exp(-x) - 2 * (y ** 3) + (y ** 2) - 3 * (z ** 3)


def ga(population_size=MAX_POPULATION_SIZE,
       num_generations=NUM_GENERATIONS,
       population_reproduction_rate=POPULATION_REPRODUCTION_RATE,
       gene_crossover_rate=GENE_CROSSOVER_RATE,
       gene_mutation_rate=GENE_MUTATION_RATE,
       survival_rate=SURVIVAL_RATE):
    population = generate_initial_population(population_size)

    fitnesses = calculate_fitnesses(population)

    # We keep track of the best fitness at each generation.
    generations_best_fitness = np.zeros(num_generations)

    # We keep track of the best individual of the population across all generations.
    generations_best_individual = np.zeros((num_generations, INDIVIDUAL_SIZE))

    for generation in range(num_generations):

        selection_probabilities = calculate_selection_probabilities(fitnesses)

        if np.random.random() <= population_reproduction_rate:
            # Select two parents for mating according to their fitness.
            # indices contains the indices of the two chosen parent chromosomes in the population.
            # This is the "roulette wheel selection" approach.
            indices = np.random.choice(population_size, 2, replace=False, p=selection_probabilities)

            parent1 = population[indices[0]]
            parent2 = population[indices[1]]

            # Cross-over the two parents to produce 2 new children.
            child1, child2 = crossover(parent1, parent2, gene_crossover_rate=gene_crossover_rate)

            # Mutate the two children.
            mutate(child1, gene_mutation_rate=gene_mutation_rate)
            mutate(child2, gene_mutation_rate=gene_mutation_rate)

            # Replace the chromosomes in the population with the lowest fitness.
            if np.random.random() < survival_rate:
                population[np.argmin(fitnesses)] = child1
                population[np.argmin(fitnesses)] = child2

        fitnesses = calculate_fitnesses(population)

        generations_best_individual[generation] = population[np.argmax(fitnesses)]
        generations_best_fitness[generation] = np.max(fitnesses)

    return generations_best_individual, generations_best_fitness


def experiment1(num_rollouts=5,
                population_sizes=[MIN_POPULATION_SIZE, MAX_POPULATION_SIZE],
                num_generations=NUM_GENERATIONS):
    # The best individual (and the corresponding fitness) across all population sizes and rollouts.
    best_individual = None
    best_fitness = None
    all_averages = []

    for population_size in population_sizes:

        # The best fitness (of an individual) across all generations for different number of rollouts.
        generations_best_fitnesses = np.zeros((num_rollouts, NUM_GENERATIONS))

        for rollout in range(num_rollouts):
            generations_best_individual, generations_best_fitness = ga(population_size=population_size,
                                                                       num_generations=num_generations)

            idx = np.argmax(generations_best_fitness)

            if best_individual is None:
                assert best_fitness is None
                best_individual = generations_best_individual[idx]
                best_fitness = np.max(generations_best_fitness)

            generations_best_fitnesses[rollout] = generations_best_fitness

        average_generations_best_fitness = np.average(generations_best_fitnesses, axis=0)
        all_averages.append(average_generations_best_fitness)

    assert best_fitness is not None
    assert best_individual is not None
    assert all_averages is not None

    return best_individual, best_fitness, all_averages


def plot_averages(all_averages, population_sizes, num_rollouts):
    plt.figure()

    for avg in all_averages:
        plt.plot(avg)

    plt.legend(population_sizes)
    plt.xlabel("Generations")
    plt.ylabel("Average fitness of {} rollouts".format(num_rollouts))
    plt.show()


def run_experiment1():
    num_rollouts = 5
    population_sizes = [MAX_POPULATION_SIZE]

    best_individual, best_fitness, all_averages = experiment1(num_rollouts=num_rollouts,
                                                              population_sizes=population_sizes)

    print("best_individual =", best_individual)
    print(get_loss(best_individual))
    plot_averages(all_averages, population_sizes, num_rollouts)


# TODO:
#   1. implement multiplicative mutation
#   2. stop searching when critical point is found
#   3. use second derivative to understand if the found critical point is maximum or minimum
#   (https://math.stackexchange.com/a/2058474/168764)
# https://en.wikipedia.org/wiki/Fitness_proportionate_selection
if __name__ == '__main__':
    run_experiment1()
