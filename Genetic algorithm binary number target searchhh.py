import random

TARGET = "10110110"

POPULATION_SIZE = 20
MUTATION_RATE = 0.01
CROSSOVER_RATE = 0.8
GENERATIONS = 100

def create_individual():
    return ''.join(random.choice('01') for _ in range(len(TARGET)))

def create_population():
    return [create_individual() for _ in range(POPULATION_SIZE)]

def fitness(individual):
    return sum(1 for i in range(len(TARGET)) if individual[i] == TARGET[i])

def selection(population):
    tournament = random.sample(population, 3)
    return max(tournament, key=fitness)

def crossover(parent1, parent2):
    if random.random() < CROSSOVER_RATE:
        point = random.randint(1, len(TARGET) - 1)
        child1 = parent1[:point] + parent2[point:]
        child2 = parent2[:point] + parent1[point:]
        return child1, child2
    return parent1, parent2

def mutation(individual):
    individual = list(individual)

    for i in range(len(individual)):
        if random.random() < MUTATION_RATE:
            individual[i] = '1' if individual[i] == '0' else '0'

    return ''.join(individual)

def genetic_algorithm():
    population = create_population()

    for generation in range(1, GENERATIONS + 1):
        best_individual = max(population, key=fitness)
        best_fitness = fitness(best_individual)

        print(f"Generation {generation}: Best = {best_individual}, Fitness = {best_fitness}/{len(TARGET)}")

        if best_individual == TARGET:
            print("\nTarget found!")
            print("Target   :", TARGET)
            print("Solution :", best_individual)
            print("Generation:", generation)
            return

        new_population = []

        while len(new_population) < POPULATION_SIZE:
            parent1 = selection(population)
            parent2 = selection(population)

            child1, child2 = crossover(parent1, parent2)

            child1 = mutation(child1)
            child2 = mutation(child2)

            new_population.append(child1)

            if len(new_population) < POPULATION_SIZE:
                new_population.append(child2)

        population = new_population

    best_individual = max(population, key=fitness)

    print("\nTarget was not found within the given generations.")
    print("Target  :", TARGET)
    print("Best    :", best_individual)
    print("Fitness :", fitness(best_individual), "/", len(TARGET))

genetic_algorithm()
