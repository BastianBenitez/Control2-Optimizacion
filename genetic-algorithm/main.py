import random
import copy
import math
import matplotlib.pyplot as plt

POPULATION_SIZE = 20
population = []
x = []
y = []
costByExecution = []

def read_tsplib_file(filename):
    coords = []
    with open(filename, 'r') as file:
        start = False
        for line in file:
            if line.startswith("NODE_COORD_SECTION"):
                start = True
                continue
            if start:
                if line.strip() == "EOF":
                    break
                parts = line.strip().split()
                if len(parts) >= 3:
                    x_coord = float(parts[1])
                    y_coord = float(parts[2])
                    coords.append((x_coord, y_coord))
    return coords

def generatePossiblePath():
    path = []
    while len(path) < CITIES_SIZE:
        rand = random.randint(1, CITIES_SIZE)
        if rand not in path:
            path.append(rand)
    population.append(path)

def generateFirstPopulation():
    for _ in range(POPULATION_SIZE):
        generatePossiblePath()

def mutate(matrix):
    for i in range(len(matrix)):
        for _ in range(len(matrix[i])):
            if random.randint(1, 100) <= 5:
                i1 = random.randint(0, CITIES_SIZE - 1)
                i2 = random.randint(0, CITIES_SIZE - 1)
                matrix[i][i1], matrix[i][i2] = matrix[i][i2], matrix[i][i1]

def generateTour():
    global tour
    tour = copy.deepcopy(population)
    for path in tour:
        path.append(path[0])

def calculateDistances():
    global distances
    distances = [0 for _ in range(POPULATION_SIZE)]
    for i in range(POPULATION_SIZE):
        for j in range(CITIES_SIZE):
            a = tour[i][j] - 1
            b = tour[i][j + 1] - 1
            distances[i] += round(dCidade[a][b], 4)
    dict_dist = {i: distances[i] for i in range(POPULATION_SIZE)}
    distances = copy.deepcopy(dict_dist)
    return sorted(distances.items(), key=lambda kv: kv[1])

def fitnessFunction():
    for i in range(CITIES_SIZE):
        for j in range(CITIES_SIZE):
            dCidade[i][j] = round(math.sqrt((x[i] - x[j])**2 + (y[i] - y[j])**2), 4)
    return calculateDistances()

def createParents(rouletteArr):
    return [rouletteArr[random.randint(0, len(rouletteArr) - 1)] for _ in range(5)]

def rouletteFunction(sorted_x):
    global parentsOne, parentsTwo
    best = [sorted_x[i][0] for i in range(10)]
    rouletteArr = []
    for j in range(len(best)):
        rouletteArr.extend([best[j]] * (10 - j))
    parentsOne = createParents(rouletteArr)
    parentsTwo = createParents(rouletteArr)

def hasDuplicity(path, usedIndexes):
    for i in range(len(path)):
        for j in range(i + 1, len(path)):
            if path[i] == path[j]:
                return j if i in usedIndexes else i
    return -1

def doCycle(sorted_x):
    global population
    children = []

    for i in range(5):
        idx1 = parentsOne[i]
        idx2 = parentsTwo[i]
        used = [random.randint(0, CITIES_SIZE - 1)]
        c1 = copy.deepcopy(population[idx1])
        c2 = copy.deepcopy(population[idx2])
        c1[used[0]], c2[used[0]] = c2[used[0]], c1[used[0]]

        while (dup := hasDuplicity(c1, used)) != -1:
            used.append(dup)
            c1[dup], c2[dup] = c2[dup], c1[dup]

        children.append(c1)
        children.append(c2)

    mutate(children)

    temp = copy.deepcopy(population)
    for i in range(10):
        population[i] = copy.deepcopy(temp[sorted_x[i][0]])
    for j in range(10, POPULATION_SIZE):
        population[j] = copy.deepcopy(children[j - 10])

def main():
    global CITIES_SIZE, TOUR_SIZE, dCidade

    coords = read_tsplib_file("a280.tsp")
    CITIES_SIZE = len(coords)
    TOUR_SIZE = CITIES_SIZE + 1
    dCidade = [[0 for _ in range(CITIES_SIZE)] for _ in range(CITIES_SIZE)]

    for xi, yi in coords:
        x.append(xi)
        y.append(yi)

    generateFirstPopulation()
    generateTour()

    for _ in range(500):  # puedes cambiar 500 por 9999 si quieres más generaciones
        sorted_x = fitnessFunction()
        rouletteFunction(sorted_x)
        doCycle(sorted_x)
        generateTour()
        costByExecution.append(sorted_x[0][1])

    sorted_x = fitnessFunction()
    print("Ciudades:", CITIES_SIZE)
    print("Mejor Costo:", sorted_x[0][1])
    print("Mejor Ruta:", population[sorted_x[0][0]])

    plt.plot(costByExecution)
    plt.title("Costo por generación")
    plt.xlabel("Generación")
    plt.ylabel("Costo")
    plt.show()

if __name__ == "__main__":
    main()
