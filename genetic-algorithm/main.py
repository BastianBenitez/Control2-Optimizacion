import random
import copy
import math
import matplotlib.pyplot as plt
import time  # Added time module

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

def read_optimal_tour(filename):
    """Reads the optimal tour from a .tour file."""
    optimal_path = []
    with open(filename, 'r') as file:
        start = False
        for line in file:
            if line.startswith("TOUR_SECTION"):
                start = True
                continue
            if start:
                if line.strip() == "-1" or line.strip() == "EOF":
                    break
                city_number = int(line.strip())
                optimal_path.append(city_number)
    return optimal_path

def calculate_tour_length(tour, distance_matrix):
    """Calculate the length of a given tour."""
    total_distance = 0
    for i in range(len(tour) - 1):
        a = tour[i] - 1  # Adjusting for 0-indexed arrays
        b = tour[i + 1] - 1
        total_distance += distance_matrix[a][b]
    # Add distance from last to first city to complete the tour
    a = tour[-1] - 1
    b = tour[0] - 1
    total_distance += distance_matrix[a][b]
    return total_distance

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
    
    model_name = "GA-TSP"  # Name of the algorithm model
    start_time = time.time()  # Start timing execution

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
    best_solution_index = sorted_x[0][0]
    best_solution_cost = sorted_x[0][1]
    best_solution_path = population[best_solution_index]
    
    # Complete the tour by adding the first city at the end
    complete_best_path = best_solution_path + [best_solution_path[0]]
    
    # Read the optimal tour
    try:
        optimal_tour = read_optimal_tour("a280.opt.tour")
        
        # Calculate distance for the optimal tour
        for i in range(CITIES_SIZE):
            for j in range(CITIES_SIZE):
                if dCidade[i][j] == 0 and i != j:  # If distance not calculated yet
                    dCidade[i][j] = round(math.sqrt((x[i] - x[j])**2 + (y[i] - y[j])**2), 4)
        
        optimal_cost = calculate_tour_length(optimal_tour, dCidade)
        
        # Calculate difference and percent error
        absolute_difference = best_solution_cost - optimal_cost
        percent_error = (absolute_difference / optimal_cost) * 100
        
        execution_time = time.time() - start_time  # Calculate execution time
        
        # Print results in the requested table format
        print("\n{:<15} {:<15} {:<30} {:<30} {:<15}".format(
            "Modelo", 
            "travelled", 
            "Distacion tola de ruta optima", 
            "Gap entre la solucion optima (%)", 
            "Tiempo(s)"
        ))
        print("-" * 105)
        print("{:<15} {:<15.2f} {:<30.2f} {:<30.2f} {:<15.2f}".format(
            model_name,
            best_solution_cost,  # The algorithm's solution distance (travelled)
            best_solution_cost,  # The algorithm's best route distance 
            percent_error,       # The percentage gap with reference optimal solution
            execution_time
        ))
        
        # Plot both the best solution and optimal solution for visual comparison
        plt.figure(figsize=(12, 10))
        
        # Plot cost by generation
        plt.subplot(2, 1, 1)
        plt.plot(costByExecution)
        plt.title("Costo por generación")
        plt.xlabel("Generación")
        plt.ylabel("Costo")
        
        # Plot the tours
        plt.subplot(2, 1, 2)
        
        # Plot cities
        plt.scatter(x, y, color='blue', s=10)
        
        # Best solution in red
        best_x = [x[city-1] for city in complete_best_path]
        best_y = [y[city-1] for city in complete_best_path]
        plt.plot(best_x, best_y, 'r-', label=f"GA Solution (cost: {best_solution_cost:.2f})")
        
        # Optimal tour in green
        opt_tour_with_return = optimal_tour + [optimal_tour[0]]
        opt_x = [x[city-1] for city in opt_tour_with_return]
        opt_y = [y[city-1] for city in opt_tour_with_return]
        plt.plot(opt_x, opt_y, 'g-', alpha=0.5, label=f"Optimal (cost: {optimal_cost:.2f})")
        
        plt.title("Comparación de rutas")
        plt.legend()
        plt.tight_layout()
        
    except FileNotFoundError:
        execution_time = time.time() - start_time  # Calculate execution time
        
        # Print results in a simplified table format (without optimal solution)
        print("\n{:<15} {:<15} {:<30} {:<30} {:<15}".format(
            "Modelo", 
            "travelled", 
            "Distacion tola de ruta optima", 
            "Gap entre la solucion optima (%)", 
            "Tiempo(s)"
        ))
        print("-" * 105)
        print("{:<15} {:<15.2f} {:<30} {:<30} {:<15.2f}".format(
            model_name,
            best_solution_cost,
            "N/A (archivo no encontrado)",
            "N/A",
            execution_time
        ))
        
        # Plot just the cost by generation
        plt.plot(costByExecution)
        plt.title("Costo por generación")
        plt.xlabel("Generación")
        plt.ylabel("Costo")
    
    plt.show()

if __name__ == "__main__":
    main()
