import math
import random
import os
import time
from matplotlib import pyplot as plt


def read_tsp_file(file_path):
    """
    Lee un archivo TSP y devuelve una lista de coordenadas de nodos
    """
    nodes = []
    reading_nodes = False
    
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if line == "NODE_COORD_SECTION":
                reading_nodes = True
                continue
            if reading_nodes and line != "EOF":
                parts = line.split()
                if len(parts) >= 3:  # Asegurarse de que hay al menos 3 valores (ID, X, Y)
                    node_id = int(parts[0])
                    x = float(parts[1])
                    y = float(parts[2])
                    nodes.append((x, y))
            if line == "EOF":
                break
    
    return nodes


def read_optimal_tour(file_path):
    """
    Lee un archivo de ruta óptima (.tour) y devuelve una lista con el orden de los nodos
    """
    tour = []
    reading_tour = False
    
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if line == "TOUR_SECTION":
                reading_tour = True
                continue
            if reading_tour and line != "-1":
                try:
                    # Los nodos en el archivo .tour están numerados desde 1
                    node_id = int(line)
                    # Convertimos a índices basados en 0 para nuestro algoritmo
                    tour.append(node_id - 1)
                except ValueError:
                    continue
            if line == "-1":
                break
    
    return tour


def calculate_tour_distance(tour, nodes):
    """
    Calcula la distancia total de una ruta dada
    """
    distance = 0.0
    num_nodes = len(tour)
    
    for i in range(num_nodes):
        from_node = tour[i]
        to_node = tour[(i + 1) % num_nodes]
        distance += math.sqrt(
            pow(nodes[from_node][0] - nodes[to_node][0], 2.0) + 
            pow(nodes[from_node][1] - nodes[to_node][1], 2.0)
        )
    
    return distance


def compare_with_optimal(algorithm_tour, optimal_tour, nodes):
    """
    Compara la ruta obtenida por el algoritmo con la ruta óptima
    """
    algorithm_distance = calculate_tour_distance(algorithm_tour, nodes)
    optimal_distance = calculate_tour_distance(optimal_tour, nodes)
    
    absolute_difference = algorithm_distance - optimal_distance
    percentage_difference = (absolute_difference / optimal_distance) * 100.0
    
    return {
        'algorithm_distance': algorithm_distance,
        'optimal_distance': optimal_distance,
        'absolute_difference': absolute_difference,
        'percentage_difference': percentage_difference
    }


class SolveTSPUsingACO:
    class Edge:
        def __init__(self, a, b, weight, initial_pheromone):
            self.a = a
            self.b = b
            self.weight = weight
            self.pheromone = initial_pheromone

    class Ant:
        def __init__(self, alpha, beta, num_nodes, edges):
            self.alpha = alpha
            self.beta = beta
            self.num_nodes = num_nodes
            self.edges = edges
            self.tour = None
            self.distance = 0.0

        def _select_node(self):
            roulette_wheel = 0.0
            unvisited_nodes = [node for node in range(self.num_nodes) if node not in self.tour]
            heuristic_total = 0.0
            for unvisited_node in unvisited_nodes:
                # Añadir una pequeña constante para evitar división por cero
                heuristic_total += self.edges[self.tour[-1]][unvisited_node].weight
                
            for unvisited_node in unvisited_nodes:
                # Evitar división por cero añadiendo una pequeña constante
                weight = max(self.edges[self.tour[-1]][unvisited_node].weight, 0.0001)
                roulette_wheel += pow(self.edges[self.tour[-1]][unvisited_node].pheromone, self.alpha) * \
                              pow((heuristic_total / weight), self.beta)
                              
            random_value = random.uniform(0.0, roulette_wheel)
            wheel_position = 0.0
            for unvisited_node in unvisited_nodes:
                # Evitar división por cero añadiendo una pequeña constante
                weight = max(self.edges[self.tour[-1]][unvisited_node].weight, 0.0001)
                wheel_position += pow(self.edges[self.tour[-1]][unvisited_node].pheromone, self.alpha) * \
                              pow((heuristic_total / weight), self.beta)
                if wheel_position >= random_value:
                    return unvisited_node
                    
            # Si por alguna razón no se seleccionó ningún nodo, retornar el primero no visitado
            return unvisited_nodes[0] if unvisited_nodes else 0

        def find_tour(self):
            self.tour = [random.randint(0, self.num_nodes - 1)]
            while len(self.tour) < self.num_nodes:
                self.tour.append(self._select_node())
            return self.tour

        def get_distance(self):
            self.distance = 0.0
            for i in range(self.num_nodes):
                self.distance += self.edges[self.tour[i]][self.tour[(i + 1) % self.num_nodes]].weight
            return self.distance

    def __init__(self, mode='ACS', colony_size=10, elitist_weight=1.0, min_scaling_factor=0.001, alpha=1.0, beta=3.0,
                 rho=0.1, pheromone_deposit_weight=1.0, initial_pheromone=1.0, steps=100, nodes=None, labels=None):
        self.mode = mode
        self.colony_size = colony_size
        self.elitist_weight = elitist_weight
        self.min_scaling_factor = min_scaling_factor
        self.rho = rho
        self.pheromone_deposit_weight = pheromone_deposit_weight
        self.steps = steps
        self.num_nodes = len(nodes)
        self.nodes = nodes
        if labels is not None:
            self.labels = labels
        else:
            self.labels = range(1, self.num_nodes + 1)
        self.edges = [[None] * self.num_nodes for _ in range(self.num_nodes)]
        for i in range(self.num_nodes):
            for j in range(i + 1, self.num_nodes):
                self.edges[i][j] = self.edges[j][i] = self.Edge(i, j, math.sqrt(
                    pow(self.nodes[i][0] - self.nodes[j][0], 2.0) + pow(self.nodes[i][1] - self.nodes[j][1], 2.0)),
                                                                initial_pheromone)
        self.ants = [self.Ant(alpha, beta, self.num_nodes, self.edges) for _ in range(self.colony_size)]
        self.global_best_tour = None
        self.global_best_distance = float("inf")

    def _add_pheromone(self, tour, distance, weight=1.0):
        pheromone_to_add = self.pheromone_deposit_weight / distance
        for i in range(self.num_nodes):
            self.edges[tour[i]][tour[(i + 1) % self.num_nodes]].pheromone += weight * pheromone_to_add

    def _acs(self):
        for step in range(self.steps):
            for ant in self.ants:
                self._add_pheromone(ant.find_tour(), ant.get_distance())
                if ant.distance < self.global_best_distance:
                    self.global_best_tour = ant.tour
                    self.global_best_distance = ant.distance
            for i in range(self.num_nodes):
                for j in range(i + 1, self.num_nodes):
                    self.edges[i][j].pheromone *= (1.0 - self.rho)

    def _elitist(self):
        for step in range(self.steps):
            for ant in self.ants:
                self._add_pheromone(ant.find_tour(), ant.get_distance())
                if ant.distance < self.global_best_distance:
                    self.global_best_tour = ant.tour
                    self.global_best_distance = ant.distance
            self._add_pheromone(self.global_best_tour, self.global_best_distance, weight=self.elitist_weight)
            for i in range(self.num_nodes):
                for j in range(i + 1, self.num_nodes):
                    self.edges[i][j].pheromone *= (1.0 - self.rho)

    def _max_min(self):
        for step in range(self.steps):
            iteration_best_tour = None
            iteration_best_distance = float("inf")
            for ant in self.ants:
                ant.find_tour()
                if ant.get_distance() < iteration_best_distance:
                    iteration_best_tour = ant.tour
                    iteration_best_distance = ant.distance
            if float(step + 1) / float(self.steps) <= 0.75:
                self._add_pheromone(iteration_best_tour, iteration_best_distance)
                max_pheromone = self.pheromone_deposit_weight / iteration_best_distance
            else:
                if iteration_best_distance < self.global_best_distance:
                    self.global_best_tour = iteration_best_tour
                    self.global_best_distance = iteration_best_distance
                self._add_pheromone(self.global_best_tour, self.global_best_distance)
                max_pheromone = self.pheromone_deposit_weight / self.global_best_distance
            min_pheromone = max_pheromone * self.min_scaling_factor
            for i in range(self.num_nodes):
                for j in range(i + 1, self.num_nodes):
                    self.edges[i][j].pheromone *= (1.0 - self.rho)
                    if self.edges[i][j].pheromone > max_pheromone:
                        self.edges[i][j].pheromone = max_pheromone
                    elif self.edges[i][j].pheromone < min_pheromone:
                        self.edges[i][j].pheromone = min_pheromone

    def run(self):
        print('Started : {0}'.format(self.mode))
        start_time = time.time()
        
        if self.mode == 'ACS':
            self._acs()
        elif self.mode == 'Elitist':
            self._elitist()
        else:
            self._max_min()
            
        execution_time = time.time() - start_time
        
        print('Ended : {0}'.format(self.mode))
        print('Sequence : <- {0} ->'.format(' - '.join(str(self.labels[i]) for i in self.global_best_tour)))
        print('Total distance travelled to complete the tour : {0}'.format(round(self.global_best_distance, 2)))
        print('Tiempo de ejecución: {0:.2f} segundos\n'.format(execution_time))
        
        return execution_time

    def plot(self, line_width=1, point_radius=math.sqrt(2.0), annotation_size=8, dpi=120, save=True, name=None):
        x = [self.nodes[i][0] for i in self.global_best_tour]
        x.append(x[0])
        y = [self.nodes[i][1] for i in self.global_best_tour]
        y.append(y[0])
        plt.plot(x, y, linewidth=line_width)
        plt.scatter(x, y, s=math.pi * (point_radius ** 2.0))
        plt.title(self.mode)
        for i in self.global_best_tour:
            plt.annotate(self.labels[i], self.nodes[i], size=annotation_size)
        if save:
            if name is None:
                name = '{0}.png'.format(self.mode)
            plt.savefig(name, dpi=dpi)
        plt.show()
        plt.gcf().clear()


if __name__ == '__main__':
    _colony_size = 10  # Aumentado de 5 a 10 para mejorar la exploración
    _steps = 100  # Aumentado de 50 a 100 para dar más tiempo al algoritmo
    
    print("Cargando datos del archivo a280.tsp...")
    tsp_file_path = os.path.join(os.path.dirname(__file__), 'a280.tsp')
    _nodes = read_tsp_file(tsp_file_path)
    print(f"Se han cargado {len(_nodes)} nodos del archivo TSP.")
    
    # Leer la ruta óptima
    print("Cargando la ruta óptima del archivo a280.opt.tour...")
    optimal_tour_path = os.path.join(os.path.dirname(__file__), 'a280.opt.tour')
    _optimal_tour = read_optimal_tour(optimal_tour_path)
    _optimal_distance = calculate_tour_distance(_optimal_tour, _nodes)
    print(f"Distancia total de la ruta óptima: {round(_optimal_distance, 2)}")
    
    # Establecer un tamaño adecuado para las anotaciones
    annotation_size = 4  # Tamaño más pequeño para evitar superposiciones
    
    # Crear una lista para almacenar resultados de comparación
    comparison_results = []
    
    print("\nEjecutando algoritmo ACS...")
    acs = SolveTSPUsingACO(mode='ACS', colony_size=_colony_size, steps=_steps, nodes=_nodes)
    acs_time = acs.run()
    acs_comparison = compare_with_optimal(acs.global_best_tour, _optimal_tour, _nodes)
    print(f"Comparación con la ruta óptima:")
    print(f"  - Distancia de la ruta óptima: {round(acs_comparison['optimal_distance'], 2)}")
    print(f"  - Distancia de la ruta ACS: {round(acs_comparison['algorithm_distance'], 2)}")
    print(f"  - Diferencia absoluta: {round(acs_comparison['absolute_difference'], 2)}")
    print(f"  - Diferencia porcentual: {round(acs_comparison['percentage_difference'], 2)}%")
    comparison_results.append(('ACS', acs_comparison, acs_time))
    acs.plot(annotation_size=annotation_size, name=os.path.join('tour_plots', 'ACS.png'))
    
    print("\nEjecutando algoritmo Elitist...")
    elitist = SolveTSPUsingACO(mode='Elitist', colony_size=_colony_size, steps=_steps, nodes=_nodes)
    elitist_time = elitist.run()
    elitist_comparison = compare_with_optimal(elitist.global_best_tour, _optimal_tour, _nodes)
    print(f"Comparación con la ruta óptima:")
    print(f"  - Distancia de la ruta óptima: {round(elitist_comparison['optimal_distance'], 2)}")
    print(f"  - Distancia de la ruta Elitist: {round(elitist_comparison['algorithm_distance'], 2)}")
    print(f"  - Diferencia absoluta: {round(elitist_comparison['absolute_difference'], 2)}")
    print(f"  - Diferencia porcentual: {round(elitist_comparison['percentage_difference'], 2)}%")
    comparison_results.append(('Elitist', elitist_comparison, elitist_time))
    elitist.plot(annotation_size=annotation_size, name=os.path.join('tour_plots', 'Elitist.png'))
    
    print("\nEjecutando algoritmo MaxMin...")
    max_min = SolveTSPUsingACO(mode='MaxMin', colony_size=_colony_size, steps=_steps, nodes=_nodes)
    maxmin_time = max_min.run()
    maxmin_comparison = compare_with_optimal(max_min.global_best_tour, _optimal_tour, _nodes)
    print(f"Comparación con la ruta óptima:")
    print(f"  - Distancia de la ruta óptima: {round(maxmin_comparison['optimal_distance'], 2)}")
    print(f"  - Distancia de la ruta MaxMin: {round(maxmin_comparison['algorithm_distance'], 2)}")
    print(f"  - Diferencia absoluta: {round(maxmin_comparison['absolute_difference'], 2)}")
    print(f"  - Diferencia porcentual: {round(maxmin_comparison['percentage_difference'], 2)}%")
    comparison_results.append(('MaxMin', maxmin_comparison, maxmin_time))
    max_min.plot(annotation_size=annotation_size, name=os.path.join('tour_plots', 'MaxMin.png'))
    
    # Resumen comparativo
    print("\n======== RESUMEN DE COMPARACIÓN ========")
    print("Algoritmo\tDistancia\tDif. Óptima\t% Dif.\t\tTiempo (s)")
    print("-" * 75)
    print(f"Óptimo\t\t{round(_optimal_distance, 2)}\t-\t\t-\t\t-")
    for name, result, execution_time in comparison_results:
        print(f"{name}\t\t{round(result['algorithm_distance'], 2)}\t{round(result['absolute_difference'], 2)}\t\t{round(result['percentage_difference'], 2)}%\t\t{execution_time:.2f}")
