from sys import argv
import time
import os
import pandas as pd
import numpy as np

from io_helper import read_tsp, normalize, read_optimal_tour
from neuron import generate_network, get_neighborhood, get_route
from distance import select_closest, euclidean_distance, route_distance
from plot import plot_network, plot_route

def main():
    if len(argv) != 2:
        print("Correct use: python src/main.py <filename>.tsp")
        return -1

    problem = read_tsp(argv[1])
    
    # Time measurement start
    start_time = time.time()
    
    route = som(problem, 100000)

    problem = problem.reindex(route)
    
    # Time measurement end
    end_time = time.time()
    execution_time = end_time - start_time

    distance = route_distance(problem)
    
    print('Route found of length {}'.format(distance))
    
    # Get the optimal tour if available
    base_name = os.path.basename(argv[1]).split('.')[0]
    optimal_tour_file = os.path.join(os.path.dirname(argv[1]), f"{base_name}.opt.tour")
    
    if os.path.exists(optimal_tour_file):
        # Read the optimal tour
        optimal_tour = read_optimal_tour(optimal_tour_file)
        
        # Filter out any negative values (like -1 at the end of the tour)
        optimal_tour = [city_id for city_id in optimal_tour if city_id > 0]
        
        # Convert to zero-based indexing if needed
        if min(optimal_tour) == 1:
            optimal_tour = [x-1 for x in optimal_tour]
        
        # Create a new DataFrame with the cities in the optimal order
        optimal_problem = problem.copy().iloc[0:0]  # Empty DataFrame with same structure
        for city_id in optimal_tour:
            # Ensure the index exists in the problem DataFrame
            if city_id in problem.index:
                optimal_problem = pd.concat([optimal_problem, problem.loc[[city_id]]], ignore_index=False)
            else:
                print(f"Warning: City ID {city_id} not found in the problem index.")
        
        # Calculate distance of optimal tour
        optimal_distance = route_distance(optimal_problem)
        
        # Calculate gap
        gap = ((distance - optimal_distance) / optimal_distance) * 100
        
        # Create comparison table
        comparison = pd.DataFrame({
            'Modelo': ['SOM-TSP'],
            'travelled': [distance],
            'Distancia total de ruta optima': [optimal_distance],
            'Gap entre la solucion optima': [f"{gap:.2f}%"],
            'Tiempo(s)': [f"{execution_time:.2f}"]
        })
        
        print("\nComparison Table:")
        print(comparison.to_string(index=False))
    else:
        print(f"\nOptimal tour file {optimal_tour_file} not found. Cannot make comparison.")


def som(problem, iterations, learning_rate=0.8):
    """Solve the TSP using a Self-Organizing Map."""

    # Obtain the normalized set of cities (w/ coord in [0,1])
    cities = problem.copy()

    cities[['x', 'y']] = normalize(cities[['x', 'y']])

    # The population size is 8 times the number of cities
    n = cities.shape[0] * 8

    # Generate an adequate network of neurons:
    network = generate_network(n)
    print('Network of {} neurons created. Starting the iterations:'.format(n))

    for i in range(iterations):
        if not i % 100:
            print('\t> Iteration {}/{}'.format(i, iterations), end="\r")
        # Choose a random city
        city = cities.sample(1)[['x', 'y']].values
        winner_idx = select_closest(network, city)
        # Generate a filter that applies changes to the winner's gaussian
        gaussian = get_neighborhood(winner_idx, n//10, network.shape[0])
        # Update the network's weights (closer to the city)
        network += gaussian[:,np.newaxis] * learning_rate * (city - network)
        # Decay the variables
        learning_rate = learning_rate * 0.99997
        n = n * 0.9997

        # Check for plotting interval
        if not i % 1000:
            plot_network(cities, network, name='diagrams/{:05d}.png'.format(i))

        # Check if any parameter has completely decayed.
        if n < 1:
            print('Radius has completely decayed, finishing execution',
            'at {} iterations'.format(i))
            break
        if learning_rate < 0.001:
            print('Learning rate has completely decayed, finishing execution',
            'at {} iterations'.format(i))
            break
    else:
        print('Completed {} iterations.'.format(iterations))

    plot_network(cities, network, name='diagrams/final.png')

    route = get_route(cities, network)
    plot_route(cities, route, 'diagrams/route.png')
    return route

if __name__ == '__main__':
    main()
