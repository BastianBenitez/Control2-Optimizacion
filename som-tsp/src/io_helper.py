import pandas as pd
import numpy as np
import re

def read_tsp(filename):
    """
    Read a file in .tsp format into a pandas DataFrame

    The .tsp files can be found in the TSPLIB project. Currently, the library
    only considers the possibility of a 2D map.
    """
    with open(filename) as f:
        node_coord_start = None
        dimension = None
        lines = f.readlines()

        # Obtain the information about the .tsp
        i = 0
        while i < len(lines) and (not dimension or not node_coord_start):
            line = lines[i]
            if line.startswith('DIMENSION :') or line.startswith('DIMENSION:'):
                dimension = int(line.split()[-1])
            if line.startswith('NODE_COORD_SECTION'):
                node_coord_start = i
            i = i+1

        if dimension is None or node_coord_start is None:
            raise ValueError(f"Could not find DIMENSION or NODE_COORD_SECTION in file {filename}")

        print('Problem with {} cities read.'.format(dimension))

        # Proceso manual de extracción de datos para manejar diferentes formatos
        cities_data = []
        for i in range(node_coord_start + 1, len(lines)):
            line = lines[i].strip()
            if line == "EOF" or not line:  # Terminar al final del archivo
                break
            
            # Eliminar múltiples espacios y dividir
            values = re.split(r'\s+', line.strip())
            if len(values) >= 3:  # Asegurarnos de que hay al menos 3 valores (id, x, y)
                city_id = values[0]
                x = float(values[1])
                y = float(values[2])
                cities_data.append({'city': city_id, 'x': x, 'y': y})
        
        # Crear DataFrame directamente desde los datos extraídos
        cities = pd.DataFrame(cities_data)

        return cities

def normalize(points):
    """
    Return the normalized version of a given vector of points.

    For a given array of n-dimensions, normalize each dimension by removing the
    initial offset and normalizing the points in a proportional interval: [0,1]
    on y, maintining the original ratio on x.
    """
    ratio = (points.x.max() - points.x.min()) / (points.y.max() - points.y.min()), 1
    ratio = np.array(ratio) / max(ratio)
    norm = points.apply(lambda c: (c - c.min()) / (c.max() - c.min()))
    return norm.apply(lambda p: ratio * p, axis=1)
