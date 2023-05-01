import numpy as np
from galopy.circuit_search import *
from os.path import exists
import math

if __name__ == "__main__":
    # Initialize parameters
    min_probability = 1. / 9.
    n_population = 2000
    n_offsprings = 1200
    # n_mutated = 2000
    n_elite = 300
    n_generations = 1000


    # Gate represented as a matrix
    matrix = np.array([[1., 0., 0., 0.],
                       [0., 1., 0., 0.],
                       [0., 0., 1., 0.],
                       [0., 0., 0., -1.]])
    # State modes:
    # (3)----------
    # (2)----------
    # (1)----------
    # (0)----------
    basic_states = np.array([[0, 2],
                             [0, 3],
                             [1, 2],
                             [1, 3]])
    output_basic_states = np.array([[0, 2],
                                    [0, 3],
                                    [1, 2],
                                    [1, 3],
                                    [0, 0], [0, 1], [1, 1], [2, 2], [2, 3], [3, 3]])
    file_number = 0
    white_list = [math.acos(0), math.acos(1 / math.sqrt(2)), math.acos(1 / math.sqrt(3)), math.acos(1 / 2),
                  math.acos(1 / math.sqrt(5)), math.asin(1 / math.sqrt(3)), math.asin(1 / math.sqrt(4)),
                  math.asin(1 / math.sqrt(5))]
    while exists("results/pure_entanglement/result" + str(file_number) + ".txt"):
        file_number += 1
    # Create an instance of search
    search = CircuitSearch('cpu', matrix, input_basic_states=basic_states,output_basic_states=output_basic_states, depth=6,
                           n_ancilla_modes=2, n_ancilla_photons=2, search_type="pure_entanglement",
                           entanglement_all_bases=True, file_number=file_number, null_phases=True)
    # Launch the search!
    search.run(min_probability, n_generations, n_offsprings, n_elite)
