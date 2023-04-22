import numpy as np
from galopy.circuit_search import *
from os.path import exists

if __name__ == "__main__":
    # Initialize parameters
    min_probability = 1. / 9.
    n_population = 2000
    n_offsprings = 400
    # n_mutated = 2000
    n_elite = 800
    n_generations = 300

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
    while exists("results/probabilities/result" + str(file_number) + ".txt"):
        file_number += 1
    # Create an instance of search
    search = CircuitSearch('cpu', matrix, input_basic_states=basic_states, output_basic_states=output_basic_states,
                           depth=6,
                           n_ancilla_modes=2, n_ancilla_photons=2, search_type="probabilities", file_number=file_number,
                           entanglement_all_bases=True, entanglement_cut=0.15)
    # Launch the search!
    search.run(min_probability, n_generations, n_offsprings, n_elite)
