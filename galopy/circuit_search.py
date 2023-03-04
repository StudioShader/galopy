import torch
from math import pi, factorial
from time import time
from itertools import product
from galopy.schmidt_decompose import rho_entropy
from galopy.progress_bar import print_progress_bar
from galopy.population import RandomPopulation, FromFilePopulation


class CircuitSearch:
    def __init__(self, device: str, matrix, input_basic_states, output_basic_states=None, depth=1,
                 n_ancilla_modes=0, n_ancilla_photons=0, n_success_measurements=1):
        """
        Algorithm searching a circuit.
        Parameters:
            device: The device on which you want to store data and perform calculations (e.g. 'cuda')

            matrix: Matrix representing desired transform in the basis of basic states

            input_basic_states: Basic states on which transform is performed

            output_basic_states: Basic states which are counted as output

            depth: Number of local two-mode unitary transforms. One transform contains two phase shifters and one
            beam splitter. Must be > 0

            n_ancilla_modes: Number of modes in which ancilla photons are

            n_ancilla_photons: Number of ancilla photons

            n_success_measurements: Count of measurements that we consider as successful gate operation. Must be > 0
        """
        if n_ancilla_modes == 0 and n_ancilla_photons > 0:
            raise Exception("If number of ancilla modes is zero, number of ancilla photons must be zero as well")

        self.device = device

        self.matrix = torch.tensor(matrix, device=self.device, dtype=torch.complex64)

        input_basic_states, _ = torch.tensor(input_basic_states, device=self.device).sort()
        self.input_basic_states = input_basic_states + n_ancilla_modes
        # Number of input basic states
        self.n_input_basic_states = self.input_basic_states.shape[0]

        if not matrix.shape[1] == self.n_input_basic_states:
            raise Exception("Number of input basic states must be equal to the number of columns in transform matrix")

        if output_basic_states is None:
            self.output_basic_states = self.input_basic_states
        else:
            output_basic_states, _ = torch.tensor(output_basic_states, device=self.device).sort()
            self.output_basic_states = output_basic_states + n_ancilla_modes
        # Number of output basic states
        self.n_output_basic_states = self.output_basic_states.shape[0]

        if not matrix.shape[0] == self.n_output_basic_states:
            raise Exception("Number of output basic states must be equal to the number of rows in transform matrix")

        self.depth = depth

        self.n_state_modes = input_basic_states.max().item() + 1

        self.n_ancilla_modes = n_ancilla_modes
        # Total number of modes in scheme
        self.n_modes = self.n_state_modes + n_ancilla_modes
        # Number of modes in which unitary transform is performed
        # It's considered that all of ancilla modes always participate in this transform
        self.n_work_modes = self.n_modes

        self.n_state_photons = input_basic_states.shape[1]
        self.n_ancilla_photons = n_ancilla_photons
        # Total number of photons
        self.n_photons = input_basic_states.shape[1] + n_ancilla_photons

        self.n_success_measurements = n_success_measurements

        # # Init indices for flexible slicing
        # self._start_idx_rz_angles = 0
        # self._start_idx_ry_angles = 2 * self.depth
        # self._start_idx_modes = 3 * self.depth
        # self._start_idx_ancilla_state_in = 5 * self.depth
        # self._start_idx_ancilla_state_out = 5 * self.depth + self.n_ancilla_photons

    #     self._precompute_matrices()
    #
    # def _precompute_matrices(self):
    #     self._construct_permutation_matrix()
    #     self._construct_normalization_matrix(self._permutation_matrix)
    #
    # def _construct_permutation_matrix(self):
    #     """
    #     The matrix for output state computing.
    #     Multiply by it state vector to sum up all like terms.
    #     For example, vector (a0 * a1 + a1 * a0) will become 2 * a0 * a1
    #     """
    #     # TODO: возможно через reshape без to_idx ?
    #     def to_idx(*modes):
    #         """Convert multi-dimensional index to one-dimensional."""
    #         res = 0
    #         for mode in modes:
    #             res = res * self.n_modes + mode
    #         return res
    #
    #     args = [list(range(self.n_modes))] * self.n_photons
    #     indices = [list(i) for i in product(*args)]
    #
    #     normalized_indices = [idx.copy() for idx in indices]
    #     for idx in normalized_indices:
    #         idx.sort()
    #
    #     all_indices = list(map(lambda x, y: [to_idx(*x), to_idx(*y)], normalized_indices, indices))
    #     vals = [1.] * len(all_indices)
    #
    #     self._permutation_matrix = torch.sparse_coo_tensor(torch.tensor(all_indices).t(), vals, device=self.device,
    #                                                        dtype=torch.complex64)
    #
    # def _construct_normalization_matrix(self, permutation_matrix):
    #     """
    #     Get matrices for transforming between two representations of state: Dirac form and operator form.
    #     It's considered that operator acts on the vacuum state.
    #
    #         First matrix:  Dirac    -> operator ( |n> -> a^n / sqrt(n!) )
    #
    #         Second matrix: operator -> Dirac    ( a^n -> sqrt(n!) * |n> )
    #     """
    #     vector = torch.ones(permutation_matrix.shape[1], 1, device=self.device, dtype=torch.complex64)
    #     vector = torch.sparse.mm(permutation_matrix, vector).to_sparse_coo()
    #
    #     indices = vector.indices()[0].reshape(1, -1)
    #     indices = torch.cat((indices, indices))
    #     c = factorial(self.n_photons)
    #
    #     self._normalization_matrix = torch.sparse_coo_tensor(indices, (vector.values() / c).sqrt(),
    #                                                          size=permutation_matrix.shape, device=self.device)
    #     self._inverted_normalization_matrix = torch.sparse_coo_tensor(indices, (c / vector.values()).sqrt(),
    #                                                                   size=permutation_matrix.shape,
    #                                                                   device=self.device)

    def __calculate_fidelity_and_probability(self, transforms):
        """Given transforms, get fidelity and probability for each one."""
        if self.n_success_measurements == 1:
            # Probabilities
            # TODO: изменить формулу ?
            dot = torch.abs(transforms.mul(transforms.conj()))  # TODO: Optimize ?
            prob_per_state = torch.sum(dot, 2)
            probabilities = prob_per_state.sum(-1) / self.n_input_basic_states
            probabilities = probabilities.reshape(-1)

            # Fidelities
            # Formula is taken from the article:
            # https://www.researchgate.net/publication/222547674_Fidelity_of_quantum_operations
            m = self.matrix.t().conj() \
                .reshape(1, 1, self.n_input_basic_states, self.n_output_basic_states).matmul(transforms)

            a = torch.abs(m.matmul(m.transpose(-1, -2).conj()))  # TODO: Optimize ?
            a = a.diagonal(offset=0, dim1=-1, dim2=-2).sum(-1)  # batched trace
            a = a.reshape(-1)

            b = m.diagonal(offset=0, dim1=-1, dim2=-2).sum(-1)  # batched trace
            b = torch.abs(b.mul(b.conj()))  # TODO: Optimize ?
            b = b.reshape(-1)

            fidelities = (a + b) / self.n_input_basic_states / (self.n_input_basic_states + 1)

            # The probability of gate is counted so to get real fidelity we should divide it to probability
            pure_fidelities = fidelities / probabilities
            pure_fidelities = torch.where(probabilities == 0, 0, pure_fidelities)

            return pure_fidelities, probabilities
        else:
            raise Exception("Not implemented yet! Number of success measurements should be 1 so far")

    def __calculate_criteria(self, transforms):
        # calculating probabilities of states on basis vectors ((1, 0, 0, 0) , (0, 1, 0, 0), ...)
        # then calculating the maximum difference of these probabilities for each transform
        new_transforms = transforms.transpose(2, 3).reshape(transforms.size()[0], 4, 4)
        sums = new_transforms.abs().square().sum(2).sqrt()
        (values_max, ind1) = sums.max(1)
        (values_min, ind2) = sums.min(1)
        basic_states_probabilities_match = torch.ones(transforms.size()[0]).sub(values_max.sub(values_min))
        # calculate maximum entanglement of states
        # TODO: OPTIMIZE for torch
        normalized_states = new_transforms / sums.unsqueeze(-1)
        entanglement_entropies = torch.tensor(
            [(1. - min(min([(rho_entropy(vector).abs().item()) for vector in matrix]), 1.)) for matrix in
             normalized_states])

        return basic_states_probabilities_match, entanglement_entropies, values_min

    def __get_fidelity_and_probability(self, population):
        """Given population of circuits, get fidelity and probability for each circuit."""
        transforms = population.construct_transforms(self.input_basic_states, self.output_basic_states)
        # return self.__calculate_fidelity_and_probability(transforms)
        return self.__calculate_criteria(transforms)

    def __calculate_fitness(self, population):
        """Compute fitness for each individual in the given population."""
        # fidelities, probabilities = self.__get_fidelity_and_probability(population)
        basic_states_probabilities_match, entanglement_entropies, probabilities = self.__get_fidelity_and_probability(population)
        first_fitness = torch.where(entanglement_entropies > 0.01, 100. * basic_states_probabilities_match, entanglement_entropies)
        return torch.where(first_fitness > 99., 1000. * probabilities, first_fitness)
        # return entanglement_entropies

    def run(self, min_probability, n_generations, n_offsprings, n_elite,
            source_file=None, result_file=None):
        """
        Launch search. The algorithm stops in one of these cases:
            * After `n_generations` generations
            * If the circuit with fidelity > 0.999 and probability > `min_probability` is found

        Parameters:
            min_probability: Minimum required probability of the gate.

            n_generations: Maximum number of generations to happen.

            n_offsprings: Number of offsprings at each generation.

            n_elite: Number of individuals with the best fitness, that are guaranteed to pass into the next
            generation.

            source_file: The file to read initial population. If is None, then random population is generated.

            result_file: The file to write the result population to. If is None, the data won't be written anywhere.

            ptype: Population type (universal or real).
        """
        n_population = n_elite + n_offsprings
        # Save start time
        start_time = time()

        # Get initial population
        if source_file is None:
            population = RandomPopulation(n_individuals=n_population, depth=self.depth, n_modes=self.n_modes,
                                          n_ancilla_modes=self.n_ancilla_modes, n_state_photons=self.n_state_photons,
                                          n_ancilla_photons=self.n_ancilla_photons,
                                          n_success_measurements=self.n_success_measurements, device=self.device)
        else:
            circuits = FromFilePopulation(source_file, device=self.device)
            n_circuits = circuits.n_individuals
            if n_circuits < n_population:
                population = RandomPopulation(n_individuals=n_population - n_circuits, depth=self.depth,
                                              n_modes=self.n_modes,
                                              n_ancilla_modes=self.n_ancilla_modes,
                                              n_state_photons=self.n_state_photons,
                                              n_ancilla_photons=self.n_ancilla_photons,
                                              n_success_measurements=self.n_success_measurements, device=self.device)
                population = circuits + population
            else:
                population = circuits

        # Calculate fitness for the initial population
        fitness = self.__calculate_fitness(population)

        print_progress_bar(None, length=40, percentage=0.)

        for i in range(n_generations):
            # Select parents
            parents, fitness = population.select(fitness, n_elite)

            # Create new generation
            # parents.mutate(0.5)
            # population = parents

            children = parents.crossover(n_offsprings)
            children.mutate(mutation_probability=0.1)
            population = parents + children

            # population = parents + parents.crossover(n_offsprings)
            # population.mutate(mutation_probability=0.5)

            # Calculate fitness for the new individuals
            fitness = self.__calculate_fitness(population)

            # print("Generation:", i + 1)
            best_fitness = fitness.max().item()
            # print("Best fitness:", best_fitness)
            print_progress_bar(best_fitness, length=40, percentage=(i + 1.) / n_generations, reprint=True)

            # If circuit with high enough fitness is found, stop
            if best_fitness > 1000. * 0.99:
                n_generations = i + 1
                break
            # if best_fitness >= 100. * min_probability:
            #     n_generations = i + 1
            #     break
        print()

        # Save result population to file
        if result_file is not None:
            population.to_file(result_file)

        # Get the best circuit
        best, fitness = population.select(fitness, 1)

        # Print result info
        print("Circuit:")
        best[0].to_loqc_tech("result")
        # best[0].print()
        basic_states_probabilities_match, entanglement_entropies, pr = self.__get_fidelity_and_probability(best)
        transforms = best.construct_transforms(self.input_basic_states, self.output_basic_states)
        print("Transform: ", transforms[0])
        print("basic_states_probabilities_match: ", basic_states_probabilities_match[0].item())
        print("entanglement_entropies: ", entanglement_entropies[0].item())
        print(f"Processed {n_generations} generations in {time() - start_time:.2f} seconds")
