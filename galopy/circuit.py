from math import pi
import pandas as pd
import torch
from galopy.schmidt_decompose import rho_entropy
import json


class Circuit:
    def __init__(self, n_modes, n_state_modes, bs_angles, ps_angles, topology, initial_ancilla_state, measurements):
        self.n_state_modes = n_state_modes
        self.n_modes = n_modes
        self.bs_angles = bs_angles.cpu().numpy()
        self.ps_angles = ps_angles.cpu().numpy()
        self.topology = topology.cpu().numpy()
        self.initial_ancilla_state = initial_ancilla_state.cpu().numpy()
        self.measurements = measurements.cpu().numpy()

    # def print(self):
    #     angles = self.bs_angles.reshape(-1).tolist()
    #     angles = [f"{180. * angle / pi:.2f}" for angle in angles]
    #
    #     topology = self.topology.tolist()
    #     topology = [f"{sublist[0]}, {sublist[1]}" for sublist in topology]
    #
    #     elements = pd.DataFrame({'Element': ['Beam splitter'] * self.topology.shape[0],
    #                              'Angles': angles,
    #                              'Modes': topology})
    #
    #     if self.initial_ancilla_state.size > 0:
    #         modes_in = self.initial_ancilla_state.reshape(-1)
    #         # TODO: print all the measurements
    #         modes_out = self.measurements[0]
    #
    #         ancillas = pd.DataFrame({'Mode in': modes_in,
    #                                  'Mode out': modes_out})
    #         ancillas.index.name = 'Ancilla photon'
    #
    #         print(elements, ancillas, sep='\n')
    #     else:
    #         print(elements)

    def to_text_file(self, filename, transform, basic_states_probabilities_match, entanglement_entropies, pr, device):
        # just for testing
        print("filename: ", filename)
        transform = transform.conj()
        # just for testing

        ZX = torch.tensor(
            [[0.5 + 0j, 0.5 + 0j, 0.5 + 0j, 0.5 + 0j], [0.5 + 0j, -0.5 + 0j, 0.5 + 0j, -0.5 + 0j],
             [0.5 + 0j, 0.5 + 0j, -0.5 + 0j, -0.5 + 0j], [0.5 + 0j, -0.5 + 0j, -0.5 + 0j, 0.5 + 0j]], device=device)
        XZ = torch.inverse(ZX)

        ZY = torch.tensor(
            [[0.5 + 0j, 0.5 + 0j, 0.5 + 0j, 0.5 + 0j], [0. + 0.5j, 0. - 0.5j, 0. + 0.5j, 0. - 0.5j],
             [0. + 0.5j, 0. + 0.5j, 0. - 0.5j, 0. - 0.5j], [-0.5 + 0j, 0.5 + 0j, 0.5 + 0j, -0.5 + 0j]], device=device)
        YZ = torch.inverse(ZY)
        reshaped = transform.reshape(4, 4)
        # add result entropy add basic_prob_match add probs
        new_transforms_Z = reshaped.transpose(1, 0)
        sums_Z = new_transforms_Z.abs().square().sum(1).sqrt()
        out_Z = new_transforms_Z / sums_Z.unsqueeze(-1)
        reses_Z = [(rho_entropy(vector)) for vector in out_Z]
        data_Z = {
            "00": (str(out_Z[0]), reses_Z[0].abs().item(), str(sums_Z[0])),
            "10": (str(out_Z[1]), reses_Z[1].abs().item(), str(sums_Z[1])),
            "01": (str(out_Z[2]), reses_Z[2].abs().item(), str(sums_Z[2])),
            "11": (str(out_Z[3]), reses_Z[3].abs().item(), str(sums_Z[3]))
        }
        print(transform)
        new_transforms_X = torch.matmul(torch.matmul(XZ, reshaped), ZX).transpose(1, 0)
        print(new_transforms_X)
        sums_X = new_transforms_X.abs().square().sum(1).sqrt()
        print(sums_X)
        out_X = new_transforms_X / sums_X.unsqueeze(-1)
        print(out_X)
        reses_X = [(rho_entropy(vector)) for vector in out_X]
        print(reses_X)
        data_X = {
            "++": (str(out_X[0]), reses_X[0].abs().item(), str(sums_X[0])),
            "+-": (str(out_X[1]), reses_X[1].abs().item(), str(sums_X[1])),
            "-+": (str(out_X[2]), reses_X[2].abs().item(), str(sums_X[2])),
            "--": (str(out_X[3]), reses_X[3].abs().item(), str(sums_X[3]))
        }
        print("NOW Y")
        new_transforms_Y = torch.matmul(torch.matmul(YZ, reshaped), ZY).transpose(1, 0)
        print(new_transforms_Y)
        sums_Y = new_transforms_Y.abs().square().sum(1).sqrt()
        print(sums_Y)
        out_Y = new_transforms_Y / sums_Y.unsqueeze(-1)
        print(out_Y)
        reses_Y = [(rho_entropy(vector)) for vector in out_Y]
        print(reses_Y)
        data_Y = {
            "y1y1": (str(out_Y[0]), reses_Y[0].abs().item(), str(sums_Y[0])),
            "y1y2": (str(out_Y[1]), reses_Y[1].abs().item(), str(sums_Y[1])),
            "y2y1": (str(out_Y[2]), reses_Y[2].abs().item(), str(sums_Y[2])),
            "y2y2": (str(out_Y[3]), reses_Y[3].abs().item(), str(sums_Y[3]))
        }
        with open(filename, 'w') as f:
            f.write("Minimum probability: " + str(pr[0].item()) + "\n")
            f.write("Maximum entanglement: " + str(entanglement_entropies[0].item()) + "\n")
            f.write("Basic_states_probabilities_match: " + str(basic_states_probabilities_match[0].item()) + "\n")
            f.write("This is the transform matrix of a circuit in Z basis:\n")
            f.write(str(new_transforms_Z[0]) + "\n")
            f.write(str(new_transforms_Z[1]) + "\n")
            f.write(str(new_transforms_Z[2]) + "\n")
            f.write(str(new_transforms_Z[3]) + "\n")
            f.write(json.dumps(data_Z, indent=10))
            f.write("This is the transform matrix of a circuit in X basis:\n")
            f.write(str(new_transforms_X[0]) + "\n")
            f.write(str(new_transforms_X[1]) + "\n")
            f.write(str(new_transforms_X[2]) + "\n")
            f.write(str(new_transforms_X[3]) + "\n")
            f.write(json.dumps(data_X, indent=10))
            f.write("This is the transform matrix of a circuit in Y basis:\n")
            f.write(str(new_transforms_Y[0]) + "\n")
            f.write(str(new_transforms_Y[1]) + "\n")
            f.write(str(new_transforms_Y[2]) + "\n")
            f.write(str(new_transforms_Y[3]) + "\n")
            f.write(json.dumps(data_Y, indent=10))

    def to_loqc_tech(self, filename):
        photon_sources = []
        for i in range(self.n_modes):
            source = {
                "id": "in" + str(self.n_modes - 1 - i),
                "type": "IN",
                "theta": "undefined",
                "phi": "undefined",
                "n": "-",
                "input_type": "0",
                "x": 50,
                "y": 50 + 85 * i
            }
            photon_sources.append(source)
        for source in photon_sources[:2]:
            source["input_type"] = "2"
        for source in photon_sources[self.n_state_modes:self.n_modes]:
            source["input_type"] = "1"

        connections = []
        n_connection = 0
        frontier = [("in" + str(i), "hybrid0") for i in range(self.n_modes)]

        bs_angles = self.bs_angles.reshape(-1, 2)

        counter = 0
        right_edge = 0
        beam_splitters = []
        for i, j in self.topology:
            if abs(bs_angles[counter][0]) < 0.0001:
                counter += 1
                continue

            # id = "bs" + str(i) + "_" + str(j) + "_" + str(counter)
            id = "bs" + str(counter)
            beam_splitter = {
                "id": id,
                "type": "BS",
                "theta": str(180. * bs_angles[counter][0] / pi),
                "phi": str(180. * bs_angles[counter][1] / pi),
                "n": "undefined",
                "input_type": "undefined",
                "x": str(int(50 + 500 * (right_edge + 1) / len(self.topology))),
                "y": str(int(40 + 85 * (self.n_modes - 1) - 42.5 * (i + j)))
            }
            beam_splitters.append(beam_splitter)

            connection0 = {
                "id": "c" + str(n_connection),
                "type": "draw2d.Connection",
                "router": "draw2d.layout.connection.ManhattanConnectionRouter",
                "source": {"node": frontier[j][0], "port": frontier[j][1]},
                "target": {"node": id, "port": "hybrid0"}
            }
            n_connection += 1
            connection1 = {
                "id": "c" + str(n_connection),
                "type": "draw2d.Connection",
                "router": "draw2d.layout.connection.ManhattanConnectionRouter",
                "source": {"node": frontier[i][0], "port": frontier[i][1]},
                "target": {"node": id, "port": "hybrid2"}
            }
            n_connection += 1
            connections.append([connection0])
            connections.append([connection1])

            frontier[i] = id, "hybrid3"
            frontier[j] = id, "hybrid1"

            counter += 1
            right_edge += 1

        ps_angles = self.ps_angles.reshape(-1)

        phase_shifters = []
        for i in range(self.n_modes):
            if abs(ps_angles[-1 - i]) < 0.0001:
                continue

            id = "ps" + str(self.n_modes - 1 - i)
            phase_shifter = {
                "id": id,
                "type": "PS",
                "theta": "undefined",
                "phi": str(180. * ps_angles[-1 - i] / pi),
                "n": "undefined",
                "input_type": "undefined",
                "x": 750,
                "y": 50 + 85 * i
            }
            phase_shifters.append(phase_shifter)

            connection = {
                "id": "c" + str(n_connection),
                "type": "draw2d.Connection",
                "router": "draw2d.layout.connection.ManhattanConnectionRouter",
                "source": {"node": frontier[-1 - i][0], "port": frontier[-1 - i][1]},
                "target": {"node": id, "port": "hybrid0"}
            }
            n_connection += 1
            connections.append([connection])

            frontier[-1 - i] = id, "hybrid1"

        photon_detections = []
        for i in range(self.n_modes):
            id = "out" + str(self.n_modes - 1 - i)
            detection = {
                "id": id,
                "type": "OUT",
                "theta": "undefined",
                "phi": "undefined",
                "n": "-",
                "input_type": "1",
                "x": 800,
                "y": 50 + 85 * i
            }
            photon_detections.append(detection)

            connection = {
                "id": "c" + str(n_connection),
                "type": "draw2d.Connection",
                "router": "draw2d.layout.connection.ManhattanConnectionRouter",
                "source": {"node": frontier[-1 - i][0], "port": frontier[-1 - i][1]},
                "target": {"node": id, "port": "hybrid0"}
            }
            n_connection += 1
            connections.append([connection])

        n_ancilla_modes = self.n_modes - self.n_state_modes

        ancillas_in = [0] * n_ancilla_modes
        if self.initial_ancilla_state.size > 0:
            ancilla_state = self.initial_ancilla_state.reshape(-1)
            for i in ancilla_state:
                ancillas_in[i] += 1
        for i in range(n_ancilla_modes):
            photon_sources[self.n_modes - 1 - i]["n"] = ancillas_in[i]

        ancillas_out = [0] * n_ancilla_modes
        if self.measurements.size > 0:
            measurements = self.measurements.reshape(-1, self.initial_ancilla_state.size)
            for i in measurements[0]:
                ancillas_out[i] += 1
        for i in range(n_ancilla_modes):
            photon_detections[self.n_modes - 1 - i]["n"] = ancillas_out[i]

        data = {
            "objects": photon_sources + beam_splitters + phase_shifters + photon_detections,
            "connections": connections
        }

        with open(filename, 'w') as f:
            f.write(json.dumps(data))
