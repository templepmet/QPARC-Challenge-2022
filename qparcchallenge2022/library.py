from typing import Counter

import numpy as np
from qulacs import QuantumCircuit
from qulacs.gate import H, Sdag


def get_terms_and_measurement_circuits(observable):
    """get_terms_and_measurement_circuits
    Returns basis-rotation circuits for measurement, along with the corresponding terms.

    Args:
        observable:
            The observable to be measured.
    Returns:
        pauli_coef:
            List of coefficients.
        pauli_target:
            List of targetted qubits.
        pauli_gate:
            List of circuits for basis-rotation.
    """
    pauli_coef = []
    pauli_target = []
    pauli_gate = []
    n_qubits = observable.get_qubit_count()
    for i_term in range(observable.get_term_count()):
        term = observable.get_term(i_term)
        pauli_coef.append(term.get_coef())
        target_list = term.get_index_list()
        pauli_target.append(target_list)
        id_list = term.get_pauli_id_list()
        circuit = QuantumCircuit(n_qubits)
        for target, id in zip(target_list, id_list):
            if id == 1:
                circuit.add_gate(H(target))
            elif id == 2:
                circuit.add_gate(Sdag(target))
                circuit.add_gate(H(target))
            elif id == 3:
                pass
            else:
                raise Exception(f"Operator {target, id} not supported")
        pauli_gate.append(circuit)
    return pauli_coef, pauli_target, pauli_gate


def get_energy(n_shots, state, hamiltonian, circuit, executor):
    """get_energy
    Returns the evaluated energy

    Args:
        theta_list:
            The parameters
        n_shots:
            The number of shots used to evaluate each term.
        depth:
            The depth of the ansatz
        state:
            The integer that defines the initial state in the computational basis.
        hamiltonian:
            The Hamiltonian to be evaluated.
    Returns:
        ret:
            The evaluated energy.
    """
    pauli_coef, pauli_target, pauli_gate = get_terms_and_measurement_circuits(
        hamiltonian
    )
    n_qubits = hamiltonian.get_qubit_count()
    ret = 0
    for coef, target, gate in zip(pauli_coef, pauli_target, pauli_gate):
        if target:
            counts = Counter(
                executor.sampling(
                    [circuit, gate],
                    state_int=state,
                    n_qubits=n_qubits,
                    n_shots=n_shots,
                )
            )
            for sample, count in counts.items():
                binary = np.binary_repr(sample).rjust(n_qubits, "0")
                measurement = np.product(
                    [-1 if binary[n_qubits - t - 1] == "1" else 1 for t in target]
                )
                ret += coef * measurement * count / n_shots
        else:
            ret += coef

    return ret.real
