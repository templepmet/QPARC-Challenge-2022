from enum import Enum
from typing import List

from qparcchallenge2022.ansatz.ry_ansatz_circuit import ry_ansatz_circuit
from qulacs import QuantumCircuit


class AnsatzType(Enum):
    ry_ansatz_circuit = 1


def get_ansatz(
    name: AnsatzType, *, n_qubits: int, depth: int, theta_list: List[int]
) -> QuantumCircuit:
    return ry_ansatz_circuit(n_qubits, depth, theta_list)
