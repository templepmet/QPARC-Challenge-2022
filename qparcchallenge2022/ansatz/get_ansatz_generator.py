from enum import Enum
from functools import partial

from qparcchallenge2022.ansatz.qulacs_ansatz_circuit import qulacs_ansatz_circuit
from qparcchallenge2022.ansatz.ry_ansatz_circuit import ry_ansatz_circuit


class AnsatzType(Enum):
    ry_ansatz_circuit = 1
    qulacs_ansatz_circuit = 2


def get_ansatz_generator(name: AnsatzType, *, n_qubits: int, depth: int):
    if name == AnsatzType.ry_ansatz_circuit:
        circuit_generator_module = ry_ansatz_circuit
    elif name == AnsatzType.qulacs_ansatz_circuit:
        circuit_generator_module = qulacs_ansatz_circuit
    return partial(circuit_generator_module, n_qubits=n_qubits, depth=depth)
