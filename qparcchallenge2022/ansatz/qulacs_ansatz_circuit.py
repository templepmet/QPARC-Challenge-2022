from qulacs import QuantumCircuit
from qulacs.gate import CZ, RY, RZ, merge

# ansatz for https://dojo.qulacs.org/ja/latest/notebooks/6.2_qulacs_VQE.html


def qulacs_ansatz_circuit(theta_list, *, n_qubits, depth):
    """qulacs_ansatz_circuit
    Returns qulacs ansatz circuit.

    Args:
        n_qubits:
            The number of qubit used
        depth:
            Depth of the circuit.
        theta_list:
            Rotation angles.
    Returns:
        circuit:
            Resulting qulacs ansatz circuit.
    """
    circuit = QuantumCircuit(n_qubits)
    for d in range(depth):
        for i in range(n_qubits):
            circuit.add_gate(
                merge(
                    RY(i, theta_list[2 * i + 2 * n_qubits * d]),
                    RZ(i, theta_list[2 * i + 1 + 2 * n_qubits * d]),
                )
            )
        for i in range(n_qubits // 2):
            circuit.add_gate(CZ(2 * i, 2 * i + 1))
        for i in range(n_qubits // 2 - 1):
            circuit.add_gate(CZ(2 * i + 1, 2 * i + 2))
    for i in range(n_qubits):
        circuit.add_gate(
            merge(
                RY(i, theta_list[2 * i + 2 * n_qubits * depth]),
                RZ(i, theta_list[2 * i + 1 + 2 * n_qubits * depth]),
            )
        )

    return circuit
