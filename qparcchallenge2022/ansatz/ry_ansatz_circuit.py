from qulacs import QuantumCircuit
from qulacs.gate import CZ, RY

# Set up the ansatz


def ry_ansatz_circuit(n_qubits, depth, theta_list):
    """ry_ansatz_circuit
    Returns Ry ansatz circuit.

    Args:
        n_qubits:
            The number of qubit used
        depth:
            Depth of the circuit.
        theta_list:
            Rotation angles.
    Returns:
        circuit:
            Resulting Ry ansatz circuit.
    """
    circuit = QuantumCircuit(n_qubits)
    params_id = 0
    for _ in range(depth):
        for i in range(n_qubits):
            circuit.add_gate(
                RY(i, theta_list[params_id]),
            )
            params_id += 1
        for i in range(n_qubits // 2):
            circuit.add_gate(CZ(2 * i, 2 * i + 1))
        for i in range(n_qubits // 2 - 1):
            circuit.add_gate(CZ(2 * i + 1, 2 * i + 2))
    for i in range(n_qubits):
        circuit.add_gate(RY(i, theta_list[params_id]))
        params_id += 1

    return circuit
