# You are not allowed to import qulacs.QuantumState,
# nor other quantum circuit simulators.

from qparcchallenge2022 import QulacsExecutor
from qparcchallenge2022.executor import run_single_experiment

# Set up the executor
executor = QulacsExecutor()

# Define input settings
n_shots = 1000
evaluation_cost_of_hamiltonian = 18
initial_state = 0b10101010
depth = 2

run_single_experiment(
    n_shots=n_shots,
    evaluation_cost_of_hamiltonian=evaluation_cost_of_hamiltonian,
    initial_state=initial_state,
    depth=depth,
    executor=executor,
)
