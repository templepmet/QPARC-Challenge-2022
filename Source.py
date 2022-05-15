# You are not allowed to import qulacs.QuantumState,
# nor other quantum circuit simulators.

from functools import partial

import numpy as np
from openfermion import FermionOperator
from openfermion.transforms import jordan_wigner
from scipy.optimize import minimize

from qparcchallenge2022.ansatz import AnsatzType, get_ansatz
from qparcchallenge2022.library import get_energy
from qparcchallenge2022.qparc import (
    QulacsExecutor,
    TotalShotsExceeded,
    create_observable_from_openfermion_text,
)

# Define functions to be used in the optimization process.


def cost(
    theta_list,
    *,
    initial_state: int,
    n_qubits: int,
    n_shots: int,
    depth: int,
    hamiltonian: FermionOperator,
    executor: QulacsExecutor
):
    circuit = get_ansatz(
        name=AnsatzType.ry_ansatz_circuit,
        n_qubits=n_qubits,
        depth=depth,
        theta_list=theta_list,
    )
    ret = get_energy(
        n_shots=n_shots,
        state=initial_state,
        hamiltonian=qulacs_hamiltonian,
        executor=executor,
        circuit=circuit,
    )
    # executor.current_value will be used as the final result when no. of hots reach the limit,
    # so set the current value as often as possible, if you find a better energy.
    if ret < executor.current_value:
        executor.current_value = ret
    return ret


def callback(theta_list, *, executor: QulacsExecutor):
    print("current val", executor.current_value)
    print("current theta", theta_list)


# def grad(theta_list):
#    ret = get_gradient(
#        theta_list=theta_list,
#        n_shots=n_shots,
#        depth=depth,
#        state=initial_state,
#        hamiltonian=qulacs_hamiltonian,
#    )
#    return ret

# Set up the executor, and get the problem hamiltonian.
# One must run quantum circuits always through the executor.
executor = QulacsExecutor()
fermionic_hamiltonian, n_qubits = executor.get_problem_hamiltonian()

# Process the Hamiltonian.
jw_hamiltonian = jordan_wigner(fermionic_hamiltonian)
qulacs_hamiltonian = create_observable_from_openfermion_text(str(jw_hamiltonian))

# Define input settings
n_shots = 10000
initial_state = 0b00001111
depth = 2


# This block runs the VQE algorithm.
# The result will be finalized when executor.record_result() is called.

init_theta_list = np.random.random(n_qubits * (depth + 1)) * 0.01
method = "Nelder-Mead"
options = {"disp": True, "maxiter": 10000}
try:
    opt = minimize(
        fun=partial(
            cost,
            n_qubits=n_qubits,
            n_shots=n_shots,
            depth=depth,
            hamiltonian=qulacs_hamiltonian,
            initial_state=initial_state,
            executor=executor,
        ),
        x0=init_theta_list,
        method=method,
        options=options,
        callback=partial(callback, executor=executor),
    )
except TotalShotsExceeded as e:
    print(e)
else:
    print("Optimization finished without reaching the shot limit")
finally:
    executor.record_result()
