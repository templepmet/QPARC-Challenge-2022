# You are not allowed to import qulacs.QuantumState,
# nor other quantum circuit simulators.

from functools import partial

import numpy as np
from openfermion import FermionOperator
from openfermion.transforms import jordan_wigner
from scipy.optimize import minimize

from qparcchallenge2022 import (
    QulacsExecutor,
    TotalShotsExceeded,
    create_observable_from_openfermion_text,
    get_energy,
    get_gradient,
)
from qparcchallenge2022.ansatz import AnsatzType, get_ansatz_generator

# Define functions to be used in the optimization process.


def cost(
    theta_list,
    *,
    initial_state: int,
    n_shots: int,
    hamiltonian: FermionOperator,
    executor: QulacsExecutor,
    circuit_generator
):
    ret = get_energy(
        n_shots=n_shots,
        state=initial_state,
        hamiltonian=qulacs_hamiltonian,
        executor=executor,
        circuit_generator=circuit_generator,
        theta_list=theta_list,
    )
    # executor.current_value will be used as the final result when no. of hots reach the limit,
    # so set the current value as often as possible, if you find a better energy.
    if ret < executor.current_value:
        executor.current_value = ret
    return ret


def callback(theta_list, *, executor: QulacsExecutor):
    print("current val", executor.current_value)
    print("current theta", theta_list)


def grad(theta_list, *, circuit_generator, n_shots, state, hamiltonian, executor):
    ret = get_gradient(
        theta_list=theta_list,
        n_shots=n_shots,
        circuit_generator=circuit_generator,
        state=initial_state,
        hamiltonian=qulacs_hamiltonian,
        executor=executor,
    )
    return ret


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

circuit_generator = get_ansatz_generator(
    name=AnsatzType.qulacs_ansatz_circuit, n_qubits=n_qubits, depth=depth
)

# This block runs the VQE algorithm.
# The result will be finalized when executor.record_result() is called.

init_theta_list = np.random.random(n_qubits * (depth + 1) * 2) * 1
method = "BFGS"
options = {"disp": True, "maxiter": 100}
try:
    opt = minimize(
        fun=partial(
            cost,
            circuit_generator=circuit_generator,
            n_shots=n_shots,
            hamiltonian=qulacs_hamiltonian,
            initial_state=initial_state,
            executor=executor,
        ),
        x0=init_theta_list,
        method=method,
        jac=partial(
            grad,
            circuit_generator=circuit_generator,
            n_shots=n_shots,
            state=initial_state,
            hamiltonian=qulacs_hamiltonian,
            executor=executor,
        ),
        options=options,
        callback=partial(callback, executor=executor),
    )
except TotalShotsExceeded as e:
    print(e)
else:
    print("Optimization finished without reaching the shot limit")
finally:
    executor.record_result()
