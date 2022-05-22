# You are not allowed to import qulacs.QuantumState,
# nor other quantum circuit simulators.

from functools import partial
from random import randint

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
from qparcchallenge2022.qparc import MAX_SHOTS
from qparcchallenge2022.WATLE import get_energy_EX, hamil_think, make_pair_patan

# Define functions to be used in the optimization process.


def cost(
    theta_list,
    *,
    initial_state: int,
    n_shots: int,
    pata_data,
    hamiltonian_data,
    executor: QulacsExecutor,
    circuit_generator,
    n_qubits
):
    ret = get_energy_EX(
        n_qubit=n_qubits,
        pata_data=pata_data,
        n_shots=n_shots,
        state=initial_state,
        hamiltonian_data=hamiltonian_data,
        executor=executor,
        circuit=circuit_generator(theta_list=theta_list),
    )
    ret = float(ret)
    # executor.current_value will be used as the final result when no. of hots reach the limit,
    # so set the current value as often as possible, if you find a better energy.
    if ret < executor.current_value:
        executor.current_value = ret
    print(ret)
    return ret


def callback(theta_list, *, executor: QulacsExecutor):
    print("current val", executor.current_value)
    print("current theta", theta_list)


# Set up the executor, and get the problem hamiltonian.
# One must run quantum circuits always through the executor.
executor = QulacsExecutor()
fermionic_hamiltonian, n_qubits = executor.get_problem_hamiltonian()

mapping = [0, 4, 1, 5, 2, 6, 3, 7]
my_opr = FermionOperator()
for aa in fermionic_hamiltonian.terms:
    kou = fermionic_hamiltonian.terms[aa]
    if abs(kou) < 1e-6:
        continue
    bb = list(aa)
    cc = []
    for enzan in bb:
        (qq, ww) = enzan
        cc.append((mapping[qq], ww))
    my_opr += FermionOperator(tuple(cc), kou)

# Process the Hamiltonian.
jw_hamiltonian = jordan_wigner(my_opr)
qulacs_hamiltonian = create_observable_from_openfermion_text(str(jw_hamiltonian))

# Define input settings
n_shots = 1000
evaluation_cost_of_hamiltonian = 18
initial_state = 0b10101010
depth = 2

circuit_generator, theta_len = get_ansatz_generator(
    name=AnsatzType.ry_ansatz_circuit, n_qubits=n_qubits, depth=depth
)

# This block runs the VQE algorithm.
# The result will be finalized when executor.record_result() is called.
pata_data = make_pair_patan(n_qubits)
hamiltonian_data = hamil_think(qulacs_hamiltonian)

cost_func = partial(
    cost,
    initial_state=initial_state,
    n_shots=n_shots,
    pata_data=pata_data,
    hamiltonian_data=hamiltonian_data,
    executor=executor,
    n_qubits=n_qubits,
    circuit_generator=circuit_generator,
)

# optimize using annealing method.
theta_list = np.random.random(theta_len) * 1.1
loop_times = int(MAX_SHOTS / (n_shots * 2 * evaluation_cost_of_hamiltonian))
current_heat = 1.5
heat_reduce_factor = 0.01 ** (1 / loop_times)

try:
    for loop_cnt in range(loop_times):
        # 1. get slope of theta_list[target_index]
        # 2. move theta_list[target_index] by current_heat * (slope of theta_list[target_index])
        # 3. reduce current_heat using heat_reduce_factor (like cooling)
        target_idx = randint(0, theta_len - 1)

        theta_list[target_idx] += np.pi * 0.5
        pata_data = make_pair_patan(n_qubits)
        cost_A = cost_func(theta_list)
        theta_list[target_idx] -= np.pi
        cost_B = cost_func(theta_list)
        theta_list[target_idx] += np.pi * 0.5
        slope = cost_A - cost_B

        change_kak = current_heat * slope
        if change_kak > 1.0:
            change_kak = 1.0
        if change_kak < -1.0:
            change_kak = -1.0
        theta_list[target_idx] -= change_kak
        current_heat *= heat_reduce_factor

except TotalShotsExceeded:
    print("Finished with totalshots exceeded")
finally:
    print(executor.current_value)
