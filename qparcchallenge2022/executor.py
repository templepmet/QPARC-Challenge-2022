# You are not allowed to import qulacs.QuantumState,
# nor other quantum circuit simulators.

import logging
from ast import List
from functools import partial

import numpy as np
from openfermion import FermionOperator
from openfermion.transforms import jordan_wigner

from qparcchallenge2022 import QulacsExecutor, create_observable_from_openfermion_text
from qparcchallenge2022.ansatz import AnsatzType, get_ansatz_generator
from qparcchallenge2022.optimizer.annealing_optimizer import heat_optimizer
from qparcchallenge2022.optimizer.scipy_optimizer import scipy_optimizer
from qparcchallenge2022.utils import get_energy_EX, hamil_think, make_pair_patan

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def run_single_experiment(
    *,
    method,
    n_shots,
    initial_state,
    depth,
    executor,
    debug_shots=0,
    fin_shots=50000,
    annel_num=4,
):
    # get the problem hamiltonian.
    qulacs_hamiltonian, n_qubits = _get_qulacs_hamiltonian(executor)

    circuit_generator, theta_len = get_ansatz_generator(
        name=AnsatzType.ry_ansatz_circuit, n_qubits=n_qubits, depth=depth
    )
    pata_data = make_pair_patan(n_qubits)
    hamiltonian_data = hamil_think(qulacs_hamiltonian)

    # This block runs the VQE algorithm.
    cost_func = partial(
        _cost,
        initial_state=initial_state,
        hamiltonian_data=hamiltonian_data,
        executor=executor,
        n_qubits=n_qubits,
        circuit_generator=circuit_generator,
        pata_data=pata_data,
        n_shots=n_shots,
    )

    # optimize using annealing method.
    if method == "heat-annealing":
        heat_optimizer(
            cost_func,
            get_slope_single,
            annel_num=annel_num,
            theta_len=theta_len,
            fin_shots=fin_shots,
            n_shots=n_shots,
            executor=executor,
            debug_shots=debug_shots,
        )
    else:
        scipy_optimizer(
            cost_func,
            theta_len=theta_len,
            executor=executor,
            method=method,
            grad=partial(get_grad, cost_func=cost_func),
        )


# below code is for internal.

# Define functions to be used in the optimization process.
def _cost(
    theta_list,
    *,
    initial_state: int,
    n_shots: int,
    pata_data,
    hamiltonian_data,
    executor: QulacsExecutor,
    circuit_generator,
    n_qubits,
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
    if ret > executor.current_value:
        executor.current_value = ret
    return ret


def get_slope_single(theta_list, cost_func, target_idx) -> float:
    theta_list[target_idx] += np.pi * 0.5
    cost_A = cost_func(theta_list)
    theta_list[target_idx] -= np.pi
    cost_B = cost_func(theta_list)
    theta_list[target_idx] += np.pi * 0.5
    return cost_A - cost_B


def get_grad(theta_list, cost_func):
    result = []
    for i in range(len(theta_list)):
        result.append(get_slope_single(theta_list, cost_func, i))
    return result


def _get_qulacs_hamiltonian(executor):
    # get the problem hamiltonian.
    fermionic_hamiltonian, n_qubits = executor.get_problem_hamiltonian()
    # change hamiltonian's qubit order so that we have near
    mapping = []

    for i in range(n_qubits // 2):
        mapping.append(i)
        mapping.append(i + n_qubits // 2)
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
    return qulacs_hamiltonian, n_qubits
