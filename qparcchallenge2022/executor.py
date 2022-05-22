# You are not allowed to import qulacs.QuantumState,
# nor other quantum circuit simulators.

from functools import partial
from random import randint

import numpy as np
from openfermion import FermionOperator
from openfermion.transforms import jordan_wigner

from qparcchallenge2022 import (
    QulacsExecutor,
    TotalShotsExceeded,
    create_observable_from_openfermion_text,
)
from qparcchallenge2022.ansatz import AnsatzType, get_ansatz_generator
from qparcchallenge2022.qparc import MAX_SHOTS
from qparcchallenge2022.WATLE import get_energy_EX, hamil_think, make_pair_patan


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
    if ret < executor.current_value:
        executor.current_value = ret
    return ret


def _get_qulacs_hamiltonian(executor):
    # get the problem hamiltonian.
    fermionic_hamiltonian, n_qubits = executor.get_problem_hamiltonian()
    # change hamiltonian's qubit order so that we have near
    mapping = []

    for i in range(n_qubits//2):
        mapping.append(i)
        mapping.append(i+n_qubits//2)
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


def run_single_experiment(
    *, n_shots, initial_state, depth, executor, debug_shots=0, fin_shots=50000, annel_num=4
):
    # get the problem hamiltonian.
    qulacs_hamiltonian, n_qubits = _get_qulacs_hamiltonian(executor)

    circuit_generator, theta_len = get_ansatz_generator(
        name=AnsatzType.ry_ansatz_circuit, n_qubits=n_qubits, depth=depth
    )
    pata_data = make_pair_patan(n_qubits)
    hamiltonian_data = hamil_think(qulacs_hamiltonian)

    # This block runs the VQE algorithm.
    # The result will be finalized when executor.record_result() is called.
    cost_func = partial(
        _cost,
        initial_state=initial_state,
        hamiltonian_data=hamiltonian_data,
        executor=executor,
        n_qubits=n_qubits,
        circuit_generator=circuit_generator,
    )

    # optimize using annealing method.
    
    #loop_times = int(MAX_SHOTS / (n_shots * 2 * evaluation_cost_of_hamiltonian))
    start_heat = 1.5
    pow_atai = 1.2

    min_score = 1e9
    scores=[]
    try:
        for aaaaa in range(annel_num):

            theta_list = np.random.random(theta_len) * 1.1
            # 1回のループでどれだけのshotを消費するか予測する

            fin_loop_ratio = fin_shots / (n_shots*2)
            loop_num = 0
            shot_per_loop = 1
            annel_shots = MAX_SHOTS/annel_num
            inn_shots = executor.total_shots

            while (executor.total_shots-inn_shots) + (fin_loop_ratio + 10) * shot_per_loop < annel_shots:
                # 1. get slope of theta_list[target_index]
                # 2. move theta_list[target_index] by current_heat * (slope of theta_list[target_index])
                # 3. reduce current_heat using heat_reduce_factor (like cooling)
                target_idx = randint(0, theta_len - 1)

                theta_list[target_idx] += np.pi * 0.5
                pata_data = make_pair_patan(n_qubits)
                cost_A = cost_func(theta_list,n_shots=n_shots,pata_data=pata_data)
                theta_list[target_idx] -= np.pi
                cost_B = cost_func(theta_list,n_shots=n_shots,pata_data=pata_data)
                theta_list[target_idx] += np.pi * 0.5
                slope = cost_A - cost_B

                loop_num+=1
                shot_per_loop = (executor.total_shots-inn_shots) / loop_num

                used_wariai = (executor.total_shots-inn_shots) / (annel_shots - fin_loop_ratio * shot_per_loop)
                current_heat = start_heat * ((1-used_wariai)** pow_atai)

                change_kak = slope * current_heat
                if change_kak > 1.0:
                    change_kak = 1.0
                if change_kak < -1.0:
                    change_kak = -1.0
                theta_list[target_idx] -= change_kak

                if loop_num % 20 == 1:
                    print ("used_wariai=" , used_wariai,"current_heat=" , current_heat)
                    print ("loop_num=",loop_num, "/" , loop_num / used_wariai)
                    if debug_shots > 1:
                        pre_debug=executor.total_shots
                        print("energy=",cost_func(theta_list,n_shots=debug_shots,pata_data=pata_data))
                        executor.total_shots=pre_debug
                        print()
                        # this is use ONLY DEBUG MODE 

            # run the optimized theta_list and record the final result.
            now_score = cost_func(theta_list,n_shots=fin_shots,pata_data=pata_data)
            if min_score > now_score:
                min_score = now_score
            print("XXXXXXXXXXXXXXXXXXXXXXXXXXXX")
            print("now_score=",now_score)
            print("XXXXXXXXXXXXXXXXXXXXXXXXXXXX")
            scores.append(now_score)
        print("XXXXXXXXXXXXXXXXXXXXXXXXXXXX")
        print("min_score=",min_score)
        print("XXXXXXXXXXXXXXXXXXXXXXXXXXXX")
    except TotalShotsExceeded:
        print(
            f"Terminated because total shots exceeded the limit of MAX_SHOTS = {MAX_SHOTS}"
        )
    finally:
        min_score
