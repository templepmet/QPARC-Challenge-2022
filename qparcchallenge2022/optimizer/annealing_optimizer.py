# You are not allowed to import qulacs.QuantumState,
# nor other quantum circuit simulators.

from logging import NullHandler, getLogger
from random import randint

import numpy as np
from qparcchallenge2022 import TotalShotsExceeded
from qparcchallenge2022.qparc import MAX_SHOTS

logger = getLogger(__name__)
logger.addHandler(NullHandler())


def heat_optimizer(
    cost_func,
    get_slope_single,
    *,
    annel_num,
    theta_len,
    fin_shots,
    n_shots,
    executor,
    debug_shots,
):
    start_heat = 1.5
    pow_atai = 1.2

    min_score = 1e9
    scores = []
    try:
        for _ in range(annel_num):

            theta_list = np.random.random(theta_len) * 1.1
            # 1回のループでどれだけのshotを消費するか予測する

            fin_loop_ratio = fin_shots / (n_shots * 2)
            loop_num = 0
            shot_per_loop = 1
            annel_shots = MAX_SHOTS / annel_num
            inn_shots = executor.total_shots

            while (executor.total_shots - inn_shots) + (
                fin_loop_ratio + 10
            ) * shot_per_loop < annel_shots:
                # 1. get slope of theta_list[target_index]
                # 2. move theta_list[target_index] by current_heat * (slope of theta_list[target_index])
                # 3. reduce current_heat using heat_reduce_factor (like cooling)
                target_idx = randint(0, theta_len - 1)

                slope = get_slope_single(theta_list, cost_func, target_idx)

                loop_num += 1
                shot_per_loop = (executor.total_shots - inn_shots) / loop_num

                used_wariai = (executor.total_shots - inn_shots) / (
                    annel_shots - fin_loop_ratio * shot_per_loop
                )
                current_heat = start_heat * ((1 - used_wariai) ** pow_atai)

                change_kak = slope * current_heat
                if change_kak > 1.0:
                    change_kak = 1.0
                if change_kak < -1.0:
                    change_kak = -1.0
                theta_list[target_idx] -= change_kak

                if loop_num % 20 == 1:
                    logger.debug(
                        f"used_wariai={used_wariai}, current_heat={current_heat}"
                    )
                    logger.debug(f"loop_num= {loop_num} / {loop_num / used_wariai}")
                    if debug_shots > 1:
                        pre_debug = executor.total_shots
                        current_energy = cost_func(theta_list, n_shots=debug_shots)
                        logger.debug(f"energy={current_energy}")
                        executor.total_shots = pre_debug
                        # this is use ONLY DEBUG MODE

            # run the optimized theta_list and record the final result.
            now_score = cost_func(theta_list, n_shots=fin_shots)
            if min_score > now_score:
                min_score = now_score
            logger.info("XXXXXXXXXXXXXXXXXXXXXXXXXXXX")
            logger.info(f"now_score={now_score}")
            logger.info("XXXXXXXXXXXXXXXXXXXXXXXXXXXX")
            scores.append(now_score)
        logger.info("XXXXXXXXXXXXXXXXXXXXXXXXXXXX")
        logger.info(f"min_score={min_score}")
        logger.info("XXXXXXXXXXXXXXXXXXXXXXXXXXXX")
    except TotalShotsExceeded:
        print(
            f"Terminated because total shots exceeded the limit of MAX_SHOTS = {MAX_SHOTS}"
        )
        logger.error(
            f"Terminated because total shots exceeded the limit of MAX_SHOTS = {MAX_SHOTS}"
        )
    finally:
        executor.current_value = min_score
        executor.record_result()
        return min_score
