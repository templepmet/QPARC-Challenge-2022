# You are not allowed to import qulacs.QuantumState,
# nor other quantum circuit simulators.

from logging import NullHandler, getLogger

import numpy as np
from qparcchallenge2022 import TotalShotsExceeded
from scipy.optimize import minimize

logger = getLogger(__name__)
logger.addHandler(NullHandler())


def scipy_optimizer(
    cost_func,
    *,
    theta_len,
    executor,
    method,
    options={"disp": True, "maxiter": 10000},
    grad=(),
):
    init_theta_list = np.random.random(theta_len) * 1.1
    try:
        result_minimize = minimize(
            cost_func,
            init_theta_list,
            method=method,
            jac=grad,
            options=options,
        )
        print("Optimization finished without reaching the shot limit.")

        print("Now Runs final evaluation for optimized value.")
        print(
            "WARNING: we changed total_shots value so that for contest we can't use this result."
        )
        final_x = result_minimize.x
        before_final_check = executor.total_shots

        executor.total_shots = 0
        result = cost_func(final_x, n_shots=50000)
        executor.current_value = result

        executor.total_shots = before_final_check
    except TotalShotsExceeded as e:
        print(e)
        print("WARNING: We don't run final evaluation for optimized value.")
        print("This indicates that result may have statistics error.")
    finally:
        executor.record_result()
