# You are not allowed to import qulacs.QuantumState,
# nor other quantum circuit simulators.

import logging
import sys

from qparcchallenge2022 import QulacsExecutor
from qparcchallenge2022.executor import run_single_experiment

# setup logger
# logger = logging.getLogger()
# logger.setLevel(logging.DEBUG)
# handler = logging.FileHandler("./test.log")
# handler.setLevel(logging.DEBUG)
# formatter = logging.Formatter("%(levelname)s  %(asctime)s  [%(name)s] %(message)s")
# handler.setFormatter(formatter)
# logger.addHandler(handler)

# Set up the executor
executor = QulacsExecutor()

# Define input settings
n_shots = 1000
initial_state = 0b10101010
depth = 2

method = sys.argv[1]
# n_shots = int(float(sys.argv[2]))

print(method)

for i in range(10):
    run_single_experiment(
        method=method,
        n_shots=n_shots,
        initial_state=initial_state,
        depth=depth,
        executor=executor,
        debug_shots=0,
    )

executor.evaluate_final_result()
