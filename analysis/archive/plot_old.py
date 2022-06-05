import matplotlib.pyplot as plt
import re
import numpy as np

allowed_shots = ['1.25e7', '2.5e7', '5e7', '1e8']
# allowed_shots = [1e7, 2e7, 3e7, 4e7, 5e7, 6e7, 8e7, 9e7, 1e8]
# methods = ['heat-annealing', 'BFGS', 'Powell', 'Nelder-Mead']
methods = ['annealing', 'BFGS', 'Powell', 'Nelder-Mead']
count = 3

x = []
for shots in allowed_shots:
  x.append(float(shots))


y = {}
y_annealing = []
y_powell = []
y_bfgs = []
y_nlder = []

for method in methods:
  y[method] = []
  for shots in allowed_shots:
    sum_energy = 0.0
    for i in range(count):
      with open('./data/{}-{}-{}.txt'.format(method, shots, i+1)) as f:
        elem_enegry = float(re.findall('Average accuracy: (.*)\n', f.read())[0])
        sum_energy += elem_enegry
    sum_energy /= count
    y[method].append(sum_energy)


plt.figure()

plt.xlabel('MAX_SHOTS')
plt.ylabel('Average accuracy')

for method in methods:
  plt.plot(x, y[method], label=method)

plt.legend()

plt.savefig('plot.png', bbox_inches='tight', pad_inches=0.05)
