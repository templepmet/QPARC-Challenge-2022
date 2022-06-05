import matplotlib.pyplot as plt
import re
import numpy as np

# allowed_shots = ['1.25e7', '2.5e7', '5e7', '1e8']
allowed_shots = ['1e7', '2e7', '3e7', '4e7', '5e7', '6e7', '8e7']
# methods = ['heat-annealing', 'BFGS', 'Powell', 'Nelder-Mead']
methods = ['heat-annealing']
count = 1

x = []
for shots in allowed_shots:
  x.append(float(shots))


y = {}
y_minerr = {}
y_maxerr = {}
y_stderr = {}

for method in methods:
  y[method] = []
  y_minerr[method] = []
  y_maxerr[method] = []
  y_stderr[method] = []
  for shots in allowed_shots:
    sum_energy = 0.0
    with open('../data/new_data/{}_{}_1.txt'.format(method, shots)) as f:
      # elem_enegry = float(re.findall('Average accuracy: (.*)\n', f.read())[0])
      text = f.read()
      result_energys = [float(res) for res in re.findall('Resulting energy = (.*)\n', text)]
      fci_energy = float(re.findall('FCI energy = (.*)\n', text)[0])
      # get_ave_energy = float(re.findall('Average accuracy: (.*)\n', text)[0])
      abs_energy = [np.abs(res - fci_energy) for res in result_energys]
      min_energy = np.min(abs_energy)
      max_energy = np.max(abs_energy)
      std_energy = np.std(abs_energy)
      ave_energy = np.average(abs_energy)
      # print(min_energy)
      # print(max_energy)
      # print(ave_energy)
      y[method].append(ave_energy)
      y_minerr[method].append(min_energy)
      y_maxerr[method].append(max_energy)
      y_stderr[method].append(std_energy)


plt.figure()

plt.xlabel('MAX_SHOTS')
plt.ylabel('Average accuracy')

for method in methods:
  # plt.errorbar(x, y[method], yerr=[y_minerr[method], y_maxerr[method]], label=method, fmt='.', markersize=10, linestyle='-', capsize=5)
  plt.errorbar(x, y[method], yerr=y_stderr[method], label=method, fmt='.', markersize=10, linestyle='-', capsize=5)

plt.legend()

plt.savefig('plot.png', bbox_inches='tight', pad_inches=0.2)
