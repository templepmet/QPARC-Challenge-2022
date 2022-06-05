import matplotlib.pyplot as plt
import re
import numpy as np

# allowed_shots = ['1.25e7', '2.5e7', '5e7', '1e8']
allowed_shots = ['1e7', '2e7', '3e7', '4e7', '5e7', '6e7', '7e7', '8e7', '9e7', '1e8']
methods = ['heat-annealing', 'BFGS', 'Powell', 'Nelder-Mead']

methods_label = {}
methods_label['heat-annealing'] = 'Proposed SA'
methods_label['BFGS'] = 'BFGS'
methods_label['Powell'] = 'Powell'
methods_label['Nelder-Mead'] = 'Nelder-Mead'

markers = {}
markers['heat-annealing'] = 'x'
markers['BFGS'] = '.'
markers['Powell'] = '*'
markers['Nelder-Mead'] = '^'

# methods = ['heat-annealing']
count = 5

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
    result_energys = []
    fci_energy = 0
    for i in range(count):
      with open('../data/{}_{}_{}.txt'.format(method, shots, i+1)) as f:
        # elem_enegry = float(re.findall('Average accuracy: (.*)\n', f.read())[0])
        text = f.read()
        result_energys += [float(res) for res in re.findall('Resulting energy = (.*)\n', text)]
        fci_energy = float(re.findall('FCI energy = (.*)\n', text)[0])
        # get_ave_energy = float(re.findall('Average accuracy: (.*)\n', text)[0])
    # print(fci_energy)
    # print(result_energys)
    abs_energy = [np.abs(res - fci_energy) for res in result_energys]

    std_energy = np.std(abs_energy)
    ave_energy = np.average(abs_energy)

    # plus_energy = []
    # minus_energy = []
    # for res in result_energys:
    #   if res > ave_energy:
    #     plus_energy.append(res - ave_energy)
    #   else:
    #     minus_energy.append(ave_energy - res)

    # min_energy = np.min(abs_energy)
    # max_energy = np.max(abs_energy)
    y[method].append(ave_energy)
    # y_minerr[method].append(min_energy)
    # y_maxerr[method].append(max_energy)

    # y_minerr[method].append(min_std_err)
    # y_maxerr[method].append(max_std_err)
    y_stderr[method].append(std_energy)

plt.rcParams["font.size"] = 16

def fig_all():
  plt.figure()
  plt.xlabel('MAX_SHOTS')
  plt.ylabel('Average accuracy')

  max_value = 0.0
  for method in methods:
    # plt.errorbar(x, y[method], yerr=[y_minerr[method], y_maxerr[method]], label=methods_label[method], fmt='.', markersize=10, linestyle='-', capsize=5)
    max_value = max(max_value, np.max(np.array(y[method]) + np.array(y_stderr[method])))
    plt.plot(x, y[method], label=methods_label[method], marker=markers[method], markersize=10, linestyle='-')
    # plt.errorbar(x, y[method], yerr=y_stderr[method], label=methods_label[method], fmt=markers[method], markersize=10, linestyle='-', capsize=5)

  plt.hlines(0.06784151163,0,1e9,linestyles='dotted',label="HF energy", color='black')
  plt.xlim(0.7e7, 1.1e8)
  plt.ylim(0.0, max_value * 1.1)
  plt.legend(bbox_to_anchor=(1, 1), loc='upper left')
  plt.savefig('energy_all.png', bbox_inches='tight', pad_inches=0.2)

def fig_all_err():
  plt.figure()
  plt.xlabel('MAX_SHOTS')
  plt.ylabel('Average accuracy')

  max_value = 0.0
  for method in methods:
    # plt.errorbar(x, y[method], yerr=[y_minerr[method], y_maxerr[method]], label=methods_label[method], fmt='.', markersize=10, linestyle='-', capsize=5)
    max_value = max(max_value, np.max(np.array(y[method]) + np.array(y_stderr[method])))
    # plt.plot(x, y[method], label=methods_label[method], marker=markers[method], markersize=10, linestyle='-')
    plt.errorbar(x, y[method], yerr=y_stderr[method], label=methods_label[method], fmt=markers[method], markersize=10, linestyle='-', capsize=5)

  plt.hlines(0.06784151163,0,1e9,linestyles='dotted',label="HF energy", color='black')
  plt.xlim(0.7e7, 1.1e8)
  plt.ylim(0.0, max_value * 1.1)
  plt.legend(bbox_to_anchor=(1, 1), loc='upper left')
  plt.savefig('energy_all_err.png', bbox_inches='tight', pad_inches=0.2)

def fig_zoom():
  plt.figure()
  plt.xlabel('MAX_SHOTS')
  plt.ylabel('Average accuracy')

  max_value = 0.0
  for method in methods:
    # plt.errorbar(x, y[method], yerr=[y_minerr[method], y_maxerr[method]], label=methods_label[method], fmt='.', markersize=10, linestyle='-', capsize=5)
    max_value = max(max_value, np.max(np.array(y[method]) + np.array(y_stderr[method])))
    plt.plot(x, y[method], label=methods_label[method], marker=markers[method], markersize=10, linestyle='-')
    # plt.errorbar(x, y[method], yerr=y_stderr[method], label=methods_label[method], fmt=markers[method], markersize=10, linestyle='-', capsize=5)

  plt.hlines(0.06784151163,0,1e9,linestyles='dotted',label="HF energy", color='black')
  plt.xlim(0.7e7, 1.1e8)
  plt.ylim(0.0, 0.3)
  plt.legend(bbox_to_anchor=(1, 1), loc='upper left')
  plt.savefig('energy_zoom.png', bbox_inches='tight', pad_inches=0.2)

def fig_zoom_err():
  plt.figure()
  plt.xlabel('MAX_SHOTS')
  plt.ylabel('Average accuracy')

  max_value = 0.0
  for method in methods:
    # plt.errorbar(x, y[method], yerr=[y_minerr[method], y_maxerr[method]], label=methods_label[method], fmt='.', markersize=10, linestyle='-', capsize=5)
    max_value = max(max_value, np.max(np.array(y[method]) + np.array(y_stderr[method])))
    # plt.plot(x, y[method], label=methods_label[method], marker=markers[method], markersize=10, linestyle='-')
    plt.errorbar(x, y[method], yerr=y_stderr[method], label=methods_label[method], fmt=markers[method], markersize=10, linestyle='-', capsize=5)

  plt.hlines(0.06784151163,0,1e9,linestyles='dotted',label="HF energy", color='black')
  plt.xlim(0.7e7, 1.1e8)
  plt.ylim(0.0, 0.3)
  plt.legend(bbox_to_anchor=(1, 1), loc='upper left')
  plt.savefig('energy_zoom_err.png', bbox_inches='tight', pad_inches=0.2)

fig_all()
fig_zoom()
fig_all_err()
fig_zoom_err()
