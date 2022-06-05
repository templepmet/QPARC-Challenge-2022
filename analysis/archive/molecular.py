import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.size"] = 16

plt.figure()
x1 = [1, 2]
y1 = [184, 919]
x2 = [1.3, 2.3]
y2 = [9, 47]
label_x = ['H4', 'H6']
# 1つ目の棒グラフ
plt.bar(x1, y1, width=0.3, label='Baseline', align="center")
# 2つ目の棒グラフ
plt.bar(x2, y2, width=0.3, label='Proposed', align="center")
# 凡例
plt.legend(loc=2)
# X軸の目盛りを置換


plt.xticks([1.15, 2.15], label_x)
plt.ylabel('Number of measurements')
plt.xlabel('Molecular')
plt.savefig('terms.png', bbox_inches='tight', pad_inches=0.20)
