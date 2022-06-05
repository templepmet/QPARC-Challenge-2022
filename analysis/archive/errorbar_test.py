from matplotlib import pyplot as plt
import random
import numpy as np

## テストデータの作成
x = [i+random.uniform(-0.2, 0.2) for i in range(5)] # 変数を初期化
y = [i+random.uniform(-0.5, 0.5) for i in range(5)]
x_err = [random.uniform(-0.5, 0.5) for i in range(5)] # 誤差範囲を乱数で生成
y_err = [random.uniform(-0.5, 0.5) for i in range(5)] 

# y軸方向にのみerrorbarを表示
plt.figure(figsize=(10,7))
plt.errorbar(x, y, yerr = y_err, capsize=5, fmt='o', markersize=10, ecolor='black', markeredgecolor = "black", color='w')
coef=np.polyfit(x, y, 1)
appr = np.poly1d(coef)(x)
plt.plot(x, appr,  color = 'black', linestyle=':')
y_pred = [coef[0]*i+coef[1] for i in x]

# plt.text(max(x)/3.7, max(y)*6/8, 'y={:.2f}x + {:.2f}, \n$R^2$={:.2f}'.format(coef[0], coef[1], r2), fontsize=15)
plt.xlabel('x', fontsize=20)
plt.ylabel('y', fontsize=20)
plt.savefig('y_error_bar.png')