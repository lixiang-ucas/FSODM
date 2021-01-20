import os
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    import sys
    # file_path = sys.argv[1]
    file_path = '../../test.CSV'

    if os.path.isfile(file_path):
        with open(file_path, 'r') as f:
            lines = f.readlines()

            X = lines[0].strip().split(',')
            X = [int(x) for x in X if x != '']

            Y = []
            Y_base = []
            for line in lines[1:]:
                y = line.strip().split(',')[1:]
                Y_base.append(float(y[-1]))
                y = [float(a) for a in y[:-1]]
                Y.append(y)
            Y = np.array(Y)

            plt.axhline(Y_base[0], ls='--', c='C0', label='airplane(all)', alpha=0.7)
            plt.axhline(Y_base[1], ls='--', c='C1', label='baseball diamond(all)', alpha=0.7)
            plt.axhline(Y_base[2], ls='--', c='C2', label='tennis court(all)', alpha=0.7)
            plt.axhline(Y_base[3], ls='--', c='C3', label='mean(all)', alpha=0.7)

            plt.plot(X, Y[0], color='C0', label='airplane', marker='^', linestyle=':')
            plt.plot(X, Y[1], color='C1', label='baseball diamond', marker='+', linestyle=':')
            plt.plot(X, Y[2], color='C2', label='tennis court', marker='*', linestyle=':')
            plt.plot(X, Y[3], color='C3', label='mean', marker='o', linestyle=':')

            plt.ylim(0, 1)
            plt.xlabel("shots")
            plt.ylabel("mAP")
            plt.legend()
            plt.savefig('../../few_base.png', bbox_inches='tight', pad_inches=0.05)
            plt.show()
