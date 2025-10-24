import pylab
import matplotlib.pyplot as plt
import numpy as np
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

lambda_values = [0.1, 0.2, 0.6]
validation_accuracy = [0.78, 0.82, 0.86]
line = ax.plot(lambda_values, validation_accuracy, label='GradM', color='red',linestyle='dashed', marker='o') 
#line = ax.plot(frac_wise_speedup_glister, frac_wise_accuracy__glister, label='Glister', color='blue',linestyle='dashed',marker='o')
#line = ax.plot(frac_wise_speedup_ours, frac_wise_accuracy__ours, label='Ours', color='green',linestyle='dashed',marker='o')
#line = ax.plot(frac_wise_speedup_craig, frac_wise_accuracy__craig, label='Craig', color='orange', linestyle='dashed',marker='o')
#ax.set_ylim([0.75,1.1])
#ax.set_xlim([0.1,1.2])
ax.set_yscale('log')
ax.set_xscale('log')
ax.legend()
#ax.set_xticks([0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
ax.set_yticks([0.78, 0.8, 0.82, 0.84, 0.86, 0.88, 0.9, 0.92, 0.94, 0.96, 0.98, 1.0])
ax.xaxis.grid(True, linestyle='--')
ax.yaxis.grid(True, linestyle='--')
ax.set_xlabel('Speedup')
ax.set_ylabel('Relative Accuracy')
pylab.show()