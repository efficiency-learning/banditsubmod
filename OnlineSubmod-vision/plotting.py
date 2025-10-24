import pylab
import matplotlib.pyplot as plt
import numpy as np
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

'''
ours [88.7(slow)@0.33, 89.6@21319, 0.943@21000 ] ==> thse are run on slow sl2
gradmatch [91.75@8429,  92@11981, 94.1@14382] ==> your logs
craig [ 0.7861@945, 0.9392@21468,  0.9478@23817] ==> your logs
glister [0.92@12182,0.94@14272, 0.95@16225] ==> your logs
'''
frac_wise_accuracy__ours = [88.7/95, 91.6/95, 94.5/95]
frac_wise_accuracy_gradm  = [91.75/95, 92/95, 94.1/95]
frac_wise_accuracy__craig = [78.6/95, 89.92/95, 94.78/95]
frac_wise_accuracy__glister = [92/95, 94/95, 93.63/95]

frac_wise_speedup_gradm = [0.158986175, 0.31797235, 0.52113]
frac_wise_speedup_glister = [0.2327188941,0.33640553,0.470046083]
frac_wise_speedup_ours = [0.14516129, 0.34483871, 0.5100]
frac_wise_speedup_craig = [0.15124424,0.35483871,0.4268254]
#ax.imshow([[.3,.3], [.1,.1]], cmap='Pastel2_r', vmin=0.14, vmax=0.5, aspect='auto', interpolation='bicubic')

line = ax.plot(frac_wise_speedup_gradm, frac_wise_accuracy_gradm, label='GradM', color='red',linestyle='dashed', marker='o',markersize=10) 
line = ax.plot(frac_wise_speedup_glister, frac_wise_accuracy__glister, label='Glister', color='blue',linestyle='dashed',marker='d',markersize=10)
line = ax.plot(frac_wise_speedup_ours, frac_wise_accuracy__ours, label='OnlineSubmod', color='green',linestyle='dashed',marker='v',markersize=10)
line = ax.plot(frac_wise_speedup_craig, frac_wise_accuracy__craig, label='Craig', color='orange', linestyle='dashed',marker='*',markersize=10)
ax.xaxis.grid(True, linestyle='--')
ax.yaxis.grid(True, linestyle='--')
ax.set_yscale('log')
ax.set_xscale('log')
ax.legend(fontsize=14)

yticks  = np.linspace(0.84, 1.02, num=10)
xticks = np.linspace(0.1, .55, num=10)
ax.set_yticklabels(['${{{:.2f}}}$'.format(ytick) for ytick in yticks], minor=True,fontsize=16)
ax.set_xticklabels(['${{{:.2f}}}$'.format(xtick) for xtick in xticks], minor=True, fontsize=16)
ax.tick_params(axis='y', labelsize=14)
ax.tick_params(axis='x', labelsize=14)

#plt.gca().xaxis.set_major_locator(plt.LogLocator(base=2, numticks=10))
#plt.gca().yaxis.set_major_locator(plt.LogLocator(base=2, numticks=10))

ax.set_xlabel('Speedup',fontsize=20)
ax.set_ylabel('Relative Accuracy',fontsize=20)
ax.xaxis.grid(True,which='minor', linestyle='--')
ax.yaxis.grid(True,which='minor',linestyle='--')
pylab.show()