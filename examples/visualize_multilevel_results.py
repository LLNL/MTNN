import numpy as np
import matplotlib.pyplot as plt
import sys

# Example usage:
# python visualize_multilevel_results.py 1level_results.txt "1Level" 1.0 2level_results.txt "2Level" 2.5

cycle_arrays = []
loss_arrays = []
WU_multipliers = []
labelnames = []
for i in range(1, len(sys.argv) - 1, 3):
    fname = sys.argv[i]
    labelname = sys.argv[i+1]
    print(labelname)
    WU_multiplier = float(sys.argv[i+2])

    f = open(fname, 'r')
    cycles = dict()
    losses = dict()
    for line in f:
        if "After" in line:
            tokens = line.split()
            cycle_num = int(tokens[3])
            hier_level = int(tokens[1].rstrip(':'))
            loss = float(tokens[8].rstrip(',')) # L2 loss
#            loss = float(tokens[16].rstrip(',')) # Linf loss
            if hier_level not in losses:
                losses[hier_level] = []
                cycles[hier_level] = []
            losses[hier_level].append(loss)
            cycles[hier_level].append(cycle_num)
    f.close()
    cycle_arrays.append(cycles)
    loss_arrays.append(losses)
    WU_multipliers.append(WU_multiplier)
    labelnames.append(labelname)

for i, cycles in enumerate(cycle_arrays):
    for hier_level in loss_arrays[i]:
        lvl_cycles = WU_multipliers[i] * np.array(cycles[hier_level])
        plt.semilogy(lvl_cycles, loss_arrays[i][hier_level], label="{}, L{}".format(labelnames[i], hier_level))
        best_loss = np.nanmin(loss_arrays[i][hier_level])
        print("For {}, level {}, best loss is {}".format(labelname[i], hier_level, best_loss))

print(labelnames)
plt.legend()
plt.xlabel("Work Units")
plt.ylabel("Loss")
plt.show()
