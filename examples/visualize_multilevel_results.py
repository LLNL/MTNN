import numpy as np
import matplotlib.pyplot as plt
import sys

# Example usage:
# python visualize_multilevel_results.py 1level_results.txt "1Level" 1.0 2level_results.txt "2Level" 2.5

def read_args(args):
    int_reader = lambda x : int(x)
    float_reader = lambda x : float(x)
    string_reader = lambda x : x
    bool_reader = lambda x : x.lower() in ("yes", "true", "t", "1")
    ensure_trailing_reader = lambda tr : lambda x : x.rstrip(tr) + tr
    array_reader = lambda element_reader : \
                   lambda x : [element_reader(z) for z in x.split(',')]

    # Define reader functions for each parameter                                                                                                                                                                                              
    reader_fns = { "file_names" : array_reader(string_reader),  # Array of data files
                   "label_names" : array_reader(string_reader), # Array of key names for the legend
                   "work_units" : array_reader(float_reader),   # Array of work unit multipliers associated with each data file
                   "save_prefix" : string_reader,               # Filename prefix to use, files will be [save_prefix]_l2.pdf and [save_prefix]_linf.pdf
                   "l2_limits" : array_reader(float_reader),    # If specified, 2-element array of upper and lower y axis limits for L2 plot.
                   "linf_limits" : array_reader(float_reader),  # If specified, 2-element array of upper and lower y axis limits for Linf plot.
                   "hier_levels" : array_reader(int_reader),    # Which hierarchy levels to display
                   "smoothing_window" : int_reader,             # Size of moving average window. Not specifying or set to 1 implies no smoothing.
                   "l2_stop_threshold": float_reader,           # If L2 loss goes over this threshold, stop reading data from that file and consider it ended there.
                   "linf_stop_threshold": float_reader}         # If Linf loss goes over this threshold, stop reading data from that file and consider it ended there.
    
    params_dict = dict()
    try:
        for a in args[1:]:
            tokens = a.split('=')
            params_dict[tokens[0]] = reader_fns[tokens[0]](tokens[1])
    except Exception as e:
        exit(str(e) + ": Unrecognized command line flag.")
    return params_dict

# Read params, input defaults
params = read_args(sys.argv)
if "smoothing_window" not in params:
    params["smoothing_window"] = 1
if "l2_stop_threshold" not in params:
    params["l2_stop_threshold"] = 1e100
if "linf_stop_threshold" not in params:
    params["linf_stop_threshold"] = 1e100

# Read data
cycle_arrays = []
l2_loss_arrays = []
linf_loss_arrays = []
WU_multipliers = []
labelnames = []
for i,fname in enumerate(params["file_names"]):
    labelname = params["label_names"][i]
    print(labelname)
    WU_multiplier = params["work_units"][i]

    f = open(fname, 'r')
    cycles = dict()
    l2_losses = dict()
    linf_losses = dict()
    should_stop_reading = False
    for line in f:
        if "After" in line:
            tokens = line.split()
            cycle_num = int(tokens[3])
            hier_level = int(tokens[1].rstrip(':'))
            l2_loss = float(tokens[8].rstrip(','))
            linf_loss = float(tokens[16].rstrip(','))
            if l2_loss > params["l2_stop_threshold"] or linf_loss > params["linf_stop_threshold"]:
                should_stop_reading = True
                continue
            if hier_level not in l2_losses:
                l2_losses[hier_level] = []
                linf_losses[hier_level] = []
                cycles[hier_level] = []
            l2_losses[hier_level].append(l2_loss)
            linf_losses[hier_level].append(linf_loss)
            cycles[hier_level].append(cycle_num)
        if should_stop_reading:
            continue
    print("Final cycle read is {}".format(cycles[0][-1]))
    f.close()
    cycle_arrays.append(cycles)
    l2_loss_arrays.append(l2_losses)
    linf_loss_arrays.append(linf_losses)
    WU_multipliers.append(WU_multiplier)
    labelnames.append(labelname)

# Smooth data for nicer display
if params["smoothing_window"] > 1:
    conv_length = params["smoothing_window"]
    conv_arr = np.ones(conv_length) / float(conv_length)
    for i in range(len(cycle_arrays)):
        for hier_level in l2_loss_arrays[i]:
            l2_loss_arrays[i][hier_level] = np.convolve(l2_loss_arrays[i][hier_level], conv_arr, mode="same")[int(conv_length/2) : -int(conv_length/2)]
            linf_loss_arrays[i][hier_level] = np.convolve(linf_loss_arrays[i][hier_level], conv_arr, mode="same")[int(conv_length/2) : -int(conv_length/2)]
            cycle_arrays[i][hier_level] = cycle_arrays[i][hier_level][int(conv_length/2) : -int(conv_length/2)]

# Plot L2 losses
for i, cycles in enumerate(cycle_arrays):
    for hier_level in l2_loss_arrays[i]:
        if hier_level not in params["hier_levels"]:
            continue
        lvl_cycles = WU_multipliers[i] * np.array(cycles[hier_level])
        plt.semilogy(lvl_cycles, l2_loss_arrays[i][hier_level], label="{}".format(labelnames[i]))
        best_loss = np.nanmin(l2_loss_arrays[i][hier_level])
        print("For {}, level {}, best loss is {}".format(labelnames[i], hier_level, best_loss))

print(labelnames)
plt.legend()
plt.xlabel("Work Units")
plt.ylabel("Loss")
if "l2_limits" in params:
    plt.ylim(params["l2_limits"])
if "save_prefix" in params:
    plt.savefig(params["save_prefix"] + "_l2.pdf", bbox_inches="tight")
    plt.cla()
else:
    plt.show()

# Plot linf losses
for i, cycles in enumerate(cycle_arrays):
    for hier_level in linf_loss_arrays[i]:
        if hier_level not in params["hier_levels"]:
            continue
        lvl_cycles = WU_multipliers[i] * np.array(cycles[hier_level])
        plt.semilogy(lvl_cycles, linf_loss_arrays[i][hier_level], label="{}".format(labelnames[i]))
        best_loss = np.nanmin(linf_loss_arrays[i][hier_level])
        print("For {}, level {}, best loss is {}".format(labelnames[i], hier_level, best_loss))

print(labelnames)
plt.legend()
plt.xlabel("Work Units")
plt.ylabel("Loss")
if "linf_limits" in params:
    plt.ylim(params["linf_limits"])
if "save_prefix" in params: 
    plt.savefig(params["save_prefix"] + "_linf.pdf", bbox_inches="tight")
    plt.cla()
else:
    plt.show()
