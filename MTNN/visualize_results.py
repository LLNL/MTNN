import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("../")
from MTNN.utils.ArgReader import VisualizationArgReader

# Example usage:
# python visualize_results.py file_names=file1.txt,file2.txt data_name=Validation_Data label_names=1Level,2Level work_units=10.0,4.5 save_prefix=result_image hier_levels=0 smoothing_window=33

def read_single_file(filename, data_name):
    # dict where keys are loss names and vals are dicts, each of which
    # has keys that are hier_levels and vals that are arrays of losses
    loss_arrays = {} 
    cycle_arrays = {}
    best_seen_dict = {}
    f = open(fname, 'r')
    for line in f:
        if "Level" in line and "Cycle" in line and "best" in line and data_name in line:
            tokens = line.split()
            cycle_num = int(tokens[4].rstrip(':'))
            if cycle_num < params["start_reading_at"]:
                continue
            hier_level = int(tokens[2].rstrip(','))
            for i in range(5, len(tokens), 6):
                name = " ".join(tokens[i : i+2])
                loss = float(tokens[i+2])
                best_seen = float(tokens[i+5].rstrip('),'))
                if name not in loss_arrays:
                    loss_arrays[name] = {}
                if name not in best_seen_dict:
                    best_seen_dict[name] = {}
                if hier_level not in loss_arrays[name]:
                    loss_arrays[name][hier_level] = []
                if hier_level not in cycle_arrays:
                    cycle_arrays[hier_level] = []
                loss_arrays[name][hier_level].append(loss)
                best_seen_dict[name][hier_level] = best_seen
            cycle_arrays[hier_level].append(cycle_num)
    f.close()
    return cycle_arrays, loss_arrays, best_seen_dict
    
arg_reader = VisualizationArgReader()
params = arg_reader.read_args(sys.argv)

# Read data
loss_arrays_by_file = {}
cycle_arrays_by_file = {}
WU_multipliers_by_file = {}
labelnames_by_file = {}
best_seen_by_file = {}
loss_names = set()
for i,fname in enumerate(params["file_names"]):
    file_cycle_arrays, file_loss_arrays, file_best_seen_dict = read_single_file(fname, params["data_name"])
    print("{} cycles in {}".format(file_cycle_arrays[0][-1], fname))
    loss_arrays_by_file[fname] = file_loss_arrays
    cycle_arrays_by_file[fname] = file_cycle_arrays
    best_seen_by_file[fname] = file_best_seen_dict
    
    labelname = params["label_names"][i]
    WU_multiplier = params["work_units"][i]
    WU_multipliers_by_file[fname] = WU_multiplier
    labelnames_by_file[fname] = labelname

    for loss_name in file_loss_arrays.keys():
        loss_names.add(loss_name)
        
# Print best losses
for fname, label in labelnames_by_file.items():
    for loss_name, best_seen_by_level in best_seen_by_file[fname].items():
        for hier_level, best_loss in best_seen_by_level.items():
            print("For {}, level {}, best {} is {}".format(label, hier_level, loss_name, best_loss))

# Smooth data for nicer display
if params["smoothing_window"] > 1:
    conv_length = params["smoothing_window"]
    conv_arr = np.ones(conv_length) / float(conv_length)
    for fname, loss_arrays_for_fname in loss_arrays_by_file.items():
        for loss_name, losses_dict in loss_arrays_for_fname.items():
            for hier_level in losses_dict.keys():
                losses_dict[hier_level] = np.convolve(losses_dict[hier_level], conv_arr, mode="same")

# For each kind of loss, plot them
print(loss_names)
for loss_name in loss_names:
    for fname, label in labelnames_by_file.items():
        WU_mult = WU_multipliers_by_file[fname]
        for hier_level, losses in loss_arrays_by_file[fname][loss_name].items():
            if hier_level not in params["hier_levels"]:
                continue
            WU_arr = WU_mult * np.array(cycle_arrays_by_file[fname][hier_level])
            plt.semilogy(WU_arr[:-1], losses[:-1], label="{}, L{}".format(labelnames_by_file[fname], hier_level))
                       
    print("printing loss {}".format(loss_name))
    plt.legend()
    plt.title(loss_name)
    plt.xlabel("Work Units")
    plt.ylabel("Loss")
    if "save_prefix" in params:
        loss_type = loss_name.split()[0]
        fname = "{}_{}.pdf".format(params["save_prefix"], loss_type)
        plt.savefig(fname, bbox_inches="tight")
        plt.cla()
    else:
        plt.show()
