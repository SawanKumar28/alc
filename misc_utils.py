from collections import defaultdict
import numpy as np
import ipdb as pdb

#Following https://github.com/jzbjyb/lm-calibration/blob/master/cal.py
def compute_ece(eval_results, conf_values, num_bins=20):
    bin_size = 1.0/num_bins
    bins = defaultdict(list)
    
    for idx in range(len(conf_values)):
        conf = max(0, conf_values[idx])
        bin_index = min(int(conf/bin_size), num_bins -1)
        bins[bin_index].append([conf, eval_results[idx]])

    ece = 0
    total_count = 0
    for bin_index, vals in bins.items():
        count = len(vals)
        if count <= 0:
            continue
        total_count = total_count + count
        mean_conf = np.mean([item[0] for item in vals]) 
        mean_acc = np.mean([item[1] for item in vals]) 
        ece = ece + count * np.abs(mean_conf-mean_acc)
    ece = ece / total_count
    return ece
