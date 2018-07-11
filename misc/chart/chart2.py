# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np

def axis_range(list_x, list_y):
    max_x = max(list_x)*1.05
    max_y = max(list_y)*1.05
    axis_ran = [0, 0, max_x, max_y]
    return axis_ran

# Open file
f  =  open("C:/Localdata/D/Note/Python/chart/table1.txt",'r')

# Read and ignore header lines
header1 = f.readline()
#print(header1)

# Loop over lines and extract variables of interest
gauges = {}
if len(gauges)==0 :
    gauges['gauge'] = []
    gauges['draw_cd'] = []
    gauges['range_min'] = []
    gauges['range_max'] = []
    gauges['plot_cd'] = []
    gauges['wafer_cd'] = [] 
    gauges['model_cd'] = []
    gauges['model_err'] = []
for line in f:
    #print(repr(line)) # repr : object representations as a whole string
    line = line.strip()
    cols = line.split('\t')
    gauges['gauge'].append(cols[0])
    gauges['draw_cd'].append(float(cols[1]))
    gauges['range_min'].append(float(cols[2]))
    gauges['range_max'].append(float(cols[3]))
    gauges['plot_cd'].append(float(cols[4]))
    gauges['wafer_cd'].append(float(cols[5]))
    gauges['model_cd'].append(float(cols[6]))
    gauges['model_err'].append(float(cols[7]))

gauge_num = len(gauges['gauge'])
index = np.arange(gauge_num)

fig =  plt.figure()
ax1 = fig.add_axes([0.1, 0.15, 0.8, 0.8], label="ax1") # rect [left, bottom, width, height]

ax1.set_xticks(index)

ax1.set_xticklabels(gauges['gauge'], rotation=270)
line1 = ax1.plot(index, np.asarray(gauges['wafer_cd']),linestyle='None', marker= 'o', markeredgewidth = 1, markeredgecolor='g', markerfacecolor='none', label='wafer_cd')
line2 = ax1.plot(index, np.asarray(gauges['model_cd']),linestyle='--', marker='v', markeredgewidth = 1, markeredgecolor='purple', markerfacecolor='none',label='model_cd')
  
ax2 = ax1.twinx()

ax2.axhline(y=0, linewidth=2, color='grey')
line3 = ax2.plot(index, np.asarray(gauges['range_min']), linestyle='-', linewidth=2, color='orange',label='spec_min')
line4 = ax2.plot(index, np.asarray(gauges['range_max']), linestyle='-', linewidth=2, color='gray',label='spec_max')

bar_width = 0.5
opacity = 0.4
error_config = {'ecolor': '0.3'}

rects1 = plt.bar(index-bar_width/2, gauges['model_err'], bar_width,
                 alpha=opacity,
                 color='b',#yerr=std_err,
                 error_kw=error_config,
                 label='err')
for idx in index:
    if abs(gauges['model_err'][idx]) > gauges['range_max'][idx] : 
        line5 = ax2.plot(idx, gauges['model_err'][idx], linestyle='None', marker= 's', markeredgewidth = 2, markeredgecolor='r', markerfacecolor='none', label='outlier')

# legend
legend_handles = []
legend_labels = []
# Two kinds of handles: matplotlib.lines.Line2D object; Container object of 22 artists
# Not able to use append, but able to use "+"
for ax in [ax1, ax2]:
    h, l = ax.get_legend_handles_labels()
    legend_handles = legend_handles + h 
    legend_labels = legend_labels +l 
ax1.legend(legend_handles, legend_labels, loc='upper left', fontsize=12, framealpha = 0.5)

# twinx axis don't have the xaxis instance ?
ax1.xaxis.grid(True, which='both')
ax2.grid(True, which='both', axis='both')
# title
ax1.set_xlabel("Gauge")
ax1.set_ylabel(r"CD($nm$)")
ax2.set_ylabel(r"err($nm$)")
ax1.set_title("isoLine", loc='center')
plt.show()