# -*- coding: utf-8 -*-
import pylab
import matplotlib.pyplot as plt
import matplotlib.path as mpath
import matplotlib.axis as maxis
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

fig = plt.figure()

ax1 = fig.add_axes([0.1, 0.1, 0.8, 0.8], label="ax1") 
ax1.yaxis.tick_right()
ax1.yaxis.set_label_position('right')
ax1.yaxis.set_offset_position('right')
ax1.xaxis.tick_top()
ax1.xaxis.set_label_position('top')
#plt.xticks(index , gauges['gauge'], rotation=90)                 
plt.xticks(index , index) 
ax1.axhline(y=0, linewidth=2, color='grey')
ax1.plot(index, np.asarray(gauges['range_min']), linestyle='-', linewidth=2, color='orange')
ax1.plot(index, np.asarray(gauges['range_max']), linestyle='-', linewidth=2, color='gray')     

bar_width = 0.5
opacity = 0.4
error_config = {'ecolor': '0.3'}

rects1 = plt.bar(index-bar_width/2, gauges['model_err'], bar_width,
                 alpha=opacity,
                 color='b',#yerr=std_err,
                 error_kw=error_config,
                 label='err')            
for idx in index:
    #if abs(gauges['model_err'][idx]) > gauges['range_max'][idx] : 
        ax1.plot(idx, gauges['model_err'][idx], linestyle='None', marker= 's', markeredgewidth = 2, markeredgecolor='r', markerfacecolor='none')

# plot wafer_cd model_cd thr pitch
#print(max(gauges['wafer_cd']))
ax2 = fig.add_axes([0.1, 0.1, 0.8, 0.8], label="ax1", frameon=False) #essential frameon=False
ax2.yaxis.tick_left()
ax2.yaxis.set_offset_position('left')
ax2.xaxis.tick_bottom()
plt.xticks(index , index)
ax2.plot(index, np.asarray(gauges['wafer_cd']),linestyle='None', marker= 'o', markeredgewidth = 1, markeredgecolor='g', markerfacecolor='none')
ax2.plot(index, np.asarray(gauges['model_cd']),linestyle='--', marker='v', markeredgewidth = 1, markeredgecolor='purple', markerfacecolor='none')

def align_xaxis(ax1, v1, ax2, v2):
    """adjust ax2 xlimit so that v2 in ax2 is aligned to v1 in ax1"""
    x1, _ = ax1.transData.transform((v1, 0)) 
    x2, _ = ax2.transData.transform((v2, 0)) 
    inv = ax2.transData.inverted() 
    dx, _ = inv.transform((0, 0)) - inv.transform((x1-x2, 0)) 
    minx, maxx = ax2.get_xlim()
    scale_x1 = ax1.get_xscale()
    scale_x2 = ax2.get_xscale()
    ax2.set_xlim(minx+dx, maxx+dx, True, True)
    #ax2.set_xscale(scale_x)
    print(dx, minx, maxx, minx+dx, maxx+dx, scale_x1, scale_x2)
    
#for idx in index: 
align_xaxis(ax1, 0, ax2, 0)
#align_xaxis(ax2, gauge_num, ax1, gauge_num)
#ax1.set_autoscalex_on(True)
#ax2.set_autoscalex_on(True)
    
plt.grid()
plt.show()