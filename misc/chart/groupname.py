# -*- coding: utf-8 -*-
# gauge statistics by group 
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
#import seaborn as sns
import re

# Open file
f  =  open("C:/Localdata/D/Note/Python/chart/gauge.txt",'r')

# Read and ignore header lines
header = f.readline()
header = header.strip()
col_names = header.split('\t')
print(col_names)
columns = len(col_names)

col_status = { 'gauge': { 'flag': False, 'index': -1},
            #'groupname': { 'flag': False, 'index': -1},
            'tone_sgn': { 'flag': False, 'index': -1}, 
            'cost_wt': { 'flag': False, 'index': -1}, 
            'draw_cd': { 'flag': False, 'index': -1}, 
            'plot_cd': { 'flag': False, 'index': -1}, 
            'range_min': { 'flag': False, 'index': -1}, 
            'range_max': { 'flag': False, 'index': -1},
            'wafer_cd': { 'flag': False, 'index': -1},
            'model_cd': { 'flag': False, 'index': -1},
            'model_error': { 'flag': False, 'index': -1} }
            
def str_clean(s):
     # Remove all non-word characters (everything except numbers and letters)
     s = re.sub(r"[^\w\s]", '', s)
     # Replace all runs of whitespace with a single dash
     s = re.sub(r"\s+", '_', s)
     return s

def group_dict_construct(group, status):
    d = {}
    for key, value in col_status.iteritems():
        if value['flag'] == True:
            d[key] = []
    return d

flag_group = False 
idx_group = -1                                      
for i in range(columns) :
    name = col_names[i]
    name = str_clean(name)
    name = name.lower()
    col_names[i] = name
    if name == 'groupname':
        flag_group = True
        idx_group = i
    for key, value in col_status.iteritems():
        if key == name :
            value['flag'] = True
            value['index'] = i
            break            
#print(col_names)

# store data into dictionary, using groupname as key
grouped_gauges = {}
if flag_group == True:
    for line in f:
        line = line.strip()
        cols = line.split('\t')
        group = cols[idx_group]
        if not grouped_gauges.has_key(group):
            grouped_gauges[group] = group_dict_construct(group, col_status)
        for key, value in col_status.iteritems(): 
            #print(key, cols)
            if value['flag'] == True:
                if key == 'gauge':
                    grouped_gauges[group][key].append(cols[value['index']])
                else:
                    grouped_gauges[group][key].append(float(cols[value['index']]))

# when iteretally removing some items by certain criteria
# It's better to do it reversely
def gauge_filter(gauges):   
    length = len(gauges['gauge'])
    del_num = 0
    i = 0
    for i in xrange(length - 1, -1, -1):
        # filter 1: cost_wt<0.000001
        if gauges.has_key("cost_wt") and float(gauges['cost_wt'][i]) < 0.000001:
            print("Delete {} for cost_wt <= 0".format(gauges['gauge'][i]))
            del_num = del_num + 1
            for key in gauges:
                del gauges[key][i]
        # filter 2: wafer_cd<0.00001
        elif gauges.has_key("wafer_cd") and float(gauges['wafer_cd'][i]) < 0.000001:
            print("Delete {} for wafer_cd <= 0".format(gauges['gauge'][i]))
            del_num = del_num + 1
            for key in gauges:
                del gauges[key][i]        
    return gauges, del_num

def sort_by_key(gauges, sort_key):
    if not gauges.has_key(sort_key):
        print("Can't found the key", sort_key)
        return gauges    
    length = len(gauges[sort_key])    
    for i in range(length) :
        swaped = False
        for j in range(length-1, i, -1):
            if gauges[sort_key][j-1] > gauges[sort_key][j]:
                swaped = True
                for key in gauges:
                    temp = gauges[key][j-1]
                    gauges[key][j-1] = gauges[key][j]
                    gauges[key][j] = temp
        if swaped == False:
            break
    return gauges

count = 0
for groupname, cur_dict in grouped_gauges.iteritems():
    gauges = {} 
    for key, cur_list in cur_dict.iteritems(): 
        gauges[key] = cur_list
    
    if groupname == 'pitch_2x':
        print(gauges)

    # gauge data filter
    gauges, del_num = gauge_filter(gauges)
    if del_num > 0 :
        print("Gauge_filter: Delete {} gauges in {}".format(del_num, groupname))
    
    # ascending sort by draw_cd or plot_cd or spec
    gauges = sort_by_key(gauges, 'plot_cd')
    gauges = sort_by_key(gauges, 'draw_cd')
    #gauges = sort_by_key(gauges, 'range_max')
    
    gauge_num = len(gauges['gauge'])
    index = np.arange(gauge_num)
    #print(index)
    #print(gauges)
    fig =  plt.figure()
    ax1 = fig.add_axes([0.1, 0.15, 0.8, 0.8], label="ax1") # rect [left, bottom, width, height]
    ax1.set_xticks(index)
    ax1.set_xticklabels(gauges['gauge'], rotation=270)
    #ax1.plot(index, np.asarray(gauges['wafer_cd']), sns.xkcd_rgb["pale red"], lw=3)
    ax1.plot(index, np.asarray(gauges['wafer_cd']),linestyle='None', marker= 'o', markeredgewidth = 1, markeredgecolor='g', markerfacecolor='none', label='wafer_cd')
    ax1.plot(index, np.asarray(gauges['model_cd']),linestyle='--', marker='v', markeredgewidth = 1, markeredgecolor='purple', markerfacecolor='none',label='model_cd')
   
    ax2 = ax1.twinx()  
    #ax2.spines['right'].set_position('center')
    ax2.axhline(y=0, linewidth=2, color='grey')  
    #ax2.set_yticks(range(-10,12,2))
    #ax2.set_yticklabels(range(-10,12,2))
    if col_status['range_min']['flag'] == True:
        ax2.plot(index, np.asarray(gauges['range_min']), linestyle='-', linewidth=2, color='orange',label='spec_min')
    if col_status['range_max']['flag'] == True:
        ymin, ymax = ax2.get_ylim()
        ylimit = max(abs(ymin), abs(ymax))
        ax2.set_ylim([-ylimit*1.2, ylimit*1.2])
        ax2.plot(index, np.asarray(gauges['range_max']), linestyle='-', linewidth=2, color='gray',label='spec_max')
    
    bar_width = 0.5
    opacity = 0.4
    error_config = {'ecolor': '0.3'}
    
    #print("{} = {}, {} = {}, {} = {}".format('index', index, 'bar_width', bar_width, "gauges['model_error']", gauges['model_error']))
    #print(type(index), type(bar_width), type(gauges['model_error']), type(np.asarray(gauges['model_error'], dtype = np.float)))
    rects1 = plt.bar(index-bar_width/2, np.asarray(gauges['model_error']), bar_width,
                    alpha=opacity,
                    color='b',#yerr=std_err,
                    error_kw=error_config#,label='err'
                    )
    if col_status['range_max']['flag'] == True:
        outlier_index = []
        outlier_errs = []
        for idx in index:
            if abs(gauges['model_error'][idx]) > gauges['range_max'][idx] :
                outlier_index.append(idx) 
                outlier_errs.append(gauges['model_error'][idx])          
        ax2.plot(outlier_index,outlier_errs, linestyle='None', marker= 's', markeredgewidth = 2, markeredgecolor='r', markerfacecolor='none', label='outlier')

    # legend
    legend_handles = []
    legend_labels = []
    # Two kinds of handles: matplotlib.lines.Line2D object; Container object of 22 artists
    # Not able to use append, but able to use "+"
    for ax in [ax1, ax2]:
        h, l = ax.get_legend_handles_labels()
        legend_handles = legend_handles + h 
        legend_labels = legend_labels +l 
        # Shrink current axis by 20% 
        box = ax.get_position()
        ax.set_position([box.x0+box.width * 0.2, box.y0, box.width * 0.8, box.height])     
    
    # model error statistics by group
    # error consider tone_sgn
    err = []
    for i in xrange(gauge_num):  
        err.append(gauges['model_error'][i]*gauges['tone_sgn'][i]) 
    err_range = round((max(err) - min(err)), 2)
    avg = round(np.average(err), 2)
    sigma = round(np.std(err), 2) 
    # create blank rectangle
    extra = Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0) 
    text = " 3 Sigma\n  {}\n\n Err Range\n  {}\n\n Avg Err\n  {}".format(3*sigma, err_range, avg)
    #plt.figtext(0, 0.3, text, fontsize=12)
    plt.annotate(text, (-0.3, 0.3), xycoords='axes fraction', fontsize=12)
    #legend_handles = legend_handles + extra 
    #legend_labels = legend_labels + text
    ax1.legend(legend_handles, legend_labels, loc='Top left', bbox_to_anchor=(-0.1, 1), fontsize=12, framealpha = 0.5)       
    
    # twinx axis don't have the xaxis instance ?
    ax1.xaxis.grid(True, which='both')
    ax2.grid(True, which='both', axis='both')
    # title
    ax1.set_xlabel("Gauge")
    ax1.set_ylabel(r"CD($nm$)")
    ax2.set_ylabel(r"err($nm$)")
    ax1.set_title(groupname, loc='center')
    plt.show()
    count = count + 1
    if count > 2:
        break                                                                 
