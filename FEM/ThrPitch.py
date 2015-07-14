# -*- coding: utf-8 -*-
# gauge statistics by group 
import os, os.path
import pandas as pd
import numpy as np
import sys
import matplotlib as mpl
import matplotlib.pyplot as plt
sys.path.append('C:\Localdata\D\Note\Python\FEM')
import plot_util as pltu

workpath = os.path.dirname(os.path.abspath(__file__))
####################################
# 01. drawXY example               #
####################################
# get the directory of gauge files
#gauge_result = os.path.join(workpath,  "selectedgauge_1D_145.txt")
#
#df = pd.read_csv(gauge_result, delimiter = "\t")
#column_name = df.columns.values.tolist()
#print( column_name)
#df_module = df.drop_duplicates(cols='ModuleId', take_last=True)
#module_uniq = df_module.loc[:, "ModuleId"].values.tolist()
#
#for module in module_uniq:
#    pltu.drawXY(df[df.ModuleId == module], "plot_CD", "draw_CD", module, save = True)


####################################
# 02. drawQuadState example        #
####################################
#
#gauge_result = "C:\Localdata\D\Note\Python\chart\gauge.txt"
#
#df = pd.read_csv(gauge_result, delimiter = "\t")
#column_name = df.columns.values.tolist()
#print( column_name)
#df_module = df.drop_duplicates(cols='GroupName', take_last=True)
#group_uniq = df_module.loc[:, "GroupName"].values.tolist()
#group_uniq
#count = 0
#for group in group_uniq:
#    count = count + 1
#    if(count > 5):
#        break
#    df_slice = df[df.GroupName == group]
#    df_slice = df_slice.sort(["plot_CD", "draw_CD"], ascending=[True, True])
#    pltu.drawQuadState(df_slice, group, save = True)
    
####################################
#        read gauge files          #
####################################
# get the directory of gauge files
workpath = os.path.dirname(os.path.abspath(__file__))
gauge_result = os.path.join(workpath,  "Selected_2D_gauges.txt")

df = pd.read_csv(gauge_result, delimiter = "\t")
print( df.columns.values.tolist())

####################################
#  gauges statistics by module id  #
####################################
df["dummy_dir"] = df.apply(lambda x: x['dir'], axis=1)
df.ix[df["dir"]==180, "dummy_dir"] = 0
df.ix[df["dir"]==90, "dummy_dir"] = 1

# get the unique module list
df_module = df.drop_duplicates(cols='ModuleId', take_last=True)
module_uniq = df_module.loc[:, "ModuleId"].values.tolist()

def f(c1, c2, c3):
    return (8*c1 + 4*c2 + c3)
df["SubGroup"] = df.apply(lambda x: f(x['dummy_dir'], x['EPSId'], x['MPId']), axis=1)
df_SubGroupId = df.drop_duplicates(cols='SubGroup', take_last=True)
SubGroupId_uniq = df_SubGroupId.loc[:, "SubGroup"].values.tolist()
print(SubGroupId_uniq)
# accending oder for SubGroupId_uniq
SubGroupId_uniq.sort()
print(SubGroupId_uniq)
# create statistic dict

gauge_columns_order = ['gauge', 'type', 'tone_sgn', 'base_x', 'base_y', 'head_x', 'head_y', 'plot_CD', 'draw_CD', 'wafer_CD', 'cost_wt', 'center_x', 'center_y', 'dir', 'coor', 'GroupName', 'ModuleId', 'short_gauge_name', 'user_label', 'background', 'StructureName', 'EPSId', 'MPId', 'SequenceId', 'Xperiod', 'Yperiod', 'p0', 'p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'p7', 'p8', 'SubGroup']
df_result = df[gauge_columns_order]
df_result.to_csv(os.path.join(workpath, "stage_result", "Selected_2D_gauges_grouped.txt"),  na_rep='NaN', sep = '\t', index=False)

module_stat_dict = {}
total_gauge = 0
count = 0
for cur_module in module_uniq:
    count = count + 1
    module_stat_dict[cur_module] = {}
    cond_gauge = (df["ModuleId"] == cur_module)
    module_gauge_num = df[cond_gauge].count()['gauge']
    module_stat_dict[cur_module]['count'] = module_gauge_num
    total_gauge = total_gauge + module_gauge_num
    #print(df[cond_gauge])
    # if we want multiple conditions with &, then bracket is mandatory 
    for key in SubGroupId_uniq:
        SubGroupId = int(key)
        dirId = SubGroupId/8
        EMId = SubGroupId%8
        EPSId = EMId/4    
        MPId = EMId%4
        module_stat_dict[cur_module][key] = df[(cond_gauge) & (df['EPSId'] == EPSId) & (df['MPId'] == MPId) & (df['dummy_dir'] == dirId)].count()['gauge']  
    print("{} {}\t{} {} {} {} {}".format(count, cur_module, module_gauge_num, SubGroupId, dirId, EPSId, MPId))        

# module gauge percentage statistics
if(total_gauge != 0):    
    for cur_module in module_stat_dict:
        percentage = (module_stat_dict[cur_module]["count"])/(total_gauge+0.)
        print("{} {} {}".format(cur_module, module_stat_dict[cur_module]["count"], percentage))
        module_stat_dict[cur_module]["percentage"] = np.round(percentage, 4)
df_module_stat = pd.DataFrame(module_stat_dict)

# sort statistics gauge by count number
df_module_stat = df_module_stat.T.sort('count', ascending=False).T
df_module_stat.T.to_csv(os.path.join(workpath, "stage_result", "statistics_by_SubGroupId.txt"), sep = '\t')
module_uniq_sorted = df_module_stat.columns.values.tolist() 

####################################
#  gauges statistics plotting      #
####################################   
     
# sns setting: the rc have to set before plotting
#sns.set_style("darkgrid")
#rc = mpl.rcParams
#rc["xtick.direction"] = 'in'
#rc["xtick.labelsize"] = 8
#rc["ytick.direction"] = 'in'
#rc["ytick.labelsize"] = 8
#sns.set_context(rc=rc)
fig =  plt.figure()
ax = fig.add_axes([0.1, 0.2, 0.8, 0.7], label="ax") # rect [left, bottom, width, height]
# statistics data
module_len = len(module_uniq)
xaxis_index = np.arange(1, module_len+1, 1)
print(xaxis_index)
color_list = ["#8721BD", "#927BCD", "#5764CD", "#0BE2F0", "#1EBF3A", "#9BEE13", "#CDB07E", "#EEC76D", "#EE8F72", "#EE2C88"]
portion_list = []
for key in SubGroupId_uniq:
    portion_list.append(df_module_stat.loc[key].values)
count = df_module_stat.loc["count"].values
percentage = df_module_stat.loc["percentage"].values
# statistics bar plotting
bar_width = 0.7 
barplot_list = []
portion_len = len(portion_list)

for portion_idx in range(portion_len):
    if portion_idx == 0:
        position = [0]*module_len
    else:
        position = position + portion_list[portion_idx-1]
    SubGroupId = SubGroupId_uniq[portion_idx]
    dirId = SubGroupId/8
    if dirId == 1:
        dirId = 90
    EMId = SubGroupId%8
    EPSId = EMId/4    
    MPId = EMId%4
    print("portion{} = {}".format(portion_idx, portion_list[portion_idx]))  
    p = ax.bar(xaxis_index-bar_width/2., portion_list[portion_idx], bar_width,  bottom= position, color=color_list[portion_idx], label = 'Dir = {}, EPSId = {}, MPId = {}'.format(dirId, EPSId, MPId)) # alpha =0.8
    barplot_list.append(p)
plt.xticks(xaxis_index, module_uniq_sorted, rotation=270 )
# auto labeling the stacked bar
def autoLabelStackedBar(rects, labels):
    # attach some text labels
    lastbar = len(rects)-1
    length = len(labels)
    for i in range(length):
        height = 0
        for stacked_bar in rects:
            height = height + stacked_bar[i].get_height()
        ax.text(rects[lastbar][i].get_x()+rects[lastbar][i].get_width()/2., height+0.04, int(labels[i]),
                    ha='center', va='bottom')
autoLabelStackedBar(barplot_list, count)
ax.set_title("Gauge Number Statistics by dir, EPSId & MPId")

# plot module wt
axx = ax.twinx()
percentage = list(percentage)
axx.plot(xaxis_index, percentage, linestyle='--', linewidth=2, color='r', marker='D', markeredgewidth = 2, markeredgecolor='#FF8C00', markerfacecolor='none',label='percentage')
ymin, ymax = axx.get_ylim()
shif_y = ymin
def autoLabelPoint(ax, xs, ys, labels):
    alignment = {'horizontalalignment':'center', 'verticalalignment':'baseline'}
    for x, y, label in zip(xs, ys, labels):
      ax.text(x, float(y)+0.01, "{0:.2f}%".format(label*100), **alignment) 
autoLabelPoint(axx, xaxis_index, percentage, percentage ) 
axx.set_ylim(-0.03, ymax)       

# legend
legend_handles = []
legend_labels = []
# Two kinds of handles: matplotlib.lines.Line2D object; Container object of 22 artists
# Not able to use append, but able to use "+"
for ax in [ax, axx]:
    h, l = ax.get_legend_handles_labels()
    legend_handles = legend_handles + h 
    legend_labels = legend_labels +l 
    # Shrink current axis by 20% 
    #box = ax.get_position()
    #ax.set_position([box.x0+box.width * 0.2, box.y0, box.width * 0.8, box.height])     
ax.legend(legend_handles, legend_labels, loc='upper right', fontsize=12, framealpha = 0.5) # bbox_to_anchor=(-0.1, 1)
ax.grid(True, which='both', axis='both')
ax.set_xlabel("ModuleId")  
ax.set_ylabel("Gauge Number")  
axx.set_ylabel("Gauge Percentage")  
axx.grid(False, which='both', axis='both')      
plt.show()
