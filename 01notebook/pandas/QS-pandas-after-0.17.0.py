# -*- coding: utf-8 -*-
# gauge statistics by group
import os, os.path
import pandas as pd
import numpy as np
import sys
import matplotlib as mpl
import matplotlib.pyplot as plt
sys.path.append('C:\Localdata\D\Note\Python\FEM\util')
import plot_util as pltu

workpath = os.path.dirname(os.path.abspath(__file__))

####################################
#        read gauge files          #
####################################
# get the directory of gauge files
workpath = os.path.dirname(os.path.abspath(__file__))
result_name = "J064P3\J064P3_vrf"
gauge_result = os.path.join(workpath,  result_name+".csv")

df = pd.read_csv(gauge_result, delimiter = ",")
column_name = df.columns.values.tolist()
print( column_name)

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

df_module = df.drop_duplicates(subset='GroupName', keep='last')
group_uniq = df_module.loc[:, "GroupName"].values.tolist()
group_uniq
count = 0
dict_outliers = {}
dict_stat_by_group = {}
for group in group_uniq:
    count += 1
#    if count ==3:
#        break
    df_slice = df[df.GroupName == group]
    df_slice = df_slice.sort_values(by=["plot_CD", "draw_CD"], ascending=[True, True])
    dict_outliers[group], dict_stat_by_group[group] = pltu.drawQuadState(df_slice, group, save = True)
    dict_stat_by_group[group]["count"] = int(len(df_slice.index))
    dict_stat_by_group[group]["avg_wt"] = np.average(df_slice["cost_wt"].values.tolist())

for group in group_uniq:
    length = len(dict_outliers[group])
    if length >= 1:
        print group
        for a in dict_outliers[group]:
            print "\t"+a

df_group = pd.DataFrame(dict_stat_by_group)
out_name = result_name + "_stat_by_group.txt"
df_group.to_csv(os.path.join(workpath, out_name), sep = '\t')

def drawQuadState(df, filter_name, **args):
    fig =  plt.figure()
    outliers = []
    rc = mpl.rcParams
    rc["xtick.labelsize"] = 8
    rc["ytick.labelsize"] = 8
    ax1 = fig.add_axes([0.1, 0.1, 0.8, 0.7], label="ax1")

    ## dataframe gauge data clean

    orig_gauge_num = len(df.index)
    df = cleanGauge(df)
    filtered_gauge_num = len(df.index)
    print("{} gauges: {} valid, {} filtered by \'cost_wt>0\' & \'model_cd>0\' ".format(filter_name, filtered_gauge_num, orig_gauge_num-filtered_gauge_num))

    ## gauge data acquisition, model error consider tone_sgn
    col_name = df.columns.values.tolist()
    wafer_cd = df.loc[:, "wafer_cd"].values.tolist()
    model_cd = df.loc[:, "model_cd"].values.tolist()
    # df["signed_error"] = df.apply(lambda x: x["model_error"]*x["tone_sgn"], axis=1)
    # model_error = df.loc[:, "signed_error"].values.tolist()
    model_error = df.loc[:, "model_error"].values.tolist()
    length = len(wafer_cd)
    index = np.arange(1, length+1)
    ax1.plot(index, wafer_cd, linestyle='None', marker= 'o', markeredgewidth = 1, markeredgecolor='g', markerfacecolor='none', label='wafer_cd')
    ax1.plot(index, model_cd, linestyle='--', color = '#DB4105', marker='v', markeredgewidth = 1, markeredgecolor='purple', markerfacecolor='none',label='model_cd')

    ax2 = ax1.twinx()
    ax2.axhline(y=0, linewidth=2, color='grey')

    # draw error rectangle box
    bar_width = 0.5
    opacity = 0.4
    error_config = {'color': '0.3'}
    plt.bar(index-bar_width/2, model_error, bar_width,
        alpha=opacity,
        color='#AEEE00',#yerr=std_err, 'b',
        error_kw=error_config#,label='err'
        )

    if "range_min" in col_name:
        range_min = wafer_cd = df.loc[:, "range_min"].values.tolist()
        ax2.plot(index, range_min, linestyle='-', linewidth=2, color='#01B0F0',label='range_min') # orange
    if "range_max" in col_name:
        outlier_index = []
        outlier_errs = []
        range_max = wafer_cd = df.loc[:, "range_max"].values.tolist()
        ax2.plot(index, range_max, linestyle='-', linewidth=2, color='g',label='range_max')
        for idx in xrange(length):
            if abs(model_error[idx]) > range_max[idx] :
                outlier_index.append(idx+1)
                outlier_errs.append(model_error[idx])
        ax2.plot(outlier_index ,outlier_errs, linestyle='None', marker= 's', markeredgewidth = 2, markeredgecolor='r', markerfacecolor='none', label='outlier')
    # axis range setting
    # ax2, 0-centered y
    ymin, ymax = ax2.get_ylim()
    ylimit = max(abs(ymin), abs(ymax))
    ax2.set_ylim([-ylimit*1.2, ylimit*1.2])
    # ax1, 1->index
    xmin, xmax = [1-bar_width, length+bar_width]
    ax2.set_xlim([xmin, xmax])

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
    ax1.legend(legend_handles, legend_labels, bbox_to_anchor=(-0.1, 1), fontsize=12, framealpha = 0.5) #  loc='Top left',

    # model error statistics by group
    # error consider tone_sgn
    err_range = round((max(model_error) - min(model_error)), 2)
    avg = round(np.average(model_error), 2)
    sigma = round(np.std(model_error), 2)
    df_index = df.index.values.tolist()
    tone_sgn = df.at[df_index[0],"tone_sgn"]
    # create blank rectangle
    text = " 3 Sigma\n  {}\n\n Err Range\n  {}\n\n Avg Err\n  {}\n\ntone_sgn\n  {}".format(3*sigma, err_range, avg, tone_sgn)
    dict_stat = {}
    dict_stat = {"err_sigma": sigma, "err_range": err_range, "err_mean": avg, "tone_sgn": tone_sgn}
    plt.text( -0.2, 0.12, text,
        horizontalalignment='right',
        verticalalignment='bottom',
        ##xycoords='axes fraction',
        transform = ax2.transAxes,
        fontsize=12)

    # twinx axis don't have the xaxis instance ?
    ax1.grid(True, which='both', axis='both')
    ax2.grid(False, which='both', axis='both')

    # ticks
    ax3= ax1.twiny()
    gauge = df.loc[:, "gauge"].values.tolist()
    plot_cd = df.loc[:, "plot_cd"].values.tolist()
    ax1.set_xticks(index)
    ax1.set_xticklabels(plot_cd, rotation=270)
    ax3.xaxis.set_tick_params(labeltop='on')
    ax3.set_xlim([xmin, xmax])
    ax3.set_xticks(index)
    ax3.set_xticklabels(gauge, rotation=90)

    # title
    ax1.set_ylabel(r"CD($nm$)")
    ax1.set_xlabel("plot_cd($nm$)")
    ax2.set_ylabel(r"err($nm$)")
    figure_title = "{}".format(filter_name)
    plt.text(-0.1, 1.1, figure_title,
         horizontalalignment='right',
         fontsize=14,
         transform = ax2.transAxes)
    if(args.has_key("save") & args["save"] == True):
        directory = os.path.join(workpath, "results")
        if not os.path.exists(directory):
            os.makedirs(directory)
        fig.savefig(os.path.join(workpath, "results", "{}_error_analysis.png".format(filter_name)), dpi=2*fig.dpi, frameon = False )
    else:
        plt.show()
    plt.close(fig)
    outlier_index = [x-1 for x in outlier_index ]
    df_out = df.iloc[outlier_index]
    outliers = df_out["gauge"].values.tolist()
    return outliers, dict_stat