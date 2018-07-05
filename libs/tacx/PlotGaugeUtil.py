"""
-*- coding: utf-8 -*-
Created: peyang, 2018-01-24 22:07:11

Last Modified by: peyang

GaugePlotUtil: plot utility module for Gauge Table
Some Complicated/Specific FEM+ gauge plot functions, like:
    * Error Source Analysis Plot
    * Process Uncertainty Plot
    * Model Error Trend Plot
    * Quad Stat Plot
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os.path
from PlatformUtil import inWindows

if inWindows():
    import seaborn as sns

def drawQuadStateCmp(df_gauges, filter_name, **args):
    # df_gauges: cell of dataframe, the dataframe the gauge results
    fig =  plt.figure()
    outliers = []
    rc = mpl.rcParams
    rc["xtick.labelsize"] = 8
    rc["ytick.labelsize"] = 8
    ax1 = fig.add_axes([0.1, 0.15, 0.8, 0.7], label="ax1")

    ## dataframe gauge data clean
    # orig_gauge_num = len(df.index)
    df = df_gauges[0]
    numGauges = len(df)
    numModels = len(df_gauges)

    ## baseline information
    df = df.sort_values(by=["draw_cd", "plot_cd"], ascending=[True, True])
    col_name = df.columns.values.tolist()
    wafer_cd = df.loc[:, "wafer_cd"].values.tolist()
    model_cd = df.loc[:, "model_cd"].values.tolist()
    model_error = df.loc[:, "model_error"].values.tolist()
    rms = calRMS(df)
    tot_wt = df.sum()["cost_wt"]
    err_range = round((max(model_error) - min(model_error)), 2)
    sigma = round(np.std(model_error), 2)

    ## reference job information, all the cost_wt from the baseline job, rms
    ref_errors = []
    txt_err_range = str(err_range)+"( "
    txt_rms = str(rms)+"( "
    txt_sigma = str(3*sigma)+"( "
    df_base_wt = pd.DataFrame({'base_wt': df['cost_wt']})
    for i in range(1, numModels, 1):
        cur_df = df_gauges[i].sort_values(by=["draw_cd", "plot_cd"], ascending=[True, True])
        cur_model_error = cur_df.loc[:, "model_error"].values.tolist()
        cur_df = cur_df.join(df_base_wt)
        if cur_df.sum()["base_wt"] == 0:
            print "break1"
        cur_rms = calRMS(cur_df)
        cur_err_range = round((max(cur_model_error) - min(cur_model_error)), 2)
        cur_sigma = round(np.std(cur_model_error), 2)
        ref_errors.append(cur_model_error)
        txt_err_range += str(cur_err_range) + " "
        txt_rms += str(cur_rms) + " "
        txt_sigma += str(3*cur_sigma) + " "
    txt_err_range += ")"
    txt_rms += ")"
    txt_sigma += ")"

    length = len(wafer_cd)
    index = np.arange(1, length+1)
    #ax1.plot(index, wafer_cd, linestyle='None', marker= 'o', markeredgewidth = 1, markeredgecolor='g', markerfacecolor='none', label='wafer_cd')
    #ax1.plot(index, model_cd, linestyle='-', color = '#DB4105', marker='v', markeredgewidth = 1, markeredgecolor='purple', markerfacecolor='none',label='model_cd')
    ax1.plot(index, model_cd, linestyle='None', marker= 'o', markeredgewidth = 1, markeredgecolor='b', markerfacecolor='none', label='model_cd')
    ax1.plot(index, wafer_cd, linestyle='-', color = 'r', label='wafer_cd')
    if 'target_cd' in df.columns:
        ax1.plot(index, df['target_cd'].values.tolist(), linestyle='--', color = 'g', label='target_cd')

    ax2 = ax1.twinx()
    ax2.axhline(y=0, linewidth=2, color='grey')

    # draw error rectangle box
    bar_width = 0.8/numModels
    opacity = 0.8
    error_config = {'color': '0.3'}
    barcolor = ["#FF0010", "#2BCE48",  "#0075DC", "#FF5005", '#AEEE00'] # Red, Green, Blue, yellow
    bar_start_pos = index-bar_width/2
    plt.bar(bar_start_pos, model_error, bar_width,
        alpha=opacity,
        color=barcolor[0],
        error_kw=error_config,
        label='error0'
    )
    for i in range(numModels-1):
        plt.bar(bar_start_pos+bar_width*(i+1), ref_errors[i], bar_width,
            alpha=opacity,
            color= barcolor[i+1],
            error_kw=error_config,
            label='error{}'.format(i+1)
        )
    outlier_index = []
    if "range_max" in col_name:
        range_min = wafer_cd = df.loc[:, "range_min"].values.tolist()
        ax2.plot(index, range_min, linestyle='-', linewidth=2, color='#01B0F0',label='range_min') # orange
        outlier_errs = []
        range_max = wafer_cd = df.loc[:, "range_max"].values.tolist()
        ax2.plot(index, range_max, linestyle='-', linewidth=2, color='g',label='range_max')
        for idx in xrange(length):
            if abs(model_error[idx]) > range_max[idx] :
                outlier_index.append(idx+1)
                outlier_errs.append(model_error[idx])
        ax2.plot(outlier_index ,outlier_errs, linestyle='None', marker= 's', markeredgewidth = 2, markeredgecolor='#FFFF00', markerfacecolor='none', label='outlier')
    draw3Sigma = False
    if draw3Sigma:
        if "3sigma" in col_name:
            sigma3 =  df.loc[:, "3sigma"].values.tolist()
            ax2.plot(index, sigma3, linestyle='--', label='$3\sigma$')
            meef =  df.loc[:, "cd_meef"].values.tolist()
            ax2.plot(index, meef, linestyle='--', label='MEEF')

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
    left_fonsize = 12
    if numModels > 1:
        left_fonsize = 12/numModels**(1/3.)
    for ax in [ax1, ax2]:
        h, l = ax.get_legend_handles_labels()
        legend_handles = legend_handles + h
        legend_labels = legend_labels +l
        # Shrink current axis by 20%
        box = ax.get_position()
        ax.set_position([box.x0+box.width * 0.2, box.y0, box.width * 0.8, box.height])
    ax1.legend(legend_handles, legend_labels, bbox_to_anchor=(-0.1, 1), fontsize=left_fonsize, framealpha = 0.5) #  loc='Top left',

    # display the 3sigma,Err range, rms Err
    df_index = df.index.values.tolist()
    tone_sgn = df.at[df_index[0],"tone_sgn"]

    text = "RMS\n  {}\n\n 3 Sigma\n  {}\n\n Err Range\n  {}\n\n Total Wt\n  {}\n\n tone_sgn\n  {}".format(txt_rms, txt_sigma, txt_err_range, tot_wt, tone_sgn)
    dict_stat = {}
    dict_stat = {"rms": txt_rms, "err_sigma": txt_sigma, "err_range": txt_err_range, "tone_sgn": tone_sgn, "tot_wt": tot_wt}
    dict_stat['type'] = df.loc[0, 'type']
    plt.text( -0.2/numModels*2, 0.12, text,
        horizontalalignment='right',
        verticalalignment='bottom',
        ##xycoords='axes fraction',
        transform = ax2.transAxes,
        fontsize=left_fonsize)

    # twinx axis don't have the xaxis instance ?
    ax1.grid(False, which='both', axis='both')
    ax2.grid(False, which='both', axis='both')
    ax2.yaxis.grid(True)

    # ticks
    ax3= ax1.twiny()
    ax3.grid(False, which='both', axis='both')
    gauge = df.loc[:, "gauge"].values.tolist()
    draw_cd = df.loc[:, "draw_cd"].values.tolist()
    ax1.set_xticks(index)
    ax1.set_xticklabels(draw_cd, rotation=270)
    ax3.xaxis.set_tick_params(labeltop='on')
    ax3.set_xlim([xmin, xmax])
    ax3.set_xticks(index)
    ax3.set_xticklabels(gauge, rotation=90)

    # title
    ax1.set_ylabel(r"CD($nm$)")
    ax1.set_xlabel("draw_cd($nm$)")
    ax2.set_ylabel(r"err($nm$)")
    figure_title = "{}".format(filter_name)
    plt.text(-0.1, 1.1, figure_title,
         horizontalalignment='right',
         fontsize=left_fonsize,
         transform = ax2.transAxes)
    if(args.has_key("save") and args["save"] == True):
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

def drawQuadState(df, filter_name, **args):
    fig =  plt.figure()
    outliers = []
    rc = mpl.rcParams
    rc["xtick.labelsize"] = 8
    rc["ytick.labelsize"] = 8
    ax1 = fig.add_axes([0.1, 0.15, 0.8, 0.7], label="ax1")

    # orig_gauge_num = len(df.index)
    numGauges = len(df.index)
    # print("{} gauges: {} valid, {} filtered by \'cost_wt>0\' & \'model_cd>0\' ".format(filter_name, filtered_gauge_num, orig_gauge_num-filtered_gauge_num))

    # if numGauges > 40:
    #    def sampleBin(numGauges):
    #        divider = 2
    #        while(True):
    #            if numGauges/divider <= 40:
    #                return numGauges/divider
    #            divider += 1
    #    numBins = sampleBin(numGauges)
    #    plt.locator_params(nbins=numBins)

    ## baseline information
    df = df.sort_values(by=["draw_cd", "plot_cd"], ascending=[True, True])
    col_name = df.columns.values.tolist()
    wafer_cd = df.loc[:, "wafer_cd"].values.tolist()
    model_cd = df.loc[:, "model_cd"].values.tolist()
    model_error = df.loc[:, "model_error"].values.tolist()
    rms = calRMS(df)
    err_range = round((max(model_error) - min(model_error)), 2)
    sigma = round(np.std(model_error), 2)
    txt_err_range = str(err_range)+"( "
    txt_rms = str(rms)+"( "
    txt_sigma = str(3*sigma)+"( "

    # axis 1: model_cd, and wafer_cd
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
        ax2.plot(outlier_index ,outlier_errs, linestyle='None', marker= 's', markeredgewidth = 2, markeredgecolor='r', markerfacecolor='none', label='outlier') # 'Yellow'



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
    ax1.legend(legend_handles, legend_labels, bbox_to_anchor=(-0.1, 1), fontsize=10, framealpha = 0.5) #  loc='Top left',

    # display the 3sigma,Err range, rms Err
    df_index = df.index.values.tolist()
    tone_sgn = df.at[df_index[0],"tone_sgn"]

    text = "RMS\n  {}\n\n 3 Sigma\n  {}\n\n Err Range\n  {}\n\n Total Wt\n  {}\n\n tone_sgn\n  {}".format(txt_rms, txt_sigma, txt_err_range, tot_wt, tone_sgn)
    dict_stat = {}
    dict_stat = {"rms": txt_rms, "err_sigma": txt_sigma, "err_range": txt_err_range, "tone_sgn": tone_sgn, "tot_wt": tot_wt}

    plt.text( -0.2, 0.12, text,
        horizontalalignment='right',
        verticalalignment='bottom',
        ##xycoords='axes fraction',
        transform = ax2.transAxes,
        fontsize=10)

    # twinx axis don't have the xaxis instance ?
    ax1.grid(False, which='both', axis='both')
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
    if(args.has_key("save") and args["save"] == True):
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

def drawErrorMeasMEEF(df, filter_name, **args):
    fig =  plt.figure()
    outliers = []
    rc = mpl.rcParams
    rc["xtick.labelsize"] = 8
    rc["ytick.labelsize"] = 8
    ax1 = fig.add_axes([0.1, 0.15, 0.8, 0.7], label="ax1")

    # orig_gauge_num = len(df.index)
    numGauges = len(df.index)

    ## baseline information
    df = df.sort_values(by=["draw_cd", "plot_cd"], ascending=[True, True])
    col_name = df.columns.values.tolist()
    wafer_cd = df.loc[:, "wafer_cd"].values.tolist()
    model_cd = df.loc[:, "model_cd"].values.tolist()
    model_error = df.loc[:, "model_error"].values.tolist()
    rms = calRMS(df)
    tot_wt = df.sum()["cost_wt"]
    err_range = round((max(model_error) - min(model_error)), 2)
    sigma = round(np.std(model_error), 2)
    txt_err_range = str(err_range)
    txt_rms = str(rms)
    txt_sigma = str(3*sigma)

    length = len(wafer_cd)
    index = np.arange(1, length+1)
    ax1.axhline(y=0, linewidth=2, color='grey')

    draw3Sigma = True
    if draw3Sigma:
        sigma3 =  df.loc[:, "3sigma"].values.tolist()
        ax1.plot(index, sigma3, linestyle='--', label='meas $3\sigma$')
        meef =  df.loc[:, "cd_meef"].values.tolist()
        ax1.plot(index, meef, linestyle='--', label='MEEF')
        corr = np.corrcoef(model_error, sigma3)
        corr_sigma = corr[0, 1]
        corr = np.corrcoef(model_error, meef)
        corr_meef = corr[0, 1]
        txt_corr_meas = str( np.round(corr_sigma,2) )
        txt_corr_meef = str( np.round(corr_meef,2) )

    # draw error rectangle box
    bar_width = 0.5
    opacity = 0.4
    error_config = {'color': '0.3'}
    plt.bar(index-bar_width/2, model_error, bar_width,
        alpha=opacity,
        color='#AEEE00',#yerr=std_err, 'b',
        error_kw=error_config#,label='err'
        )


    # axis range setting
    # ax1, 0-centered y
    ymin, ymax = ax1.get_ylim()
    ylimit = max(abs(ymin), abs(ymax))
    ax1.set_ylim([-ylimit*1.2, ylimit*1.2])
    # ax1, 1->index
    xmin, xmax = [1-bar_width, length+bar_width]
    ax1.set_xlim([xmin, xmax])

    # legend
    legend_handles = []
    legend_labels = []
    for ax in [ax1]:
        h, l = ax.get_legend_handles_labels()
        legend_handles = legend_handles + h
        legend_labels = legend_labels +l
        # Shrink current axis by 20%
        box = ax.get_position()
        ax.set_position([box.x0+box.width * 0.2, box.y0, box.width * 0.8, box.height])
    ax1.legend(legend_handles, legend_labels, bbox_to_anchor=(-0.1, 1), fontsize=10, framealpha = 0.5) #  loc='Top left',

    # display the 3sigma,Err range, rms Err
    df_index = df.index.values.tolist()
    tone_sgn = df.at[df_index[0],"tone_sgn"]

    text = "RMS\n  {}\n Err 3sigma\n  {}\n Err Range\n  {}\n Total Wt\n  {}\n tone_sgn\n  {}\n Corr Meas\n {}\n Corr MEEF\n {}".format(txt_rms, txt_sigma, txt_err_range, tot_wt, tone_sgn, txt_corr_meas, txt_corr_meef)
    dict_stat = {}
    dict_stat = {"rms": txt_rms, "err_sigma": txt_sigma, "err_range": txt_err_range, "tone_sgn": tone_sgn, "tot_wt": tot_wt}

    plt.text( -0.2, 0.12, text,
        horizontalalignment='right',
        verticalalignment='bottom',
        ##xycoords='axes fraction',
        transform = ax1.transAxes,
        fontsize=10)

    # twinx axis don't have the xaxis instance ?
    ax1.grid(True, which='both', axis='both')

    # ticks
    gauge = df.loc[:, "gauge"].values.tolist()
    plot_cd = df.loc[:, "plot_cd"].values.tolist()
    ax1.set_xticks(index)
    ax1.set_xticklabels(gauge, rotation=60)

    # title
    ax1.set_ylabel(r"err($nm$)")
    figure_title = "{}".format(filter_name)
    plt.text(-0.1, 1.1, figure_title,
         horizontalalignment='right',
         fontsize=14,
         transform = ax1.transAxes)
    if(args.has_key("save") and args["save"] == True):
        directory = os.path.join(workpath, "results")
        if not os.path.exists(directory):
            os.makedirs(directory)
        fig.savefig(os.path.join(workpath, "results", "{}_error_analysis.png".format(filter_name)), dpi=2*fig.dpi, frameon = False )
    else:
        plt.show()
    plt.close(fig)
    outliers = [0]
    return outliers, dict_stat


def dfProcessUncertPlot(df, column_list, col_xlabel, figure_name, **args):
    df.reset_index(drop=True, inplace=True)
    sns.set(style="darkgrid")
    nGauges = len(df)
    df_slice = df[column_list]

    withModelErr = True
    withStacked = True
    withUncertMSE = True
    withILS = False
    ax = df_slice.plot(kind='area', stacked=withStacked);
    #ax.set_xticks(df.index)
    stepsize = 8
    #stepsize = min([len(df)/10, 30])
    start, end = ax.get_xlim()
    ax.xaxis.set_ticks(np.arange(start, end, stepsize), )
    xticklabels_all = df.loc[:, col_xlabel].values.tolist()
    xticklabels = [xticklabels_all[i] for i in range(nGauges) if i%stepsize==0]
    xlabelfontsize = 8
    ax.set_xticklabels(xticklabels, rotation=60, fontsize=xlabelfontsize)
    ax.set_xlabel(col_xlabel)
    ax.set_ylabel("Delta_CD (nm)")
    ax.set_title("Simulated CD Sensitivity Plot")

    #legend
    legend_handles = []
    legend_labels = []
    # Two kinds of handles: matplotlib.lines.Line2D object; Container object of 22 artists
    # Not able to use append, but able to use "+"
    left_fonsize = 10
    for ax in [ax]:
        h, l = ax.get_legend_handles_labels()
        legend_handles = legend_handles + h
        legend_labels = legend_labels +l
        # Shrink current axis by 20%
        box = ax.get_position()
        # ax.set_position([box.x0, box.y0, box.width * 0.9, box.height])
    ax.legend(legend_handles, legend_labels, bbox_to_anchor=(0.9, 1), fontsize=left_fonsize, framealpha = 0.5) #  loc='upper right lower left',

    #UNCERTAINTY_MSE plot
    if withUncertMSE:
        plt.figure(2)
        if withModelErr:
            maxYLim = 0
            df.loc[:, 'abs_err'] = df.apply(lambda x: abs(x['model_error']), axis=1)
            df_slice = df['abs_err']
            maxYLim = max(maxYLim, df_slice.max())
            ax2 = df_slice.plot(kind='area', stacked=False, legend="model_err", alpha=1)
            df_slice = df["uncertainty_mse"]
            maxYLim = max(maxYLim, df_slice.max())
            df_slice.plot(kind='area', stacked=False, ax=ax2, legend="uncertainty", alpha=1)
            #xx = range(len(df_slice))
            #spec = 1.1
            #yy = [spec for i in xx]
            df_slice = df["range_max"]
            maxYLim = max(maxYLim, df_slice.max())
            spec = np.round(df["range_max"].mean(), 2)
            df_slice.plot(ax=ax2, legend="spec={}".format(spec))
            ax2.set_ylim([0, maxYLim*1.2])
        else:
            df_slice = df["uncertainty_mse"]
            ax2 = df_slice.plot(kind='area', stacked=False, legend="uncertainty", alpha=1)
        if withILS:
            df["12-ils_result_(1/um)/5"] = 11.-df["ils_result_(1/um)"]/5.
            df["12-ils_result_(1/um)/5"].plot(legend="ils", ax=ax2, linewidth=2)
            ax2.set_ylim([0, 11])

        start, end = ax2.get_xlim()
        ax2.xaxis.set_ticks(np.arange(start, end, stepsize))
        ax2.set_xticklabels(xticklabels, rotation=60, fontsize=xlabelfontsize)
        ax2.set_xlabel(col_xlabel)

        ax2.set_ylabel("Estimated CD uncertainty (nm)")
        ax2.set_title("Estimated Process Uncertainty Plot")
        fig2 = ax2.get_figure()

    fig = ax.get_figure()
    if(args.has_key("save") and args["save"] == True):
        directory = os.path.join(workpath, "results")
        if not os.path.exists(directory):
            os.makedirs(directory)
        fig.savefig(os.path.join(workpath, "results", "{}_uncert_analysis.png".format(figure_name)), dpi=2*fig.dpi, frameon = False )
        if withUncertMSE:
            fig2.savefig(os.path.join(workpath, "results", "{}_uncert_MSE.png".format(figure_name)), dpi=2*fig.dpi, frameon = False )
    else:
        plt.show()
    plt.close(fig)
    plt.close(fig2)


# Only suitable for NTD, BF. Draw resist bias trend by group or use overall
def NTDResistBiasTrend(df, **args):
    ## args:
    ##      group, draw by the groups, default is "overall"
    ##      save, whether to save the image

    hasSave = False
    if(args.has_key("save") and args["save"] == True):
        hasSave = True

    groupname = "overall"
    if args.has_key("group"):
        groupname = args["group"]

    isNTDProcess = True
    if args.has_key("PTD") & args["PTD"] == 1:
        isNTDProcess = False

    def col_line(tone_sgn, model_cd):
        if tone_sgn == 1:
            isMaskLine = True
        else:
            isMaskLine = False
        isResistLine = (isNTDProcess != isMaskLine)
        if isResistLine:
            return model_cd
        else:
            return 0
    def col_trench(tone_sgn, model_cd):
        if tone_sgn == 1:
            isMaskLine = True
        else:
            isMaskLine = False
        isResistLine = (isNTDProcess != isMaskLine)
        if False == isResistLine:
            return model_cd
        else:
            return  0
    df.loc[:, "trench"] = df.apply(lambda x: col_trench(x["tone_sgn"], x["model_cd"]), axis=1)
    df.loc[:, "line"] = df.apply(lambda x: col_line(x["tone_sgn"], x["model_cd"]), axis=1)
    df.loc[:, "model_cd-ai_cd"] = df.apply(lambda x: x["model_cd"] - x["ai_cd"], axis=1)
    df.loc[:, "wafer_cd-ai_cd"] = df.apply(lambda x: x["wafer_cd"] - x["ai_cd"], axis=1)
    df_trench = df[df["trench"] > 0]
    df_line = df[df["line"] > 0]
    numTrench = len(df_trench.index)
    numLine = len(df_line.index)

    directory = os.path.join(workpath, "results")
    if not os.path.exists(directory):
        os.makedirs(directory)

    fig1 = plt.figure()
    ax1 = fig1.add_axes([0.1, 0.1, 0.8, 0.8], label="ax1")
    fig2 = plt.figure()
    ax2 = fig2.add_axes([0.1, 0.1, 0.8, 0.8], label="ax2")

    if numTrench > 0:
        rms = calRMS(df_trench)
        df_trench = df_trench.sort_values(by=["trench"], ascending=[True])
#        p1 = ax1.plot(df_trench["trench"].values.tolist(), df_trench["wafer_cd-ai_cd"].values.tolist(), linestyle='None', marker= 'o', markersize=4, markerfacecolor='b',  alpha=.5, label='wafer_cd-ai_cd')
#        p2 = ax1.plot(df_trench["trench"].values.tolist(), df_trench["model_cd-ai_cd"].values.tolist(), linestyle='None', marker= 'o', markersize=4, markerfacecolor='g', label='model_cd-ai_cd')
        p1 = ax1.plot(df_trench["trench"].values.tolist(), df_trench["wafer_cd-ai_cd"].values.tolist(), linestyle='None', marker= 'o', markersize=4, markeredgewidth = 1, alpha=.5, markeredgecolor='b', markerfacecolor='none', label='wafer_cd-ai_cd')
        p2 = ax1.plot(df_trench["trench"].values.tolist(), df_trench["model_cd-ai_cd"].values.tolist(), linestyle='None', marker= 'o', markersize=4, markeredgewidth = 1, markeredgecolor='g', markerfacecolor='none', label='model_cd-ai_cd')
        # title
        ax1.set_xlabel("trench($nm$)")
        ax1.set_ylabel(r"resist bias($nm$)")
        figure_title = "{} resist bias vs trench({} $nm$)".format(groupname, rms)
        fig1.suptitle(figure_title, fontsize=14)
        # legend
        h, l = ax1.get_legend_handles_labels()
        ax1.legend(h, l, loc='upper right', fontsize=12, framealpha = 0.5)

        if hasSave:
            fig1.savefig(os.path.join(workpath, "results", "{}_resist_bias_vs_trench_trend.png".format(groupname)), dpi=2*fig1.dpi, frameon = False )
        else:
            plt.show()
    plt.close(fig1)

    if numLine > 0:
        rms = calRMS(df_line)
        df_line = df_line[["line", "model_cd-ai_cd", "wafer_cd-ai_cd"]]
        df_line = df_line.sort_values(by=["line"], ascending=[True])

        p1 = ax2.plot(df_line["line"].values.tolist(), df_line["wafer_cd-ai_cd"].values.tolist(), linestyle='None', marker= 'o', markersize=4, markeredgewidth = 1, alpha=.5, markeredgecolor='b', markerfacecolor='none', label='wafer_cd-ai_cd')
        p2 = ax2.plot(df_line["line"].values.tolist(), df_line["model_cd-ai_cd"].values.tolist(), linestyle='None', marker= 'o', markersize=4, markeredgewidth = 1, markeredgecolor='g', markerfacecolor='none', label='model_cd-ai_cd')

        # title
        ax2.set_xlabel("line($nm$)")
        ax2.set_ylabel(r"resist bias($nm$)")
        figure_title = "{} resist bias vs line({} $nm$)".format(groupname, rms)
        fig2.suptitle(figure_title, fontsize=14)
        # legend
        h, l = ax2.get_legend_handles_labels()
        ax2.legend(h, l, loc='upper right', fontsize=12, framealpha = 0.5)

        if hasSave:
            fig2.savefig(os.path.join(workpath, "results", "{}_resist_bias_vs_line_trend.png".format(groupname)), dpi=2*fig2.dpi, frameon = False )
        else:
            plt.show()
    plt.close(fig2)

def PTDResistBiasTrend(df, **args):
    ## args:
    ##      group, draw by the groups, default is "overall"
    ##      save, whether to save the image

    hasSave = False
    if(args.has_key("save") and args["save"] == True):
        hasSave = True
    if args.has_key("group"):
        groupname = args["group"]
    else:
        groupname = "overall"

    df.loc[:, "model_cd-ai_cd"] = df.apply(lambda x: x["model_cd"] - x["ai_cd"], axis=1)
    df.loc[:, "wafer_cd-ai_cd"] = df.apply(lambda x: x["wafer_cd"] - x["ai_cd"], axis=1)


    directory = os.path.join(workpath, "results")
    if not os.path.exists(directory):
        os.makedirs(directory)

#    fig1, ax1 = plt.subplots(1, 1)
    fig1 = plt.figure()
    ax1 = fig1.add_axes([0.1, 0.2, 0.8, 0.7], label="ax1")

    numGauges = len(df.index)

    if numGauges > 0:
        rms = calRMS(df)
        if numGauges > 40:
            plt.locator_params(nbins=24)
        if numGauges < 40:
            plt.locator_params(nbins=numGauges)
        df.loc[:, "plot_cd"] = df["plot_cd"].apply(lambda x: int(x))
        df = df.sort_values(by=["draw_cd", "plot_cd"], ascending=[True, True])
        df.plot(x = ["plot_cd", "draw_cd"], y = "wafer_cd-ai_cd", ax = ax1, rot=60, linestyle = 'None', marker= 'o', markersize=2.5, markerfacecolor='r', label='wafer_cd-ai_cd') # markeredgewidth = 1, markeredgecolor='b', blue
        df.plot(x = ["plot_cd", "draw_cd"], y = "model_cd-ai_cd", ax = ax1, rot=60, linestyle='-', linewidth=0.5, color = 'g', marker='o', markersize=2.5,  markerfacecolor='g', label='model_cd-ai_cd') # markeredgewidth = 1, markeredgecolor='g', markerfacecolor='none', green
        fig1 = ax1.get_figure()
        ax1.set_xlabel("(plot_cd, draw_cd)($nm$)")
        ax1.set_ylabel(r"resist bias($nm$)")
        figure_title = "{} resist bias vs plot_cd, draw_cd ({} $nm$)".format(groupname, rms)
        fig1.suptitle(figure_title, fontsize=14)
        # legend
        h, l = ax1.get_legend_handles_labels()
        ax1.legend(h, l, loc='upper right', fontsize=10, framealpha = 0.5)
        ax1.tick_params(axis='both', which='major', labelsize=8)

        if hasSave:
            fig1.savefig(os.path.join(workpath, "results", "{}_resist_bias_vs_draw_cd_trend.png".format(groupname)), dpi=2*fig1.dpi, frameon = False )
        else:
            plt.show()
#    plt.close(fig1)

def cmpResistBiasTrend(df1, df2, **args):
    ## args:
    ##      group, draw by the groups, default is "overall"
    ##      save, whether to save the image
    hasSave = False
    if(args.has_key("save") and args["save"] == True):
        hasSave = True
    df1.loc[:, "model_cd-ai_cd"] = df1.apply(lambda x: x['tone_sgn']*(x["model_cd"] - x["ai_cd"]), axis=1)
    df1.loc[:, "wafer_cd-ai_cd"] = df1.apply(lambda x: x['tone_sgn']*(x["wafer_cd"] - x["ai_cd"]), axis=1)
    df1 = df1[abs(df1["wafer_cd-ai_cd"])<50]
    df2.loc[:, "model_cd-ai_cd"] = df2.apply(lambda x: x['tone_sgn']*(x["model_cd"] - x["ai_cd"]), axis=1)
    df2.loc[:, "wafer_cd-ai_cd"] = df2.apply(lambda x: x['tone_sgn']*(x["wafer_cd"] - x["ai_cd"]), axis=1)
    df2 = df2[abs(df2["wafer_cd-ai_cd"])<50]

    if( args.has_key("path") ):
        directory = args["path"]
    else:
        directory = os.path.join(workpath, "results")
    if not os.path.exists(directory):
        os.makedirs(directory)

    if( args.has_key("name") ):
        fig_name = args["name"]
    else:
        fig_name = "overall"

    fig1 = plt.figure()
    ax1 = fig1.add_axes([0.1, 0.2, 0.8, 0.8], label="ax1")

    numGauges = len(df1.index)
    if numGauges > 0:
        rms1 = calRMS(df1)
        rms2 = calRMS(df2)
#        if numGauges > 40:
#            plt.locator_params(nbins=24)
#        if numGauges < 40:
#            plt.locator_params(nbins=numGauges)

        # target for tone_sgn = 1 gauge
#        df1.plot(x = "ai_cd", y = "wafer_cd-ai_cd", ax = ax1, rot=60, linestyle = '--', marker= 'o', markersize=2.5, markerfacecolor='b', label='wafer_cd-ai_cd') # markeredgewidth = 1, markeredgecolor='b', blue
#        df1.plot(x = "ai_cd",  y = "model_cd-ai_cd", ax = ax1, rot=60, linestyle='-', linewidth=0.5, color = 'r', marker='o', markersize=2.5,  markerfacecolor='r', label='model_cd-ai_cd') # markeredgewidth = 1, markeredgecolor='g', markerfacecolor='none', green
        xlabel = 'wafer_cd'
        ylabel = 'ils' #
#        ylabel = "wafer_cd-ai_cd"
        df1.plot(x = xlabel, y = ylabel, ax = ax1, rot=60, linestyle = 'None', marker= 'o', markersize=5, markeredgecolor='r', markeredgewidth = 2, markerfacecolor='none', label='cal: {}'.format(ylabel)) # markeredgewidth = 1, markeredgecolor='b', blue
        df2.plot(x = xlabel, y = ylabel, ax = ax1, rot=60, linestyle= 'None', marker='o', markersize=10,  markerfacecolor='g', label='ver: {}'.format(ylabel)) # markeredgewidth = 1, markeredgecolor='g', markerfacecolor='none', green
#        df1.plot(x = xlabel,  y = "model_cd-ai_cd", ax = ax1, rot=60, linestyle='None',  color = 'r', marker='o', markersize=2.5,  markerfacecolor='r', label='cal:model_cd-ai_cd') # markeredgewidth = 1, markeredgecolor='g', markerfacecolor='none', green
#        df2.plot(x = xlabel,  y = "model_cd-ai_cd", ax = ax1, rot=60, linestyle='None', color = 'g', marker='o', markersize=2.5,  markerfacecolor='g', label='model_cd-ai_cd') # markeredgewidth = 1, markeredgecolor='g', markerfacecolor='none', green

        ymin = min(df1.min()[ylabel]-0.2, df2.min()[ylabel]-0.2)
        ymax = max(df1.max()[ylabel]+0.2, df2.max()[ylabel]+0.2)
        ax1.set_ylim([ymin, ymax])
        xmin = min(df1.min()[xlabel]-2, df2.min()[xlabel]-2)
        xmax = max(df1.max()[xlabel]+2, df2.max()[xlabel]+2)
        ax1.set_ylim([ymin, ymax])
        ax1.set_xlim([xmin, xmax])
        fig1 = ax1.get_figure()
        ax1.set_xlabel("{}($nm$)".format(xlabel))
        ax1.set_ylabel(r"{}($nm$)".format(ylabel))
        figure_title = "{} {} trend, RMS={}$nm$ ({} $nm$)".format(fig_name, ylabel, rms1, rms2)
        fig1.suptitle(figure_title, fontsize=14)
        # legend
        h, l = ax1.get_legend_handles_labels()
        ax1.legend(h, l, loc='best', fontsize=10, framealpha = 0.5)
        ax1.tick_params(axis='both', which='major', labelsize=8)
        if hasSave:
            fig_file_name = "{}_{}_trend_cmp".format(fig_name, ylabel)
            fig1.savefig(os.path.join(workpath, "results", fig_file_name), dpi=2*fig1.dpi, frameon = False )
        else:
            plt.show()
#    plt.close(fig1)

