import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import pandas as pd
import seaborn as sns
import numpy as np
import os
import math
import string

workpath = os.path.dirname(os.path.abspath(__file__))

##############################################################
#########      function list                ##################
# cleanGauge(df): change the column name to lower case, replace space with underscore and clean zero cost_wt || model_cd gauges
# setPalette(): create palette from distinguishable colors: https://en.wikipedia.org/wiki/Help:Distinguishable_colors
# drawXY(df, x_col_name, y_col_name, filter_name, **args): draw XY plotting by two column of dataframe
# drawQuadState(df, filter_name, **args): GF Quad State plotting, model error, range_min, range_max, model_cd, wafer_cd
# drawStackedBar(df, grouped_col, stacked_col, **args): group statistics, stacked by subgroups, also display the proportion of groups
# def drawTwoGroupsCmp(df1, df2, rows_col, value_col, **args):: http://pandas.pydata.org/pandas-docs/stable/visualization.html kind='bar'
# def dfProcessUncertPlot(df, column_list, col_xlabel, filter_name, **args), plot a cumulative area plotting
def stdCol(name, illchars = string.punctuation+string.whitespace):
    newname = name.lower()
    for char in illchars:
        newname = newname.replace(char, '_')
    if newname=='drawn_cd':
        newname =  'draw_cd'
    if "ils" in newname:
        newname = 'ils'
    return newname

def stdDFCols(df0):
    df = df0.copy()
    for name in df.columns.values:
        df.rename(columns={name: stdCol(name)}, inplace=True)
    return df

def cleanGauge(df0):
    df = df0.copy()
    for name in df.columns.values:
        df.rename(columns={name: stdCol(name)}, inplace=True)
    df = df[df["cost_wt"] > 1e-6 ]
#    df = df[df['model_cd'] > 1e-6]
    df = df[df['wafer_cd'] > 1e-6]
#    df = df[df['ai_cd'] > 1e-6]
    df.index = range(len(df))
    return df

def setPalette(style=''):
    colors = ['#F0A3FF', '#0075DC', '#993F00', '#4C005C', '#191919', '#005C31', '#2BCE48', '#FFCC99', '#808080', '#94FFB5', '#8F7C00', '#9DCC00', '#C20088', '#003380', '#FFA405', '#FFA8BB', '#426600', '#FF0010', '#5EF1F2', '#00998F', '#E0FF66', '#740AFF', '#990000', '#FFFF00', '#FF5005']
    colors = colors[::-1]
#    if sytle=='deep':
    if style == 'various':
        colors = [
        "#FF34FF", "#FF4A46", "#008941", "#006FA6", "#A30059",
        "#7A4900", "#0000A6", "#63FFAC", "#B79762", "#004D43", "#8FB0FF", "#997D87",
        "#5A0007", "#809693", "#1B4400", "#4FC601", "#3B5DFF", "#4A3B53", "#FF2F80",
        "#61615A", "#BA0900", "#6B7900", "#00C2A0", "#FFAA92", "#FF90C9", "#B903AA", "#D16100",
        "#DDEFFF", "#000035", "#7B4F4B", "#A1C299", "#300018", "#0AA6D8", "#013349", "#00846F",
        "#372101", "#FFB500", "#C2FFED", "#A079BF", "#CC0744", "#C0B9B2", "#C2FF99", "#001E09",
        "#00489C", "#6F0062", "#0CBD66", "#EEC3FF", "#456D75", "#B77B68", "#7A87A1", "#788D66",
        "#885578", "#FAD09F", "#FF8A9A", "#D157A0", "#BEC459", "#456648", "#0086ED", "#886F4C",

        "#34362D", "#B4A8BD", "#00A6AA", "#452C2C", "#636375", "#A3C8C9", "#FF913F", "#938A81",
        "#575329", "#00FECF", "#B05B6F", "#8CD0FF", "#3B9700", "#04F757", "#C8A1A1", "#1E6E00",
        "#7900D7", "#A77500", "#6367A9", "#A05837", "#6B002C", "#772600", "#D790FF", "#9B9700",
        "#549E79", "#FFF69F", "#201625", "#72418F", "#BC23FF", "#99ADC0", "#3A2465", "#922329",
        "#5B4534", "#FDE8DC", "#404E55", "#0089A3", "#CB7E98", "#A4E804", "#324E72", "#6A3A4C",
        "#83AB58", "#001C1E", "#D1F7CE", "#004B28", "#C8D0F6", "#A3A489", "#806C66", "#222800",
        "#BF5650", "#E83000", "#66796D", "#DA007C", "#FF1A59", "#8ADBB4", "#1E0200", "#5B4E51",
        "#C895C5", "#320033", "#FF6832", "#66E1D3", "#CFCDAC", "#D0AC94", "#7ED379", "#012C58",

        "#7A7BFF", "#D68E01", "#353339", "#78AFA1", "#FEB2C6", "#75797C", "#837393", "#943A4D",
        "#B5F4FF", "#D2DCD5", "#9556BD", "#6A714A", "#001325", "#02525F", "#0AA3F7", "#E98176",
        "#DBD5DD", "#5EBCD1", "#3D4F44", "#7E6405", "#02684E", "#962B75", "#8D8546", "#9695C5",
        "#E773CE", "#D86A78", "#3E89BE", "#CA834E", "#518A87", "#5B113C", "#55813B", "#E704C4",
        "#00005F", "#A97399", "#4B8160", "#59738A", "#FF5DA7", "#F7C9BF", "#643127", "#513A01",
        "#6B94AA", "#51A058", "#A45B02", "#1D1702", "#E20027", "#E7AB63", "#4C6001", "#9C6966",
        "#64547B", "#97979E", "#006A66", "#391406", "#F4D749", "#0045D2", "#006C31", "#DDB6D0",
        "#7C6571", "#9FB2A4", "#00D891", "#15A08A", "#BC65E9", "#FFFFFE", "#C6DC99", "#203B3C",

        "#671190", "#6B3A64", "#F5E1FF", "#FFA0F2", "#CCAA35", "#374527", "#8BB400", "#797868",
        "#C6005A", "#3B000A", "#C86240", "#29607C", "#402334", "#7D5A44", "#CCB87C", "#B88183",
        "#AA5199", "#B5D6C3", "#A38469", "#9F94F0", "#A74571", "#B894A6", "#71BB8C", "#00B433",
        "#789EC9", "#6D80BA", "#953F00", "#5EFF03", "#E4FFFC", "#1BE177", "#BCB1E5", "#76912F",
        "#003109", "#0060CD", "#D20096", "#895563", "#29201D", "#5B3213", "#A76F42", "#89412E",
        "#1A3A2A", "#494B5A", "#A88C85", "#F4ABAA", "#A3F3AB", "#00C6C8", "#EA8B66", "#958A9F",
        "#BDC9D2", "#9FA064", "#BE4700", "#658188", "#83A485", "#453C23", "#47675D", "#3A3F00",
        "#061203", "#DFFB71", "#868E7E", "#98D058", "#6C8F7D", "#D7BFC2", "#3C3E6E", "#D83D66",

        "#2F5D9B", "#6C5E46", "#D25B88", "#5B656C", "#00B57F", "#545C46", "#866097", "#365D25",
        "#252F99", "#00CCFF", "#674E60", "#FC009C", "#92896B", "#000000", "#FFFF00", "#1CE6FF"]

    #pal = sns.xkcd_palette(colors)
    pal = sns.color_palette(colors)
    return pal

def drawXY(df, x_col_name, y_col_name, filter_name, **args):
    fig =  plt.figure()
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8], label="ax") # rect [left, bottom, width, height]
    df = df.sort([x_col_name, y_col_name], ascending=[True, True])
    x = df.loc[:, x_col_name].values.tolist()
    y = df.loc[:, y_col_name].values.tolist()
    ax.plot(x, y, linestyle='None', marker= 'o', markeredgewidth = 2, markeredgecolor='g', markerfacecolor='none')
    #plt.xticks(x, x)
    scale = 0.1
    (xmin, xmax) =  ax.get_xlim()
    (ymin, ymax) =  ax.get_ylim()
    x_range = xmax-xmin
    y_range = ymax-ymin
    ax.set_xlim([xmin-x_range*scale, xmax+x_range*scale])
    ax.set_ylim([ymin-y_range*scale, ymax+y_range*scale])
    ax.set_xlabel("{}($nm$)".format(x_col_name))
    ax.set_ylabel(r"{}($nm$)".format(y_col_name))
    ax.set_title("{}: {} Through {} Trend".format(filter_name, y_col_name, x_col_name))
    plt.show()
    plt.close(fig)

    if(args.has_key("save") and args["save"] == True):
        directory = os.path.join(workpath, "results")
        if not os.path.exists(directory):
            os.makedirs(directory)
        fig.savefig(os.path.join(workpath, "results", "{}_{}_Thr_{}.png".format(filter_name, y_col_name, x_col_name)), frameon = False )

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

def drawStackedBar(df, grouped_col, stacked_col, **args):
    ##################################################
    #  gauges statistics by grouped_col&stacked_col  #
    ##################################################
    df_grouped_col = df.drop_duplicates(cols = grouped_col, take_last=True)
    group_uniq = df_grouped_col[grouped_col].values.tolist()
    df_stacked_col = df.drop_duplicates(cols = stacked_col, take_last=True)
    stacked_col_uniq = df_stacked_col[stacked_col].values.tolist()
    stacked_col_uniq.sort()
    group_stat_dict = {}
    total_gauge = 0
    count = 0
    for cur_group in group_uniq:
        count = count + 1
        group_stat_dict[cur_group] = {}
        cond_gauge = (df[grouped_col] == cur_group)
        group_gauge_num = df[cond_gauge].count()['gauge']
        group_stat_dict[cur_group]['count'] = group_gauge_num
        total_gauge = total_gauge + group_gauge_num
        for key in stacked_col_uniq:
             group_stat_dict[cur_group][key] = df[cond_gauge & (df[stacked_col] == key)].count()['gauge']
        print("{} {}\t{}".format(count, cur_group, group_gauge_num))
    # add percentage
    if(total_gauge != 0):
        for cur_group in group_stat_dict:
            percentage = (group_stat_dict[cur_group]["count"])/(total_gauge+0.)
            print("{} {} {}".format(cur_group, group_stat_dict[cur_group]["count"], percentage))
            group_stat_dict[cur_group]["percentage"] = np.round(percentage, 4)

    df_group_stat = pd.DataFrame(group_stat_dict)
    # sort statistics gauge by count number
    df_group_stat = df_group_stat.T.sort('count', ascending=False).T
    group_uniq_sorted = df_group_stat.columns.values.tolist()

    ####################################
    #  gauges statistics plotting      #
    ####################################
    fig =  plt.figure()
    ax = fig.add_axes([0.1, 0.2, 0.8, 0.7], label="ax") # rect [left, bottom, width, height]
    # statistics data
    group_len = len(group_uniq)
    xaxis_index = np.arange(1, group_len+1, 1)
    print(xaxis_index)
    #color_list = ["#8721BD", "#927BCD", "#5764CD", "#0BE2F0", "#1EBF3A", "#9BEE13", "#CDB07E", "#EEC76D", "#EE8F72", "#EE2C88"]
    color_list = ["#00261C", "#044C29", "#167F39", "#45BF55", "#96ED89", "#00305A", "#004B8D", "#0074D9", "#4192D9", "#7ABAF2"]
    portion_list = []
    for key in stacked_col_uniq:
        portion_list.append(df_group_stat.loc[key].values)
    count = df_group_stat.loc["count"].values
    percentage = df_group_stat.loc["percentage"].values

    # statistics bar plotting
    bar_width = 0.7
    barplot_list = []
    portion_len = len(portion_list)
    color_index_bias = 0
    if(portion_len < 4):
        color_index_bias = 2
    for portion_idx in range(portion_len):
        if portion_idx == 0:
            position = [0]*group_len
        else:
            position = position + portion_list[portion_idx-1]
        SubGroupId = stacked_col_uniq[portion_idx]
        print("portion{} = {}".format(portion_idx, portion_list[portion_idx]))
        p = ax.bar(xaxis_index-bar_width/2., portion_list[portion_idx], bar_width,  bottom= position, color=color_list[color_index_bias+portion_idx], label = '{}={}'.format(stacked_col, SubGroupId)) # alpha =0.8
        barplot_list.append(p)
    plt.xticks(xaxis_index, group_uniq_sorted, rotation=270 )
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
    ax.set_title("Gauge Number Statistics by {} & {}".format(grouped_col, stacked_col))

    # plot group wt
    axx = ax.twinx()
    percentage = list(percentage)
    axx.plot(xaxis_index, percentage, linestyle='--', linewidth=2, color='r', marker='D', markeredgewidth = 2, markeredgecolor='#FF8C00', markerfacecolor='none',label='percentage')
    ymin, ymax = axx.get_ylim()
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

    # label
    ax.grid(True, which='both', axis='both')
    axx.grid(False, which='both', axis='both')
    ax.set_ylabel(r"Gauge Number")
    axx.set_xlabel("{}".format(grouped_col))
    axx.set_ylabel(r"Gauge Percentage")

    if(args.has_key("save") and args["save"] == True):
        directory = os.path.join(workpath, "results")
        if not os.path.exists(directory):
            os.makedirs(directory)
        df_group_stat.T.to_csv(os.path.join(workpath, "results", "statistics_by_{}_{}.txt".format(grouped_col, stacked_col)), sep = '\t')
        fig.savefig(os.path.join(workpath, "results", "statistics_by_{}_{}.png".format(grouped_col, stacked_col)), frameon = False ) # dpi = 318
    else:
        plt.show()
    plt.close(fig)

def drawStatisticsCompare(df1, df2, rows_col, value_col, **args):
    # Default keyword arguments
    df1_label = args.get("label1", "group1")
    df2_label = args.get("label2", "group2")

    pivot1 = df1.pivot_table(values = value_col, rows=rows_col, aggfunc=lambda x: len(x.unique()))
    df_pivot1 = pd.DataFrame({"count": pivot1})
    # reset index, then the rows_col became a column of the pivot table
    df_pivot1 = df_pivot1.reset_index()
    pivot2 = df2.pivot_table(values = value_col, rows=rows_col, aggfunc=lambda x: len(x.unique()))
    df_pivot2 = pd.DataFrame({"count": pivot2})
    df_pivot2 = df_pivot2.reset_index()
    #print(df_pivot1.head(3))
    #print(df_pivot2.head(3))
    df_pivot1 = df_pivot1.sort([rows_col], ascending=[True])
    df_pivot2 = df_pivot2.sort([rows_col], ascending=[True])

    df_pivot1["group"] = df1_label
    df_pivot2["group"] = df2_label

    df = df_pivot1
    df = df.append(df_pivot2)
    df['ID'] = df.index
    print(df.head(3))

    # Draw a nested barplot to show survival for class and sex
    sns.set(style="whitegrid")
    g = sns.factorplot(x = rows_col , y="count", hue="group", data=df,
                    size=6, kind="bar", palette="muted")
    g.despine(left=True)
    g.set_ylabels("{} number".format(value_col))
    g.set_xticklabels(rotation=270)
    plt.show()
    plt.close(fig)

def drawTwoGroupsCmp(df1, df2, rows_col, value_col, **args):
    # Default keyword arguments
    df1_label = args.get("label1", "group1")
    df2_label = args.get("label2", "group2")

    pivot1 = df1.pivot_table(values = value_col, rows=rows_col, aggfunc=lambda x: len(x.unique()))
    df_pivot1 = pd.DataFrame({df1_label: pivot1})
    # reset index, then the rows_col became a column of the pivot table
    df_pivot1 = df_pivot1.reset_index()
    pivot2 = df2.pivot_table(values = value_col, rows=rows_col, aggfunc=lambda x: len(x.unique()))
    df_pivot2 = pd.DataFrame({df2_label: pivot2})
    df_pivot2 = df_pivot2.reset_index()
    #print(df_pivot1.head(3))
    #print(df_pivot2.head(3))
    df = pd.merge(df_pivot1, df_pivot2, left_on = rows_col, right_on = rows_col, how = 'outer')
    print(df)

    # Draw a nested barplot to show survival for class and sex
    sns.set(style="darkgrid")
    ax = df.plot(kind='bar', width=0.8)
    for p in ax.patches:
        ax.annotate(int(p.get_height()), (p.get_x()+p.get_width()/2., p.get_height()), ha='center', va='center', xytext=(0, 10), textcoords='offset points')
    #for p in ax.patches:
    #    ax.annotate(str(p.get_height()), (p.get_x() * 1.005, p.get_height() * 1.005))
    ax.set_ylabel("{} number".format(value_col))
    xlist = df[rows_col].values.tolist()
    ax.set_xticklabels(xlist, rotation=270)
    plt.show()
    plt.close(fig)

def drawAreaPlot(df, column_list, col_xlabel, filter_name, **args):
#    df = cleanGauge(df)
    sns.set(style="darkgrid")
    ncol = len(column_list)
    y = [ [] for x in range(ncol)]

    df_slice = df[column_list]
    df_slice = df_slice.abs()

    # read the column lists
    for i in range(ncol):
        y[i] = df_slice.loc[:, column_list[i]].values.tolist()

    # convert to a turple
    #y_tuple = tuple(y)

    nrow = len(y[0])
    #y_2dlist = np.row_stack(y_tuple) # ncol*nrow

    # this call to 'cumsum' (cumulative sum), passing in your y data,
    # is necessary to avoid having to manually order the datasets
    xlabel = df.loc[:, col_xlabel].values.tolist()
    x = np.arange(nrow)
    y_stack =  np.cumsum(y, axis=0)
#    print("y = ", y)
#    print("y_stack = ", y_stack)
    #print(x, xlabel, y, y_tuple, y_2dlist, y_stack)
    fig = plt.figure()
    ax1 = fig.add_axes([0.1, 0.1, 0.8, 0.7], label="ax1")
    dict_stack = {"index": df.loc[:, col_xlabel].values.tolist(), "y_stack":  y_stack[ncol-1, :] }
    df_stack = pd.DataFrame(dict_stack)
    df_stack_slice = df_stack[ df_stack.y_stack > 10]
#    print(df_stack_slice)
    colors = ['#F0A3FF', '#0075DC', '#993F00', '#4C005C', '#191919', '#005C31', '#2BCE48', '#FFCC99', '#808080', '#94FFB5', '#8F7C00', '#9DCC00', '#C20088', '#003380', '#FFA405', '#FFA8BB', '#426600', '#FF0010', '#5EF1F2', '#00998F', '#E0FF66', '#740AFF', '#990000', '#FFFF00', '#FF5005']
    for i in range(ncol):
        if i == 0:
            ax1.fill_between(x, 0, y_stack[0,:], facecolor=colors[i], alpha=.7)
        else:
            ax1.fill_between(x, y_stack[i-1,:], y_stack[i,:], facecolor=colors[i], alpha=.7)

    if(args.has_key("save") and args["save"] == True):
        directory = os.path.join(workpath, "results")
        if not os.path.exists(directory):
            os.makedirs(directory)
        fig.savefig(os.path.join(workpath, "results", "{}_uncert_analysis.png".format(filter_name)), dpi=2*fig.dpi, frameon = False )
    else:
        plt.show()
    plt.close(fig)

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

def drawQuadStateCmp(df_gauges, filter_name, **args):
    # df_gauges: cell of dataframe, the dataframe the gauge results
    fig =  plt.figure()
    outliers = []
    rc = mpl.rcParams
    rc["xtick.labelsize"] = 8
    rc["ytick.labelsize"] = 8
    ax1 = fig.add_axes([0.1, 0.15, 0.8, 0.7], label="ax1")

    ## dataframe gauge data clean
#    orig_gauge_num = len(df.index)
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

def validTitle(title, illChars=string.punctuation+string.whitespace):
    newTitle = title.lower()
    newTitle = newTitle.replace('<', '_le_')
    newTitle = newTitle.replace('>', '_gt_')
    for x in illChars:
        newTitle=newTitle.replace(x,'_')
    return newTitle

def mergeColumns(df_input, columns, newcolname,**args):
    '''merge listed columns into a new column in the dataframe, another added column is category'''
    df = df_input.copy(deep=True)
    df['category'] = ''
    firstInstance = True
    for colname in columns:
        df['category'] = colname
        df[newcolname] = df[colname]
        if firstInstance:
            df_result = df.copy(deep=True)
            firstInstance = False
        else:
            df_result = df_result.append(df)
    print "YPC: merged DF"
    print df_result.pivot_table(values='gauge', columns='category', aggfunc=np.count_nonzero )
    return df_result

def groupsort(df, mode=''):
    '''sort DataFrame group by group, uring pre-defined mode:
            'cost_wt': largest cost_wt come first
    '''
    if len(df)==0:
        return df
    grouplabel = [x for x in df.columns if 'group' in x.lower()][0]
    grouped = df.groupby([grouplabel])
    if mode=='count':
        pass
    else: # default mode, 'cost-wt'
        wt_label = [x for x in df.columns if 'wt' in x.lower()][0]
        sumwt = grouped.apply(lambda x: x[wt_label].sum())
        sumwt = sumwt.sort_values(ascending=False)
        sortedgroups = sumwt
        #print type(sumwt),'\n',sumwt
    first = True
    result = pd.DataFrame({})
    for group in sortedgroups.index:
        # print group
        curdf = df[df[grouplabel]==group]
        if first:
            result = curdf
            first = False
        else:
            result = result.append(curdf)
    return result

def errDistributionPlotSort(df):
    grouplabel = [x for x in df.columns if 'group' in x.lower()][0]
    drawcd_label = [x for x in df.columns if 'draw' in x.lower()][0]
    gauge_label = [x for x in df.columns if 'gauge' in x.lower()][0]
    type_label = [x for x in df.columns if 'type' in x.lower()][0]
    gauge1D = df[df[type_label]=='1D']
    gauge2D = df[df[type_label]=='2D']
    gauge1D = gauge1D.sort_values(by=[drawcd_label, gauge_label])
    gauge1D = groupsort(gauge1D)
    gauge2D = gauge2D.sort_values(by=[drawcd_label, gauge_label])
    gauge2D = groupsort(gauge2D)
    result = gauge1D.append(gauge2D)
    groups = result[grouplabel].unique()
    return result, groups

def calRMS(df0, option=''):
    df = df0.copy()
    wt_label = [x for x in df.columns if 'wt' in x.lower()][0]
    df = df[df[wt_label]>1e-6]
    if len(df)==0:
        return 0
    err_label = [x for x in df.columns if 'error' in x.lower()][0]
    if option=='unweighted':
        df.ix[:, wt_label]=1
    df.ix[:, 'err^2'] = df.apply(lambda x: (x[err_label])**2, axis=1)
    rms = math.sqrt(np.inner(df['err^2'], df[wt_label]) /  df.sum()[wt_label])
    rms = round(rms, 3)
    return rms

def getSpecName(df):
    specnames=[x for x in df.columns if 'spec'==x.lower() or 'range_max'==x.lower()]
    if len(specnames)<1:
        raise KeyError('Fail, neither spec nor range_max column exist')
    else:
        sort_order = {}
        for col in specnames:
            sort_order[col] = 3
            if col.lower() == 'range_max':
                sort_order[col] = 2
            if col.lower() == 'spec':
                sort_order[col] = 1
        specnames = sorted(specnames, key=sort_order.__getitem__)
    return specnames[0]

def countInSpecRatio(df0):
    df = cleanGauge(df0)
    errorlabel = [x for x in df.columns if 'error' in x.lower()][0]
    speclabel = getSpecName(df)
    inspecfilter = "({}>-{}) and ({}<{})".format(errorlabel, speclabel, errorlabel, speclabel)
    nGauges = len(df)
    nInSpecGauges = len(df.query(inspecfilter))
    stat_dict = {"InSpecNum": nInSpecGauges,
                 "All": nGauges}
    if nGauges != 0:
        stat_dict["InSpecRatio"] = "{percent:.2%}".format(percent=1.*nInSpecGauges/nGauges)
    else:
        stat_dict["InSpecRatio"] = "{percent:.2%}".format(percent=0.)
    return stat_dict

def calInSpecRatio(df0):
    df=stdDFCols(df0)

    df_valid = df.query('cost_wt > 0')
    df_1D = df_valid.query("type=='1D'")
    df_2D = df_valid.query("type=='2D'")
    stat_dict = {}
    stat_dict["1D"] = countInSpecRatio(df_1D)
    stat_dict["2D"] = countInSpecRatio(df_2D)
    stat_dict["All"]= countInSpecRatio(df_valid)
    return pd.DataFrame(stat_dict)

def countErrorSourceAnalysisTags(df):
    tag_label = [x for x in df.columns if x.lower()=='tag'][0]
    tags = ['In-Spec', 'In-Spec-with-Potential-Risk', 'Out-of-Spec-with-Known-Reason', 'Out-of-Spec-with-Possible-Reason', 'Out-of-Spec-with-Unknown-Reason']
    stat_dict = {}
    nGauges = len(df)
    for tag in tags:
        if nGauges == 0:
            stat_dict[tag]='0(0.0%)'
            continue
        mask = "{}=='{}'".format(tag_label, tag)
        numCurTag = len(df.query(mask))
        stat_dict[tag]= "{}({percent:.2%})".format(numCurTag, percent=1.*numCurTag/nGauges)
    stat_dict['Valid-Gauges']="{}({percent:.2%})".format(nGauges, percent=1.)
    return stat_dict

def calErrorSourceAnalysisRatio(df):
    '''based on the tags of error Source Analysis'''
    df_valid = df.query('cost_wt > 0')
    df_1D = df_valid.query("type=='1D'")
    df_2D = df_valid.query("type=='2D'")
    stat_dict = {}
    stat_dict['All'] = countErrorSourceAnalysisTags(df_valid)
    stat_dict['1D'] = countErrorSourceAnalysisTags(df_1D)
    stat_dict['2D'] = countErrorSourceAnalysisTags(df_2D)
    return pd.DataFrame(stat_dict)

def adjustWtByErrorSourceAnalysis1(df0):
    df = df0.copy()
    def adjWt(row, tag_label, wt_label):
        wt = row[wt_label]
        if row[tag_label]=='Out-of-Spec-with-Known-Reason':
            wt = 0
        return np.round(wt, 2)
    tag_label = [x for x in df.columns if x.lower()=='tag'][0]
    wt_label = [x for x in df.columns if 'wt' in x.lower()][0]
    df.ix[:, wt_label] = df.apply(lambda x: adjWt(x, tag_label, wt_label), axis=1)
    return df

def adjustWtByErrorSourceAnalysis2(df0):
    df = df0.copy()
    def adjWt(row, tag_label, wt_label):
        wt = row[wt_label]
        if row[tag_label]=='In-Spec-with-Potential-Risk':
            wt /= 2.
        elif row[tag_label]=='Out-of-Spec-with-Known-Reason':
            wt = 0
        elif row[tag_label]=='Out-of-Spec-with-Possible-Reason':
            wt /= 3.
        elif row[tag_label]=='Out-of-Spec-with-Unknown-Reason':
            wt *= 4.
        return np.round(wt, 2)
    tag_label = [x for x in df.columns if x.lower()=='tag'][0]
    wt_label = [x for x in df.columns if 'wt' in x.lower()][0]
    df.ix[:, wt_label] = df.apply(lambda x: adjWt(x, tag_label, wt_label), axis=1)
    return df


def errorSourceAnalysisPlot(gauges):
    pal = setPalette('various')

    grouplabel = [x for x in gauges.columns if 'group' in x.lower()][0]
    ylabel = 'Model Error'
    type_label = [x for x in gauges.columns if 'type' in x.lower()][0]
    spec_columns = [x for x in gauges.columns if any( y==x.lower() for y in ['spec', 'range_max'])]
    if 'range_max' in spec_columns:
        spec_label = 'range_max'
    else:
        spec_label = 'spec'

    '''solution 1: seaborn'''
    df = gauges.copy(True)
#    df = df[df['cost_wt']>1e-6]
    df, groups = errDistributionPlotSort(df)
    df.ix[:, 'count'] = range(1, len(df)+1)
    num1D = len(df[df[type_label]=='1D'])
    df.ix[:, 'count'] = df.apply(lambda x: x['count'] if x[type_label]=='1D' else x['count']-num1D, axis=1)

    gauge1D = df[df[type_label]=='1D']
    gauge2D = df[df[type_label]=='2D']

    sns.set_style("ticks", {"xtick.major.size": 8, "ytick.major.size": 8})
    lm = sns.lmplot('count', ylabel, data=df, hue=grouplabel, col=type_label, fit_reg=False, legend=False, sharex=False, sharey=False, palette=pal)

    # axis
    for nAxis, ax in enumerate(lm.axes[0]):
        if (nAxis==0):
            curdf = gauge1D
            if len(curdf)==0:
                continue
            title = "1D Error Distribution, RMS={}nm".format(calRMS(gauge1D))
            legend_xshift = 0
        else:
            curdf = gauge2D
            if len(curdf)==0:
                continue
            title = "2D Error Distribution, RMS={}nm".format(calRMS(gauge2D))
            legend_xshift = 0.2

        ax.set_xlim(0, len(curdf)+1)
        ymax = max(abs(df.max()[ylabel]), abs(df.min()[ylabel]))+0.1
        ax.set_ylim(-ymax, ymax)
        ax.plot(curdf['count'], curdf[spec_label], label='+spec')
        ax.plot(curdf['count'], -curdf[spec_label], label='-spec')
        InSpec = curdf[curdf['Tag']=='In-Spec']
        InSpecWiRisk = curdf[curdf['Tag']=='In-Spec-with-Potential-Risk']
        OutSpecWiReason = curdf[curdf['Tag']=='Out-of-Spec-with-Known-Reason']
        OutSpecWoReason = curdf[curdf['Tag']=='Out-of-Spec-with-Unknown-Reason']
        OutSpecWiPossibleReason = curdf[curdf['Tag']=='Out-of-Spec-with-Possible-Reason']
        firstInSpecWiRisk, firstOutSpecWiReason, firstOutSpecWoReason, firstOutSpecWiPossibleReason = [True for i in range(4)]
        for ngroup, group in enumerate(groups):
            curcolor = pal[ngroup]
            curInSpecWiRisk = InSpecWiRisk[InSpecWiRisk[grouplabel]==group]
            if( isinstance(curInSpecWiRisk, pd.DataFrame) and len(curInSpecWiRisk) > 0):
                if firstInSpecWiRisk:
                    legend_label = 'In-Spec-with-Potential-Risk'
                    firstInSpecWiRisk = False
                else:
                    legend_label = ylabel
                ax.plot(curInSpecWiRisk['count'], curInSpecWiRisk[ylabel], linestyle='None', marker='o', markeredgewidth=1,
                    markeredgecolor=curcolor, markerfacecolor='None', label=legend_label)
            curOutSpecWiReason = OutSpecWiReason[OutSpecWiReason[grouplabel]==group]
            if( isinstance(curOutSpecWiReason, pd.DataFrame) and len(curOutSpecWiReason) > 0):
                if firstOutSpecWiReason:
                    legend_label = 'Out-of-Spec-with-Known-Reason'
                    firstOutSpecWiReason = False
                else:
                    legend_label = ylabel
                ax.plot(curOutSpecWiReason['count'], curOutSpecWiReason[ylabel], linestyle='None', marker='x', markeredgewidth=2,
                    markeredgecolor=curcolor, markerfacecolor=curcolor, label=legend_label)
            curOutSpecWoReason = OutSpecWoReason[OutSpecWoReason[grouplabel]==group]
            if( isinstance(curOutSpecWoReason, pd.DataFrame) and len(curOutSpecWoReason) > 0):
                if firstOutSpecWoReason:
                    legend_label = 'Out-of-Spec-with-Unknown-Reason'
                    firstOutSpecWoReason = False
                else:
                    legend_label = ylabel
                ax.plot(curOutSpecWoReason['count'], curOutSpecWoReason[ylabel], linestyle='None', marker='d', markeredgewidth=1,
                    markeredgecolor=curcolor, markerfacecolor='None', label=legend_label)
            curOutSpecWiPossibleReason = OutSpecWiPossibleReason[OutSpecWiPossibleReason[grouplabel]==group]
            if( isinstance(curOutSpecWiPossibleReason, pd.DataFrame) and len(curOutSpecWiPossibleReason) > 0):
                if firstOutSpecWiPossibleReason:
                    legend_label = 'Out-of-Spec-with-Possible-Reason'
                    firstOutSpecWiPossibleReason = False
                else:
                    legend_label = ylabel
                ax.plot(curOutSpecWiPossibleReason['count'], curOutSpecWiPossibleReason[ylabel], linestyle='None', marker='s', markeredgewidth=1,
                    markeredgecolor=curcolor, markerfacecolor='None', label=legend_label)
        # title
        ax.set_title(title)
        # legend
        h, l = ax.get_legend_handles_labels()
        legend_handles, legend_labels = [], []
        art_handles, art_labels = [], []
        tags = ['In-Spec-with-Potential-Risk', 'Out-of-Spec-with-Known-Reason', 'Out-of-Spec-with-Unknown-Reason', 'Out-of-Spec-with-Possible-Reason', '+spec', '-spec']
        for nlabel, label in enumerate(l):
            if (label != ylabel) and (label not in tags):
                legend_labels.append(label)
                legend_handles.append(h[nlabel])
            elif label in tags:
                art_labels.append(label)
                art_handles.append(h[nlabel])
        box = ax.get_position()
        # Shrink current axis by 10% in the width
        ax.set_position([box.x0, box.y0, box.width * 0.9, box.height])
        ax.legend(legend_handles, legend_labels, bbox_to_anchor=(0.20+0.05*i, 1), fontsize=10, borderaxespad=0.,) # framealpha=0.5  loc='upper right lower left', bbox_to_anchor=(1, 1),
        if nAxis == 1:
            plt.gca().add_artist(plt.gcf().legend(art_handles, art_labels, bbox_to_anchor=(1.00, 0.95), fontsize=10, framealpha=0.5)) # framealpha=0.5  loc='upper right lower left', bbox_to_anchor=(1, 1),
    plt.show()

def setAxisLim(ax, scale=0.1):
    (xmin, xmax) =  ax.get_xlim()
    (ymin, ymax) =  ax.get_ylim()
    x_range = xmax-xmin
    y_range = ymax-ymin
    ax.set_xlim([xmin-x_range*scale, xmax+x_range*scale])
    ax.set_ylim([ymin-y_range*scale, ymax+y_range*scale])


