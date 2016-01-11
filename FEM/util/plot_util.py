import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import pandas as pd
import seaborn as sns
import numpy as np
import os
import math

workpath = os.path.dirname(os.path.abspath(__file__))

##############################################################
#########      function list                ##################
# cleanGauge(df): change the column name to lower case, replace space with underscore and clean zero cost_wt || model_cd gauges
# setPalette(): create palette from distinguishable colors: https://en.wikipedia.org/wiki/Help:Distinguishable_colors
# drawXY(df, x_col_name, y_col_name, filter_name, **args): draw XY plotting by two column of dataframe
# drawQuadState(df, filter_name, **args): GF Quad State plotting, model error, range_min, range_max, model_cd, wafer_cd
# drawStackedBar(df, grouped_col, stacked_col, **args): group statistics, stacked by subgroups, also display the proportion of groups
# def drawTwoGroupsCmp(df1, df2, rows_col, value_col, **args):: http://pandas.pydata.org/pandas-docs/stable/visualization.html kind='bar'
# def dfAreaPlot(df, column_list, col_xlabel, filter_name, **args), plot a cumulative area plotting


def cleanGauge(df):
    col_names = df.columns.values.tolist()
    for gauge_col in col_names:
        df.rename(columns={gauge_col: gauge_col.lower()}, inplace=True)
    col_names = df.columns.values.tolist()
    for name in col_names:
        if " " in name:
            newName = name.replace(" ", "_")
            df.rename(columns={name: newName}, inplace=True)
    if "drawn_cd" in name:
        df.rename(columns={"drawn_cd": "draw_cd"}, inplace=True)
    df = df[df["cost_wt"] > 0.0001 ]
    if "model_cd" in col_names:
        df=  df[['model_cd'] > 0.0001]
    df.reset_index(inplace = True)
    return df

def setPalette():
    colors = ['#F0A3FF', '#0075DC', '#993F00', '#4C005C', '#191919', '#005C31', '#2BCE48', '#FFCC99', '#808080', '#94FFB5', '#8F7C00', '#9DCC00', '#C20088', '#003380', '#FFA405', '#FFA8BB', '#426600', '#FF0010', '#5EF1F2', '#00998F', '#E0FF66', '#740AFF', '#990000', '#FFFF00', '#FF5005']
    #colors = reversed(colors)
    colors_r = colors[::-1]
    #pal = sns.xkcd_palette(colors)
    pal = sns.color_palette(colors_r)
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

    if(args.has_key("save") & args["save"] == True):
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

    ## dataframe gauge data clean
#    orig_gauge_num = len(df.index)
    numGauges = len(df.index)
#    print("{} gauges: {} valid, {} filtered by \'cost_wt>0\' & \'model_cd>0\' ".format(filter_name, filtered_gauge_num, orig_gauge_num-filtered_gauge_num))

#    if numGauges > 40:
#        def sampleBin(numGauges):
#            divider = 2
#            while(True):
#                if numGauges/divider <= 40:
#                    return numGauges/divider
#                divider += 1
#        numBins = sampleBin(numGauges)
#        plt.locator_params(nbins=numBins)
    ## gauge data acquisition, model error consider tone_sgn
    df = df.sort_values(by=["draw_cd", "plot_cd"], ascending=[True, True])
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

    if(args.has_key("save") & args["save"] == True):
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

    if(args.has_key("save") & args["save"] == True):
        directory = os.path.join(workpath, "results")
        if not os.path.exists(directory):
            os.makedirs(directory)
        fig.savefig(os.path.join(workpath, "results", "{}_uncert_analysis.png".format(filter_name)), dpi=2*fig.dpi, frameon = False )
    else:
        plt.show()
    plt.close(fig)

def dfAreaPlot(df, column_list, col_xlabel, filter_name, **args):
    sns.set(style="darkgrid")
    df_slice = df[column_list]
    df_slice = df_slice.abs()
    ax = df_slice.plot(kind='area')
    ax.set_xticks(df.index)
    ax.set_xticklabels(df.loc[:, col_xlabel], rotation=90)
    ax.set_xlabel(col_xlabel)
    ax.set_ylabel("Uncertainty")
    ax.set_title("Process Uncertainty plotting")

    fig = ax.get_figure()
    if(args.has_key("save") & args["save"] == True):
        directory = os.path.join(workpath, "results")
        if not os.path.exists(directory):
            os.makedirs(directory)
        fig.savefig(os.path.join(workpath, "results", "{}_uncert_analysis.png".format(filter_name)), dpi=2*fig.dpi, frameon = False )
    else:
        plt.show()
    plt.close(fig)



# Only suitable for NTD, BF. Draw resist bias trend by group or use overall
def NTDResistBiasTrend(df, **args):
    ## args:
    ##      group, draw by the groups, default is "overall"
    ##      save, whether to save the image

    hasSave = False
    if(args.has_key("save") & args["save"] == True):
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
    df.loc[:, "err^2"] = df.apply(lambda x: x["cost_wt"]*x["model_error"]*x["model_error"], axis=1)
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
        rms = math.sqrt(df_trench.sum()['err^2'] /  df_trench.sum()["cost_wt"])
        rms = np.round(rms, 3)
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
        rms = math.sqrt(df_line.sum()['err^2'] /  df_line.sum()["cost_wt"])
        rms = np.round(rms, 3)
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

# Only suitable for PTD,  Draw resist bias trend vs drawn_cd by group or use overall
def PTDResistBiasTrend0(df, **args):
    ## args:
    ##      group, draw by the groups, default is "overall"
    ##      save, whether to save the image

    hasSave = False
    if(args.has_key("save") & args["save"] == True):
        hasSave = True
    if args.has_key("group"):
        groupname = args["group"]
    else:
        groupname = "overall"

    df.reset_index(inplace = True)
    df.loc[:, "model_cd-ai_cd"] = df.apply(lambda x: x["model_cd"] - x["ai_cd"], axis=1)
    df.loc[:, "wafer_cd-ai_cd"] = df.apply(lambda x: x["wafer_cd"] - x["ai_cd"], axis=1)

    directory = os.path.join(workpath, "results")
    if not os.path.exists(directory):
        os.makedirs(directory)

    fig1 = plt.figure()
    ax1 = fig1.add_axes([0.1, 0.1, 0.8, 0.8], label="ax1")

    numGauges = len(df.index)

    if numGauges > 0:
        df = df.sort_values(by=["draw_cd", "plot_cd"], ascending=[True, True])
        index = range(1, numGauges+1, 1)
        ax1.plot(index, df["wafer_cd-ai_cd"].values.tolist(), linestyle='--', color = '#990000', marker= '^', markeredgewidth = 1, markeredgecolor='r', markerfacecolor='none', label='wafer_cd-ai_cd') ## color wine
        ax1.plot(index, df["model_cd-ai_cd"].values.tolist(), linestyle = 'None', color = '#9DCC00', marker='o', markeredgewidth = 1, markeredgecolor='g', markerfacecolor='none', label='model_cd-ai_cd')
        # title
        # plot_cd = df["plot_cd"].values.tolist()
        draw_cd = df["draw_cd"].values.tolist()
        ax1.set_xticks(index)
        ax1.set_xticklabels(draw_cd, rotation=270)
        ax1.set_xlabel("draw_cd($nm$)")
        ax1.set_ylabel(r"resist bias($nm$)")
        figure_title = "{} resist bias vs draw_cd".format(groupname)
        fig1.suptitle(figure_title, fontsize=14)
        # legend
        h, l = ax1.get_legend_handles_labels()
        ax1.legend(h, l, loc='upper right', fontsize=12, framealpha = 0.5)

        if hasSave:
            fig1.savefig(os.path.join(workpath, "results", "{}_resist_bias_vs_draw_cd_trend.png".format(groupname)), dpi=2*fig1.dpi, frameon = False )
        else:
            plt.show()
    plt.close(fig1)

def PTDResistBiasTrend(df, **args):
    ## args:
    ##      group, draw by the groups, default is "overall"
    ##      save, whether to save the image

    hasSave = False
    if(args.has_key("save") & args["save"] == True):
        hasSave = True
    if args.has_key("group"):
        groupname = args["group"]
    else:
        groupname = "overall"

    df.reset_index(inplace = True)
    df.loc[:, "model_cd-ai_cd"] = df.apply(lambda x: x["model_cd"] - x["ai_cd"], axis=1)
    df.loc[:, "wafer_cd-ai_cd"] = df.apply(lambda x: x["wafer_cd"] - x["ai_cd"], axis=1)
    df.loc[:, "err^2"] = df.apply(lambda x: x["cost_wt"]*x["model_error"]*x["model_error"], axis=1)


    directory = os.path.join(workpath, "results")
    if not os.path.exists(directory):
        os.makedirs(directory)

#    fig1, ax1 = plt.subplots(1, 1)
    fig1 = plt.figure()
    ax1 = fig1.add_axes([0.1, 0.2, 0.8, 0.7], label="ax1")

    numGauges = len(df.index)

    if numGauges > 0:
        rms = math.sqrt(df.sum()['err^2'] /  df.sum()["cost_wt"])
        rms = np.round(rms, 3)
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
    plt.close(fig1)

def cmpResistBiasTrend(df1, df2, **args):
    ## args:
    ##      group, draw by the groups, default is "overall"
    ##      save, whether to save the image
    hasSave = False
    if(args.has_key("save") & args["save"] == True):
        hasSave = True

    df1.reset_index(inplace = True)
    df1.loc[:, "model_cd-ai_cd"] = df1.apply(lambda x: x["model_cd"] - x["ai_cd"], axis=1)
    df1.loc[:, "wafer_cd-ai_cd"] = df1.apply(lambda x: x["wafer_cd"] - x["ai_cd"], axis=1)
    df1.loc[:, "err^2"] = df1.apply(lambda x: x["cost_wt"]*x["model_error"]*x["model_error"], axis=1)


    df2.reset_index(inplace = True)
    df2.loc[:, "model_cd-ai_cd"] = df2.apply(lambda x: x["model_cd"] - x["ai_cd"], axis=1)
    df2.loc[:, "wafer_cd-ai_cd"] = df2.apply(lambda x: x["wafer_cd"] - x["ai_cd"], axis=1)
    df2.loc[:, "err^2"] = df2.apply(lambda x: x["cost_wt"]*x["model_error"]*x["model_error"], axis=1)

    if( args.has_key("path") ):
        directory = args["path"]
    else:
        directory = os.path.join(workpath, "results")
    if not os.path.exists(directory):
        os.makedirs(directory)

    fig1 = plt.figure()
    ax1 = fig1.add_axes([0.1, 0.2, 0.8, 0.7], label="ax1")

    numGauges = len(df1.index)
    if numGauges > 0:
        rms1 = math.sqrt(df1.sum()['err^2'] /  df1.sum()["cost_wt"])
        rms1 = np.round(rms1, 3)
        rms2 = math.sqrt(df2.sum()['err^2'] /  df2.sum()["cost_wt"])
        rms2 = np.round(rms2, 3)
        if numGauges > 40:
            plt.locator_params(nbins=24)
        if numGauges < 40:
            plt.locator_params(nbins=numGauges)

        # target for tone_sgn = 1 gauge
        df1.plot(x = "ai_cd", y = "wafer_cd-ai_cd", ax = ax1, rot=60, linestyle = 'None', marker= 'o', markersize=2.5, markerfacecolor='b', label='wafer_cd-ai_cd') # markeredgewidth = 1, markeredgecolor='b', blue
        df1.plot(x = "ai_cd",  y = "model_cd-ai_cd", ax = ax1, rot=60, linestyle='None', linewidth=0.5, color = 'r', marker='o', markersize=2.5,  markerfacecolor='r', label='model_cd-ai_cd') # markeredgewidth = 1, markeredgecolor='g', markerfacecolor='none', green
        df2.plot(x = "ai_cd",  y = "model_cd-ai_cd", ax = ax1, rot=60, linestyle='None', linewidth=0.5, color = 'g', marker='o', markersize=2.5,  markerfacecolor='g', label='model_cd-ai_cd') # markeredgewidth = 1, markeredgecolor='g', markerfacecolor='none', green

        fig1 = ax1.get_figure()
        ax1.set_xlabel("ai_cd($nm$)")
        ax1.set_ylabel(r"resist bias($nm$)")
        figure_title = "Resist bias trend through trench, RMS={}$nm$ ({} $nm$)".format( rms1, rms2)
        fig1.suptitle(figure_title, fontsize=14)
        # legend
        h, l = ax1.get_legend_handles_labels()
        ax1.legend(h, l, loc='best', fontsize=10, framealpha = 0.5)
        ax1.tick_params(axis='both', which='major', labelsize=8)
        if hasSave:
            fig1.savefig(os.path.join(workpath, "results", "cmp_resist_bias_vs_trench_trend.png"), dpi=2*fig1.dpi, frameon = False )
        else:
            plt.show()
    plt.close(fig1)