# -*- coding: utf-8 -*-
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

my_dpi = 96
NM_defocus = -0.056

df = pd.read_csv('C:/Localdata/D/Note/Python/BF/bossung2.txt', delim_whitespace = True)
#df.loc[:, "delta_defocus"] = df.loc[:, "delta_defocus"] + 
print(df)

df_m=df[ df["legend"]=="model_cd"]
df_w=df[ df["legend"]=="wafer_cd"]
defocus = df_m.loc[:, "defocus"].values
model_cd = df_m.loc[:, "cd"].values
wafer_cd = df_w.loc[:, "cd"].values 
print(model_cd)
#wafer_cd = df.loc[:, "cd", df.loc[;, "legend"]=="wafer_cd"].values
length = len(model_cd)

## the rc have to set before plotting
#sns.set_style("darkgrid")
rc = mpl.rcParams
rc["xtick.direction"] = 'in'
rc["xtick.labelsize"] = 8
rc["ytick.direction"] = 'in'
rc["ytick.labelsize"] = 8
#sns.set_context(rc=rc)
##print(rc)
#
fig = plt.figure()
ax = fig.add_axes([0.1, 0.1, 0.8, 0.8], label="ax")
##fig = plt.figure(figsize=(15,6))
##with sns.color_palette("PuBuGn_d"): 
##    ax1 = fig.add_subplot(121)
##with sns.color_palette("RdBu_r"):   
##    ax2 = fig.add_subplot(122)
#
## lmplot defocus , model_cd
order=["wafer_cd", "model_cd"]
g_mcd = sns.lmplot("defocus", "cd", hue = "legend", data=df, order=2,
           ci=None, size=4, ax = ax, palette="PuBuGn_d", 
           hue_order = order) #,  scatter_kws = { "color": "#80013f" }, line_kws={"color": sns.xkcd_rgb["denim blue"]}, ,legend_out=False
g_mcd.despine(left=True)
plt.legend(loc='upper left')
ax.set_xticks(defocus) 
#ymin1, ymax1 = ax.get_ylim()
#
## lmplot defocus , wafer_cd
#g_wcd = sns.lmplot("defocus", "wafer_cd", data=df, order=2,
#           ci=None, line_kws={"color": "indianred"},
#           size=4, ax = ax, legend=True) # palette="RdBu_r"
#ymin2, ymax2 = ax.get_ylim()
#
#ymin, ymax = [min(ymin1, ymin2), max(ymax1, ymax2)]
#ax.set_ylim([ymin, ymax])
#
## fitting: defocus , model_cd
#coeff1 = np.polyfit(defocus, model_cd, 2)
#a1 = coeff1[0]
#b1 = coeff1[1]
#c1 = coeff1[2]
#x_peak1 = -b1/(2*a1)
#y_peak1 = a1*x_peak1*x_peak1 + b1*x_peak1 + c1
#ax.scatter([x_peak1, x_peak1], [ymin, y_peak1], 50, color='b')
#ax.plot([x_peak1, x_peak1], [ymin, y_peak1], color='blue', linewidth=1, linestyle="--" )
#ax.annotate(r'$Model\_BF = {}$'.format(np.round(x_peak1,4)),
#    xy=(x_peak1, ymin1), xycoords='data',
#    xytext=(-110, +20), textcoords='offset points', fontsize=10, arrowprops=dict(arrowstyle="->", connectionstyle="arc3, rad=0.1"))  
#     
## fitting: defocus , wafer_cd
#coeff2 = np.polyfit(defocus, wafer_cd, 2)
#a2 = coeff2[0]
#b2 = coeff2[1]
#c2 = coeff2[2]
#x_peak2 = -b2/(2*a2)
#y_peak2 = np.polyval(coeff2, x_peak2)
##y_peak1 = a1*x_peak1*x_peak1 + b1*x_peak1 + c1
#ax.scatter([x_peak2, x_peak2], [ymin, y_peak2], 50, color='r')
#ax.plot([x_peak2, x_peak2], [ymin, y_peak2], color='indianred', linewidth=1, linestyle="--" )
#ax.annotate(r'$Wafer\_BF = {}$'.format(np.round(x_peak2, 4)),
#    xy=(x_peak2, ymin2), xycoords='data',
#    xytext=(+10, +30), textcoords='offset points', fontsize=10, arrowprops=dict(arrowstyle="->", connectionstyle="arc3, rad=-0.1"))
#
## fig show and save
##fig.suptitle("pitch_08B7_O5_1 bossung GUI")
#ax.set_title("pitch_08B7_O5_1 bossung GUI")
fig.show()
#
#filename = "bossung_gui.png"
#directory = os.path.dirname(os.path.abspath(__file__))
#savepath = os.path.join(directory, filename)
##plt.rc('savefig', dpi=300)
#fig.savefig(savepath,frameon = False )  #plt.savefig(savepath, dpi=my_dpi*2, frameon = False , transparent = True) grid has conflicts with transparent