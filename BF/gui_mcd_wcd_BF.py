# -*- coding: utf-8 -*-
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

my_dpi = 96
NM_defocus = -0.056

df = pd.read_csv('C:/Localdata/D/Note/Python/BF/bossung0.txt', delim_whitespace = True)
df['defocus'] = pd.Series(df.loc[:, "delta_defocus"]+NM_defocus, index=df.index)
#df.loc[:, "delta_defocus"] = df.loc[:, "delta_defocus"] + 
print(df)
defocus = df.loc[:, "defocus"].values
model_cd = df.loc[:, "model_cd"].values
wafer_cd = df.loc[:, "wafer_cd"].values
length = len(model_cd)

# the rc have to set before plotting
sns.set_style("darkgrid")
rc = mpl.rcParams
rc["xtick.direction"] = 'in'
rc["xtick.labelsize"] = 8
rc["ytick.direction"] = 'in'
rc["ytick.labelsize"] = 8
sns.set_context(rc=rc)
#print(rc)

fig = plt.figure()
ax = fig.add_axes([0.1, 0.1, 0.8, 0.8], label="ax")
#fig = plt.figure(figsize=(15,6))
#with sns.color_palette("PuBuGn_d"): 
#    ax1 = fig.add_subplot(121)
#with sns.color_palette("RdBu_r"):   
#    ax2 = fig.add_subplot(122)

# lmplot defocus , model_cd
g_mcd = sns.regplot("defocus", "model_cd", data=df, order=2,
           ci=None, scatter_kws = { "color": sns.xkcd_rgb["azure"] }, 
           line_kws={"color": sns.xkcd_rgb["denim blue"]}, 
          ax = ax, label="model_cd") #, palette="PuBuGn_d",
ax.set_xticks(defocus) 
ymin1, ymax1 = ax.get_ylim()

# lmplot defocus , wafer_cd
g_wcd = sns.lmplot("defocus", "wafer_cd", data=df, order=2,
           ci=None,  scatter_kws = { "color": sns.xkcd_rgb["rose"] }, 
           line_kws={"color": "indianred"},
           ax = ax, label="wafer_cd") # palette="RdBu_r"
ymin2, ymax2 = ax.get_ylim()

ymin, ymax = [min(ymin1, ymin2), max(ymax1, ymax2)]
ax.set_ylim([ymin, ymax])

# fitting: defocus , model_cd
coeff1 = np.polyfit(defocus, model_cd, 2)
a1 = coeff1[0]
b1 = coeff1[1]
c1 = coeff1[2]
x_peak1 = -b1/(2*a1)
y_peak1 = a1*x_peak1*x_peak1 + b1*x_peak1 + c1
ax.scatter([x_peak1, x_peak1], [ymin, y_peak1], 50, color='b')
ax.plot([x_peak1, x_peak1], [ymin, y_peak1], color=sns.xkcd_rgb["azure"], linewidth=1, linestyle="--" )
ax.annotate(r'$Model\_BF = {}$'.format(np.round(x_peak1,4)),
    xy=(x_peak1, ymin1), xycoords='data',
    xytext=(-110, +20), textcoords='offset points', fontsize=10, arrowprops=dict(arrowstyle="->", connectionstyle="arc3, rad=0.1"))  
     
# fitting: defocus , wafer_cd
coeff2 = np.polyfit(defocus, wafer_cd, 2)
a2 = coeff2[0]
b2 = coeff2[1]
c2 = coeff2[2]
x_peak2 = -b2/(2*a2)
y_peak2 = np.polyval(coeff2, x_peak2)
#y_peak1 = a1*x_peak1*x_peak1 + b1*x_peak1 + c1
ax.scatter([x_peak2, x_peak2], [ymin, y_peak2], 50, color='r')
ax.plot([x_peak2, x_peak2], [ymin, y_peak2], color='indianred', linewidth=1, linestyle="--" )
ax.annotate(r'$Wafer\_BF = {}$'.format(np.round(x_peak2, 4)),
    xy=(x_peak2, ymin2), xycoords='data',
    xytext=(+10, +30), textcoords='offset points', fontsize=10, arrowprops=dict(arrowstyle="->", connectionstyle="arc3, rad=-0.1"))

# fig show and save

# legend
ax.legend()
#legend_handles = []
#legend_labels = []
## Two kinds of handles: matplotlib.lines.Line2D object; Container object of 22 artists
## Not able to use append, but able to use "+"
#for ax in [ax]:
#    h, l = ax.get_legend_handles_labels()
#    legend_handles = legend_handles + h 
#    legend_labels = legend_labels +l 
#    # Shrink current axis by 20% 
#    box = ax.get_position()
#    ax.set_position([box.x0+box.width * 0.2, box.y0, box.width * 0.8, box.height])     

#fig.suptitle("pitch_08B7_O5_1 bossung GUI")
ax.set_title("pitch_08B7_O5_1 bossung GUI")
fig.show()

filename = "bossung_gui.png"
directory = os.path.dirname(os.path.abspath(__file__))
savepath = os.path.join(directory, filename)
#plt.rc('savefig', dpi=300)
fig.savefig(savepath,frameon = False )  #plt.savefig(savepath, dpi=my_dpi*2, frameon = False , transparent = True) grid has conflicts with transparent