# -*- coding: utf-8 -*-
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

my_dpi = 96

mcd_focus_autocal = [
	{
		"defocus":-0.062,
		"mcd":48.405823788135
	},
	{
		"defocus":-0.111,
		"mcd":45.544566153783
	},
	{
		"defocus":-0.103,
		"mcd":46.534481925309
	},
	{
		"defocus":-0.086,
		"mcd":47.895842246053
	},
	{
		"defocus":-0.054,
		"mcd":48.21835746602
	},
	{
		"defocus":-0.046,
		"mcd":47.833852285557
	},
	{
		"defocus":-0.078,
		"mcd":48.242967828974
	},
	{
		"defocus":-0.038,
		"mcd":47.28796476212
	},
	{
		"defocus":-0.094,
		"mcd":47.373765772479
	},
	{
		"defocus":-0.07,
		"mcd":48.427337173056
	},
	{
		"defocus":-0.022,
		"mcd":45.543697598066
	},
	{
		"defocus":-0.03,
		"mcd":46.522089716588
	}
]
mcd_autocal = []
focus_autocal = []

NM_wcd = 0
for cur_dict in mcd_focus_autocal: # list only have index
    mcd_autocal.append(cur_dict["mcd"])
    focus_autocal.append(cur_dict["defocus"])
    #if cur_dict['defocus'] == 0:
    #    NM_wcd = cur_dict['wcd']

print(mcd_autocal)
print(focus_autocal)

length = len(mcd_autocal)
bossung_mcd = {"model_cd": pd.Series(mcd_autocal, index=range(length)), "defocus": pd.Series(focus_autocal, index=range(length))}
df_mcd = pd.DataFrame(bossung_mcd)
print(df_mcd)

# the rc have to set before plotting
sns.set_style("darkgrid")
rc = mpl.rcParams
rc["xtick.direction"] = 'in'
rc["xtick.labelsize"] = 8
rc["ytick.direction"] = 'in'
rc["ytick.labelsize"] = 8
sns.set_context(rc=rc)
#print(rc)

fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(15,6), sharey=True) #
#fig = plt.figure(figsize=(15,6))
#with sns.color_palette("PuBuGn_d"): 
#    ax1 = fig.add_subplot(121)
#with sns.color_palette("RdBu_r"):   
#    ax2 = fig.add_subplot(122)

# lmplot model_cd
g_mcd = sns.lmplot("defocus", "model_cd", data=df_mcd, order=2,
           ci=None, scatter_kws = { "color": "#80013f" }, 
           line_kws={"color": sns.xkcd_rgb["denim blue"]},  
           size=4, ax = ax1, legend=False) #, palette="PuBuGn_d",
defocus_auto = df_mcd.loc[:,"defocus"].values
mcd_auto = df_mcd.loc[:,"model_cd"].values
print(type(defocus_auto))
coeff1 = np.polyfit(defocus_auto, mcd_auto, 2)
a1 = coeff1[0]
b1 = coeff1[1]
c1 = coeff1[2]
x_peak1 = -b1/(2*a1)
y_peak1 = a1*x_peak1*x_peak1 + b1*x_peak1 + c1
#g_mcd.set(xticks=focus_autocal)
ax1.set_xticks(focus_autocal)
xmin1 = min(focus_autocal)
xmax1 = max(focus_autocal)
points = len(focus_autocal)
if(points != 0):
    temp_min = xmin1
    xmin1 = xmin1 - (xmax1-temp_min)/points
    xmax1 = xmax1 + (xmax1-temp_min)/points
ax1.set_xlim(xmin1, xmax1)  
ymin1, ymax1 = ax1.get_ylim()
ax1.set_title("Model_cd through defocus")
    
# lmplot wafer_cd
df_wcd = pd.read_csv('C:/Localdata/D/Note/Python/BF/bossung.txt', delim_whitespace = True)
print(df_wcd)
delta_defocus = df_wcd.loc[:, "delta_defocus"].tolist()
delta_defocus = np.round(delta_defocus, 3)
print(delta_defocus)
g_wcd = sns.lmplot("delta_defocus", "wafer_cd", data=df_wcd, order=2,
           ci=None, line_kws={"color": "indianred"}, size=4, ax = ax2, legend_out=False) # palette="RdBu_r",
ax2.set_xticks(delta_defocus)
ax2.set_title("Wafer_cd through delta_defocus")

ymin2, ymax2 = ax2.get_ylim()
ymin = min(ymin1, ymin2)
ymax = max(ymax1, ymax2)
print("ymin1 {} ymax1 {} ymin2 {} ymax2 {}".format(ymin1, ymax1, ymin2, ymax2))
ax1.set_ylim(ymin, ymax)


# plot the model_BF 
defocus_auto = df_mcd.loc[:,"defocus"].values
mcd_auto = df_mcd.loc[:,"model_cd"].values
print(type(defocus_auto))
coeff1 = np.polyfit(defocus_auto, mcd_auto, 2)
a1 = coeff1[0]
b1 = coeff1[1]
c1 = coeff1[2]
x_peak1 = -b1/(2*a1)
y_peak1 = a1*x_peak1*x_peak1 + b1*x_peak1 + c1
ax1.scatter([x_peak1, x_peak1], [ymin, y_peak1], 50, color='#80013f')
ax1.plot([x_peak1, x_peak1], [ymin, y_peak1], color='red', linewidth=1, linestyle="--" )
ax1.annotate(r'$Model\_BF = {}$'.format(np.round(x_peak1,4)),
    xy=(x_peak1, ymin), xycoords='data',
    xytext=(+10, +20), textcoords='offset points', fontsize=10, arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))
    
    
# plot the NM_BF_Shift 
delta_defocus = df_wcd.loc[:,"delta_defocus"].values
wcd_auto = df_wcd.loc[:,"wafer_cd"].values
coeff2 = np.polyfit(delta_defocus, wcd_auto, 2)
a2 = coeff2[0]
b2 = coeff2[1]
c2 = coeff2[2]
x_peak2 = -b2/(2*a2)
y_peak2 = np.polyval(coeff2, x_peak2)
#y_peak1 = a1*x_peak1*x_peak1 + b1*x_peak1 + c1
ax2.scatter([x_peak2, x_peak2], [ymin, y_peak2], 50, color='b')
ax2.plot([x_peak2, x_peak2], [ymin, y_peak2], color='g', linewidth=1, linestyle="--" )
ax2.annotate(r'$NM\_BF\_Shift = {}$'.format(np.round(x_peak2, 4)),
    xy=(x_peak2, ymin), xycoords='data',
    xytext=(+10, +20), textcoords='offset points', fontsize=10, arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))

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
ax1.legend(legend_handles, legend_labels, loc='best',  fontsize=12, framealpha = 0.5)  #bbox_to_anchor=(-0.1, 1),
print(legend_labels)
     
#plt.tight_layout()
fig.suptitle("pitch_08B7_O5_1 bossung curve")
fig.show()

filename = "bossung_auto.png"
directory = os.path.dirname(os.path.abspath(__file__))
savepath = os.path.join(directory, filename)
#plt.rc('savefig', dpi=300)
fig.savefig(savepath,frameon = False )  #plt.savefig(savepath, dpi=my_dpi*2, frameon = False , transparent = True) grid has conflicts with transparent
