# -*- coding: utf-8 -*-
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

my_dpi = 96

pitch_04A7_K13_1  =[
  {
    "wcd" :  57.458394,
    "defocus" :   0.045
  },
  {
    "wcd" :  61.283071,
    "defocus" :   -0.015
  },
  {
    "wcd" :  59.16074,
    "defocus" :   -0.045
  },
  {
    "wcd" :  59.012457,
    "defocus" :   0.03
  },
  {
    "wcd" :  59.89526,
    "defocus" :   0.015
  },
  {
    "wcd" :  61.060977,
    "defocus" :   0
  },
  {
    "wcd" :  60.525337,
    "defocus" :   -0.03
  }
]
wcd = []
defocus = []
NM_wcd = 0
for cur_dict in pitch_04A7_K13_1: # list only have index
    wcd.append(cur_dict['wcd'])
    defocus.append(cur_dict['defocus'])
    if cur_dict['defocus'] == 0:
        NM_wcd = cur_dict['wcd']

print(wcd)
print(defocus)

length = len(wcd)
bossung = {"wcd": pd.Series(wcd, index=range(length)), "defocus": pd.Series(defocus, index=range(length))}
df = pd.DataFrame(bossung)
print(df)

# the rc have to set before plotting
sns.set_style("darkgrid")
rc = mpl.rcParams
rc["xtick.direction"] = 'in'
rc["xtick.labelsize"] = 8
rc["ytick.direction"] = 'in'
rc["ytick.labelsize"] = 8
sns.set_context(rc=rc)
#print(rc)

g = sns.lmplot("defocus", "wcd", data=df, order=2,
           ci=None, palette="PuBuGn_d", size=4)
g.set(xticks=defocus , title = "pitch_04A7_K13_1 bossung curve")
ymin, ymax = plt.ylim()
g.set(ylim=(ymin, None))
plt.annotate(r'$-\frac {b}{2a}$',
    xy=(0, (ymax+ymin)/2), xycoords='data',
    xytext=(+10, +30), textcoords='offset points', fontsize=16)
    #arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))

plt.plot([0, 0],[ymin, NM_wcd], color='red', linewidth=1, linestyle="--")
plt.plot([0, 0],[ymin, NM_wcd], color='red', linewidth=1, linestyle="--")  

#plt.tight_layout()
plt.show()

filename = "bossung.png"
directory = os.path.dirname(os.path.abspath(__file__))
savepath = os.path.join(directory, filename)
#plt.rc('savefig', dpi=300)
plt.savefig(savepath,frameon = False )  #plt.savefig(savepath, dpi=my_dpi*2, frameon = False , transparent = True) grid has conflicts with transparent
