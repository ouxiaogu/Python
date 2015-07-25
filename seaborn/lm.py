import seaborn as sns
import matplotlib.pyplot as plt
import os, os.path
import pandas as pd

sns.set(style="ticks", context="talk")
workpath = os.path.dirname(os.path.abspath(__file__))
filepath = os.path.join(workpath, "data",  "tips.txt" )

tips = pd.read_csv(filepath, sep = '\t')
print(tips)
days = ["Thur", "Fri", "Sat", "Sun"]
#pal = sns.cubehelix_palette(4, 1.5, .75, light=.6, dark=.2)
#g = sns.lmplot("total_bill", "tip", hue="day", data=tips,
#                palette=pal, size=6) #hue_order=days,
#g.set_axis_labels("Total bill ($)", "Tip ($)")

colors = ['#F0A3FF', '#0075DC', '#993F00', '#4C005C', '#191919', '#005C31', '#2BCE48', '#FFCC99', '#808080', '#94FFB5', '#8F7C00', '#9DCC00', '#C20088', '#003380', '#FFA405', '#FFA8BB', '#426600', '#FF0010', '#5EF1F2', '#00998F', '#E0FF66', '#740AFF', '#990000', '#FFFF00', '#FF5005']
#colors_r =  reversed(colors)
colors_r = colors[::-1]
pal = sns.palplot(colors_r)
plt.show()