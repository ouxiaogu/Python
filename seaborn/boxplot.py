import seaborn as sns
import matplotlib.pyplot as plt
import os

directory = os.path.dirname(os.path.abspath(__file__))

sns.set(style="ticks")

tips = sns.load_dataset("tips")
days = ["Thur", "Fri", "Sat", "Sun"]
print(tips)
g = sns.factorplot("day", "total_bill", "sex", tips, kind="box",
                   palette="PRGn", aspect=1.25, order=days,  legend_out=False)
g.despine(offset=10, trim=True)
g.set_axis_labels("Day", "Total Bill")

plt.show()
filename = "boxplot.png"
savepath = os.path.join(directory, filename)
#plt.savefig(savepath)
plt.savefig(savepath, transparent=True, bbox_inches='tight', pad_inches=0)