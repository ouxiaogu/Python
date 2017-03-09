import seaborn as sns
import matplotlib.pyplot as plt
import os

sns.set(style="whitegrid")

titanic = sns.load_dataset("titanic")

g = sns.factorplot("class", "survived", "sex",
                    data=titanic, kind="bar",
                    size=6, palette="muted",
                   legend_out=False)
g.despine(left=True)
g.set_ylabels("survival probability")

plt.show()

filename = "bars.png"
directory = os.path.dirname(os.path.abspath(__file__))
savepath = os.path.join(directory, filename)
#plt.savefig(savepath)
plt.savefig(savepath, transparent=True, bbox_inches='tight', pad_inches=0)