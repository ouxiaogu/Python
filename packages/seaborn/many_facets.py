import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="ticks")

rs = np.random.RandomState(4)

pos = rs.randint(-1, 2, (20, 5)).cumsum(axis=1)
pos -= pos[:, 0, np.newaxis]
step = np.tile(range(5), 20)
walk = np.repeat(range(20), 5)

df = pd.DataFrame(np.c_[pos.flat, step, walk],
                  columns=["position", "step", "walk"])

grid = sns.FacetGrid(df, col="walk", hue="walk", col_wrap=5, size=1.5)
grid.map(plt.axhline, y=0, ls=":", c=".5")
grid.map(plt.plot, "step", "position", marker="o", ms=4)
grid.set(xticks=np.arange(5), yticks=[-3, 3],
         xlim=(-.5, 4.5), ylim=(-3.5, 3.5))
grid.fig.tight_layout(w_pad=1)

plt.show()