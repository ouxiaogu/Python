import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def sinplot(flip=1):
    x = np.linspace(0, 14, 100)
    for i in range(1, 7):
        plt.plot(x, np.sin(x + i * .5) * (7 - i) * flip)

sinplot()

#sns.set_style("ticks", {'xtick.direction': 'in','ytick.direction': 'in', "xtick.minor.size": 50, "ytick.major.size": 50})
#sns.set_context("paper")
#plt.figure(figsize=(8, 6))
#sinplot()
plt.show()