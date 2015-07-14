import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#plt.rcParams
rc={'xtick.labelsize': 16, 'ytick.labelsize': 16, 'legend.fontsize': 16, 'axes.titlesize': 16}
#rc={'font.size': 16}
#plt.rcParams.update(**rc)

sns.set(rc=rc)


plt.plot([2,1,3], label='one')
plt.title('test')
plt.legend()
plt.show() 