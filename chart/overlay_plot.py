import numpy as np
import matplotlib.pyplot as plt

x1 = np.arange(10)
y1 = x1**2
x2 = np.arange(100,200)
y2 = x2

fig = plt.figure()

ax1 = fig.add_axes([0.1, 0.1, 0.8, 0.8], label="ax1")
ax2 = fig.add_axes([0.1, 0.1, 0.8, 0.8], label="ax2", frameon=False)

ax1.yaxis.tick_left()
ax1.xaxis.tick_bottom()


ax2.yaxis.tick_right()
ax2.yaxis.set_label_position('right')
ax2.yaxis.set_offset_position('right')
ax2.xaxis.tick_top()
ax2.xaxis.set_label_position('top')

ax1.spines['right'].set_color('red')
ax1.spines['top'].set_color('red')

for ylabel, xlabel in zip(ax2.get_yticklabels(), ax2.get_xticklabels()):
     ylabel.set_color("red")
     xlabel.set_color("red")


ax1.plot(x1,y1)
ax2.plot(x2,y2, 'r')

plt.show()