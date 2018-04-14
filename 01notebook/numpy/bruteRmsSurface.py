import pandas as pd
#from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D
import numpy as np
from scipy.interpolate import griddata
import sys
sys.path.append(r'C:\Users\peyang\Perforce\peyang_LT324319_3720\app\mxp\scripts\util')
from FileUtil import gpfs2WinPath

INFILE = r'rms.txt'
INFILE2 = r'rms.txt'
INFILE = gpfs2WinPath(INFILE)
INFILE2 = gpfs2WinPath(INFILE2)

if __name__ == '__main__':
    fig = plt.figure()
    ax = Axes3D(fig)

    # Plot the surface.
#    surf = ax.contour(X, Y, Z, cmap=plt.cm.YlGnBu_r, extend3d=True)

    _, _, x, y, z = np.loadtxt(INFILE, unpack=True, skiprows=1)
    _, _, x2, y2, z2 = np.loadtxt(INFILE2, unpack=True, skiprows=1)

    # If necessary you can pass vmin and vmax to define the colorbar range,
    # cmap can Greens, YlGnBu_r
    surf = ax.plot_trisurf(x, y, z, cmap=plt.cm.YlGnBu_r, linewidth=0.1, vmin=0, vmax=40)
    # Add a color bar which maps values to colors.

    ind = np.unravel_index(np.argmin(z, axis=None), z.shape)
    print("z min rms {} at ({}, {})".format(z[ind], x[ind], y[ind]) )

    ax.text2D(x[ind], y[ind], "minZ:{}".format(z[ind]), transform=ax.transAxes)
    ax.clabel(surf, fontsize=9, inline=1)
    fig.colorbar(surf, shrink=0.5, aspect=5)

    '''plot GN path
    '''
    ax.plot(x2, y2, z2, color='b')
    for i, zz in enumerate(z2):
        # zz = 10*zz
        # ax.scatter(x2[i], y2[i], zz, color='b')
        ax.text(x2[i], y2[i], zz,  '%s' % (str(i+1)), size=10, zorder=1,
        color='k')
    ind = np.unravel_index(np.argmin(z2, axis=None), z2.shape)
    print("z2 min rms {} at ({}, {})".format(z2[ind], x2[ind], y2[ind]) )

    ax.set_xlabel('xshift')
    ax.set_ylabel('yshift')
    ax.set_zlabel('rms')
    plt.show()