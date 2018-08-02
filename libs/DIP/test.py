# -*- coding: utf-8 -*-
"""
Created on Thu Aug  2 11:11:23 2018

@author: peyang
"""

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    sns.set_style("whitegrid")
    x = np.arange(100 )
    y = np.sin(np.pi*x/10)
    fig = plt.figure()
    plt.plot(x, y, '-*')
    