# -*- coding: utf-8 -*-
import numpy as np
import os, os.path
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

x = np.arange(12).reshape((3,4))
#df = pd.DataFrame({'R':x[0],'G':x[1],'B':x[2]})
df = pd.DataFrame(x, index = ['R', 'G', 'B'])
#df = df[["R", "G", "B"]]
print(df)
# the first running vertically downwards, row by row(axis 0) 
# the second running horizontally rightwards, column by column(axis 1)
median_axis0 = np.median(x, axis=0)
median_axis1 = np.median(x, axis=1)
print(median_axis0, median_axis1)