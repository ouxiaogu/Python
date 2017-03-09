import random
import pandas as pd
import numpy as np

y = np.random.rand(2,5)
df = pd.DataFrame(y, columns = ["y1", "y2", "y3", "y4", "y5"])
print(df)
def f_polyfit(y1, y2, y3, y4, y5, degree):
    y = [y1, y2, y3, y4, y5]
    x = [1, 2, 3, 4, 5]
    coeffs = np.polyfit(x, y, degree)   
    coeffs = coeffs.tolist()
    # constructe the polynomial formula
    p = np.poly1d(coeffs)
    # fit values, and mean
    y_fit = p(x)                        
    y_avg = np.sum(y)/len(y)          
    ssreg = np.sum((y_fit-y_avg)**2)   
    sstot = np.sum((y - y_avg)**2)  
    R2 = ssreg / sstot
    return coeffs[0], R2
# df["slope"], df["R2"] = zip(df.apply(lambda x:f_polyfit(["y1"], ["y2"], ["y3"], ["y4"], ["y5"], degree = 1),  axis = 1))

g = lambda x: pd.Series(f_polyfit(x.y1, x.y2, x.y3, x.y5, x.y5, degree=1))
df[['slope', 'R2']] = df.apply(g, axis=1)
print(df)
