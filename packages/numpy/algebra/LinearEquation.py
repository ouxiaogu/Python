import numpy as np
import pandas as pd


def Cal_NM_BF(a, b):
    coeff = np.linalg.solve(a, b)
    #print(coeff)
    #b_fit = np.dot(a, coeff)
    #print(b_fit)
    NM_BF = -coeff[1]/coeff[0]
    print(NM_BF)
    #print(-33*coeff[0]+coeff[1])

df = pd.read_csv("C:/Localdata/D/Note/Python/numpy/algebra/defocus_points.txt", delim_whitespace = True)
print(df)
for index, row in df.iterrows():
    #print(row["NM_DF1"]) 
    a = [[row['NM_DF1'], 1], [row['NM_DF2'], 1]]
    b = [row['AVG_BF_DIFF1'], row['AVG_BF_DIFF2']]
    a = np.array(a)
    b = np.array(b)
    Cal_NM_BF(a, b)