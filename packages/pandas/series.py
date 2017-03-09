import numpy as np
import pandas as pd


length = 4

def genNaNList(length):
    List = range(length)
    for i in range(length):
        List[i] = np.nan
    return List
    
def genBinList(length, isAnchor):
    List = range(length)
    for i in range(length):
        if isAnchor:
            List[i] = 1
        else:
            List[i] = 0      
    return List

list_a = genNaNList(length)
sr_a = pd.Series(list_a, index=range(length))
list_b = ['A0', 'A1', 'A2', 'A3']
sr_b = pd.Series(list_b, index=range(length))
df = pd.DataFrame({"a":sr_a, "b": sr_b})


dict1 = {"a": list_a, "b": list_b, "c":['B0', 'B1', 'B2', 'B3']}

length = len(list_a)
List1_Cal = genBinList(length, False)
print(List1_Cal)
dict1["Cal"] = List1_Cal

df1  = pd.DataFrame(dict1)
#print(df)
print(df1)

df2 = pd.read_csv("C:\Localdata\D\Note\Python\pandas\NaN.txt", delim_whitespace = True)
columns2 = list(df2.columns.values)

orig_col = list(df1.columns.values)
print(orig_col)
for col in orig_col:
    print(col)
    if col not in columns2:
        length =  df2.shape[0]
        if col == "Cal":
            cur_list = genBinList(length, True)
        else:
            cur_list = genNaNList(length)
    else:
        cur_list = list(df2.loc[:, col].values)
    print(cur_list)
    temp = dict1[col]
    
    dict1[col] =  temp + cur_list
        
    #print(merged_col)
    
print(dict1)
    


