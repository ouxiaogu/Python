Pandas DataFrame Usage in Gauge File

1. IO
2. Indexing / Selecting Data
3. Editing 
4. Merge, join, and concatenate
5. Reshaping, sorting, transposing

Pandas 

The python code in this blog assume these module are imported:

```py
import numpy as np
import os, os.path
import pandas as pd
```

## 1. IO


### 1.0 DataFrame Declaration

1. DataFrame by constructor 

    - `DataFrame([data, index, columns, dtype, copy])` : Two-dimensional size-mutable, potentially heterogeneous tabular data structure with labeled axes (rows and columns).

    ```py
    d = {'col1': ts1, 'col2': ts2}
    df = DataFrame(data=d, index=index)
    df2 = DataFrame(np.random.randn(10, 5))
    df3 = DataFrame(np.random.randn(10, 5), columns=['a', 'b', 'c', 'd', 'e'])
    ```

2. DataFrame by other data structure

    - From dictionary 
    
    ```py
    df_module_stat = pd.DataFrame(module_stat_dict)
    df_module_stat = 
    ```


### 1.1 read 

1. Get the work directory and the data file under a folder

    ```py
    workpath = os.path.dirname(os.path.abspath(__file__))
    gaugedir = workpath+"\gauges"
    gauge_files = []
    gauge_files_subname = os.listdir(gaugedir)
    for subname in gauge_files_subname:
        cur_file = os.path.join(gaugedir, subname)
        gauge_files.append(cur_file) 
    ```

2. Flat file

    - `read_table(filepath_or_buffer[, sep, ...])`    Read general delimited file into DataFrame
    - `read_csv(filepath_or_buffer[, sep, dialect, ...])`   Read CSV (comma-separated) file into DataFrame

    [read_csv](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_csv.html): read comma-delimited text file into DataFrame

    ```py
    df = pd.read_csv(gauge_file, sep = ',', delimiter = '\t', skip )
    ```

    - *delimiter* is Alternative argument name for *sep*


### 1.2 write


1. `DataFrame.to_csv([path_or_buf, sep, na_rep, ...])`   Write DataFrame to a comma-separated values (csv) file
    
    ```py
    df.to_csv(os.path.join(workpath, "02.PS", "gauge_checked_filtered.txt"), na_rep='NaN', sep = '\t', index=False)
    ```
2. `DataFrame.to_dict(*args, **kwargs)`   Convert DataFrame to dictionary.

    ```py
    module_dict = df_deck.set_index("Module").T.to_dict('list') # set_index('Module'): use existed column "Module" as row index; .T: trnaspose; to_dict options str {‘dict’, ‘list’, ‘series’, ‘split’, ‘records’}
    ```

## 2. Indexing / Selecting Data

1. Indexing

    - `DataFrame.ix`    A primarily label-location based indexer, with integer position fallback.
    - `DataFrame.loc`   Purely label-location based indexer for selection by label.
    - `DataFrame.iloc`  Purely integer-location based indexing for selection by position.

    e.g. 

    ```py
    portion1 = df_group_stat.loc["dir_0"].values
    df.ix[cond_gauge, 'base_x']=  df.ix[cond_gauge, 'bas
e_x'] + shift_x
    ```

2. Subset

    There are several API to create a subset of DataFrame.

    - `DataFrame.isin(values)`  Return boolean DataFrame showing whether each element in the DataFrame is contained in values.
    - `DataFrame.where(cond[, other, inplace, ...])`    Return an object of same shape as self and whose corresponding entries are from self where cond is True and otherwise are from other.
    - `DataFrame.mask(cond[, other, inplace, axis, ...])`   Return an object of same shape as self and whose corresponding entries are from self where cond is False and otherwise are from other.
    - `DataFrame.query(expr, **kwargs)` Query the columns of a frame with a boolean expression.
    - `DataFrame.head([n])`   Returns first n rows

    Here list some very common scenario of gauge file subsets:

    1. Multiple condition: math inequation:
    
    `df = df[(df["ILS Result (1/um)"] >= 22) & (df["AI_CD"] >= 40) & (df["Model CD"] >= 45) ]`

    2. Multiple condition: `np.logical_not`,  `df.isin`: 

    ```py
    filter_module = ["BWALL__CH_Trench_chop", "BWALL__Calibration_chop"]
    MASK = np.logical_not( (df["ModuleId"].isin(filter_module)) & (df["pre_num"] == '0') & (df["tail_num"] == '2') & (df["plot_CD"] == df["draw_CD"])  )
    df = df.ix[ MASK ]
    ```

    ```py
    module = "BWALL__CH_AR__Row_chop"
    group0_mask = (df.ModuleId == module) & (df.SubGroup ==5)
    mask = ( group0_mask & (abs(df.CDxy_Ratio - 0.5)<0.001) )
    ```

    ```py
    module_list = ['BWall_01', 'BWALL__CH_AR__CHE', 'CH_SR_01', 'CH_DR_01', 'CHE_STAG_01', 'BWALL__CH_AR__Row']
    group0_mask = (df.ModuleId == module) & (df.SubGroup == 0)
    group5_mask = (df.ModuleId == module) & (df.SubGroup == 5)
    masks[module] = ( group0_mask & (abs(df.CDxy_Ratio -2./3.) < 0.0001) & (df.CDx == 66) & (df.plot_CD.isin(pitch_list1)) ) | ( group5_mask & (abs(df.CDxy_Ratio - 1.5) < 0.0001) & (df.CDx == 46) & (df.plot_CD.isin(pitch_list2)) ) 
    ```

3. DataFrame column, index

    - The list of column name

    ```py
    print( df.columns.values.tolist() )
    ```

    - uniq list of a specific column 
    
    ```py
    df_module = df.drop_duplicates(cols='ModuleId', take_last=True)
    module_uniq = df_module.loc[:, "ModuleId"].values.tolist()
    ```

    - complete list of a specific column

    ```py
    gauges = df.gauge.values.tolist()
    x_bls = df.x_bl.values.tolist()
    ```

    - set specific column as index
    
    ```py
    df = df.set_index(['gauge'])
    ```



## 3. Editing

1. Add a new column by formula of other columns:

    - simple arithmetic
    
    ```py
    df_module_stat["percentage"] =  df_module_stat['c
ount'].apply(lambda x: (x+0.0)/count_sum)
    df["pre_num"] =  df['gauge'].apply(lambda x: x[len(x)-8:len(x)-7])
    ```

    - construct a function

    ```py
    def f(c1, c2, c3):
        return (8*c1 + 4*c2 + c3)
    df["SubGroup"] = df.apply(lambda x: f(x['dummy_dir'], x['EPSId'], x['MPId']), axis=1)
    ```

2. change value of selected column to:

    - To a constant
    
    ```py
    df["CAL"] = 0
    module_list = ["Block_SAQP_01_NoOPC", "Cut_01_NoOPC", "Cut_SAQP_01_NoOPC"]
    df.ix[df.ModuleId.isin(module_list), 'CAL'] = 1
    ```

    - arithmetic from current column

    ```py
    df.ix[cond_gauge, 'base_x']=  df.ix[cond_gauge, 'base_x'] + shift_x
    ```

    - formula from other column

    ```py
    def f_xy(x1, x2, flag, isX):
       if ((flag ==  0) and (isX == False)) or ((flag ==  1) and (isX == True)):
           return x1
       elif((flag ==  1) and (isX == False)) or ((flag ==  0) and (isX == True)):
           return x2       
    mask = ((df.ModuleId == cur_module)&(df.SubGroup == int(sub_group)))
    df['CDx'] = np.nan
    df.ix[mask, 'CDx'] = df[mask].apply(lambda x: f_xy(x['p2'], x['p3'], flag, isX = True), axis=1)
    ```

3. change column name

    ```py
    df_result = df_result.rename(columns={'group': 'GroupName'})
    df_result.rename(columns={'group': 'GroupName'}, inplace=True)
    ```
    
## 4. Merge, join, and concatenate

1. `merge`: Use one column or index as mapping key:

    `merge(left, right[, how, on, left_on, right_on ...])`  Merge DataFrame objects by performing a database-style join operation by columns or indexes.

    This method is simple and convenient:

    ```py
    df_result = merge(df_gauge, df_lmc, left_on = 'gauge', right_on = 'gauge', how = 'outer')
    ```
    - how : {‘left’, ‘right’, ‘outer’, ‘inner’}, default ‘inner’
        - left: use only keys from left frame (SQL: left outer join)
        - right: use only keys from right frame (SQL: right outer join)
        - outer: use union of keys from both frames (SQL: full outer join)
        - inner: use intersection of keys from both frames (SQL: inner join)

    To achieve equivalent outcome by `concatenate`, we need:

    ```py
    df_gauge = df_gauge.set_index("gauge")
    df_lmc = df_lmc.set_index("gauge")
    df_result = pd.concat([df_gauge, df_lmc], axis=1)
    df_result.reset_index(inplace = True)
    ```

2. `concat`: Concatenate mulitple pandas objects.

    - `concat(objs[, axis, join, join_axes, ...])`  Concatenate pandas objects along a particular axis with optional set logic along the other axes.


3. The method of DataFrame

    - DataFrame.append(other[, ignore_index, ...])    Append rows of other to the end of this frame, returning a new object.
    - DataFrame.assign(**kwargs)  Assign new columns to a DataFrame, returning a new object (a copy) with all the original columns in addition to the new ones.
    - DataFrame.join(other[, on, how, lsuffix, ...])  Join columns with other DataFrame either on index or on a key column.
    - DataFrame.merge(right[, how, on, left_on, ...]) Merge DataFrame objects by performing a database-style join operation by columns or indexes.
    - DataFrame.update(other[, join, overwrite, ...]) Modify DataFrame in place using non-NA values from passed DataFrame.


## 5. Reshaping, sorting, transposing



## Appendix I

1. The axis in DataFrame

    Axis in DataFrame is inherited from [numpy](http://docs.scipy.org/doc/numpy/glossary.html). 

    This code piece can get the meaning of axis in DataFrame:

    ```py
    x = np.arange(12).reshape((3,4))
    #df = pd.DataFrame({'R':x[0],'G':x[1],'B':x[2]})
    #df = df[["R", "G", "B"]]
    df = pd.DataFrame(x, index = ['R', 'G', 'B'])
    print(df)
    median_axis0 = np.median(x, axis=0)
    median_axis1 = np.median(x, axis=1)
    print(median_axis0, median_axis1)
    ```

    - **axis 0**: the first running vertically downwards, row by row
    - **axis 1**the second running horizontally rightwards, column by column


