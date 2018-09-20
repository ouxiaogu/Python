Pandas DataFrame Usage in Gauge File

1. IO
2. Indexing / Selecting Data
3. Editing 
4. Merge, join, and concatenate

Pandas 

The python code in this blog assume these module are imported:

```py
import numpy as np
import os, os.path
import pandas as pd
```

## 1. IO


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
    - `read_fwf(filepath_or_buffer[, colspecs, widths])`    Read a table of fixed-width formatted lines into DataFrame

    [read_csv](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_csv.html): read tab-delimited text file into DataFrame

    ```py
    df = pd.read_csv(gauge_file, sep = ',', delimiter = '\t', skip )
    ```

    - *delimiter* is Alternative argument name for *sep*

3. Excel

    - `read_excel(io[, sheetname])` Read an Excel table into a pandas DataFrame
    - `ExcelFile.parse([sheetname, header, ...])`   Read an Excel table into DataFrame

4. SQL

    - `read_sql_table(table_name, con[, schema, ...])`  Read SQL database table into a DataFrame.
    - `read_sql_query(sql, con[, index_col, ...])`  Read SQL query into a DataFrame.
    - `read_sql(sql, con[, index_col, ...])`    Read SQL query or database table into a DataFrame.

### 1.2 write


## series

1. how to get True values of the Boolean Series

```shell
flt = (gg.ptnid==2)

flt.loc[flt>0]
Out[46]: 
812    True
dtype: bool

flt.loc[flt>0].index
Out[47]: Int64Index([812], dtype='int64')
```





