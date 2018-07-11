import numpy as np

def mse_wt(df, val_col_name, wt_col_name, **kwargs ):
    def f(x, wt):
        return wt*x*x
    df["temp"] = df.apply(lambda x: f(x[val_col_name], x[wt_col_name]), axis = 1)
    value = np.array(df.loc[:, "temp"].values)
    wt = np.array(df.loc[:, wt_col_name].values)+0.
    mse = np.sqrt( sum(value) / sum(wt) )
    print_log = kwargs.get("log", False)
    if(print_log):
        print("{} mse={}".format(val_col_name, mse))
    return mse

    