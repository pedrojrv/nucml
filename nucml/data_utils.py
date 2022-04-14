"""General utilities used by the data utilities modules."""


def copy_data_from_df_to_df(df, new_df, start=None, end=None, ignore_cols=[]):
    for i in list(df.columns)[start:end]:
        if i in ignore_cols:
            continue
        new_df[i] = df[i].values[0]
    return new_df
