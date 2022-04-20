"""General utilities used by the data utilities modules."""


def copy_data_from_df_to_df(df, new_df, start=None, end=None, ignore_cols=[]):
    """Copy data from one dataframe into another."""
    for i in list(df.columns)[start:end]:
        if i in ignore_cols:
            continue
        new_df[i] = df[i].values[0]
    return new_df


def _filter_df_with_za(df, Z, A):
    if Z is not None:
        df = df[(df["Z"] == Z)]
    if A is not None:
        df = df[(df["A"] == A)]
    return df
