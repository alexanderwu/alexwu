"""
Alex Wu's utility functions for data processing, visualization, and reporting

Author: Alexander Wu
Email: alexander.wu7@gmail.com
Date Modified: August 2023
"""
import inspect
import itertools
import time
import warnings
from functools import reduce, wraps
from typing import List

import numpy as np
import pandas as pd
import scipy.stats as st
from IPython import get_ipython
from IPython.display import HTML, Markdown, display


def wrap_series(fn):
    """Allows Pandas series operations to apply for other input"""
    def wrapper(series, *args):
        not_series = False
        if not isinstance(series, pd.Series):
            not_series = True
            series = pd.Series(series)
        res = fn(series, *args)
        if not_series:
            res = res.iloc[0]
        return res
    return wrapper

def decorator(func):
    @wraps(func)
    def wrapper_decorator(*args, **kwargs):
        # Do something before
        value = func(*args, **kwargs)
        # Do something after
        return value
    return wrapper_decorator

def ignore_warnings(func):
    @wraps(func)
    def wrapper_decorator(*args, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            value = func(*args, **kwargs)
            return value
    return wrapper_decorator

def debug(fn):
    def wrapper(*args):
        t1 = time.time()
        result = fn(*args)
        t2 = time.time()
        print(f'{fn.__name__}{args} : {result} ({t2-t1:.1f} s)')
        return result
    return wrapper

################################################################################
# (For Jupyter)
################################################################################

# Source: https://stackoverflow.com/questions/15411967/how-can-i-check-if-code-is-executed-in-the-ipython-notebook/39662359#39662359
def isnotebook() -> bool:
    """Detect if code is running in Jupyter Notebook"""
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreter

def disp(df, caption='', k=2, na_rep='-'):
    """
    (For Jupyter)
    Prints newlines instead of '\n' characters for easier reading.
    Optionally, you can label dataframes with caption and round numbers
    """
    assert isnotebook()
    # Ensure row names and column names are unique
    df = df.pipe(df_enumerate)
    df = df.style if hasattr(df, 'style') else df
    df_captioned = (df.format(lambda x: str_round(x, k=k),
                              na_rep=na_rep,
                              subset=df.data.select_dtypes(exclude=object).columns)
                      .set_properties(**{'white-space': 'pre-wrap', 'text-align': 'left'})
                      .set_table_attributes("style='display:inline'")
                      .set_caption(caption))
    return df_captioned

# Derived from: https://stackoverflow.com/a/57832026
def displays(*args, captions: List[str] = None, k=2, na_rep='-'):
    """
    (For Jupyter)
    Display tables side by side to save vertical space.
    Prints newlines instead of '\n' characters for easier reading.
    Optionally, you can label dataframes with captions

    Input:
        args: list of pandas.DataFrame
        captions: list of table captions
    """
    assert isnotebook()
    if captions is None:
        captions = []
    if isinstance(captions, str):
        captions = [captions]
    if k is None:
        k = []

    args = (*args, pd.DataFrame())
    args = [arg.to_frame().style.hide_index() if isinstance(arg, pd.Series) else arg for arg in args]
    k_list = [k]*len(args) if isinstance(k, int) else k
    k_list.extend([None] * (len(args) - len(k_list)))
    captions.extend([''] * (len(args) - len(captions)))
    captioned_tables = [df.pipe(disp, caption, k, na_rep)._repr_html_()
                        for caption, df, k in zip(captions, args, k_list)]
    display(HTML('\xa0\xa0\xa0'.join(captioned_tables)))

# Source: https://docs.python.org/3/library/itertools.html#itertools-recipes
def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)

def display100(df, I=10, N=100, na_rep=None):
    if isinstance(df, pd.Series):
        df = df.to_frame()
    N = min(N, len(df))
    displays(*[df.iloc[a:b] for a, b in pairwise(range(0, N+I, I))], na_rep=na_rep)
disp100 = display100
d100 = display100

def display_code(code: str, language: str = 'python'):
    markdown_code = f'```{language}\n{code}\n```'
    if isnotebook():
        display(Markdown(markdown_code))
    else:
        print(markdown_code)

# https://stackoverflow.com/questions/624926/how-do-i-detect-whether-a-python-variable-is-a-function
def show(item, hide_docstring: bool = False):
    """(For Jupyter) Displays function source code or JSON output"""
    if callable(item):
        code = inspect.getsource(item)
        if hide_docstring:
            function_text = [code.split('"""')[0], *code.split('"""')[2:]]
            code = ''.join([x.rstrip() for x in function_text])
        display_code(code)
    elif isnotebook():
        if isinstance(item, dict):
            try:
                import plotly.io as pio
                pio.show(item, 'json', False)
            except:
                display_code(item, 'json')
        elif isinstance(item, str):
            display(Markdown(item))
        else:
            return type(item)
    else:
        return type(item)

def percent(pd_series, caption='', display_false=False):
    df = pd.value_counts(pd_series).to_frame().T
    if True not in df:
        df[True] = 0
    if False not in df:
        df[False] = 0
    df['Total'] = len(pd_series)
    df['%'] = 100*df[True] / df['Total']
    if not display_false:
        df = df.rename(columns={True: 'N'})
        df = df.drop(columns=[False])
    styled_df = (df.style.hide()
            .bar(vmin=0, vmax=100, color='#543b66', subset=['%'])
            .format('{:,.1f}', subset=['%']))
    if caption:
        styled_df = styled_df.set_caption(caption)
    return styled_df
perc = percent

def append_percent(df, col=None, vmax=None, verbose=True, inplace=False):
    if not inplace:
        df = df.copy()
    if col is None:
        col = df.shape[1] - 1

    if isinstance(col, int):
        pd_series = df.iloc[:,col]
        col_i = col + 1
    else:
        pd_series = df[col]
        col_i = df.columns.to_list().index(col) + 1

    pd_series = pd.to_numeric(pd_series, errors='coerce')
    if vmax is None:
        vmax = pd_series.sum()

    percent_series = 100 * pd_series / vmax
    df.insert(col_i, '%', percent_series)

    if verbose:
        # print(f'Total: {vmax:,}')
        styled_df = (df.style
            .bar(vmin=0, vmax=100, color='#543b66', subset=['%'])
            .format('{:,.1f}', subset=['%'])
        )
        return styled_df
    if not inplace:
        return df
append_perc = append_percent

def vcounts(pd_series, cutoff=20, vmax=None, sort_index=False, verbose=True, **kwargs):
    data = pd_series.value_counts(**kwargs)
    if sort_index:
        data = data.sort_index()
    if vmax is None:
        vmax = data.sum()
    if len(data) > cutoff:
        other = pd.Series([data[cutoff:].sum()], index=['(Other)'])
        if isinstance(pd_series, pd.DataFrame):
            # other.index = pd.MultiIndex.from_tuples([('(Other)',) * data.index.nlevels])
            other_index = (*['-']*(data.index.nlevels-1), '(Other)')
            other.index = pd.MultiIndex.from_tuples([other_index])
        data = pd.concat([data[:cutoff], other])

    data_df = data.reset_index()
    if isinstance(pd_series, pd.Series):
        data_df.rename(columns={'index': pd_series.name}, inplace=True)
    data_df.columns = [*data_df.columns[:-1], 'N']
    data_df.index += 1
    value_counts_df = data_df.pipe(append_percent, vmax=vmax, verbose=verbose)
    return value_counts_df

def describe(pd_series, caption='', count=False):
    df = pd_series.describe().to_frame().T
    df['IQR'] = df['75%'] - df['25%']
    num_missing = pd_series.isna().sum()
    if num_missing > 0:
        df['N/A'] = num_missing
    if not count:
        df = df.drop(columns=['count'])
    if caption:
        styled_df = df.style.hide().set_caption(caption).format(
            lambda x: int(x) if x == int(x) else round(x, 4))
        display(styled_df)
    else:
        return df

def uniq(x_list):
    unique_list = list(set([x for x in x_list if pd.notnull(x)]))
    return unique_list

def duplicates(df, subset=None, keep=False):
    duplicates_df = df[df.duplicated(subset=subset, keep=keep)]
    return duplicates_df

def df_index(df, verbose=False, k=False):
    if hasattr(df, 'data'):
        df = df.data
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)
    index_name = tuple(['-']*df.columns.nlevels) if df.columns.nlevels > 1 else '-'
    index_df = df.iloc[:,0].map(lambda x: '').to_frame(name=index_name)
    if verbose:
        displays(index_df, df.style.hide(), k=k)
    else:
        return index_df

def df_enumerate(df, rows=None, columns=None, inplace=False):
    enumerate_rows = (not df.index.is_unique or rows) and rows != False
    enumerate_columns = (not df.columns.is_unique or columns) and columns != False
    if not inplace and (enumerate_rows or enumerate_columns):
        df = df.copy()
    if enumerate_rows:
        df.index = pd.MultiIndex.from_tuples([(i, *x) if isinstance(x, tuple) else (i,x) for i, x in enumerate(df.index, 1)])
        # df = df.set_index(pd.MultiIndex.from_tuples([(i, *x) if isinstance(x, tuple) else (i,x) for i, x in enumerate(df.columns, 1)]))
    if enumerate_columns:
        df.columns = pd.MultiIndex.from_tuples([(i, *x) if isinstance(x, tuple) else (i,x) for i, x in enumerate(df.columns, 1)])
    if not inplace:
        return df

def combo_sizes(set_list, set_names=None, vmax=None, sort=True):
    """Summarary table of set combinations sizes"""
    if vmax is None:
        vmax = len(reduce(lambda x, y: x | y, set_list))
    if set_names is None:
        set_names = list(range(len(set_list)))
    combo_list = [()]
    sizes_list = [vmax]
    for k in range(1, len(set_list)+1):
        for indices_combo in itertools.combinations(enumerate(set_list), k):
            indices, combo = zip(*indices_combo)
            size = len(reduce(lambda x, y: x & y, combo))
            sizes_list.append(size)
            combo_list.append(indices)
    combo_df = pd.DataFrame([['Yes' if i in i_list else '-' for i in range(len(set_names))]
                             for i_list in combo_list], columns=set_names)
    combo_df['Size'] = sizes_list
    combo_df['%'] = 100*combo_df['Size'] / vmax
    if sort:
        combo_df = combo_df.sort_values('Size', ascending=False)
    combo_df.index += 1

    def highlight(s):
        return ['background-color: green' if v else '' for v in s == 'Yes']
    combo_df_styled = (combo_df
            .style.apply(highlight)
            .bar(color='#543b66', vmin=0, vmax=100, subset=['%'])
            .format(precision=1))
    return combo_df_styled

def combo_sizes2(set_list, set_names=None, vmax=None, sort=True):
    """Summarary table of set combinations sizes"""
    if vmax is None:
        vmax = len(reduce(lambda x, y: x | y, set_list))
    if set_names is None:
        set_names = list(range(len(set_list)))
    combo_list = [()]
    sizes_list = [vmax]

    for k in range(1, len(set_list)+1):
        for indices, combo, other_combo in zip(
            itertools.combinations(range(len(set_list)), k),
            itertools.combinations(set_list, k),
            list(itertools.combinations(set_list, len(set_list) - k))[::-1]
        ):
            row_vals = reduce(lambda x, y: x & y, combo)
            if other_combo:
                row_vals = row_vals - reduce(lambda x, y: x | y, other_combo)
            size = len(row_vals)
            sizes_list.append(size)
            combo_list.append(indices)

    combo_df = pd.DataFrame([
            # First row is union of all values
            ['-']*len(set_names),
            # All other rows proceed normally as combinations of 'Yes', 'No'
            *[['Yes' if i in i_list else 'No' for i in range(len(set_names))]
                for i_list in combo_list[1:]]
        ], columns=set_names)
    combo_df['Size'] = sizes_list
    combo_df['%'] = 100*combo_df['Size'] / vmax
    if sort:
        combo_df = combo_df.sort_values('Size', ascending=False)
    combo_df.index += 1

    def highlight(s):
        return ['background-color: green' if v == 'Yes' else 'background-color: darkred' if v == 'No' else '' for v in s]
    combo_df_styled = (combo_df
            .style.apply(highlight)
            .bar(color='#543b66', vmin=0, vmax=100, subset=['%'])
            .format(precision=1))
    return combo_df_styled

# def highlight(v, color='DarkSlateGray'):
#     """Example usage: df.style.applymap(aw.highlight(1))"""
#     f = lambda x: 'background-color: DarkSlateGray' if x == v else ''
#     return f

def highlight(df, v, color='DarkSlateGray', subset=None):
    """Example usage: df.pipe(highlight, 1, color='k')"""
    if hasattr(df, 'style'):
        df = df.style
    if color == 'w': color = 'white'
    if color == 'k': color = 'black'
    return df.applymap(lambda x: f'background-color: {color}' if x == v else '', subset=subset)

def str_contains(pd_series, *regex_str_list, **kwargs):
    '''
    Filters Pandas Series strings using regex patterns from `regex_str_list`

    Parameters
    ----------
    pat : str
        Character sequence or regular expression.
    case : bool, default True
        If True, case sensitive.
    flags : int, default 0 (no flags)
        Flags to pass through to the re module, e.g. re.IGNORECASE.
    na : scalar, optional
        Fill value for missing values. The default depends on dtype of the
        array. For object-dtype, ``numpy.nan`` is used. For ``StringDtype``,
        ``pandas.NA`` is used.
    regex : bool, default True
        If True, assumes the pat is a regular expression.

        If False, treats the pat as a literal string.
    '''
    if 'case' not in kwargs:
        kwargs['case'] = False

    match pd_series:
        case pd.Series():
            pass
        case str():
            pd_series = pd.Series([pd_series])
        case _:
            raise ValueError

    mask_list = [pd_series.str.contains(x, **kwargs) for x in regex_str_list]
    pd_series_masked = pd_series[reduce(lambda x,y: x|y, mask_list)]
    return pd_series_masked

################################################################################
# Statistical
################################################################################

# https://stackoverflow.com/questions/26102867/python-weighted-median-algorithm-with-pandas
def median(df, val, weight=None):
    if weight is None:
        return df[val].median()
    df_sorted = df.sort_values(val)
    cumsum = df_sorted[weight].cumsum()
    cutoff = df_sorted[weight].sum() / 2.
    return df_sorted[cumsum >= cutoff][val].iloc[0]

def chi2_table(table_df, prob=0.99, verbose=True):
    if not isinstance(table_df, pd.DataFrame):
        table_df = pd.DataFrame(table_df)
    stat, p_val, dof, expected = st.chi2_contingency(table_df)
    if verbose:
        expected_df = pd.DataFrame(expected, index=table_df.index, columns=table_df.columns)
        displays(
            table_df,
            expected_df,
            table_df - expected_df,
            captions=['Table', 'Expected', 'Difference']
        )
        # interpret test-statistic
        critical = st.chi2.ppf(prob, dof)
        print(f'dof={dof}, probability={prob:.3f}, critical={critical:.3f}, stat={stat:.3f}')
        if abs(stat) >= critical:
            print('Dependent (reject H0)')
        else:
            print('Independent (fail to reject H0)')
    return p_val

def chi2_pair(treatment, control):
    assert isinstance(treatment, (list, tuple)) and len(treatment) == 2
    assert isinstance(control, (list, tuple)) and len(control) == 2
    treatment_n, treatment_size = treatment
    control_n, control_size = control
    treatment_vs_control_df = pd.DataFrame({
        'Table 1': [treatment_n, treatment_size - treatment_n],
        'Table 2': [control_n, control_size - control_n],
    })
    p_val = chi2_table(treatment_vs_control_df, verbose=False)
    return p_val

@ignore_warnings
def p_value(pd_series1, pd_series2, verbose=False, mww=False):
    if isinstance(pd_series1, pd.DataFrame):
        assert all(pd_series1.columns == pd_series2.columns)
        pval_list = [p_value(pd_series1[col], pd_series2[col], mww=mww) for col in pd_series1.columns]
        pval_df = pd.DataFrame(pval_list, index=pd_series1.columns, columns=['p-val'])
        return pval_df
    try:
        if isinstance(pd_series1, (list, tuple)) and len(pd_series1) == 2:
            p_val = chi2_pair(pd_series1, pd_series2)
            return p_val
        if pd_series1.dtype.name == 'bool':
            pd_series1 = pd_series1.astype('category')
            pd_series2 = pd_series2.astype('category')
        if pd_series1.dtype.name == 'category':
            table_df = pd.concat([
                pd_series1.value_counts(sort=False, dropna=False).rename('Table 1'),
                pd_series2.value_counts(sort=False, dropna=False).rename('Table 2'),
            ], axis=1).fillna(0)
            # Remove rows with all zeros
            table_df = table_df.loc[(table_df != 0).any(axis=1)]
            p_val = chi2_table(table_df, verbose=verbose)
            return p_val
        if pd_series1.dtype.name == 'datetime64[ns]':
            pd_series1 = pd_series1.astype(int)
            pd_series2 = pd_series2.astype(int)
        if mww:
            statistic, pvalue = st.mannwhitneyu(pd_series1.dropna(), pd_series2.dropna())
        else:
            statistic, pvalue = st.ttest_ind(pd_series1.dropna(), pd_series2.dropna(), equal_var=False)
        if verbose:
            print(f'stat={statistic:.3f}')
            # print(f'dof={dof}, probability={prob:.3f}, critical={critical:.3f}, stat={statistic:.3f}')
    except ValueError:
        return np.nan
    return pvalue

# Inspiration: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.trim_mean.html
def trim(pd_series, proportiontocut=0.05, conservative=True):
    from math import ceil
    if pd_series.size == 0:
        return pd_series

    nobs = pd_series.shape[0]
    lowercut = int(proportiontocut * nobs) if conservative else ceil(proportiontocut * nobs)
    uppercut = nobs - lowercut
    if lowercut > uppercut:
        raise ValueError("Proportion too big.")

    atmp = np.partition(pd_series, (lowercut, uppercut - 1))
    return pd.Series(atmp[lowercut:uppercut])

# Source: https://stackoverflow.com/questions/15033511/compute-a-confidence-interval-from-sample-data
def mean_confidence_interval(data, confidence=0.95):
    a = np.array(data, dtype=float)
    a = a[~np.isnan(a)]
    n = len(a)
    m, se = np.mean(a), st.sem(a)
    h = se * st.t.ppf((1 + confidence) / 2, n-1)
    return m, m-h, m+h

# Derived from: https://cran.r-project.org/web/packages/stddiff/stddiff.pdf
def stddiff_categorical(treatment, control):
    T = pd.value_counts(treatment, sort=False, normalize=True)[1:].values
    C = pd.value_counts(control, sort=False, normalize=True)[1:].values
    assert len(T) == len(C)
    K_1 = len(T)
    if K_1 == 1:
        S = np.array([[(( T[0]*(1-T[0]) + C[0]*(1-C[0]) ) / 2)]])
    else:
        S = np.zeros([K_1, K_1])
        for k, l in itertools.product(range(K_1), range(K_1)):
            if k == l:
                S[k, l] = ( T[k]*(1-T[k]) + C[k]*(1-C[k]) ) / 2
            else:
                S[k, l] = -( T[k]*T[l] + C[k]*C[l] ) / 2

    try:
        smd = np.sqrt( (T-C).T @ np.linalg.inv(S) @ (T-C) )
    except np.linalg.LinAlgError:
        smd = np.nan
    return smd

def stddiff_numerical(treatment, control):
    if treatment.dtype.name == 'datetime64[ns]' and control.dtype.name == 'datetime64[ns]':
        treatment = treatment.astype(int)
        control = control.astype(int)
    numer = treatment.mean() - control.mean()
    denom = np.sqrt((treatment.var() + control.var()) / 2)
    with np.errstate(divide='ignore', invalid='ignore'):
        smd = numer / denom
    return smd

def stddiff_pair(treatment, control):
    assert isinstance(treatment, (list, tuple)) and len(treatment) == 2
    assert isinstance(control, (list, tuple)) and len(control) == 2
    treatment_n, treatment_size = treatment
    control_n, control_size = control
    treatment_array = np.array([[1]*treatment_n + [0]*(treatment_size - treatment_n)])
    control_array = np.array([[1]*control_n + [0]*(control_size - control_n)])
    smd = stddiff_numerical(treatment_array, control_array)
    return smd

@ignore_warnings
def stddiff(treatment, control, ci=False, k=3):
    if isinstance(treatment, pd.DataFrame):
        assert all(treatment.columns == control.columns)
        smd_list = [stddiff(treatment[col], control[col], ci=ci) for col in treatment.columns]
        smd_df_colnames = ['SMD', 'lower', 'upper'] if ci else ['SMD']
        smd_df = pd.DataFrame(smd_list, index=treatment.columns, columns=smd_df_colnames)
        return smd_df
    if isinstance(treatment, (list, tuple)) and len(treatment) == 2:
        smd = stddiff_pair(treatment, control)
    elif treatment.dtype.name == 'category':
        smd = stddiff_categorical(treatment, control)
    else:
        smd = stddiff_numerical(treatment, control)
    if k:
        smd = round(smd, k)

    if ci:
        n1, n2 = len(treatment), len(control)
        sigma = np.sqrt( (n1+n2)/(n1*n2) + (smd**2)/(2*(n1+n2)) )
        smd_l = smd - 1.96*sigma
        smd_u = smd + 1.96*sigma
        if k:
            smd_l, smd_u = round(smd_l, k), round(smd_u, k)
        return smd, smd_l, smd_u
    else:
        return smd

@ignore_warnings
def compare_dfs(df1, df2, cols=None, each=True, fillna=False, style=True):
    rows = {}
    if cols is None:
        assert isinstance(df1, pd.Series) and isinstance(df2, pd.Series)
        if df1.name == df2.name:
            cols = df1.name
        else:
            cols = f'{df1.name} vs {df2.name}'
        df1 = df1.to_frame(name=cols)
        df2 = df2.to_frame(name=cols)
    if isinstance(cols, str):
        cols = [cols]
    for col in cols:
        if df1[col].dtype.name == 'bool':
            rows[(col, '')] = ['\n', '']
            rows[(col, 'Yes')] = [stddiff(df1[col], df2[col]), str(p_value(df1[col], df2[col]))]
            rows[(col, 'No')] = ['\n', '']
        elif df1[col].dtype.name == 'category':
            # rows[(col, '')] = ['\n', '']
            rows[(col, '')] = [stddiff(df1[col], df2[col]), str(p_value(df1[col], df2[col]))]
            if each:
                rows.update({(col, k): [stddiff(df1[col] == k, df2[col] == k), str(p_value(df1[col] == k, df2[col] == k))] for k in df1[col].cat.categories})
                rows[(col, '(nan)')] = [stddiff(df1[col].isna(), df2[col].isna()), str(p_value(df1[col].isna(), df2[col].isna()))]
            else:
                rows.update({(col, k): ['\n', ''] for k in df1[col].cat.categories})
                rows[(col, '-')] = [stddiff(df1[col], df2[col]), str(p_value(df1[col], df2[col]))]
        else:
            rows[(col, ' (N)')] = [stddiff((sum(~np.isnan(df1[col])), df1[col].shape[0]), (sum(~np.isnan(df2[col])), df2[col].shape[0])),
                                  str(p_value((sum(~np.isnan(df1[col])), df1[col].shape[0]), (sum(~np.isnan(df2[col])), df2[col].shape[0])))]
            if fillna:
                df1[col] = df1[col].fillna(0)
                df2[col] = df2[col].fillna(0)
            rows[(col, ' (mean, SD, 95% CI)')] = [stddiff(df1[col], df2[col]), str(p_value(df1[col], df2[col]))]
            rows[(col, ' (median, [Q1, Q3], [min, max])')] = [' ', str(p_value(df1[col], df2[col], mww=True))]
    res = pd.DataFrame(
        rows,
        index=['SMD', 'p-val']
    ).T
    if style:
        def highlight_smd(s):
            s = pd.to_numeric(s, errors='coerce')
            return ['background-color: darkslateblue' if v else ''
                    for v in (s <= -0.1) | (s > 0.1)]
        def highlight_pval(s):
            return ['background-color: brown' if v else ''
                   for v in s.astype(float) < 0.05]
        return res.style.apply(highlight_smd, subset=['SMD']).apply(highlight_pval, subset=['p-val'])
    return res

################################################################################
# Reporting results
################################################################################

def str_round(x, k=None):
    if k == False:
        return str(x)
    try:
        x = float(x)
        if x == int(x):
            return f'{int(x):,}'
        elif k is None:
            return f'{x:,}'
        else:
            return f'{round(x, k):,}'
    except OverflowError:
        return str(x)
    except ValueError:
        return str(x)
    except TypeError:
        return str(x)

def bracket_str(q1, q3, k=2, is_date=False):
    if is_date:
        # if not isinstance(k, str):
        #     k = '1s'
        # q1q3_str = f'[{q1.round(k)}, {q3.round(k)}]'
        q1q3_str = f'[{q1.date()}, {q3.date()}]'
    else:
        q1q3_str = f'[{str_round(q1, k)}, {str_round(q3, k)}]'
    return q1q3_str

def q1q3(pd_series, k=2, is_date=False):
    try:
        q1, q3 = pd_series.describe()[['25%', '75%']].values
    except KeyError:
        q1, q3 = pd_series.astype(float).describe()[['25%', '75%']].values
    q1q3_str = bracket_str(q1, q3, k, is_date)
    return q1q3_str

def report_categorical(pd_series, dropna=True, style=True):
    if pd_series.dtype.name == 'bool':
        pd_series = pd.Categorical(pd_series, categories=[True, False]).rename_categories({True: 'Yes', False: 'No'})
    vcounts = pd.value_counts(pd_series, sort=False, dropna=dropna)
    vcount_dict = dict(zip(vcounts.index.to_list(), [[x, '', ''] for x in vcounts]))
    if not dropna:
        vcount_dict[np.nan] = vcount_dict.get(np.nan, [0, '', ''])
        vcount_dict['(N/A)'] = vcount_dict.pop(np.nan)

    if style:
        vsum = vcounts.sum()
        vcount_dict = {k: [v[0], f'({100*v[0]/vsum:.1f}%)', f'{100*v[0]/vsum:.1f}%']
                    for k, v in vcount_dict.items()}
    return vcount_dict

def report_numerical(pd_series, name='', k=2, proportiontocut=0, fillna=False, N=True):
    if proportiontocut > 0:
        pd_series = trim(pd_series, proportiontocut)
    report_numerical_dict = {}
    if N:
        report_numerical_dict[f'{name} (N)'] = [pd_series.notna().sum(), f'({100*pd_series.notna().mean():.1f}%)', f'{100*pd_series.notna().mean():.1f}%']
    if fillna:
        pd_series = pd_series.fillna(0)
    if isinstance(pd_series, pd.Series) and pd_series.dtype == 'datetime64[ns]':
        report_numerical_dict[f'{name} (mean, SD)'] = [pd_series.mean().date(), f'{pd_series.std().days} days', '-']
        report_numerical_dict[f'{name} (median, [Q1, Q3], [min, max])'] = [pd_series.median().date(), q1q3(pd_series, is_date=True), bracket_str(pd_series.min(), pd_series.max(), is_date=True)]
    else:
        mean, ci_left, ci_right = mean_confidence_interval(pd_series)
        report_numerical_dict[f'{name} (mean, SD, 95% CI)'] = [str_round(mean, k), str_round(pd_series.std(), k), bracket_str(ci_left, ci_right, k=k)]
        report_numerical_dict[f'{name} (median, [Q1, Q3], [min, max])'] = [str_round(pd_series.median(), k), q1q3(pd_series, k=k), bracket_str(pd_series.min(), pd_series.max(), k=k)]
    # report_numerical_dict = {
    #     f'{name} (mean, SD, 95% CI)': [mean, pd_series.std(), bracket_str(ci_left, ci_right, k=k)],
    #     f'{name} (median, [Q1, Q3], [min, max])': [pd_series.median(), q1q3(pd_series, k=k), bracket_str(pd_series.min(), pd_series.max(), k=k)],
    # }
    return report_numerical_dict

def report_rows(df, cols=None, dropna=False, k=2, proportiontocut=0, fillna=False, style=True):
    rows = {}
    if isinstance(df, list):
        rows_list = [report_rows(d) for d in df]
        return {k: v for x in rows_list for k, v in x.items()}
    if isinstance(df, pd.Series):
        df = df.to_frame(name='')
    if cols is None:
        cols = list(df.columns)
    for col in cols:
        if col.startswith('-'):
            rows[('-', col[1:])] = ['\n', '', '']
        elif df[col].dtype.name == 'object':
            rows[(col, f'(N, uniq)')] = [df[col].notna().sum(), df[col].nunique(), '']
        elif df[col].dtype.name in ('bool', 'category'):
            rows[(col, f'{col} N (%)')] = ['\n', '', '']
            rows.update({(col, k): v for k, v in report_categorical(df[col], dropna=dropna, style=style).items()})
        else:
            # rows[(col, '-')] = ['\n', '', '']
            rows.update({(col, k): v for k, v in report_numerical(df[col], k=k, proportiontocut=proportiontocut, fillna=fillna).items()})
    return rows

def report_rows_df(df, cols=None, dropna=False, k=2, proportiontocut=0, fillna=False, style=True):
    INDEX_COLS = ['N', '%/SD/IQR', '95% CI/Range']
    res = pd.DataFrame(
        report_rows(df, cols, dropna=dropna, k=k, proportiontocut=proportiontocut, fillna=fillna, style=style),
        index=INDEX_COLS
    ).T
    if style:
        def bar_percent(x, color='#543b66'):
            if str(x).endswith('%'):
                x = float(x[:-1])
                return f'background: linear-gradient(90deg, {color} {x}%, transparent {x}%); width: 10em; color: rgba(0,0,0,0);'
        return res.style.applymap(bar_percent, color='steelblue', subset=['95% CI/Range'])
    return res

# Source: https://docs.python.org/3/library/itertools.html#itertools-recipes
def grouper(iterable, n, fillvalue=None):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx"
    args = [iter(iterable)] * n
    return itertools.zip_longest(*args, fillvalue=fillvalue)

@ignore_warnings
def report(func, *df_args, compare=False, concat=False, k_list=2, fillna=False, **kwargs):
    if isinstance(func, list):
        assert len(df_args) > 0
        cols = func
        func = lambda df: report_rows_df(df, cols=cols, fillna=fillna, **kwargs)
    if not callable(func):
        df_args = [func, *df_args]
        cols = None
        func = lambda df: report_rows_df(df, cols=cols, fillna=fillna, **kwargs)

    if compare and len(df_args) % 2 == 1:
        df_args = (*df_args[:-1], df_args[0], df_args[-1])
    if compare:
        if cols is None:
            cols = list(df_args[0].columns)
        compared_dfs = [(func(df1), func(df2), compare_dfs(df2, df1, cols, fillna=fillna)) for df1, df2 in grouper(df_args, 2)]
        result_dfs = list(itertools.chain(*compared_dfs))
    else:
        result_dfs = [func(df) for df in df_args]

    if k_list is None:
        k_list = [None] * len(result_dfs)
    if isinstance(k_list, int):
        k_list = [k_list] * len(result_dfs)
    k_list = [None, *k_list]

    if concat:
        result_dfs = [df.data if hasattr(df, 'data') else df for df in result_dfs]
        # result_dfs = [df.applymap(lambda x: str_round(x, k=k)) for df, k in zip(result_dfs, k_list[1:])]
        ## Ensure only one new-line per empty row
        result_dfs = [result_dfs[0]] + [df.replace('\n', '') for df in result_dfs[1:]]
        result_dfs = pd.concat(result_dfs, axis=1).pipe(df_enumerate)
        result_dfs.pipe(df_index, verbose=True, k=k_list)
    else:
        index_df = result_dfs[0].pipe(df_index)
        displays(index_df, *[df.style.hide() if hasattr(df, 'style') else df.hide()
                             for df in result_dfs], k=k_list)
