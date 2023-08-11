"""
Alex Wu's utility functions for data processing, visualization, and reporting

Author: Alexander Wu
Email: alexander.wu7@gmail.com
Date Modified: August 2023
"""
import sys
import warnings
from functools import cache
from pathlib import Path

import pandas as pd


def reload(copy_clipboard=False):
    if copy_clipboard:
        copy(f'%load_ext autoreload\n%autoreload 2')
    print(f'%load_ext autoreload\n%autoreload 2')

def dirr(arg, like=None):
    def get_attr(arg, x):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            try:
                return getattr(arg, x)
            except AttributeError:
                return '!'
    print(type(arg))
    dirr_list = [x for x in dir(arg) if not x.startswith('_')]
    dirr_df = pd.DataFrame({'attr': dirr_list})
    dirr_df['type'] = [type(get_attr(arg, x)) for x in dirr_list]
    if like is not None:
        dirr_df = dirr_df[dirr_df['attr'].str.contains(like)]
    dirr_df['doc'] = [get_attr(arg, attr).__doc__ if str(tt) == "<class 'method'>" else ''
                      for attr, tt in zip(dirr_df['attr'], dirr_df['type'])]
    dirr_df['doc'] = dirr_df['doc'].astype(str).str.split(r'\.\n').str[0].str.strip()
    dirr_df['doc'] = [get_attr(arg, attr) if str(tt) != "<class 'method'>" else doc
                      for attr, tt, doc in zip(dirr_df['attr'], dirr_df['type'], dirr_df['doc'])]
    return dirr_df

def ls(path='.', resolve=False):
    match path:
        case Path():
            pass
        case '~':
            path = Path.home()
        case str():
            path = Path(path)
        case _:
            raise ValueError('invalid path')
    if resolve:
        path = path.resolve()
    df = DF({path: path.iterdir()})
    df.index += 1
    def g(self, row=1):
        return self.loc[row].iloc[0]
    def open(self, row=None):
        import subprocess
        from pathlib import PureWindowsPath
        posix_path = self.loc[row].iloc[0].resolve() if row is not None else PureWindowsPath(path.resolve())
        windows_path = PureWindowsPath(posix_path)
        subprocess.run(['explorer.exe', windows_path])
    df.g = g.__get__(df)
    df.open = open.__get__(df)
    return df

def mkdir(path, **kwargs):
    match path:
        case Path():
            pass
        case str():
            path = Path(path)
        case _:
            raise ValueError('invalid path')
    if 'parents' not in kwargs:
        kwargs['parents'] = True
    if 'exist_ok' not in kwargs:
        kwargs['exist_ok'] = True
    path.mkdir(**kwargs)


def S(*args, **kwargs):
    df = pd.Series(*args, **kwargs).convert_dtypes(dtype_backend='pyarrow')
    return df
def DF(*args, **kwargs):
    df = pd.DataFrame(*args, **kwargs).convert_dtypes(dtype_backend='pyarrow')
    return df

def copy(text):
    # Source: https://stackoverflow.com/questions/11063458/python-script-to-copy-text-to-clipboard
    try:
        import pyperclip # type: ignore
        pyperclip.copy(text)
    except Exception:
        sys.stderr.write("Cannot copy. Try `pip install pyperclip`\n")

def get_sessions(pd_series, diff=pd.Timedelta(30, 'min')):
    """Compute groups (sessions) chained together by `diff` units. Assumes pd_series is sorted."""
    assert pd_series.is_monotonic_increasing

    current_session = pd_series.iloc[0]
    sessions = [current_session]

    for item in pd_series.iloc[1:]:
        if sessions[-1] + diff <= item:
            current_session = item
        sessions.append(current_session)
    return pd.Series(sessions)

def date2name(pd_series):
    """Convert to date to day of week"""
    DAY_NAMES = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    day_name_series = pd.Categorical(pd_series.dt.day_name(), categories=DAY_NAMES)
    return day_name_series

def add_prefix(df, prefix, subset=None, regex=None):
    cols = list(df.columns)
    if regex is not None:
        cols = list(df.columns.str.contains(regex))
    if isinstance(subset, str):
        subset = [subset]
    if hasattr(subset, '__contains__'):
        cols = [col for col in cols if col in subset]
    df_prefix = df.rename(columns={col: f'{prefix}{col}' for col in cols})
    return df_prefix

def add_suffix(df, suffix, subset=None, regex=None):
    cols = list(df.columns)
    if regex is not None:
        cols = list(df.columns.str.contains(regex))
    if isinstance(subset, str):
        subset = [subset]
    if hasattr(subset, '__contains__'):
        cols = [col for col in cols if col in subset]
    df_suffix = df.rename(columns={col: f'{col}{suffix}' for col in cols})
    return df_suffix

def overlaps(x_interval, y_interval):
    # TODO: Optimize O(XY) to O(X + Y) algo
    '''Compute overlaps. Assumes non_overlapping_monotonic_increasing.

    There are 9 ways Y_interval and overlap with X_interval

        |   XXX   | X_interval   | ( 3,   5 ) |                                       |
        |---------|--------------|------------|---------------------------------------|
        | _YY_    | left         | (_2 , _4 ) | (x_begin < y_begin) & (x_end < y_end) |
        | _YYYY   | left_spill   | (_2 ,  5 ) | (x_begin < y_begin) & (x_end = y_end) |
        | _YYYYY_ | superset     | (_2 ,  6_) | (x_begin < y_begin) & (x_end > y_end) |
        |   Y_    | left_subset  | ( 3 , _4 ) | (x_begin = y_begin) & (x_end < y_end) |
        |   YYY   | equal        | ( 3 ,  5 ) | (x_begin = y_begin) & (x_end = y_end) |
        |   YYYY_ | right_spill  | ( 3 ,  6_) | (x_begin = y_begin) & (x_end > y_end) |
        |    _    | subset       | (_4_, _4_) | (x_begin > y_begin) & (x_end < y_end) |
        |    _Y   | right_subset | (_4 ,  5 ) | (x_begin > y_begin) & (x_end = y_end) |
        |    _YY_ | right        | (_4 ,  6_) | (x_begin > y_begin) & (x_end > y_end) |
        |---------|--------------|------------|---------------------------------------|
        | __      | no_overlap   | (_2 ,  6_) | (x_begin > y_end)                     |
        |      __ | no_overlap   | (_2 ,  6_) | (x_end   < y_begin)                   |
        |---------|--------------|------------|---------------------------------------|
        | 1234567 |              |            |                                       |
    '''
    overlaps_list = []
    for _, (x_begin, x_end) in enumerate(x_interval):
        x_overlap_list = []
        for y_i, (y_begin, y_end) in enumerate(y_interval):
            # Case: no_overlap
            #if x_begin > y_end or x_end < y_begin:
            if x_begin >= y_end or x_end <= y_begin:
                continue

            begin_order = '>' if x_begin < y_begin else '<' if x_begin > y_begin else '='
            end_order = '>' if x_end < y_end else '<' if x_end > y_end else '='
            overlap_str = f'{begin_order}{end_order}'
            overlap_tuple = (y_i, overlap_str)
            x_overlap_list.append(overlap_tuple)

        overlaps_list.append(x_overlap_list)
    return overlaps_list


def df_overlaps(df1, df2, suffixes=('1', '2')):
    assert 'start' in df1.columns and 'end' in df1.columns and 'i' not in df1.columns
    assert 'start' in df2.columns and 'end' in df2.columns and 'i' not in df2.columns
    assert df1['start'].is_monotonic_increasing & df1['end'].is_monotonic_increasing
    assert df2['start'].is_monotonic_increasing & df2['end'].is_monotonic_increasing
    assert all(df1['start'] <= df1['end'])
    assert all(df2['start'] <= df2['end'])
    df1 = df1.reset_index(names='i')
    df2 = df2.reset_index(names='i')
    X_interval = list(df2[['start', 'end']].itertuples(index=False, name=None))
    Y_interval = list(df1[['start', 'end']].itertuples(index=False, name=None))
    overlaps_list = overlaps(X_interval, Y_interval)
    index_list = [[y_i for y_i, _ in x_list] for x_list in overlaps_list]
    overlap_list = [[overlap for _, overlap in x_list] for x_list in overlaps_list]
    i1, _ = f'i{suffixes[0]}', f'i{suffixes[1]}'
    overlap_df = (df2.pipe(add_suffix, suffixes[1])
                  .assign(**{i1: index_list, 'overlap': overlap_list})
                  .explode([i1, 'overlap']))
    overlap_df = overlap_df.merge(df1.pipe(add_suffix, suffixes[0]), on=i1, how='outer')
    return overlap_df

def itertuples(df, **kwargs):
    # Roughly same as `.itertuples(index=False, name=None))`
    kwargs['index'] = False
    kwargs['name'] = None
    df_list = list(df.itertuples(**kwargs))
    return df_list

################################################################################
# Utility functions
################################################################################

def size(num, prefix='', deep=True, verbose=True):
    """Human readable file size (ex: 123.4 KB)"""
    x = num
    if not isinstance(x, (int, float)):
        num = len(num)
    if isinstance(x, (str, set, dict, list)):
        return print(f'{num:,}') if verbose else f'{num:,}'
    if isinstance(x, pd.DataFrame):
        x = x.memory_usage(deep=deep).sum()
    if isinstance(x, pd.Series):
        x = x.memory_usage(deep=deep)

    for unit in ('bytes', 'KB', 'MB', 'GB', 'TB'):
        if abs(x) < 1024:
            return print(f'{prefix}: {num:,}  ({x:3.1f}+ {unit})') if verbose else (f'{num:,}  ({x:3.1f}+ {unit})')
        x /= 1024
    print(f'{prefix}: {num:,}  ({x:.1f}+ PB)') if verbose else (f'{num:,}  ({x:.1f}+ PB)')

@cache
def _read_file(filename, base='data', verbose=True, **kwargs):
    match filename:
        case Path():
            base = filename.parent
            filename = filename.name
        case str():
            pass
        case _:
            raise ValueError
    if '.' not in filename:
        filename = f'{filename}.feather'
    P_READ = Path(base) / filename
    assert P_READ.exists()
    if filename.endswith('.feather'):
        df = pd.read_feather(P_READ, **kwargs)
    elif filename.endswith('.parquet'):
        df = pd.read_parquet(P_READ, **kwargs)
    elif filename.endswith('.parquet.gzip'):
        df = pd.read_parquet(P_READ, **kwargs)
    elif filename.endswith('.pkl'):
        df = pd.read_pickle(P_READ, **kwargs)
    elif filename.endswith('.csv'):
        df = pd.read_csv(P_READ, **{'dtype_backend': 'pyarrow', **kwargs})
    else:
        raise ValueError
    if verbose:
        df.pipe(size, prefix=filename)
    return df

# https://stackoverflow.com/questions/56544334/disable-functools-lru-cache-from-inside-function
def read_file(filename, overwrite=False, base='data', verbose=True, **kwargs):
    """Read serialized file, caching the result"""
    if overwrite:
        return _read_file.__wrapped__(filename, base=base, verbose=verbose, **kwargs)
    return _read_file(filename, base=base, verbose=verbose, **kwargs)

def rm_file(filename, base='data', verbose=True):
    match filename:
        case Path():
            base = filename.parent
            filename = filename.name
        case str():
            pass
        case _:
            raise ValueError
    if '.' not in filename:
        filename = f'{filename}.feather'
    P_REMOVE = Path(base) / filename
    if P_REMOVE.exists():
        if verbose:
            size(P_REMOVE.stat().st_size, prefix=f'Deleting "{P_REMOVE}"')
        P_REMOVE.unlink()
    else:
        print(f'"{P_REMOVE}" does not exist...')
    if P_REMOVE.parent.exists() and not any(P_REMOVE.parent.iterdir()):
        print(f'Removing empty directory: "{P_REMOVE.parent}"...')
        P_REMOVE.parent.rmdir()

def read_csv(*args, **kwargs):
    kwargs['dtype_backend'] = 'pyarrow'
    df = pd.read_csv(*args, **kwargs)
    return df

def write_file(df, filename, overwrite=False, base='data', verbose=True, **kwargs):
    """Write serialized file"""
    match filename:
        case Path():
            base = filename.parent
            filename = filename.name
        case str():
            pass
        case _:
            raise ValueError
    if '.' not in filename:
        df = DF(df)
        filename = f'{filename}.feather'
    P_WRITE = Path(base) / filename
    if overwrite or not P_WRITE.exists():
        P_WRITE.parent.mkdir(parents=True, exist_ok=True)
        if filename.endswith('.feather'):
            df.to_feather(P_WRITE, **kwargs)
        elif filename.endswith('.parquet'):
            df.to_parquet(P_WRITE, **kwargs)
        elif filename.endswith('.parquet.gzip'):
            df.to_parquet(P_WRITE, **{'compression': 'gzip', **kwargs})
        elif filename.endswith('.pkl'):
            df.to_pkl(P_WRITE, **kwargs)
        elif filename.endswith('.csv'):
            df.to_csv(P_WRITE, **{'index': False, **kwargs})
        else:
            raise ValueError
        df.pipe(size, prefix='(DataFrame rows)')
    if verbose:
        size(P_WRITE.stat().st_size, prefix=P_WRITE)
