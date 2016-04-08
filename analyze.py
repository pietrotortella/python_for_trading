
import os
import pandas as pd
import numpy as np
import time
import random
import datetime
import matplotlib.pyplot as plt
import scipy.optimize as spo

from sys import argv


def symbol_to_path(symbol, base_dir='C:/Users/Pietro T/Documenti/ML_and_finance/exercises/data'):
    """Returns the file path given the symbol"""
    return os.path.join(base_dir, '{}.csv'.format(str(symbol)))


def plot_data(df, title='Portfolio Value vs Symbol'):
    ax = df.plot(title=title, fontsize=10)
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    
    plt.show()
  
    return ax


def normalize_data(df):
    '''Normalize stock prices to have initial value equal to 1'''
    return df/df.ix[0,:]


def get_data(symbo, dates, col='Adj Close', printerror=True):
    """"For the given sybols, returns the values of cols in the range dates"""
    df = pd.DataFrame(index=dates)
    symbols = symbo.copy()
    if 'SPY' not in symbols:
        symbols.insert(0, 'SPY')
    
    for s in symbols:
        try:
            df_temp = pd.read_csv(symbol_to_path(s), index_col='Date', parse_dates=True, usecols=['Date', col],
                             na_values=['nan'])

            df_temp = df_temp.rename(columns={col:s})
            df = df.join(df_temp)
        except OSError:
            if printerror:
                print('File {}.csv not found, skipping the associated column.'.format(symbo))
        
        if s == 'SPY':
            df = df.dropna(subset=['SPY'])
        
    df.fillna(method='ffill', inplace=True)
    df.fillna(method='bfill', inplace=True)
    
    return df


def read_values(filename):
    df_raw = pd.read_csv(filename, header=-1, skipinitialspace=True)
    dates = []
    vals = []
    
    for i in df_raw.index:
        loc_date = str(int(df_raw.ix[i][0])) + '-' + str(int(df_raw.ix[i][1])) + '-' + str(int(df_raw.ix[i][2]))
        dates += [loc_date]
        vals += [df_raw.ix[i][3]]
    
    dates = pd.to_datetime(dates)
    
    df_good = pd.DataFrame(index=dates, columns=['Value'])
    df_good['Value'] = vals
        
    return df_good


def get_daily_returns(df):
    dr = df.copy()
    dr = (df/ df.shift(1)) - 1
    dr.ix[0,:] = 0
    
    return dr


def get_daily_returns_mean(df):
    dr = get_daily_returns(df)
    return dr.mean()


def get_risk(df):
    dr = get_daily_returns(df)
    return dr.std()


def get_sharp_ratio(df):
    dr = get_daily_returns(df)
    return np.sqrt(252) * dr.mean()/dr.std()



scriptname, values_file, compare_symb = argv
values_file = str(values_file)
compare_symb = str(compare_symb)
base_dir='C:/Users/Pietro T/Documenti/ML_and_finance/exercises/'


values_df = read_values(os.path.join(base_dir, values_file))

try:
    tocompare_df = get_data([compare_symb], values_df.index)[compare_symb]
    total_df = values_df.join(tocompare_df)
    total_df.fillna(method='ffill', inplace=True)
    total_df.fillna(method='bfill', inplace=True)
    
    print('The average daily return of the values is:', get_daily_returns_mean(total_df))
    print('The standard deviation of the values is:', get_risk(total_df))
    print('The Sharp ratio of the values is:', get_sharp_ratio(total_df))
    
    title_plot = 'The values against ' + compare_symb
    plot_data(normalize_data(total_df), title=title_plot)
    
    
except KeyError:
    print('Datas for {} not found, job aborted.'.format(compare_symb))