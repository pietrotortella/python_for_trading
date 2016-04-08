
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


def get_symbols_from_txt(filename, base_dir='C:/Users/Pietro T/Documenti/ML_and_finance/exercises/data'):
    """Read a list of symbols from a file to a list"""
    f = open(os.path.join(base_dir, filename))
    sy = f.read().splitlines()
    f.close()
    
    return sy


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


def get_daily_returns(df):
    dr = df.copy()
    dr = (df/ df.shift(1)) - 1
    dr.ix[0,:] = 0
    
    return dr


class Portfolio:
    """Portfolio object. In the amount mode stores the quantities of each stocks"""
    
    def __init__(self, initial_money=0, symbols=[], order_list=[]):
        """initializes with initial money and a list of symbols.
        Order list should be a list of dicts, with key equal to the symbol, with keys 'date', 'symb', 'is_buy', 'amount'.
        """
        
        #create the symbols list, adding 'Cash'
        self.symbols = list(symbols)
        if 'Cash' not in self.symbols:
            self.symbols.insert(0, 'Cash')
        
        #creates the dict where store the participation of the portfolio
        self.amount = dict()
        for s in self.symbols:
            self.amount[s] = 0
        self.amount['Cash'] = initial_money
        
        #initializes with the order lists
        for order in order_list:
            self.execute_order(date=order['date'], symbol=order['symbol'], is_buy=order['is_buy'],
                              amount=order['amount'], p_output=True)            
            
            
    def execute_order(self, date, symbol, is_buy, amount, p_output=True):
        try:
            dates = pd.date_range(date, date)
            asset_value = float(get_data([symbol], dates, printerror=False)[symbol].values)
            old_cash = self.amount['Cash']
            old_asset = self.amount[symbol]
            
            if np.isnan(asset_value):
                print('Symbol {} not trading on day {}. Order aborted.'.format(symbol, date))
            else:            
                if is_buy:
                    if self.amount[symbol] >= 0:
                        self.amount['Cash'] -= amount * asset_value
                    elif self.amount[symbol] <= - amount:
                        self.amount['Cash'] += amount * asset_value
                    else:
                        self.amount['Cash'] -= (amount - 2*abs(self.amount[symbol])) * asset_value
                    self.amount[symbol] += amount

                else:
                    if self.amount[symbol] <= 0:
                        self.amount['Cash'] -= amount * asset_value
                    elif self.amount[symbol] >= amount:
                        self.amount['Cash'] += amount * asset_value
                    else:
                        self.amount['Cash'] += (amount - 2*abs(self.amount[symbol])) * asset_value
                    self.amount[symbol] -= amount

                if self.amount['Cash'] < 0:
                    if p_output == True:
                        print('Not enough liquid money to purchase ', symbol, ', order aborted.')
                    self.amount['Cash'] = old_cash
                    self.amount[symbol] = old_asset
                else:
                    if p_output == True:
                        if is_buy:
                            o_type = 'bought'
                        else:
                            o_type = 'sold'
                        print('Order executed: {} stocks of {} {} for {} each.'.format(amount, symbol, o_type, asset_value))

        except (OSError, KeyError):
            if p_output:
                print('File corresponding to symbol {} not found. Associated order aborted.'.format(symbol))
            else:
                pass
    
     
    def get_portfolio_value(self, dates):
        """Computes the portfolio value on a specific day.
        If df containing the data is provided, it reads from there, otherwise it reads from file.
        Returns the portfolio value, and the boolean some_nan_value, that tells if some of the stock
        had a NaN value."""
        
        u_symb = list(self.symbols)
        for s in self.symbols:
            if self.amount[s] == 0:
                u_symb.remove(s)
        u_symb.remove('Cash')
        
        df_value = pd.DataFrame(index=dates, columns=['Value'])
        df_data = get_data(u_symb, dates)
        
        df_value['Value'] = self.amount['Cash']
        
        for s in u_symb:
            df_value['Value'] += abs(self.amount[s]) * df_data[s] 
            
        df_value = df_value.dropna()
                    
        return df_value
    
    
    def get_daily_returns(self, dates):
        d_value = self.get_portfolio_value(dates)
        
        return get_daily_returns(d_value)    
    
    
    def get_daily_returns_mean(self, dates):
        dr = self.get_daily_returns(dates)
        return dr.mean()
    
    
    def get_risk(self, dates):
        dr = self.get_daily_returns(dates)
        return dr.std()
    
    
    def get_sharp_ratio(self, dates):
        dr = self.get_daily_returns(dates)
        return np.sqrt(252) * dr.mean()/dr.std()
        
    
    def print_exposition(self):
        for s in self.symbols:
            if self.amount[s] != 0:
                print(s, ': ', self.amount[s])
                
                

def get_order_list(filename):
    df_olist = pd.read_csv(filename, header=-1, skipinitialspace=True)
    olist = []
    
    for i in df_olist.index:
        loc_dict = dict()
        loc_dict['date'] = str(df_olist.ix[i][0]) + '-' + str(df_olist.ix[i][1]) + '-' + str(df_olist.ix[i][2])
        loc_dict['symbol'] = str(df_olist.ix[i][3])
        if str(df_olist.ix[i][4]) == 'BUY':
            loc_dict['is_buy'] = True
        elif str(df_olist.ix[i][4]) == 'SELL':
            loc_dict['is_buy'] = False
        else:
            loc_dict['is_buy'] = None
        loc_dict['amount'] = df_olist.ix[i][5]
        
        olist += [loc_dict]
        
    return olist


def save_df_to_file(dataframe, filename):
    file = open(filename, 'w')
    
    for i in range(len(dataframe.index)):
        dateline = str(dataframe.index[i].year) + ', ' + str(dataframe.index[i].month) + ', ' + str(dataframe.index[i].day)
        file.write(dateline)
        for c in dataframe.columns:
            line = ', ' + str(dataframe.ix[i, c])
            file.write(line)
        if i != len(dataframe.index) - 1:
            file.write('\n')
    
    file.close()
    


scriptname, initial_cash, orders_file, output_file = argv
initial_cash = float(initial_cash)
orders_file = str(orders_file)
output_file = str(output_file)
base_dir='C:/Users/Pietro T/Documenti/ML_and_finance/exercises/'

symbols = get_symbols_from_txt('SP500-symbols.txt')

orders_list = get_order_list(os.path.join(base_dir, orders_file))

pfolio = Portfolio(initial_cash, symbols, orders_list)

start_date = '2015-01-01'
end_date = '2015-02-28'
dates = pd.date_range(start_date, end_date)
pf_value = pfolio.get_portfolio_value(dates)

save_df_to_file(pf_value, output_file)