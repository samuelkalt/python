"""
Created a long time ago
@author: joshua.leonard@transmarketgroup.com
"""

import sys
import os
import pytz
import numpy as np
import datetime as dt
import pandas as pd
from pandas.tseries.frequencies import to_offset
import mysql.connector as mdb
import tempfile
import TickP

###############################################################################
# Constants
###############################################################################

SECMASTER_CONNECTION_PARAMS = {'host': 'chd-dev-vsecst01', 
                               'user': 'read', 
                               'passwd': 'only', 
                               'db': 'secmaster'}

DST = (dt.datetime.now(tz=pytz.timezone('America/Chicago')).dst() != dt.timedelta(0))
CENTRAL_TIME_OFFSET = np.array((6 - (1 if DST else 0))*60*60*1e9, dtype=np.longlong)
DEFAULT_EXCHANGES = ['CME', 'ICE', 'TOCOM', 'TOCOM_OLD', 'CFE', 'BVMF', 'BTEC', 'ASX', 'TMX']
DEFAULT_TICKDB_PATH = '/media/tickdb/TickFS/'

###############################################################################
# Secmaster Symbol Info Functions
###############################################################################


def from_sql(query_str):
    rows = []
    cols = None
    try:
        con = mdb.connect(**SECMASTER_CONNECTION_PARAMS)
        cur = con.cursor()
        cur.execute(query_str)
        rows = cur.fetchall()
        cols = [x[0] for x in cur.description]

        if cur and con:
            cur.close()
            con.close()
            del cur
            del con
    except mdb.Error as e:
        print(str('Error {}: {}').format(e.args[0], e.args[1]))
        return None
    return pd.DataFrame(data=list(rows), columns=cols)


def symbol_ids(symbols, with_exchange=True, no_index=False, exchanges=None):
    symbols = [symbols] if type(symbols) not in [list, np.ndarray] else symbols
    exchanges = _set_exchanges(exchanges)
    query = "select distinct symbol, id, endpoint from market_instrument where symbol in " \
            "('{}') and endpoint in ('{}')".format("', '".join(symbols), "', '".join(exchanges))
    out = from_sql(query)
    out = out if no_index else out.set_index('symbol').loc[symbols]
    if len(out.index.unique()) < len(out.index):
        out = out.reset_index().drop_duplicates(subset='symbol', keep='last').set_index('symbol')

    if with_exchange:
        if 'TOCOM_OLD' in out['endpoint'].values:
            out.loc[(out['endpoint'] == 'TOCOM_OLD'), 'endpoint'] = 'TOCOM'
    return out if with_exchange else out[['id']]


def pnl_multipliers(symbols, no_index=False, exchanges=None):
    symbols = [symbols] if type(symbols) not in [list, np.ndarray] else symbols
    exchanges = _set_exchanges(exchanges)
    query = "select distinct symbol, multiplier from market_instrument where symbol in " \
            "('{}') and endpoint in ('{}')".format("', '".join(symbols), "', '".join(exchanges))
    out = from_sql(query)
    out = out if no_index else out.set_index('symbol').loc[symbols]
    # Take last index if index is not unique
    if len(out.index.unique()) < len(out.index):
        out = out.reset_index().drop_duplicates(subset='symbol', keep='last').set_index('symbol')
    return out


def get_active_outrights(product, endpoint, num_outrights=1, effective_date=None, product_is_regex=False):
    if product_is_regex:
        product_sql_regex = product
    else:
        if endpoint == 'CME':
            product_sql_regex = '{}__'.format(product)
        elif endpoint == 'ICE':
            product_sql_regex = '{} ___00__!'.format(product)
        elif endpoint == 'TOCOM':
            product_sql_regex = '1FUT\_{}\_______'.format(product)
        else:
            product_sql_regex = product

    effective_date = (dt.datetime.today() if effective_date is None
                      else pd.to_datetime(effective_date)).strftime('%Y-%m-%d')

    query = ("select m.id, m.symbol, m.endpoint, f.expiration from market_instrument m, futures f "
             "where m.endpoint='{}' and m.type='F' and m.symbol like '{}' and f.instrument_id=m.id and "
             "f.expiration > '{}' order by id limit 100".format(endpoint, product_sql_regex, effective_date))
    out = from_sql(query)
    return out.iloc[:num_outrights][['symbol', 'endpoint', 'id', 'expiration']]


def get_active_spreads(product, endpoint, duration=None, num_spreads=1, effective_date=None, product_is_regex=False):
    if product_is_regex:
        product_sql_regex = product
    else:
        if endpoint == 'CME':
            product_sql_regex = '{0}__-{0}__'.format(product)
        elif endpoint == 'ICE':
            product_sql_regex = '{0} ___00__-{0} ___00__'.format(product)
        elif endpoint == 'TOCOM':
            product_sql_regex = '3FSPR\_{}\_______/______'.format(product)
        else:
            product_sql_regex = product

    effective_date = (dt.datetime.today() if effective_date is None
                      else pd.to_datetime(effective_date)).strftime('%Y-%m-%d')

    inner_query = ("select m.id, m.symbol, m.endpoint, s.expiration, s.leg_instrument_id, s.leg_side, "
                   "s.leg_quantity from market_instrument m, spreads s where m.endpoint='{}' and m.symbol "
                   "like '{}' and s.instrument_id=m.id and s.expiration > '{}' "
                   "order by id".format(endpoint, product_sql_regex, effective_date))

    def _side_query(side):
        _query = ("select t.id, t.symbol, t.endpoint, t.expiration, t.leg_instrument_id, t.leg_side, "
                  "t.leg_quantity, m.symbol as leg_symbol, f.expiration as leg_expiration from ({}) as t, "
                  "market_instrument m, futures f where t.leg_instrument_id=m.id and "
                  "f.instrument_id=m.id and t.leg_side='{}'".format(inner_query, side))
        return _query

    buys_query = _side_query('B')
    sells_query = _side_query('S')
    query = ("select buys.symbol, buys.endpoint, buys.id, "
             "abs((year(sells.leg_expiration)*12 + month(sells.leg_expiration)) - "
             "(year(buys.leg_expiration)*12 +month(buys.leg_expiration))) as duration, "
             "buys.leg_symbol as front_leg_symbol, sells.leg_symbol as back_leg_symbol, "
             "buys.leg_expiration as front_leg_expiration, sells.leg_expiration as back_leg_expiration "
             "from ({}) as buys, ({}) as sells "
             "where buys.id=sells.id order by duration, buys.id limit 1000".format(buys_query, sells_query))
    out = from_sql(query)
    if duration is not None:
        out = out.loc[(out['duration'] == duration)]
    return out.iloc[:num_spreads]


def get_spread_durations(symbols):
    symbol_ids_frame = symbol_ids(symbols)
    endpoints = symbol_ids_frame['endpoint'].unique()
    out = []
    for endpoint in endpoints:
        symbols_by_endpoint = symbol_ids_frame.loc[symbol_ids_frame['endpoint'] == endpoint].index.values.astype('S35')
        symbols_string = "'{}'".format("', '".join(symbols_by_endpoint))

        inner_query = ("select m.id, m.symbol, m.endpoint, s.expiration, s.leg_instrument_id, s.leg_side, "
                       "s.leg_quantity from market_instrument m, spreads s where m.endpoint='{}' and m.symbol "
                       "in ({}) and s.instrument_id=m.id and order by id".format(endpoint, symbols_string))

        def _side_query(side):
            _query = ("select t.id, t.symbol, t.endpoint, t.expiration, t.leg_instrument_id, t.leg_side, "
                      "t.leg_quantity, m.symbol as leg_symbol, f.expiration as leg_expiration from ({}) as t, "
                      "market_instrument m, futures f where t.leg_instrument_id=m.id and "
                      "f.instrument_id=m.id and t.leg_side='{}'".format(inner_query, side))
            return _query

        buys_query = _side_query('B')
        sells_query = _side_query('S')
        query = ("select buys.symbol, buys.endpoint, buys.id, "
                 "abs((year(sells.leg_expiration)*12 + month(sells.leg_expiration)) - "
                 "(year(buys.leg_expiration)*12 +month(buys.leg_expiration))) as duration, "
                 "buys.leg_symbol as front_leg_symbol, sells.leg_symbol as back_leg_symbol, "
                 "buys.leg_expiration as front_leg_expiration, sells.leg_expiration as back_leg_expiration "
                 "from ({}) as buys, ({}) as sells "
                 "where buys.id=sells.id order by duration, buys.id limit 1000".format(buys_query, sells_query))
        query_frame = from_sql(query)
        query_frame = query_frame.set_index('symbol')
        tmp = pd.DataFrame(index=symbols_by_endpoint, columns=['duration'])
        tmp.loc[query_frame.index, 'duration'] = query_frame.loc[:, 'duration']
        tmp.replace(np.nan, 0, inplace=True)
        out.append(tmp)
    out = pd.concat(out)
    return out.loc[symbols]


###############################################################################
# TickP Input Functions
###############################################################################


def symbol_exists_in_tickdb(symbol_id, endpoint, trade_date=None, tickdb_path=DEFAULT_TICKDB_PATH):
    trade_date = (dt.datetime.today() if trade_date is None else pd.to_datetime(trade_date)).strftime('%Y_%m_%d')
    path = '{}/{}/{}/{}.tickdb'.format(tickdb_path, endpoint, trade_date, symbol_id)
    return os.path.exists(path)


def exchange_symbol_id_list(symbols, exchanges=None):
    symbols = [symbols] if type(symbols) not in [list, np.ndarray] else symbols
    d = symbol_ids(symbols, exchanges=exchanges)
    out = ['{}:{}'.format(d.loc[s, 'endpoint'], d.loc[s, 'id']) for s in symbols]
    return out


def _set_exchanges(exchanges_list=None):
    if exchanges_list is None:
        exchanges = DEFAULT_EXCHANGES
    else:
        exchanges = [exchanges_list] if type(exchanges_list) not in [list, np.ndarray] else exchanges_list
        exchanges = [str(x).upper() for x in exchanges]
    return exchanges


def tickp_args(symbols, start_datetime, end_datetime, tickdb_path=DEFAULT_TICKDB_PATH, exchanges=None):
    start_datetime = pd.to_datetime(start_datetime).to_datetime()
    end_datetime = pd.to_datetime(end_datetime).to_datetime()

    startDate = start_datetime.strftime('%Y-%m-%d')
    startTime = start_datetime.strftime('%H:%M')

    endDate = end_datetime.strftime('%Y-%m-%d')
    endTime = end_datetime.strftime('%H:%M')

    symbolArg = exchange_symbol_id_list(symbols, exchanges=exchanges)
    out = (startDate, startTime, endDate, endTime, tickdb_path, symbolArg)
    return out


def tickp_functions(books_type, with_depth):
    out = None
    if books_type.lower() == 'direct':
        if with_depth:
            out = TickP.get_books_with_depth_direct
        else:
            out = TickP.get_inside_books_direct
    elif books_type.lower() == 'implied':
        if with_depth:
            out = TickP.get_books_with_depth_implied
        else:
            out = TickP.get_inside_books_implied
    elif books_type.lower() == 'merged':
        if with_depth:
            out = TickP.get_books_with_depth_merged
        else:
            out = TickP.get_inside_books_merged
    else:
        print(str('Invalid books_type: books_type must be direct, implied, or merged. Exiting.'))
    return out


def tickdb_data_dates(tickdb_path=DEFAULT_TICKDB_PATH, test_exchange='CME'):
    dates = [f.replace('_', '-') for f in os.listdir('{}/{}'.format(tickdb_path, test_exchange))
             if ('20' in f and '_' in f and len(f) == 10)]
    return [min(dates), max(dates)]


###############################################################################
# TickP Output Functions
###############################################################################


def posix_to_datetime_index(timestamps, nanostamps=None, central_time_offset=CENTRAL_TIME_OFFSET):
    timestamps = np.array([timestamps]) if type(timestamps) is not np.ndarray else timestamps
    nanostamps = np.zeros(len(timestamps), module_type='f8') if nanostamps is None else nanostamps
    nanostamps = np.array([nanostamps]) if type(nanostamps) is not np.ndarray else nanostamps
    scalar = (len(timestamps) == 1)

    # Nanosecond arithmetic requires 19 digits of precision. Use 128 bit integers for all nanosecond operations.
    preciseTimestamps = np.zeros((len(timestamps), 2), dtype=np.longlong)
    preciseTimestamps[:, 0] = np.trunc(timestamps)*1e9
    preciseTimestamps[:, 1] = nanostamps
    out = pd.to_datetime(np.sum(preciseTimestamps, 1) - central_time_offset)
    return out[0] if scalar else out


def resample(x, ts, remove_trades=True):
    t = x if type(x) is list else [x]
    out = []
    for i in xrange(len(t)):
        if remove_trades:
            ind = np.isnan(t[i]['trade_price'])
        else:
            ind = np.ones(len(t[i]), dtype='bool')
        xInd = np.searchsorted(t[i][ind]['timestamp'], ts, side='right') - 1
        xInd = np.where(xInd < 0, 0, xInd)
        if len(t[i][ind]) > 0:
            out.append(t[i][ind][xInd])
            out[i]['timestamp'] = ts[...]
        else:
            out.append(np.zeros(0, dtype=t[i].dtype))
    return out


def weighted_mids(x):
    def _weighted_mids(x):
        if len(x[0]['bid_price'].shape) == 0:
            return (x['bid_size']*x['ask_price'] + x['ask_size']*x['bid_price'])/(x['bid_size'] + x['ask_size'])
        else:
            return (x['bid_size'][:, 0]*x['ask_price'][:, 0] +
                    x['ask_size'][:, 0]*x['bid_price'][:, 0])/(x['bid_size'][:, 0] + x['ask_size'][:, 0])
    out = None
    if type(x) is list:
        out = [_weighted_mids(x[i]) for i in xrange(len(x))]
    else:
        out = _weighted_mids(x)
    return out


def simple_mids(x):
    def _simple_mids(x):
        if len(x[0]['bid_price'].shape) == 0:
            return .5*(x['bid_price']+x['ask_price'])
        else:
            return .5*(x['bid_price'][:,0]+x['ask_price'][:,0])
    out = None
    if type(x) is list:
        out = [_simple_mids(x[i]) for i in xrange(len(x))]
    else:
        out = _simple_mids(x)
    return out


def simple_pricing_vectors(size):
    out = np.zeros((size, size))
    out[:, 0] = 1.
    for i in xrange(1, size):
        out[i, 1:(i + 1)] = -1.
    return out


def empty_ticks_array():
    dtype = np.dtype([('timestamp', '<f8'), ('nanostamp', '<i4'), ('instrumentId', '<u8'), ('symbol', 'S32'),
                      ('trade_price', '<f8'), ('trade_size', '<f8'), ('bid_price', '<f8'), ('ask_price', '<f8'),
                      ('bid_size', '<f8'), ('ask_size', '<f8'), ('bid_orders', '<i4'), ('ask_orders', '<i4')])
    return np.zeros(0, dtype=dtype)

###############################################################################
# User-friendly Functions
###############################################################################


def raw_ticks_dataframe(symbols, start_datetime, end_datetime, tickdb_path=DEFAULT_TICKDB_PATH, books_type='direct', exchanges=None):
    ''' Loads dataframe of ticks for all symbols with a datetime index created
        from timestamp/nanostamp fields.
        -----------------------------------------------------------------------
        Argument examples:
            symbols = ['CLG6', 'BRN FMG0016!']
            start_datetime = '2015-12-30 08:00:00'
            end_datetime = '2015-12-30 14:00:00'
            books_type in ['direct', 'implied', 'merged']
    '''
    # This only works for books without depth.
    x = raw_ticks(symbols, start_datetime, end_datetime, tickdb_path=tickdb_path, books_type=books_type, with_depth=False, exchanges=exchanges)
    ts = posix_to_datetime_index(x['timestamp'], x['nanostamp'])
    out = pd.DataFrame.from_records(data=x, index=ts, exclude=['timestamp', 'nanostamp'])
    out['mid'] = 0.5*(out['bid_price'] + out['ask_price'])
    out['wtd_mid'] = (out['bid_price']*out['ask_size'] + out['ask_price']*out['bid_size'])/(out['bid_size'] + out['ask_size'])
    return out


def raw_ticks(symbols, start_datetime, end_datetime, tickdb_path=DEFAULT_TICKDB_PATH, books_type='direct', with_depth=False, exchanges=None):
    ''' Loads array of ticks for all symbols sorted by timestamp/nanostamp.
        -------------------------------------------------------------------
        Argument examples:
            symbols = ['CLG6', 'BRN FMG0016!']
            start_datetime = '2015-12-30 08:00:00'
            end_datetime = '2015-12-30 14:00:00'
            books_type in ['direct', 'implied', 'merged']
            with_depth in [True, False]
    '''
    symbols = [symbols] if type(symbols) not in [list, np.ndarray] else symbols
    fn = tickp_functions(books_type, with_depth)

    # TickP's output contains symbol IDs. Change these back to symbol names later.
    ids = symbol_ids(symbols, with_exchange=True, exchanges=exchanges)

    # Filter symbols that don't appear in tickdb on both start_date and end_date.
    start_date = pd.to_datetime(start_datetime).strftime('%Y-%m-%d')
    end_date = pd.to_datetime(end_datetime).strftime('%Y-%m-%d')
    ids['symbol_exists_on_start_date'] = True
    ids['symbol_exists_on_end_date'] = True
    ids['symbol_exists'] = True
    for i in xrange(len(ids)):
        ids.ix[i, 'symbol_exists_on_start_date'] = symbol_exists_in_tickdb(ids.ix[i, 'id'], ids.ix[i, 'endpoint'],
                                                                           start_date, tickdb_path)
        ids.ix[i, 'symbol_exists_on_end_date'] = symbol_exists_in_tickdb(ids.ix[i, 'id'], ids.ix[i, 'endpoint'],
                                                                         end_date, tickdb_path)
        ids.ix[i, 'symbol_exists'] = (ids.ix[i, 'symbol_exists_on_start_date'] &
                                      ids.ix[i, 'symbol_exists_on_end_date'])
        if not ids.ix[i, 'symbol_exists']:
            print(('Symbol/ID: <{}/{}> was not found in tickdb folder: <{}> '
                   'on days: <{}>. This symbol will be removed from '
                   'calls to TickP.'.format(ids.index[i], ids.ix[i, 'id'],
                                            '{}/{}'.format(tickdb_path, ids.ix[i, 'endpoint']),
                                            ', '.join(np.unique([start_date, end_date])))))
    filtered_symbols = ids.loc[ids['symbol_exists']].index.values.flatten()

    if len(filtered_symbols) == 0:
        return empty_ticks_array()

    args = tickp_args(filtered_symbols, start_datetime, end_datetime, tickdb_path, exchanges=exchanges)
    out = fn(*args)

    symbol_names = np.empty(out.shape, dtype='S27')
    for i in xrange(len(filtered_symbols)):
        mask = (out['symbol'] == str(ids.loc[filtered_symbols[i], 'id']))
        symbol_names[mask] = filtered_symbols[i]
    out['symbol'] = symbol_names
    return out


def resample_ticks(symbols, start_datetime, end_datetime, tickdb_path=DEFAULT_TICKDB_PATH, books_type='direct', frequency='1T', contract_pricing_matrix=None, use_weighted_mids=False, fill_nans=False, central_time_offset=CENTRAL_TIME_OFFSET, exchanges=None):
    ''' Loads dataframe of simple midpoints for all symbols downsampled 
        according to frequency. Downsampling is performed by taking the 
        last book update before each timestamp in the date range from 
        start_datetime to end_datetime with 
        (((end_datetime - start_datetime)/frequency) + 1) evenly-spaced points.
        If contract_pricing_matrix is specified, midpoints will be multiplied 
        by the transpose of the matrix.
        -----------------------------------------------------------------------
        Argument examples:
            symbols = ['CLG6', 'BRN FMG0016!']
            start_datetime = '2015-12-30 08:00:00'
            end_datetime = '2015-12-30 14:00:00'
            books_type in ['direct', 'implied', 'merged']
            frequency codes: H = hours, T = minutes, S = seconds, L = milliseconds, U = microseconds, N = nanoseconds
            frequency = '1T'
            contract_pricing_matrix in [None, 'simple', (numpy array with correct
            dimensions (data contracts x outrights). Mids will be multiplied by the transpose of this matrix.)]
    '''
    symbols = [symbols] if type(symbols) not in [list, np.ndarray] else symbols
    # Pull raw ticks from one frequency period before start_datetime to avoid NaNs.
    earlyStartDatetime = (pd.to_datetime(start_datetime) - to_offset(frequency)).strftime('%Y-%m-%d %H:%M:%S')
    rawTicks = raw_ticks(symbols, earlyStartDatetime, end_datetime, tickdb_path=tickdb_path, books_type=books_type, exchanges=exchanges)
#    separatedTicks = [rawTicks[rawTicks['symbol'] == symbols[i]] for i in xrange(len(symbols))]
    separatedTicks = [rawTicks]
    ts = pd.date_range(start_datetime, end_datetime, freq=frequency)
    # Fine to use normal float division here because downsampling doesn't require high precision.
    floatTs = (ts.values.astype(np.longdouble) + central_time_offset)/1.e9
    aligned = resample(separatedTicks, floatTs)
    mids = np.array(weighted_mids(aligned)) if use_weighted_mids else np.array(simple_mids(aligned))
    columns = symbols
    if contract_pricing_matrix is not None:
        if type(contract_pricing_matrix) is str and contract_pricing_matrix.lower() == 'simple':
            cpm = simple_pricing_vectors(len(symbols))
            columns = [symbols[0]] + [symbols[i].split('-')[1] for i in xrange(1, len(symbols))]
        else:
            cpm = np.array(contract_pricing_matrix, dtype='f8')
        mids = np.dot(mids.T, cpm.T).T
    out = pd.DataFrame(data=mids.T, index=ts, columns=columns)
    if fill_nans:
        out = out.ffill().bfill()
    return out

###############################################################################
# RVModelBacktest Dataset Creation
###############################################################################


def create_ticks_frame(symbols, start_datetime, end_datetime, exchanges=None):
    exchanges = _set_exchanges(exchanges)
    dateFormatStr = '%Y-%m-%d %H:%M:%S'
    startDatetime = pd.to_datetime(start_datetime).to_datetime().strftime(dateFormatStr)
    endDatetime = pd.to_datetime(end_datetime).to_datetime().strftime(dateFormatStr)
    symbolIDs = symbol_ids(symbols, with_exchange=True, exchanges=exchanges)

    out = pd.DataFrame(symbols, columns=['symbol'])
    out['endpoint'] = symbolIDs.loc[symbols, 'endpoint'].values.flatten()
    out['symbol_id'] = symbolIDs.loc[symbols, 'id'].values.flatten()
    out['start_datetime'] = startDatetime
    out['end_datetime'] = endDatetime
    return out


def _pandas_backwards_compatible_resample(data_frame, frequency, how='last', fill_method=None, closed='right', label='right'):
    if pd.__version__ >= '0.18':
        resampler = data_frame.resample(frequency, closed=closed, label=label)
        if how == 'last':
            out = resampler.last()
        elif how == 'sum':
            out = resampler.sum()
        elif how == 'ohlc':
            out = resampler.ohlc()
        elif how == 'count':
            out = resampler.count()
        elif how == 'mean':
            out = resampler.mean()

        if fill_method in ['pad', 'ffill']:
            out = out.ffill()
    else:
        out = data_frame.resample(frequency, how=how, fill_method=fill_method, closed=closed, label=label)
    return out


def resample_ticks_with_depth(uploadFrame, tickdb_path=DEFAULT_TICKDB_PATH, books_type='direct', frequency='1T', tick_data_depth=5, file_path=None, table_name=None, use_weighted_mids=False, fill_nan=True, central_time_offset=CENTRAL_TIME_OFFSET, safe_append=False, exchanges=None):
    cols = ['bid_price', 'bid_size', 'bid_orders', 'ask_price', 'ask_size', 'ask_orders']
    columnNames = []
    for name in cols:
        columnNames += ['{}_{}'.format(name, (i + 1)) for i in xrange(tick_data_depth)]

    out = {}

    # Most of the code below is from Mark Cheng's TickDataPopulator.py file
    for i in xrange(len(uploadFrame)):
        earlyStartDatetime = (pd.to_datetime(uploadFrame.ix[i, 'start_datetime']) - to_offset(frequency)).strftime('%Y-%m-%d %H:%M:%S')
        ts = pd.date_range(uploadFrame.ix[i, 'start_datetime'], uploadFrame.ix[i, 'end_datetime'], freq=frequency)
        # floatTs = (ts.values.astype(np.longdouble) + central_time_offset)/1.e9
        try:
            ticks = raw_ticks(uploadFrame.ix[i, 'symbol'], earlyStartDatetime, uploadFrame.ix[i, 'end_datetime'],
                              tickdb_path=tickdb_path, books_type=books_type, with_depth=True, exchanges=exchanges)

            print('Loaded raw ticks for: <{}:{}> from: <{}> to: <{}>.'.format(uploadFrame.ix[i, 'endpoint'],
                                                                              uploadFrame.ix[i, 'symbol'],
                                                                              uploadFrame.ix[i, 'start_datetime'],
                                                                              uploadFrame.ix[i, 'end_datetime']))

            ticksFrame = pd.DataFrame(data=ticks[['timestamp', 'nanostamp', 'trade_price', 'trade_size']])
            tmp = np.concatenate((ticks['bid_price'][:, :tick_data_depth], ticks['bid_size'][:, :tick_data_depth],
                                  ticks['bid_orders'][:, :tick_data_depth].astype('f8'),
                                  ticks['ask_price'][:, :tick_data_depth], ticks['ask_size'][:, :tick_data_depth],
                                  ticks['ask_orders'][:, :tick_data_depth].astype('f8')), axis=1)
            del ticks

            depthFrame = pd.DataFrame(tmp, columns=columnNames)
            ticksFrame = pd.concat([ticksFrame, depthFrame], axis=1)

            del tmp
            del depthFrame
        except Exception as e:
            print(str('Error: {}').format(str(e)))
            continue
    
        if len(ticksFrame) == 0:
            print(str('<{}> has no data.').format(uploadFrame.ix[i, 'symbol']))
            continue
        
        ticksFrame.ix[:,'timestamp'] = (ticksFrame.ix[:,'timestamp'] - central_time_offset/1.e9).apply(lambda x: np.datetime64(int(x*1000000.0), 'us'))
        #ticksFrame.ix[:,'timestamp'] = posix_to_datetime_index(ticksFrame.ix[:,'timestamp'].values.flatten(), ticksFrame.ix[:,'nanostamp'].values.flatten())
        ticksFrame['timestamp_index'] = ticksFrame['timestamp']
        ticksFrame.set_index('timestamp_index', inplace=True)

        for j in xrange(tick_data_depth - 1):
            ticksFrame.ix[pd.notnull(ticksFrame['bid_price_{0}'.format(j+1)]) & pd.notnull(ticksFrame['bid_price_{0}'.format(j+2)]) & (ticksFrame['bid_price_{0}'.format(j+1)] <= ticksFrame['bid_price_{0}'.format(j+2)]), ['bid_price_{0}'.format(j+1), 'bid_price_{0}'.format(j+2)]] = np.nan
            ticksFrame.ix[pd.notnull(ticksFrame['ask_price_{0}'.format(j+1)]) & pd.notnull(ticksFrame['ask_price_{0}'.format(j+2)]) & (ticksFrame['ask_price_{0}'.format(j+1)] >= ticksFrame['ask_price_{0}'.format(j+2)]), ['ask_price_{0}'.format(j+1), 'ask_price_{0}'.format(j+2)]] = np.nan

        # Filter crossed books
        ticksFrame.ix[ticksFrame['bid_price_1'] >= ticksFrame['ask_price_1'], ['bid_price_1', 'ask_price_1']] = np.nan        
        ticksFrame.ix[(pd.notnull(ticksFrame['ask_price_1']) == False) | (pd.notnull(ticksFrame['bid_price_1']) == False), ['bid_price_1', 'ask_price_1']] = np.nan

        midQuoteFrame = _pandas_backwards_compatible_resample(ticksFrame, frequency, how='last', fill_method='pad', closed='right', label='right')
        ohlcFrame = _pandas_backwards_compatible_resample(ticksFrame['trade_price'], frequency, how='ohlc', closed='right', label='right')

        ohlcFrame = ohlcFrame.rename(columns={'open':'open_price', 'close':'close_price', 'high':'high_price', 'low':'low_price'})
        if use_weighted_mids:
            ohlcFrame.ix[:, 'midquote_price'] = (midQuoteFrame['ask_price_1']*midQuoteFrame['bid_size_1'] + midQuoteFrame['bid_price_1']*midQuoteFrame['ask_size_1']) / (midQuoteFrame['ask_size_1'] + midQuoteFrame['bid_size_1'])
        else:
            ohlcFrame.ix[:, 'midquote_price'] = (midQuoteFrame['ask_price_1'] + midQuoteFrame['bid_price_1']) / 2
        ohlcFrame.ix[:, 'total_volume'] = _pandas_backwards_compatible_resample(ticksFrame['trade_size'], frequency, how='sum', closed='right', label='right')
        ohlcFrame.ix[:, 'average_traded_price'] = _pandas_backwards_compatible_resample(ticksFrame['trade_price'], frequency, how='mean', closed='right', label='right')
        ohlcFrame.ix[:, 'number_of_trades'] = _pandas_backwards_compatible_resample(ticksFrame['trade_size'], frequency, how='count', closed='right', label='right')
        ohlcFrame.ix[:, 'number_of_book_updates'] = _pandas_backwards_compatible_resample(ticksFrame['bid_price_1'], frequency, how='count', closed='right', label='right') - ohlcFrame.ix[:, 'number_of_trades']

        for j in xrange(tick_data_depth):
            idx = j+1
            ohlcFrame.ix[:, 'bid_price_{0}'.format(idx)] = midQuoteFrame['bid_price_{0}'.format(idx)]
            ohlcFrame.ix[:, 'ask_price_{0}'.format(idx)] = midQuoteFrame['ask_price_{0}'.format(idx)]
            ohlcFrame.ix[:, 'bid_size_{0}'.format(idx)] = _pandas_backwards_compatible_resample(ticksFrame['bid_size_{0}'.format(idx)], frequency, how='last', closed='right', label='right')
            ohlcFrame.ix[:, 'ask_size_{0}'.format(idx)] = _pandas_backwards_compatible_resample(ticksFrame['ask_size_{0}'.format(idx)], frequency, how='last', closed='right', label='right')
            ohlcFrame.ix[:, 'bid_orders_{0}'.format(idx)] = _pandas_backwards_compatible_resample(ticksFrame['bid_orders_{0}'.format(idx)], frequency, how='last', closed='right', label='right')
            ohlcFrame.ix[:, 'ask_orders_{0}'.format(idx)] = _pandas_backwards_compatible_resample(ticksFrame['ask_orders_{0}'.format(idx)], frequency, how='last', closed='right', label='right')

        ohlcFrame = ohlcFrame.where(pd.notnull(ohlcFrame), None)
        ohlcFrame = ohlcFrame.iloc[1:, :]

        if fill_nan:
            ohlcFrame = ohlcFrame.ffill().bfill()

        out[uploadFrame.ix[i, 'symbol']] = ohlcFrame.astype('f8')

        del ohlcFrame
        del midQuoteFrame
        del ticksFrame
        delete_cached_tickdb_files()

    if file_path is not None:
        table_name = 'ticks_panel' if table_name is None else str(table_name)
        store = pd.HDFStore(file_path, 'a')
        if safe_append:
            append_ticks_to_hdfstore(store, pd.Panel(out), table_name)
        else:
            store.append(table_name, pd.Panel(out), append=True)
        store.close()
        print(str('Wrote ticks to {} table in: {}').format(table_name, file_path))
    return pd.Panel(out)


def delete_cached_tickdb_files():
    tempDir = tempfile.gettempdir()
    filesToDelete = [f for f in os.listdir(tempDir) if 'tickdb' in f]
    out = True
    for i in xrange(len(filesToDelete)):
        try:
            os.remove(os.path.join(tempDir, filesToDelete[i]))
        except Exception as e:
            print(str('Error: {}'.format(str(e.message))))
            out = False
    return out


def append_ticks_to_hdfstore(store, ticks_panel, table_name):
    if '/{}'.format(table_name.strip('/')) in store.keys():
        mask = (ticks_panel.major_axis > store[table_name].major_axis[-1])
    else:
        mask = np.ones(len(ticks_panel.major_axis), dtype='bool')
    store.append(table_name, ticks_panel.loc[:, mask, :], append=True)
    return
