from datetime import datetime, timedelta
import logging
from typing import Optional, Union
import freqtrade.vendor.qtpylib.indicators as qtpylib
import talib.abstract as ta
import pandas_ta as pta
from freqtrade.persistence import Trade
from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame
from freqtrade.strategy import DecimalParameter, IntParameter
from functools import reduce

logger = logging.getLogger(__name__)

def ewo(dataframe, ema_length=5, ema2_length=35):
    #df = dataframe.copy()
    df = dataframe
    ema1 = ta.EMA(df, timeperiod=ema_length)
    ema2 = ta.EMA(df, timeperiod=ema2_length)
    emadif = (ema1 - ema2) / df['low'] * 100
    return emadif


class SavannahLeverage(Savannah):

    # Futures
    custom_leverage = 10.0

