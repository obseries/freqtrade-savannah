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
from Savannah import Savannah

logger = logging.getLogger(__name__)


class SavannahLeverage(Savannah):

    # Futures
    custom_leverage = 10.0

