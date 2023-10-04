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


class Savannah(IStrategy):
    minimal_roi = {
        "0": 10
    }

    timeframe = '5m'

    # TODO verificare se impostarlo a False, sembra che faccia le operazioni solo ogni 5 minuti
    #process_only_new_candles = True
    process_only_new_candles = False
    
    startup_candle_count = 20

    order_types = {
        'entry': 'market',
        'exit': 'market',
        'emergency_exit': 'market',
        'force_entry': 'market',
        'force_exit': "market",
        'stoploss': 'market',
        'stoploss_on_exchange': False,

        'stoploss_on_exchange_interval': 60,
        'stoploss_on_exchange_market_ratio': 0.99
    }

    # Trailing stoploss (not used)
    trailing_stop = False
    trailing_stop_positive = 0
    trailing_stop_positive_offset = 0.0
    trailing_only_offset_is_reached = False
  
    max_open_trades = 3
    
    # Disabled
    stoploss = -1.0
    
    # Max Trade Duration (da rivedere in base alla leva)
    max_trade_duration = 300
    
    # Futures
    custom_leverage = 1.0

    # DCA
    position_adjustment_enable = True
    max_entry = 2
    first_entry_ratio = 0.65
    
    # Custom stoploss
    use_custom_stoploss = True

    is_optimize_ewo = True
    buy_rsi_fast = IntParameter(35, 50, default=42, space='buy', optimize=is_optimize_ewo)
    buy_rsi = IntParameter(15, 35, default=35, space='buy', optimize=is_optimize_ewo)
    buy_ewo = DecimalParameter(-6.0, 5, default=-5.836, space='buy', optimize=is_optimize_ewo)
    buy_ema_low = DecimalParameter(0.9, 0.99, default=0.956, space='buy', optimize=is_optimize_ewo)
    buy_ema_high = DecimalParameter(0.95, 1.2, default=1.043, space='buy', optimize=is_optimize_ewo)

    is_optimize_32 = True
    buy_rsi_fast_32 = IntParameter(20, 70, default=40, space='buy', optimize=is_optimize_32)
    buy_rsi_32 = IntParameter(15, 50, default=29, space='buy', optimize=is_optimize_32)
    buy_sma15_32 = DecimalParameter(0.900, 1, default=0.975, decimals=3, space='buy', optimize=is_optimize_32)
    buy_cti_32 = DecimalParameter(-1, 0, default=-0.55, decimals=2, space='buy', optimize=is_optimize_32)

    is_optimize_deadfish = True
    sell_deadfish_bb_width = DecimalParameter(0.03, 0.75, default=0.359, space='sell', optimize=is_optimize_deadfish)
    sell_deadfish_profit = DecimalParameter(-0.15, -0.05, default=-0.05, space='sell', optimize=is_optimize_deadfish)
    sell_deadfish_bb_factor = DecimalParameter(0.90, 1.20, default=0.928, space='sell', optimize=is_optimize_deadfish)
    sell_deadfish_volume_factor = DecimalParameter(1, 2.5, default=2.45, space='sell', optimize=is_optimize_deadfish)

    sell_fastx = IntParameter(50, 100, default=64, space='sell', optimize=True)

    plot_config = {
        'main_plot': {
            'EWO': {},
            'ema_8': {'color': 'red'},
            'ema_16': {'color': 'white'},
            'sma_15': {'color': 'yellow'},
        },
        'subplots': {
            "RSI": {
                'rsi': {'color': 'yellow'},
                'rsi_fast': {'color': 'red'},
                'rsi_slow': {'color': 'blue'},
            }
        }
    }

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        # buy_1 indicators
        dataframe['sma_15'] = ta.SMA(dataframe, timeperiod=15)
        dataframe['cti'] = pta.cti(dataframe["close"], length=20)
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        dataframe['rsi_fast'] = ta.RSI(dataframe, timeperiod=4)
        dataframe['rsi_slow'] = ta.RSI(dataframe, timeperiod=20)

        # ewo indicators
        dataframe['ema_8'] = ta.EMA(dataframe, timeperiod=8)
        dataframe['ema_16'] = ta.EMA(dataframe, timeperiod=16)
        dataframe['EWO'] = ewo(dataframe, 50, 200)

        # profit sell indicators
        stoch_fast = ta.STOCHF(dataframe, 5, 3, 0, 3, 0)
        dataframe['fastd'] = stoch_fast['fastd']
        dataframe['fastk'] = stoch_fast['fastk']

        # loss sell indicators
        bollinger2 = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        dataframe['bb_lowerband2'] = bollinger2['lower']
        dataframe['bb_middleband2'] = bollinger2['mid']
        dataframe['bb_upperband2'] = bollinger2['upper']

        dataframe['bb_width'] = (
                (dataframe['bb_upperband2'] - dataframe['bb_lowerband2']) / dataframe['bb_middleband2'])

        dataframe['volume_mean_12'] = dataframe['volume'].rolling(12).mean().shift(1)
        dataframe['volume_mean_24'] = dataframe['volume'].rolling(24).mean().shift(1)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        conditions = []
        dataframe.loc[:, 'enter_tag'] = ''

        is_ewo = (
                (dataframe['rsi_fast'] < self.buy_rsi_fast.value) &
                (dataframe['close'] < dataframe['ema_8'] * self.buy_ema_low.value) &
                (dataframe['EWO'] > self.buy_ewo.value) &
                (dataframe['close'] < dataframe['ema_16'] * self.buy_ema_high.value) &
                (dataframe['rsi'] < self.buy_rsi.value)
        )

        buy_1 = (
                (dataframe['rsi_slow'] < dataframe['rsi_slow'].shift(1)) &
                (dataframe['rsi_fast'] < self.buy_rsi_fast_32.value) &
                (dataframe['rsi'] > self.buy_rsi_32.value) &
                (dataframe['close'] < dataframe['sma_15'] * self.buy_sma15_32.value) &
                (dataframe['cti'] < self.buy_cti_32.value)
        )

        conditions.append(is_ewo)
        dataframe.loc[is_ewo, 'enter_tag'] += 'ewo'

        conditions.append(buy_1)
        dataframe.loc[buy_1, 'enter_tag'] += 'buy_1'

        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x | y, conditions),
                'enter_long'] = 1

        return dataframe

    def custom_stoploss(self, pair: str, trade: Trade, current_time: datetime, current_rate: float,
                        current_profit: float, **kwargs) -> float:

        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        current_candle = dataframe.iloc[-1].squeeze()

        enter_tag = ''
        if hasattr(trade, 'enter_tag') and trade.enter_tag is not None:
            enter_tag = trade.enter_tag
        enter_tags = enter_tag.split()

        if "ewo" in enter_tags:
            if current_profit >= 0.05 * self.custom_leverage:
                return -0.005 * self.custom_leverage

        if current_profit > 0:
            if current_candle["fastk"] > self.sell_fastx.value:
                return -0.001 * self.custom_leverage

            if current_candle["rsi"] > 80:
                return -0.001 * self.custom_leverage

        if current_profit < 0:
            if current_candle["rsi"] > 90:
                return -0.001 * self.custom_leverage

        return self.stoploss * self.custom_leverage

    def custom_exit(self, pair: str, trade: Trade, current_time: datetime, current_rate: float,
                    current_profit: float, **kwargs) -> Optional[Union[str, bool]]:

        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        current_candle = dataframe.iloc[-1].squeeze()

        # stoploss - deadfish
        if ((current_profit < self.sell_deadfish_profit.value)
                and (current_candle['bb_width'] < self.sell_deadfish_bb_width.value)
                and (current_candle['close'] > current_candle['bb_middleband2'] * self.sell_deadfish_bb_factor.value)
                and (current_candle['volume_mean_12'] < current_candle[
                    'volume_mean_24'] * self.sell_deadfish_volume_factor.value)):
            logger.info(f"{pair} sell_stoploss_deadfish at {current_profit*100}")
            return "sell_stoploss_deadfish"
        
        # trade expired
        trade_duration = (current_time - trade.open_date_utc).seconds / 60

        if trade_duration > self.max_trade_duration:
            logger.info(f"{pair} trade_expired at {current_profit*100}")
            return "trade_expired"
        
        #TODO liquidation protection
        



    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe.loc[(), ['exit_long', 'exit_tag']] = (0, 'long_out')

        return dataframe

    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                 proposed_leverage: float, max_leverage: float, entry_tag: Optional[str], side: str,
                 **kwargs) -> float:
        """
        Customize leverage for each new trade. This method is only called in futures mode.

        :param pair: Pair that's currently analyzed
        :param current_time: datetime object, containing the current datetime
        :param current_rate: Rate, calculated based on pricing settings in exit_pricing.
        :param proposed_leverage: A leverage proposed by the bot.
        :param max_leverage: Max leverage allowed on this pair
        :param entry_tag: Optional entry_tag (buy_tag) if provided with the buy signal.
        :param side: 'long' or 'short' - indicating the direction of the proposed trade
        :return: A leverage amount, which is between 1.0 and max_leverage.
        """
        return self.custom_leverage

    # Let unlimited stakes leave funds open for DCA orders
    def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float,
                            proposed_stake: float, min_stake: Optional[float], max_stake: float,
                            entry_tag: Optional[str], side: str, **kwargs) -> float:
        self.proposed_stake = proposed_stake
        return proposed_stake * self.first_entry_ratio

    def adjust_trade_position(self, trade: Trade, current_time: datetime,
                              current_rate: float, current_profit: float, min_stake: float,
                              max_stake: float, **kwargs) -> Optional[float]:
        if current_profit > -0.05:
            return None

        filled_entries = trade.select_filled_orders(trade.entry_side)
        count_of_entries = len(filled_entries)
        if count_of_entries >= self.max_entry: return None

        dca_amount = self.proposed_stake * (1 - self.first_entry_ratio)
        logger.info(f"DCA {trade.pair} with stake amount of: {dca_amount}")
        return dca_amount
