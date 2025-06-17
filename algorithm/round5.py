
import json
from typing import Any, Dict, List, Tuple, Optional
# Allowed libraries: pandas, numpy, statistics, math, typing, jsonpickle (using json)
import pandas as pd
import numpy as np
import statistics
import math
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState

class Logger:
    # ... (Paste the full Logger class code here) ...
    def __init__(self) -> None: self.logs = ""; self.max_log_length = 3750
    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None: self.logs += sep.join(map(str, objects)) + end
    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]], conversions: int, trader_data: str) -> None:
        log_output = self.logs; persistent_data = state.traderData if state.traderData is not None else ""
        # Simplified length estimation for performance
        temp_compressed_state = [state.timestamp, len(persistent_data)]; compressed_orders = self.compress_orders(orders)
        base_length = len(self.to_json([temp_compressed_state, compressed_orders, conversions, "", ""])) + 500
        available_length = self.max_log_length - base_length; max_item_length = max(0, available_length // 3)
        output_list = [self.compress_state(state, self.truncate(persistent_data, max_item_length)), compressed_orders, conversions, self.truncate(trader_data, max_item_length), self.truncate(log_output, max_item_length)]
        print(self.to_json(output_list)); self.logs = ""
    def compress_state(self, state: TradingState, trader_data: str) -> list[Any]: return [state.timestamp, trader_data, self.compress_listings(state.listings), self.compress_order_depths(state.order_depths), self.compress_trades(state.own_trades), self.compress_trades(state.market_trades), state.position, self.compress_observations(state.observations)]
    def compress_listings(self, listings: dict[Symbol, Listing]) -> list[list[Any]]: return [[l.symbol, l.product, l.denomination] for l in listings.values()]
    def compress_order_depths(self, order_depths: dict[Symbol, OrderDepth]) -> dict[Symbol, list[Any]]: return {s: [list(od.buy_orders.items()), list(od.sell_orders.items())] for s, od in order_depths.items()}
    def compress_trades(self, trades: dict[Symbol, list[Trade]]) -> list[list[Any]]: return [[t.symbol, t.price, t.quantity, getattr(t, 'buyer', None), getattr(t, 'seller', None), t.timestamp] for arr in trades.values() for t in arr]
    def compress_observations(self, observations: Observation) -> list[Any]:
        conv_obs = {}; raw_conv_obs = getattr(observations, 'conversionObservations', {})
        if raw_conv_obs:
            for p, o in raw_conv_obs.items(): conv_obs[p] = [getattr(o, k, None) for k in ['bidPrice', 'askPrice', 'transportFees', 'exportTariff', 'importTariff', 'sunlight', 'humidity', 'sugarPrice']]
        pvo = getattr(observations, 'plainValueObservations', {}) or {}; serializable_pvo = {k: (v if isinstance(v, (int, float, str, bool, list, dict)) else repr(v)) for k, v in pvo.items()}
        return [serializable_pvo, conv_obs]
    def compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[list[Any]]: return [[o.symbol, o.price, o.quantity] for arr in orders.values() for o in arr]
    def default_serializer(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64)): return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)): return float(obj)
        elif isinstance(obj, (np.ndarray,)): return obj.tolist();
        elif isinstance(obj, (np.bool_)): return bool(obj);
        elif isinstance(obj, (np.void)): return None
        try: return repr(obj)
        except Exception: return f"<unserializable type: {type(obj).__name__}>"
    def to_json(self, value: Any) -> str:
        try: return json.dumps(value, cls=ProsperityEncoder, default=self.default_serializer, separators=(",", ":"))
        except NameError: return json.dumps(value, default=self.default_serializer, separators=(",", ":"))
    def truncate(self, value: Any, max_length: int) -> str:
        value_str = str(value);
        if max_length <= 0: return ""
        return value_str if len(value_str) <= max_length else value_str[:max_length - 3] + "..."

logger = Logger()

R = 0.0
# --- Paste normal_cdf, bs_d1, bs_d2, bs_call_price, implied_volatility_bisection ---
_p = 0.2316419; _b1 = 0.319381530; _b2 = -0.356563782; _b3 = 1.781477937; _b4 = -1.821255978; _b5 = 1.330274429; _inv_sqrt_2pi = 0.3989422804014327
def normal_cdf(x):
    if x < -8.0: return 0.0;
    if x > 8.0: return 1.0
    t = 1.0 / (1.0 + _p * abs(x)); t2 = t*t; t3 = t2*t; t4 = t3*t; t5 = t4*t
    cdf_approx = _inv_sqrt_2pi * math.exp(-0.5 * x * x) * (_b1*t + _b2*t2 + _b3*t3 + _b4*t4 + _b5*t5)
    return 1.0 - cdf_approx if x >= 0 else cdf_approx
def bs_d1(S, K, T, sigma):
    if T <= 1e-9 or sigma <= 1e-9 or S <= 1e-9 or K <= 1e-9: return np.nan
    if not all(math.isfinite(v) for v in [S, K, T, sigma]): return np.nan
    sigma_sqrt_T = sigma * math.sqrt(T)
    if sigma_sqrt_T < 1e-9: return 0.0 if abs(S-K) < 1e-9 else float('inf') * (1 if S > K else -1)
    try: log_S_K = math.log(S / K)
    except ValueError: return np.nan
    return (log_S_K + (R + 0.5 * sigma**2) * T) / sigma_sqrt_T
def bs_d2(S, K, T, sigma):
    if T <= 1e-9 or sigma <= 1e-9 or S <= 1e-9 or K <= 1e-9: return np.nan
    if not all(math.isfinite(v) for v in [S, K, T, sigma]): return np.nan
    d1 = bs_d1(S, K, T, sigma);
    if np.isnan(d1): return np.nan
    sigma_sqrt_T = sigma * math.sqrt(T);
    if sigma_sqrt_T < 1e-9: return d1
    return d1 - sigma_sqrt_T
def bs_call_price(S, K, T, sigma):
    if abs(T) < 1e-9 : return max(0.0, S - K)
    if sigma <= 1e-9 or S <= 1e-9 or K <= 1e-9: return max(0.0, S - K) if sigma <= 1e-9 else np.nan
    if not all(math.isfinite(v) for v in [S, K, T, sigma]): return np.nan
    d1 = bs_d1(S, K, T, sigma); d2 = bs_d2(S, K, T, sigma)
    if np.isnan(d1) or np.isnan(d2): return max(0.0, S-K) if abs(sigma) < 1e-6 else (max(0.0, S - K) if d1 == float('inf') else (0.0 if d1 == float('-inf') else np.nan))
    try: price = (S * normal_cdf(d1) - K * math.exp(-R * T) * normal_cdf(d2))
    except Exception: return np.nan
    return min(max(0.0, price), S)
def implied_volatility_bisection(V, S, K, T, tol=1e-4, max_iter=25): # Faster settings
    if V < 1e-6 or T < 1e-9: return np.nan
    min_price = max(0.0, S - K * math.exp(-R * T)); max_price = S
    bound_tol = max(tol*10, tol * K * 0.01); intrinsic_tol = max(tol, tol * K * 0.001)
    if V < min_price - bound_tol or V > max_price + bound_tol: return np.nan
    if abs(V - min_price) < intrinsic_tol: return 1e-4
    low_sigma, high_sigma = 1e-4, 5.0
    price_low = bs_call_price(S, K, T, low_sigma)
    if np.isnan(price_low): return np.nan
    if V < price_low + tol: return low_sigma
    price_high = bs_call_price(S, K, T, high_sigma)
    if np.isnan(price_high):
         high_sigma = 15.0; price_high = bs_call_price(S, K, T, high_sigma)
         if np.isnan(price_high): return np.nan
    if V > price_high - tol: return high_sigma
    for _ in range(max_iter):
        mid_sigma = (low_sigma + high_sigma) / 2.0;
        if mid_sigma < 1e-5: mid_sigma = 1e-5
        price_mid = bs_call_price(S, K, T, mid_sigma)
        if np.isnan(price_mid):
             high_sigma = mid_sigma if V < (S/2.0) else high_sigma
             low_sigma = mid_sigma if V > (S/2.0) else low_sigma
             continue
        diff = price_mid - V
        if abs(diff) < tol: return mid_sigma
        if diff < 0: low_sigma = mid_sigma
        else: high_sigma = mid_sigma
        if (high_sigma - low_sigma) < tol: return (low_sigma + high_sigma) / 2.0
    final_sigma = (low_sigma + high_sigma) / 2.0
    if final_sigma < 1e-5: return 1e-5;
    if final_sigma > 20.0: return np.nan
    return final_sigma

# --- TTE Calculation ---
def calculate_specific_tte(timestamp):
    tte = (3.0 - (timestamp / 1_000_000)) / 30.0
    return max(tte, 1e-9)

# --- Fair Value Calculation using Dynamic Curve ---
def get_fair_volatility(St, K, TTE, coeffs):
    """Calculate fair volatility using given curve coefficients [a, b, c]."""
    if TTE <= 1e-9 or St <= 1e-9 or K <= 1e-9: return np.nan
    sqrt_TTE = math.sqrt(TTE + 1e-9)
    if sqrt_TTE < 1e-9: return np.nan
    try: log_K_S = math.log(K / St)
    except ValueError: return np.nan
    m_t = log_K_S / sqrt_TTE
    if not math.isfinite(m_t): return np.nan
    a, b, c = coeffs
    fair_vol = a * m_t**2 + b * m_t + c
    return max(1e-3, min(fair_vol, 2.0)) # Bounds

def calculate_mid_price(order_depth: Optional[OrderDepth]) -> Optional[float]:
    """Calculates mid price safely, returns None if invalid."""
    if order_depth is None or not order_depth.buy_orders or not order_depth.sell_orders:
        return None
    best_bid = max(order_depth.buy_orders.keys())
    best_ask = min(order_depth.sell_orders.keys())
    # Check for crossed or invalid book
    if best_bid >= best_ask:
        return None
    return (best_bid + best_ask) / 2.0

def generate_orders_with_slippage(symbol: Symbol, target_pos: int, current_pos: int, order_depth: Optional[OrderDepth], slippage_ticks: int) -> List[Order]:
    """
    Generates orders to move towards target_pos, considering slippage.
    Returns a list of Order objects (potentially empty).
    Ensures integer price and quantity in orders.
    """
    orders: List[Order] = []
    if not order_depth:
        return orders

    slippage_ticks = max(1, slippage_ticks) # Ensure positive slippage
    delta = target_pos - current_pos
    if delta == 0:
        return orders

    book = order_depth.sell_orders if delta > 0 else order_depth.buy_orders
    if not book:
        return orders # Cannot fill if the required side is empty

    remaining_qty = abs(delta)
    sorted_book_items = sorted(book.items()) if delta > 0 else sorted(book.items(), reverse=True)

    if not sorted_book_items:
        return orders

    best_price = sorted_book_items[0][0]
    limit_price = best_price + slippage_ticks if delta > 0 else best_price - slippage_ticks

    total_available_at_limit_or_better = 0
    for price_level, available_qty_signed in sorted_book_items:
        available_qty = abs(available_qty_signed) # Quantity in order depth is negative for asks
        if delta > 0: # Buying
            if price_level <= limit_price:
                total_available_at_limit_or_better += available_qty
            else:
                # Since asks are sorted ascending, no more levels will match
                break
        else: # Selling
            # We can sell at any price GREATER THAN OR EQUAL TO our limit price
            if price_level >= limit_price:
                total_available_at_limit_or_better += available_qty
            else:
                # Since bids are sorted descending, no more levels will match
                break

    trade_qty = min(remaining_qty, total_available_at_limit_or_better)

    if trade_qty > 0:
        # Ensure quantity is integer and has the correct sign
        actual_delta = int(trade_qty) if delta > 0 else -int(trade_qty)
        # Ensure price is integer for the order
        final_order_price = int(round(limit_price))
        orders.append(Order(symbol, final_order_price, actual_delta))
        # logger.print(f"DEBUG GenOrder: {symbol} {actual_delta} @ {final_order_price} (Target:{target_pos}, Curr:{current_pos}, Avail:{total_available_at_limit_or_better})")

    return orders

class TriosStrategy:
    def __init__(self):
        logger.print("Initializing TriosStrategy...")
        self.assets = ["JAMS", "DJEMBES"]
        # --- Parameters ---
        self.global_limits = { "JAMS": 350, "DJEMBES": 60 }
        self.norm_offsets = {"JAMS": 280.29, "DJEMBES": 0.0}
        self.norm_ratios = {"JAMS": 2.0, "DJEMBES": 1.0}
        self.dev_ma_long_window = 400
        self.gradient_window = 10
        self.slope_threshold: Dict[Symbol, float] = { "JAMS": 0.001, "DJEMBES": 0.001 }
        self.dev_threshold: Dict[Symbol, float] = { "JAMS": 50, "DJEMBES": 25 }
        self.liq_short_dev_threshold: Dict[Symbol, float] = { "JAMS": 10, "DJEMBES": 10 }
        self.liq_long_dev_threshold: Dict[Symbol, float] = { "JAMS": -10,  "DJEMBES": -10 }
        self.go_streak_required: Dict[Symbol, int] = { "JAMS": 7, "DJEMBES": 7 }
        self.liq_streak_required: Dict[Symbol, int] = { "JAMS": 7, "DJEMBES": 7 }
        self.slippage_ticks = 2
        # --- History / State ---
        self.max_streak_lookback = max(max(self.go_streak_required.values()), max(self.liq_streak_required.values()))
        self.ma_history_len = self.gradient_window + self.max_streak_lookback
        self.dev_history_len = self.dev_ma_long_window + self.ma_history_len
        self.deviation_history: Dict[Symbol, List[Optional[float]]] = {asset: [] for asset in self.assets}
        self.ma300_history: Dict[Symbol, List[Optional[float]]] = {asset: [] for asset in self.assets}
        self.slope_history: Dict[Symbol, List[Optional[float]]] = {asset: [] for asset in self.assets}
        self.intended_state: Dict[Symbol, str] = {asset: "FLAT" for asset in self.assets}
        self._all_required_assets = sorted(list(self.assets)) # Keep sorted list for consistency
        logger.print("TriosStrategy Initialized.")

    # --- Helper Methods (Asset/Strategy Specific) ---
    def _calculate_all_prices(self, state: TradingState) -> Dict[Symbol, Optional[float]]:
        """Calculate mid prices for all assets managed by this strategy."""
        prices = {}
        for symbol in self._all_required_assets:
            # Use the global helper function
            prices[symbol] = calculate_mid_price(state.order_depths.get(symbol))
        return prices

    # --- State Management ---
    def _load_previous_state(self, trader_data_str: str):
        """Loads the intended_state dictionary from the traderData string."""
        self.intended_state = {asset: "FLAT" for asset in self.assets} # Default
        if trader_data_str:
            try:
                data = json.loads(trader_data_str)
                loaded_state = data.get("intended_state", {})
                for asset in self.assets:
                    self.intended_state[asset] = loaded_state.get(asset, "FLAT") if loaded_state.get(asset) in ["FLAT", "TARGET_LONG", "TARGET_SHORT"] else "FLAT"
                # Note: History is not loaded/saved in the original Trios logic, kept in memory
            except json.JSONDecodeError:
                logger.print(f"TRIOS WARNING: JSONDecodeError loading traderData: {trader_data_str[:100]}...")
            except Exception as e:
                 logger.print(f"TRIOS WARNING: Error loading traderData: {e}")

    def _save_current_state(self) -> str:
        """Saves the intended_state dictionary to a JSON string."""
        state_to_save = {"intended_state": self.intended_state.copy()}
        try:
            return json.dumps(state_to_save, default=logger.default_serializer, separators=(",", ":"))
        except Exception as e:
            logger.print(f"TRIOS ERROR: encoding intended_state: {e}")
            return "{}"

    # --- Calculation Methods (Unchanged from previous step) ---
    def _calculate_normalized_prices(self, prices: Dict[Symbol, Optional[float]]) -> Dict[Symbol, Optional[float]]:
        norm_prices = {}
        for asset in self.assets:
            price = prices.get(asset); ratio = self.norm_ratios.get(asset, 1.0); offset = self.norm_offsets.get(asset, 0.0)
            if price is not None and isinstance(price, (int, float)): norm_prices[asset] = ratio * price + offset
            else: norm_prices[asset] = None
        return norm_prices

    def _calculate_consensus_and_deviations(self, norm_prices: Dict[Symbol, Optional[float]]) -> Tuple[Optional[float], Dict[Symbol, Optional[float]]]:
        valid_norm_prices = [p for p in norm_prices.values() if p is not None]
        consensus_mean = np.mean(valid_norm_prices) if len(valid_norm_prices) >= 2 else None
        current_deviations = {}
        for asset in self.assets:
            norm_price = norm_prices.get(asset)
            if norm_price is not None and consensus_mean is not None:
                current_deviations[asset] = norm_price - consensus_mean
            else: current_deviations[asset] = None
        return consensus_mean, current_deviations

    def _update_histories(self, current_deviations: Dict[Symbol, Optional[float]]):
        # (Logic unchanged - updates in-memory lists)
        current_ma300s = {}
        current_slopes = {}
        for asset in self.assets:
            dev = current_deviations.get(asset)
            self.deviation_history[asset].append(dev)
            ma300 = None
            if len(self.deviation_history[asset]) >= self.dev_ma_long_window:
                valid_devs = [d for d in self.deviation_history[asset][-self.dev_ma_long_window:] if d is not None]
                if len(valid_devs) >= self.dev_ma_long_window * 0.8:
                     ma300 = np.mean(valid_devs)
            self.ma300_history[asset].append(ma300)
            current_ma300s[asset] = ma300
            slope = None
            if len(self.ma300_history[asset]) >= self.gradient_window:
                valid_mas = [m for m in self.ma300_history[asset][-self.gradient_window:] if m is not None]
                if len(valid_mas) == self.gradient_window:
                    try: slope = np.gradient(np.array(valid_mas))[-1]
                    except Exception: slope = None
            self.slope_history[asset].append(slope)
            current_slopes[asset] = slope
            # Trim Histories
            if len(self.deviation_history[asset]) > self.dev_history_len:
                self.deviation_history[asset] = self.deviation_history[asset][-self.dev_history_len:]
            if len(self.ma300_history[asset]) > self.ma_history_len:
                self.ma300_history[asset] = self.ma300_history[asset][-self.ma_history_len:]
            if len(self.slope_history[asset]) > self.max_streak_lookback + 10:
                self.slope_history[asset] = self.slope_history[asset][-(self.max_streak_lookback + 10):]
        return current_ma300s, current_slopes

    # --- Condition Checkers (Unchanged) ---
    def _is_go_short_cond(self, asset: Symbol, dev: Optional[float], slope: Optional[float]) -> bool:
        if dev is None or slope is None: return False
        return dev > self.dev_threshold[asset] and slope < -self.slope_threshold[asset]
    def _is_liq_short_cond(self, asset: Symbol, dev: Optional[float], slope: Optional[float]) -> bool:
        if dev is None: return False # Slope not used in original logic here
        return dev < self.liq_short_dev_threshold[asset]
    def _is_go_long_cond(self, asset: Symbol, dev: Optional[float], slope: Optional[float]) -> bool:
        if dev is None or slope is None: return False
        return dev < -self.dev_threshold[asset] and slope > self.slope_threshold[asset]
    def _is_liq_long_cond(self, asset: Symbol, dev: Optional[float], slope: Optional[float]) -> bool:
        if dev is None: return False # Slope not used in original logic here
        return dev > self.liq_long_dev_threshold[asset]

    # --- Streak Checking (Unchanged) ---
    def _check_historical_streak(self, asset: Symbol, condition_func, length: int) -> bool:
        dev_hist = self.deviation_history[asset]
        slope_hist = self.slope_history[asset]
        if len(dev_hist) < length or len(slope_hist) < length: return False
        for i in range(length):
            idx = -1 - i
            if abs(idx) > len(dev_hist) or abs(idx) > len(slope_hist): return False # Safety check
            past_dev = dev_hist[idx]
            past_slope = slope_hist[idx]
            if not condition_func(asset, past_dev, past_slope): return False
        return True

    # --- Target Position Logic (Unchanged) ---
    def _get_target_positions(self, current_positions: Dict[Symbol, int]) -> Tuple[Dict[Symbol, int], Dict[str, Any]]:
        # (Logic unchanged from previous step)
        final_target_positions = {}
        arb_log_state = {} # For detailed logging

        for asset in self.assets:
            current_pos = current_positions.get(asset, 0)
            limit = self.global_limits[asset]
            current_intent = self.intended_state.get(asset, "FLAT")
            target_pos = current_pos # Default: Hold
            active_signal = "HOLD"
            new_intent = current_intent

            go_req = self.go_streak_required[asset]
            liq_req = self.liq_streak_required[asset]

            go_short_signal_active = self._check_historical_streak(asset, self._is_go_short_cond, go_req)
            liq_short_signal_active = self._check_historical_streak(asset, self._is_liq_short_cond, liq_req)
            go_long_signal_active = self._check_historical_streak(asset, self._is_go_long_cond, go_req)
            liq_long_signal_active = self._check_historical_streak(asset, self._is_liq_long_cond, liq_req)

            should_liquidate = False
            if (current_intent == "TARGET_SHORT" or current_pos < 0) and liq_short_signal_active:
                active_signal = f"LIQUIDATE_SHORT (Strk={liq_req})"
                target_pos = 0
                new_intent = "FLAT"
                should_liquidate = True
            elif (current_intent == "TARGET_LONG" or current_pos > 0) and liq_long_signal_active:
                active_signal = f"LIQUIDATE_LONG (Strk={liq_req})"
                target_pos = 0
                new_intent = "FLAT"
                should_liquidate = True

            elif not should_liquidate and (current_intent == "FLAT" or current_pos == 0):
                if go_short_signal_active:
                    active_signal = f"GO_SHORT (Strk={go_req})"
                    target_pos = -limit
                    new_intent = "TARGET_SHORT"
                elif go_long_signal_active:
                    active_signal = f"GO_LONG (Strk={go_req})"
                    target_pos = limit
                    new_intent = "TARGET_LONG"

            elif not should_liquidate:
                if new_intent == "TARGET_SHORT" and current_pos > -limit:
                     active_signal = "FILL_UP_SHORT"
                     target_pos = -limit
                elif new_intent == "TARGET_LONG" and current_pos < limit:
                     active_signal = "FILL_UP_LONG"
                     target_pos = limit
                else:
                     target_pos = current_pos
                     active_signal = f"HOLD (Intent: {current_intent})"

            if new_intent == "FLAT" and not should_liquidate:
                 target_pos = 0
                 if current_intent == "FLAT": # If it was already flat
                      active_signal = "HOLD (Intent: FLAT)"


            self.intended_state[asset] = new_intent # Update persistent intent

            final_target_positions[asset] = target_pos
            # Logging details
            arb_log_state[f"tgt_{asset[:2].upper()}"] = target_pos
            arb_log_state[f"{asset[:2]}_signal"] = active_signal
            current_dev = self.deviation_history[asset][-1] if self.deviation_history[asset] else None
            current_slope = self.slope_history[asset][-1] if self.slope_history[asset] else None
            arb_log_state[f"{asset[:2]}_dev"] = round(current_dev, 2) if current_dev is not None else None
            arb_log_state[f"{asset[:2]}_slope"] = round(current_slope, 4) if current_slope is not None else None
            arb_log_state[f"{asset[:2]}_Intent"] = new_intent

        return final_target_positions, arb_log_state

    # --- Main Execution Method ---
    def run_strategy(self, state: TradingState, incoming_trader_data_str: str) -> Tuple[Dict[Symbol, List[Order]], int, Dict[str, Any], str]:
        """ Executes the Trios trading logic. """
        final_orders: Dict[Symbol, List[Order]] = {asset: [] for asset in self.assets}
        conversions = 0
        detail_log_dict = {}

        current_positions = {asset: state.position.get(asset, 0) for asset in self.assets}
        self._load_previous_state(incoming_trader_data_str)
        prices = self._calculate_all_prices(state)
        norm_prices = self._calculate_normalized_prices(prices)
        consensus_mean, current_deviations = self._calculate_consensus_and_deviations(norm_prices)
        current_ma300s, current_slopes = self._update_histories(current_deviations)
        final_target_positions, arb_log_details = self._get_target_positions(current_positions)

        for symbol in self.assets:
            target_pos = final_target_positions.get(symbol, 0)
            current_pos_start_tick = current_positions.get(symbol, 0)
            if target_pos != current_pos_start_tick:
                 order_depth = state.order_depths.get(symbol)
                 # Use the global helper function for orders
                 symbol_orders = generate_orders_with_slippage(symbol, target_pos, current_pos_start_tick, order_depth, self.slippage_ticks)
                 if symbol_orders:
                     final_orders[symbol].extend(symbol_orders)

        for asset in self.assets:
             detail_log_dict[f"pos_{asset[:2].upper()}"] = current_positions.get(asset, 0)
        detail_log_dict['consensus_mean'] = round(consensus_mean, 2) if consensus_mean is not None else None
        detail_log_dict['arb_details'] = arb_log_details

        outgoing_trader_data_str = self._save_current_state()

        final_orders_filtered = {sym: ord_list for sym, ord_list in final_orders.items() if ord_list}
        return final_orders_filtered, conversions, detail_log_dict, outgoing_trader_data_str
    
class ResinStrategy:
    def __init__(self):
        logger.print("Initializing ResinStrategy...")
        self.product = "RAINFOREST_RESIN"
        self.position_limit = 50
        self.mm_bid_price = 9994
        self.mm_ask_price = 10006
        self.take_buy_threshold = 10000
        self.take_sell_threshold = 10000
        self.base_mm_order_size = 30
        logger.print("ResinStrategy Initialized.")

    def run_strategy(self, state: TradingState, incoming_trader_data_str: str) -> Tuple[Dict[Symbol, List[Order]], int, Dict[str, Any], str]:
        """ Executes the Resin trading logic. """
        orders: Dict[Symbol, List[Order]] = {self.product: []}
        conversions = 0
        detail_log_dict = {}
        outgoing_trader_data_str = "" # No state persistence

        order_depth = state.order_depths.get(self.product)
        if not order_depth:
            return {}, conversions, detail_log_dict, outgoing_trader_data_str

        current_position = state.position.get(self.product, 0)
        generated_orders: List[Order] = []

        position_skew = current_position / self.position_limit if self.position_limit != 0 else 0
        buy_size_factor = max(0, 1 - position_skew)
        sell_size_factor = max(0, 1 + position_skew)
        mm_buy_size = int(self.base_mm_order_size * buy_size_factor)
        mm_sell_size = int(self.base_mm_order_size * sell_size_factor)
        potential_buy_room = self.position_limit - current_position
        potential_sell_room = self.position_limit + current_position
        actual_mm_buy_size = max(0, min(mm_buy_size, potential_buy_room))
        actual_mm_sell_size = max(0, min(mm_sell_size, potential_sell_room))

        if actual_mm_buy_size > 0:
            generated_orders.append(Order(self.product, self.mm_bid_price, actual_mm_buy_size))
        if actual_mm_sell_size > 0:
            generated_orders.append(Order(self.product, self.mm_ask_price, -actual_mm_sell_size))

        # --- Market Taking ---
        best_ask: Optional[int] = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None
        best_bid: Optional[int] = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else None

        if best_ask is not None and best_ask < self.take_buy_threshold:
            best_ask_amount = abs(order_depth.sell_orders[best_ask])
            space_to_buy = self.position_limit - current_position
            take_buy_quantity = max(0, min(best_ask_amount, space_to_buy))
            if take_buy_quantity > 0:
                generated_orders.append(Order(self.product, best_ask, take_buy_quantity))

        if best_bid is not None and best_bid > self.take_sell_threshold:
            best_bid_amount = order_depth.buy_orders[best_bid]
            space_to_sell = self.position_limit + current_position
            take_sell_quantity = max(0, min(best_bid_amount, space_to_sell))
            if take_sell_quantity > 0:
                generated_orders.append(Order(self.product, best_bid, -take_sell_quantity))

        if generated_orders:
            orders[self.product] = generated_orders

        detail_log_dict = {
            "pos_RE": current_position, "pos_skew": round(position_skew, 3),
            "mm_buy_size": actual_mm_buy_size, "mm_sell_size": actual_mm_sell_size,
            "best_bid": best_bid, "best_ask": best_ask
        }

        final_orders_filtered = {sym: ord_list for sym, ord_list in orders.items() if ord_list}
        return final_orders_filtered, conversions, detail_log_dict, outgoing_trader_data_str

class KelpStrategy:
    def __init__(self):
        logger.print("Initializing KelpStrategy...")
        self.symbol = "KELP"
        self.position_limit = 50
        self.target_volume = 11 # Original logic used 11 for both bid/ask
        logger.print("KelpStrategy Initialized.")

    def run_strategy(self, state: TradingState, incoming_trader_data_dict: Dict[str, Any]) -> Tuple[Dict[Symbol, List[Order]], int, Dict[str, Any], Dict[str, Any]]:
        """ Executes the Kelp market making logic. """
        orders: Dict[Symbol, List[Order]] = {self.symbol: []}
        conversions = 0
        detail_log_dict = {}
        outgoing_trader_data_dict = {} # Kelp is stateless

        order_depth = state.order_depths.get(self.symbol)
        if not order_depth:
            detail_log_dict["status"] = "No order depth"
            return {}, conversions, detail_log_dict, outgoing_trader_data_dict # Cannot trade without depth

        current_position = state.position.get(self.symbol, 0)
        generated_orders: List[Order] = []

        # Get best bid/ask prices
        best_bid_price = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else None
        best_ask_price = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None

        detail_log_dict["pos_KE"] = current_position
        detail_log_dict["best_bid"] = best_bid_price
        detail_log_dict["best_ask"] = best_ask_price

        # Apply mild skewing logic (Original logic)
        bid_skew = 0
        ask_skew = 0

        if current_position > 30:
            bid_skew = -1  # Lower bid price to slow down more buying
        elif current_position < -30:
            ask_skew = 1   # Raise ask price to slow down more selling

        detail_log_dict["bid_skew"] = bid_skew
        detail_log_dict["ask_skew"] = ask_skew

        # Place BID order
        if best_bid_price is not None:
            bid_price = int(best_bid_price + bid_skew) # Ensure integer price
            if current_position < self.position_limit: # Check if we can increase position
                # Calculate volume based on remaining room, up to target_volume
                buy_volume = min(self.target_volume, self.position_limit - current_position)
                if buy_volume > 0:
                    generated_orders.append(Order(self.symbol, bid_price, buy_volume))
                    detail_log_dict["bid_order"] = f"{buy_volume}@{bid_price}"

        # Place ASK order
        if best_ask_price is not None:
            ask_price = int(best_ask_price + ask_skew) # Ensure integer price
            if current_position > -self.position_limit: # Check if we can decrease position (increase negative)
                 # Calculate volume based on remaining room, up to target_volume
                sell_volume = min(self.target_volume, self.position_limit + current_position)
                if sell_volume > 0:
                    generated_orders.append(Order(self.symbol, ask_price, -sell_volume))
                    detail_log_dict["ask_order"] = f"{-sell_volume}@{ask_price}"

        if generated_orders:
            orders[self.symbol] = generated_orders

        # Filter empty lists (although logic above should prevent this unless no orders placed)
        final_orders_filtered = {sym: ord_list for sym, ord_list in orders.items() if ord_list}
        return final_orders_filtered, conversions, detail_log_dict, outgoing_trader_data_dict


class CroissantStrategy:
    def __init__(self):
        """Initializes the Croissant Strategy."""
        logger.print("Initializing CroissantStrategy...")
        self.symbol: Symbol = "CROISSANTS"
        self.position_limit: int = 250 # Specific limit for Croissants
        logger.print(f"Strategy: CroissantStrategy (Olivia Follower Aggressive v2) for {self.symbol}")
        logger.print(f"Position Limit: +/- {self.position_limit}")
        logger.print("CroissantStrategy Initialized.")
        # No persistent state needed for this strategy

    def run_strategy(self, state: TradingState, incoming_trader_data_dict: Dict[str, Any]) -> Tuple[Dict[Symbol, List[Order]], int, Dict[str, Any], Dict[str, Any]]:
        """
        Executes the Croissant trading logic.
        Aggressively follows Olivia's trades for CROISSANTS.
        Returns: orders, conversions, detail_log, new_state_dict
        """
        # Initialize return values
        orders: Dict[Symbol, List[Order]] = {self.symbol: []}
        conversions: int = 0
        detail_log_dict: Dict[str, Any] = {} # For orchestrator compatibility
        outgoing_trader_data_dict: Dict[str, Any] = {} # Stateless

        # Retrieve data for the target symbol
        order_depth: Optional[OrderDepth] = state.order_depths.get(self.symbol)
        market_trades: List[Trade] = state.market_trades.get(self.symbol, [])
        current_position: int = state.position.get(self.symbol, 0)

        detail_log_dict['pos_CR'] = current_position # Add basic logging

        # Early return if no order depth data for the symbol
        if order_depth is None or (not order_depth.buy_orders and not order_depth.sell_orders):
            logger.print(f"CROISSANT: No order depth data found for {self.symbol}. No action.")
            detail_log_dict['status'] = "No order depth"
            return {}, conversions, detail_log_dict, outgoing_trader_data_dict # Return empty orders dict

        # Detect Olivia's actions in recent market trades for the symbol
        olivia_buying: bool = False
        olivia_selling: bool = False
        olivia_trades_log = [] # For detailed logging
        if market_trades: # Check if there are any market trades for the symbol
            for trade in market_trades:
                if trade.buyer == "Olivia":
                    olivia_buying = True
                    olivia_trades_log.append(f"BUY {trade.quantity}@{trade.price}")
                    # logger.print(f"CROISSANT: Detected Olivia BUYING {trade.quantity} @ {trade.price}") # Minimal logging
                if trade.seller == "Olivia":
                    olivia_selling = True
                    olivia_trades_log.append(f"SELL {trade.quantity}@{trade.price}")
                    # logger.print(f"CROISSANT: Detected Olivia SELLING {trade.quantity} @ {trade.price}") # Minimal logging

        detail_log_dict['olivia_trades'] = olivia_trades_log if olivia_trades_log else "None"
        detail_log_dict['olivia_buying'] = olivia_buying
        detail_log_dict['olivia_selling'] = olivia_selling

        # --- AGGRESSIVE TRADING LOGIC ---
        generated_orders: List[Order] = []
        action_taken = "HOLD"

        # --- Logic if Olivia is BUYING (Aggressively Buy by taking asks) ---
        if olivia_buying and not olivia_selling:
            remaining_buy_capacity = self.position_limit - current_position
            action_taken = f"FOLLOW_BUY (Cap:{remaining_buy_capacity})"
            # logger.print(f"CROISSANT: Action: AGGRESSIVE_FOLLOW_OLIVIA_BUY. Remaining buy capacity: {remaining_buy_capacity}")

            if remaining_buy_capacity > 0 and order_depth.sell_orders:
                sorted_asks = sorted(order_depth.sell_orders.items())
                orders_placed_log = []
                for ask_price, ask_volume_signed in sorted_asks:
                    if remaining_buy_capacity <= 0: break
                    ask_volume = abs(ask_volume_signed)
                    volume_to_buy = min(remaining_buy_capacity, ask_volume)
                    if volume_to_buy > 0:
                        order_price = int(ask_price)
                        order_volume = int(volume_to_buy)
                        buy_order = Order(self.symbol, order_price, order_volume)
                        generated_orders.append(buy_order)
                        remaining_buy_capacity -= order_volume
                        orders_placed_log.append(f"{order_volume}@{order_price}")
                        # logger.print(f"CROISSANT: Placed BUY Order: {order_volume}@{order_price}")
                detail_log_dict['orders_placed'] = f"BUY: {', '.join(orders_placed_log)}" if orders_placed_log else "None"
                if remaining_buy_capacity > 0:
                    detail_log_dict['capacity_remaining'] = remaining_buy_capacity

            elif remaining_buy_capacity <= 0:
                 action_taken = "HOLD (Buy limit reached)"
            elif not order_depth.sell_orders:
                 action_taken = "HOLD (No asks)"

        # --- Logic if Olivia is SELLING (Aggressively Sell by hitting bids) ---
        elif olivia_selling and not olivia_buying:
            remaining_sell_capacity = current_position + self.position_limit
            action_taken = f"FOLLOW_SELL (Cap:{remaining_sell_capacity})"
            # logger.print(f"CROISSANT: Action: AGGRESSIVE_FOLLOW_OLIVIA_SELL. Remaining sell capacity: {remaining_sell_capacity}")

            if remaining_sell_capacity > 0 and order_depth.buy_orders:
                sorted_bids = sorted(order_depth.buy_orders.items(), reverse=True)
                orders_placed_log = []
                for bid_price, bid_volume in sorted_bids:
                    if remaining_sell_capacity <= 0: break
                    volume_to_sell = min(remaining_sell_capacity, bid_volume)
                    if volume_to_sell > 0:
                        order_price = int(bid_price)
                        order_volume = int(volume_to_sell)
                        sell_order = Order(self.symbol, order_price, -order_volume)
                        generated_orders.append(sell_order)
                        remaining_sell_capacity -= order_volume
                        orders_placed_log.append(f"{-order_volume}@{order_price}")
                        # logger.print(f"CROISSANT: Placed SELL Order: {-order_volume}@{order_price}")
                detail_log_dict['orders_placed'] = f"SELL: {', '.join(orders_placed_log)}" if orders_placed_log else "None"
                if remaining_sell_capacity > 0:
                    detail_log_dict['capacity_remaining'] = remaining_sell_capacity

            elif remaining_sell_capacity <= 0:
                 action_taken = "HOLD (Sell limit reached)"
            elif not order_depth.buy_orders:
                 action_taken = "HOLD (No bids)"

        else:
            # Handles: No Olivia trades OR Olivia both bought/sold (noise).
            # logger.print("CROISSANT: No clear single Olivia action. No aggressive orders placed.")
            action_taken = "HOLD (No clear Olivia signal)"


        detail_log_dict['action'] = action_taken

        # Add generated orders to the result dictionary
        if generated_orders:
            orders[self.symbol] = generated_orders
            # logger.print(f"CROISSANT: Total orders generated for {self.symbol}: {len(generated_orders)}")

        # Filter empty lists just in case
        final_orders_filtered = {sym: ord_list for sym, ord_list in orders.items() if ord_list}

        # Return adhering to the orchestrator's expected format
        return final_orders_filtered, conversions, detail_log_dict, outgoing_trader_data_dict

class InkStrategy:
    def __init__(self):
        """Initializes the Ink Strategy."""
        logger.print("Initializing InkStrategy...")
        self.symbol: Symbol = "SQUID_INK"
        self.position_limit: int = 50
        logger.print(f"Strategy: InkStrategy (Olivia Follower Aggressive v2) for {self.symbol}")
        logger.print(f"Position Limit: +/- {self.position_limit}")
        logger.print("InkStrategy Initialized.")
        # No persistent state needed for this strategy

    def run_strategy(self, state: TradingState, incoming_trader_data_dict: Dict[str, Any]) -> Tuple[Dict[Symbol, List[Order]], int, Dict[str, Any], Dict[str, Any]]:
        """
        Executes the Squid Ink trading logic.
        Aggressively follows Olivia's trades for SQUID_INK.
        Returns: orders, conversions, detail_log, new_state_dict
        """
        # Initialize return values
        orders: Dict[Symbol, List[Order]] = {self.symbol: []}
        conversions: int = 0
        detail_log_dict: Dict[str, Any] = {} # For orchestrator compatibility
        outgoing_trader_data_dict: Dict[str, Any] = {} # Stateless

        # Retrieve data for the target symbol
        order_depth: Optional[OrderDepth] = state.order_depths.get(self.symbol)
        market_trades: List[Trade] = state.market_trades.get(self.symbol, [])
        current_position: int = state.position.get(self.symbol, 0)

        detail_log_dict['pos_INK'] = current_position # Add basic logging

        # Early return if no order depth data for the symbol
        if order_depth is None or (not order_depth.buy_orders and not order_depth.sell_orders):
            logger.print(f"INK: No order depth data found for {self.symbol}. No action.")
            detail_log_dict['status'] = "No order depth"
            return {}, conversions, detail_log_dict, outgoing_trader_data_dict # Return empty orders dict

        # Detect Olivia's actions in recent market trades for the symbol
        olivia_buying: bool = False
        olivia_selling: bool = False
        olivia_trades_log = [] # For detailed logging
        if market_trades: # Check if there are any market trades for the symbol
            for trade in market_trades:
                 # Optional: Check trade timestamp if needed
                if trade.buyer == "Olivia":
                    olivia_buying = True
                    olivia_trades_log.append(f"BUY {trade.quantity}@{trade.price}")
                    # logger.print(f"INK: Detected Olivia BUYING {trade.quantity} @ {trade.price}") # Keep logging minimal
                if trade.seller == "Olivia":
                    olivia_selling = True
                    olivia_trades_log.append(f"SELL {trade.quantity}@{trade.price}")
                    # logger.print(f"INK: Detected Olivia SELLING {trade.quantity} @ {trade.price}") # Keep logging minimal

        detail_log_dict['olivia_trades'] = olivia_trades_log if olivia_trades_log else "None"
        detail_log_dict['olivia_buying'] = olivia_buying
        detail_log_dict['olivia_selling'] = olivia_selling

        # --- AGGRESSIVE TRADING LOGIC ---
        generated_orders: List[Order] = []
        action_taken = "HOLD"

        # --- Logic if Olivia is BUYING (Aggressively Buy by taking asks) ---
        if olivia_buying and not olivia_selling:
            remaining_buy_capacity = self.position_limit - current_position
            action_taken = f"FOLLOW_BUY (Cap:{remaining_buy_capacity})"
            # logger.print(f"INK: Action: AGGRESSIVE_FOLLOW_OLIVIA_BUY. Remaining buy capacity: {remaining_buy_capacity}")

            if remaining_buy_capacity > 0 and order_depth.sell_orders:
                sorted_asks = sorted(order_depth.sell_orders.items())
                # logger.print(f"INK: Available Asks: {sorted_asks}")
                orders_placed_log = []
                for ask_price, ask_volume_signed in sorted_asks:
                    if remaining_buy_capacity <= 0: break # Filled capacity
                    ask_volume = abs(ask_volume_signed)
                    volume_to_buy = min(remaining_buy_capacity, ask_volume)
                    if volume_to_buy > 0:
                        order_price = int(ask_price)
                        order_volume = int(volume_to_buy)
                        buy_order = Order(self.symbol, order_price, order_volume)
                        generated_orders.append(buy_order)
                        remaining_buy_capacity -= order_volume
                        orders_placed_log.append(f"{order_volume}@{order_price}")
                        # logger.print(f"INK: Placed BUY Order: {order_volume}@{order_price}")
                detail_log_dict['orders_placed'] = f"BUY: {', '.join(orders_placed_log)}" if orders_placed_log else "None (No fillable asks?)"
                if remaining_buy_capacity > 0:
                    # logger.print(f"INK: Could not fill entire buy capacity. Remaining: {remaining_buy_capacity}")
                    detail_log_dict['capacity_remaining'] = remaining_buy_capacity

            elif remaining_buy_capacity <= 0:
                 # logger.print(f"INK: Already at or above positive limit ({self.position_limit}). No buy.")
                 action_taken = "HOLD (Buy limit reached)"
            elif not order_depth.sell_orders:
                 # logger.print("INK: No asks available to buy from.")
                 action_taken = "HOLD (No asks)"

        # --- Logic if Olivia is SELLING (Aggressively Sell by hitting bids) ---
        elif olivia_selling and not olivia_buying:
            remaining_sell_capacity = current_position + self.position_limit
            action_taken = f"FOLLOW_SELL (Cap:{remaining_sell_capacity})"
            # logger.print(f"INK: Action: AGGRESSIVE_FOLLOW_OLIVIA_SELL. Remaining sell capacity: {remaining_sell_capacity}")

            if remaining_sell_capacity > 0 and order_depth.buy_orders:
                sorted_bids = sorted(order_depth.buy_orders.items(), reverse=True)
                # logger.print(f"INK: Available Bids: {sorted_bids}")
                orders_placed_log = []
                for bid_price, bid_volume in sorted_bids:
                    if remaining_sell_capacity <= 0: break # Filled capacity
                    volume_to_sell = min(remaining_sell_capacity, bid_volume)
                    if volume_to_sell > 0:
                        order_price = int(bid_price)
                        order_volume = int(volume_to_sell)
                        sell_order = Order(self.symbol, order_price, -order_volume)
                        generated_orders.append(sell_order)
                        remaining_sell_capacity -= order_volume
                        orders_placed_log.append(f"{-order_volume}@{order_price}")
                        # logger.print(f"INK: Placed SELL Order: {-order_volume}@{order_price}")
                detail_log_dict['orders_placed'] = f"SELL: {', '.join(orders_placed_log)}" if orders_placed_log else "None (No fillable bids?)"
                if remaining_sell_capacity > 0:
                    # logger.print(f"INK: Could not fill entire sell capacity. Remaining: {remaining_sell_capacity}")
                    detail_log_dict['capacity_remaining'] = remaining_sell_capacity

            elif remaining_sell_capacity <= 0:
                 # logger.print(f"INK: Already at or below negative limit ({-self.position_limit}). No sell.")
                 action_taken = "HOLD (Sell limit reached)"
            elif not order_depth.buy_orders:
                 # logger.print("INK: No bids available to sell to.")
                 action_taken = "HOLD (No bids)"

        else:
            # Handles: No Olivia trades OR Olivia both bought/sold (noise).
            # logger.print("INK: No clear single Olivia action. No aggressive orders placed.")
            action_taken = "HOLD (No clear Olivia signal)"


        detail_log_dict['action'] = action_taken

        # Add generated orders to the result dictionary
        if generated_orders:
            orders[self.symbol] = generated_orders
            # logger.print(f"INK: Total orders generated for {self.symbol}: {len(generated_orders)}")

        # Filter empty lists just in case
        final_orders_filtered = {sym: ord_list for sym, ord_list in orders.items() if ord_list}

        # Return adhering to the orchestrator's expected format
        return final_orders_filtered, conversions, detail_log_dict, outgoing_trader_data_dict

# --- End of InkStrategy class ---

class MacaronStrategy:
    def __init__(self):
        logger.print("Initializing Simplified MacaronSunlightStrategy...")
        self.product = "MAGNIFICENT_MACARONS"
        self.position_limit = 75
        self.max_conversion_per_tick = 10
        self.price_ma_window = 250
        self.sunlight_trigger_threshold = 49
        self.sunlight_severe_dip_threshold = 32
        self.max_slippage = 5  # Maximum slippage in points

        # State variables
        self.strategy_state = "NORMAL"  # "NORMAL", "SUN_TRIGGER_LONG", "SUN_TRIGGER_SHORT"
        self.had_severe_dip = False  # Flag if sunlight went below severe_dip_threshold
        self.sunlight_history = []
        self.price_history = []
        self.history_len = self.price_ma_window + 5

    def _load_state(self, trader_data):
        self.strategy_state = trader_data.get("strategy_state", "NORMAL")
        self.had_severe_dip = trader_data.get("had_severe_dip", False)
        
        self.sunlight_history = trader_data.get("sunlight_history", [])
        self.price_history = trader_data.get("price_history", [])
        
        # Trim if needed
        if len(self.sunlight_history) > self.history_len:
            self.sunlight_history = self.sunlight_history[-self.history_len:]
        if len(self.price_history) > self.history_len:
            self.price_history = self.price_history[-self.history_len:]

    def _save_state(self):
        return {
            "strategy_state": self.strategy_state,
            "had_severe_dip": self.had_severe_dip,
            "sunlight_history": self.sunlight_history[-self.history_len:] if len(self.sunlight_history) > self.history_len else self.sunlight_history,
            "price_history": self.price_history[-self.history_len:] if len(self.price_history) > self.history_len else self.price_history
        }

    def _update_history(self, sunlight, price):
        # Only keep last 4 points of sunlight history
        if sunlight is not None and math.isfinite(sunlight):
            self.sunlight_history.append(sunlight)
        else:
            self.sunlight_history.append(None)
        
        # Trim sunlight history to just what we need
        if len(self.sunlight_history) > 4:
            self.sunlight_history = self.sunlight_history[-4:]
            
        # Only keep necessary price history for MA calculation
        if price is not None and math.isfinite(price):
            self.price_history.append(price)
        else:
            self.price_history.append(None)
        
        # Trim price history to just what we need for MA + 1 (for slope)
        if len(self.price_history) > self.price_ma_window + 1:
            self.price_history = self.price_history[-(self.price_ma_window + 1):]

    def _calculate_indicators(self):
        # Sunlight slope
        sunlight_slope = None
        if len(self.sunlight_history) >= 2:
            curr_sun = self.sunlight_history[-1]
            prev_sun = self.sunlight_history[-2]
            if curr_sun is not None and prev_sun is not None:
                sunlight_slope = curr_sun - prev_sun

        # Calculate price MA
        price_ma = None
        if len(self.price_history) >= self.price_ma_window:
            valid_prices = [p for p in self.price_history[-self.price_ma_window:] if p is not None]
            if len(valid_prices) >= self.price_ma_window * 0.8:
                price_ma = sum(valid_prices) / len(valid_prices)

        # Calculate price MA slope
        price_ma_slope = None
        if len(self.price_history) >= self.price_ma_window + 1:
            valid_prices_curr = [p for p in self.price_history[-self.price_ma_window:] if p is not None]
            valid_prices_prev = [p for p in self.price_history[-(self.price_ma_window+1):-1] if p is not None]
            
            if (len(valid_prices_curr) >= self.price_ma_window * 0.8 and 
                len(valid_prices_prev) >= self.price_ma_window * 0.8):
                curr_ma = sum(valid_prices_curr) / len(valid_prices_curr)
                prev_ma = sum(valid_prices_prev) / len(valid_prices_prev)
                price_ma_slope = curr_ma - prev_ma

        return sunlight_slope, price_ma, price_ma_slope

    def _generate_aggressive_orders(self, product, target_pos, current_pos, order_depth, max_slippage):
        orders = []
        position_delta = target_pos - current_pos
        
        if position_delta == 0:
            return orders
            
        if position_delta > 0:  # Need to buy
            asks = sorted(order_depth.sell_orders.items()) if order_depth and order_depth.sell_orders else []
            remaining = position_delta
            
            for price, volume in asks:
                if price > asks[0][0] + max_slippage:  # Stop if slippage exceeds limit
                    break
                    
                buy_volume = min(abs(volume), remaining)
                if buy_volume > 0:
                    orders.append(Order(product, price, buy_volume))
                    remaining -= buy_volume
                    
                if remaining <= 0:
                    break
                    
        else:  # Need to sell
            bids = sorted(order_depth.buy_orders.items(), reverse=True) if order_depth and order_depth.buy_orders else []
            remaining = -position_delta
            
            for price, volume in bids:
                if price < bids[0][0] - max_slippage:  # Stop if slippage exceeds limit
                    break
                    
                sell_volume = min(volume, remaining)
                if sell_volume > 0:
                    orders.append(Order(product, price, -sell_volume))
                    remaining -= sell_volume
                    
                if remaining <= 0:
                    break
                    
        return orders

    def run_strategy(self, state, trader_data):
        orders = []
        conversions = 0
        logs = {}
        
        try:
            # Load state
            self._load_state(trader_data)
            logs["start_state"] = self.strategy_state
            
            # Get current data
            current_position = state.position.get(self.product, 0)
            order_depth = state.order_depths.get(self.product)
            current_price = calculate_mid_price(order_depth)
            
            obs = state.observations.conversionObservations.get(self.product, None)
            current_sunlight = None
            if obs:
                current_sunlight = getattr(obs, 'sunlightIndex', getattr(obs, 'sunlight', None))
            
            logs["position"] = current_position
            logs["price"] = current_price
            logs["sunlight"] = current_sunlight
            
            # Update history
            self._update_history(current_sunlight, current_price)
            
            # Calculate indicators
            sunlight_slope, price_ma, price_ma_slope = self._calculate_indicators()
            logs["sun_slope"] = sunlight_slope
            logs["price_ma_slope"] = price_ma_slope
            
            # Check for severe dip
            if current_sunlight is not None and current_sunlight < self.sunlight_severe_dip_threshold:
                self.had_severe_dip = True
                logs["severe_dip_detected"] = True
            
            # State machine logic
            target_pos = current_position  # Default: hold position
            
            # State transitions and logic
            if self.strategy_state == "NORMAL":
                # Check for SUN_TRIGGER entry
                if (current_sunlight is not None and sunlight_slope is not None and 
                    current_sunlight < self.sunlight_trigger_threshold and sunlight_slope < 0):
                    logger.print(f"MACARON: SUN_TRIGGER LONG activated! Sun={current_sunlight}, Slope={sunlight_slope:.1f}")
                    self.strategy_state = "SUN_TRIGGER_LONG"
                    self.had_severe_dip = False  # Reset dip flag
                    target_pos = self.position_limit  # Go long
                    logs["signal"] = "ENTER_LONG"
                else:
                    # Normal arbitrage logic
                    logs["signal"] = "NORMAL_ARB"
                    
                    # Sell order logic
                    island_ask = obs.askPrice if obs else None
                    if island_ask is not None and current_position > -self.position_limit:
                        sell_price = island_ask - 2  # Aggressive selling
                        size = min(7, self.position_limit + current_position)
                        if size > 0:
                            orders.append(Order(self.product, int(sell_price), -size))
                            logs["sell_order"] = f"{-size}@{int(sell_price)}"
                    
                    # Conversion logic - Only if position is short
                    if current_position < 0 and obs:
                        import_tariff = getattr(obs, 'importTariff', 0)
                        transport_fees = getattr(obs, 'transportFees', 0)
                        total_cost = island_ask + import_tariff + transport_fees if island_ask else None
                        
                        if total_cost is not None and current_price is not None and total_cost < current_price + 1:
                            conversions = min(self.max_conversion_per_tick, -current_position)
                            logs["conversions"] = conversions
                            logger.print(f"MACARON: Converting {conversions} at cost {total_cost:.2f}")
            
            elif self.strategy_state == "SUN_TRIGGER_LONG":
                logs["signal"] = "HOLDING_LONG"
                target_pos = self.position_limit  # Maintain full long
                
                # Check exit condition
                if (current_sunlight is not None and sunlight_slope is not None and 
                    price_ma_slope is not None and current_sunlight > self.sunlight_trigger_threshold and 
                    sunlight_slope > 0 and price_ma_slope < 0):
                    
                    logger.print(f"MACARON: Exiting LONG position. Sun={current_sunlight}, SunSlope={sunlight_slope:.1f}")
                    
                    if self.had_severe_dip:
                        logger.print("MACARON: Severe dip detected, flipping to SHORT")
                        self.strategy_state = "SUN_TRIGGER_SHORT"
                        target_pos = -self.position_limit  # Go short
                        logs["signal"] = "EXIT_LONG_TO_SHORT"
                    else:
                        self.strategy_state = "NORMAL"
                        target_pos = 0  # Liquidate
                        logs["signal"] = "EXIT_LONG_TO_NORMAL"
                    
                    self.had_severe_dip = False  # Reset flag
            
            elif self.strategy_state == "SUN_TRIGGER_SHORT":
                logs["signal"] = "HOLDING_SHORT"
                target_pos = -self.position_limit  # Maintain full short
                
                # Check exit condition (same as long exit)
                if (current_sunlight is not None and sunlight_slope is not None and 
                    price_ma_slope is not None and current_sunlight > self.sunlight_trigger_threshold and 
                    sunlight_slope <= 0 and price_ma_slope > 0):
                    
                    logger.print(f"MACARON: Exiting SHORT position. Sun={current_sunlight}, SunSlope={sunlight_slope:.1f}")
                    self.strategy_state = "NORMAL"
                    target_pos = 0  # Liquidate
                    logs["signal"] = "EXIT_SHORT_TO_NORMAL"
            
            # Generate orders if position needs adjustment
            if target_pos != current_position and self.strategy_state != "NORMAL":
                position_orders = self._generate_aggressive_orders(
                    self.product, target_pos, current_position, order_depth, self.max_slippage
                )
                orders.extend(position_orders)
                logs["position_orders"] = f"Moving {current_position} → {target_pos}"
            
            logs["end_state"] = self.strategy_state
            
        except Exception as e:
            logger.print(f"MACARON ERROR: {e}")
            logs["error"] = str(e)
            orders = []
            conversions = 0
        
        # Format return values
        final_orders = {self.product: orders} if orders else {}
        return final_orders, conversions, logs, self._save_state()
    
class BasketStrategy:
    def __init__(self):
        logger.print("Initializing BasketStrategy...")
        self.pb1 = "PICNIC_BASKET1"
        self.pb2 = "PICNIC_BASKET2"
        self.assets = [self.pb1, self.pb2]
        # --- Parameters ---
        self.position_limits = { self.pb1: 50, self.pb2: 100 }
        self.pb2_weight = 2.0
        self.pb2_offset = -1869.0
        self.spread_ma_window = 400
        self.min_valid_points_for_ma = int(self.spread_ma_window * 0.5)
        self.abs_spread_entry_threshold = 75.0
        self.ma_slope_entry_threshold = 0.001
        self.base_required_ma_slope_ticks = 5
        self.adaptive_slope_aggressiveness = 0.8
        self.abs_spread_exit_threshold = 5.0
        self.slippage_ticks = 4
        # --- History / State ---
        # Ensure history_len is sufficient for the longest MA + slope check + buffer
        self.history_len = self.spread_ma_window + self.base_required_ma_slope_ticks + 15
        self.spread_history: List[Optional[float]] = [] # Managed via traderData
        self.instant_liquidate = False # Internal flag
        logger.print("BasketStrategy Initialized.")

    # --- State Management ---
    def _load_state(self, incoming_trader_data_dict: Dict[str, Any]):
        self.spread_history = incoming_trader_data_dict.get("spread_history", [])
        # Trim excessively long history on load
        max_hist_len = self.history_len * 1.5 # Keep reasonable buffer
        if len(self.spread_history) > max_hist_len:
             self.spread_history = self.spread_history[-int(max_hist_len):]

    def _save_state(self) -> Dict[str, Any]:
        # Trim history before saving
        if len(self.spread_history) > self.history_len:
            self.spread_history = self.spread_history[-self.history_len:]
        return {"spread_history": self.spread_history}

    # --- Helper Methods (Strategy Specific) ---
    def _calculate_prices(self, state: TradingState) -> Dict[Symbol, Optional[float]]:
        """Calculate mid prices for the basket assets."""
        prices = {}
        for symbol in self.assets:
            # Use the global helper function
            prices[symbol] = calculate_mid_price(state.order_depths.get(symbol))
        return prices

    def _calculate_historical_ma(self, history: List[Optional[float]], window: int, end_index: int) -> Optional[float]:
        """Calculates the MA retroactively from history (Specific to Basket's needs)."""
        if end_index < 0 or end_index >= len(history): return None
        start_index = end_index - window + 1
        if start_index < 0: return None
        window_slice = history[start_index : end_index + 1]
        valid_points = [p for p in window_slice if p is not None]
        if len(valid_points) < self.min_valid_points_for_ma: return None
        return np.mean(valid_points)

    def _update_spread_history(self, prices: Dict[Symbol, Optional[float]]) -> Optional[float]:
        """Calculates spread, updates history. Returns current spread."""
        price_pb1 = prices.get(self.pb1)
        price_pb2 = prices.get(self.pb2)
        current_spread = None
        if price_pb1 is not None and price_pb2 is not None:
            try:
                adj_price_pb2 = self.pb2_weight * price_pb2 + self.pb2_offset
                current_spread = price_pb1 - adj_price_pb2
            except Exception: current_spread = None # Handle potential math errors

        self.spread_history.append(current_spread)
        # Trimming happens in _save_state
        return current_spread

    def _calculate_adaptive_slope_ticks(self, current_abs_spread: float) -> int:
        """Calculates adaptive required ticks for slope check."""
        spread_excess = max(0, current_abs_spread - self.abs_spread_entry_threshold)
        reduction = int(spread_excess * self.adaptive_slope_aggressiveness)
        return max(1, self.base_required_ma_slope_ticks - reduction)

    def _check_ma_slope_streak_retroactive(self, required_direction: str, length: int, slope_threshold: float, log_details: Dict[str, Any]) -> bool:
        """Checks MA slope streak retroactively."""
        min_hist_needed = self.spread_ma_window + length + 1
        if len(self.spread_history) < min_hist_needed:
            log_details['slope_check_fail'] = f"Hist<Min ({len(self.spread_history)}<{min_hist_needed})"
            return False

        slopes_checked = []
        for i in range(length):
            idx_current = len(self.spread_history) - 1 - i
            idx_previous = idx_current - 1 # Equivalent to len(self.spread_history) - 2 - i

            # Ensure previous index is valid
            if idx_previous < 0:
                 log_details['slope_check_fail'] = f"Prev Idx < 0 @ step {i+1}"
                 return False # Cannot calculate slope for the first point

            current_ma = self._calculate_historical_ma(self.spread_history, self.spread_ma_window, idx_current)
            previous_ma = self._calculate_historical_ma(self.spread_history, self.spread_ma_window, idx_previous)

            if current_ma is None or previous_ma is None:
                log_details['slope_check_fail'] = f"MA Fail @ step {i+1}"
                return False

            ma_slope = current_ma - previous_ma
            slopes_checked.append(round(ma_slope, 4))

            if required_direction == "decreasing" and not (ma_slope <= -slope_threshold):
                log_details['slope_check_fail'] = f"Decr Fail:{ma_slope:.4f}<={-slope_threshold}=F @{i+1}"
                log_details['slopes_chk'] = slopes_checked
                return False
            elif required_direction == "increasing" and not (ma_slope >= slope_threshold):
                log_details['slope_check_fail'] = f"Incr Fail:{ma_slope:.4f}>={slope_threshold}=F @{i+1}"
                log_details['slopes_chk'] = slopes_checked
                return False

        log_details.pop('slope_check_fail', None)
        log_details['slopes_chk'] = slopes_checked
        return True

    # --- Target Position Logic ---
    def _get_target_positions(self, current_positions: Dict[Symbol, int],
                              current_spread: Optional[float]) -> Tuple[Dict[Symbol, int], Dict[str, Any]]:
        """Determines target positions for the basket pair."""
        target_positions = current_positions.copy()
        pair_log_details = {}
        active_signal = "HOLD"
        target_pb1 = current_positions.get(self.pb1, 0)
        target_pb2 = current_positions.get(self.pb2, 0)
        self.instant_liquidate = False # Reset flag

        # Calculate current MA for logging
        current_ma = self._calculate_historical_ma(self.spread_history, self.spread_ma_window, len(self.spread_history) - 1)
        pair_log_details['spread'] = round(current_spread, 2) if current_spread is not None else None
        pair_log_details['spread_ma'] = round(current_ma, 2) if current_ma is not None else None

        if current_spread is not None:
            # --- Exit Logic ---
            if abs(current_spread) < self.abs_spread_exit_threshold:
                 if target_pb1 != 0 or target_pb2 != 0: # Only liquidate if holding a position
                     active_signal = f"LIQUIDATE (S|{abs(current_spread):.1f}|<{self.abs_spread_exit_threshold})"
                     target_pb1 = 0
                     target_pb2 = 0
                     self.instant_liquidate = True
                 # else: HOLD (Already flat)
            else:
                # --- Entry Logic ---
                if abs(current_spread) > self.abs_spread_entry_threshold:
                    required_ticks = self._calculate_adaptive_slope_ticks(abs(current_spread))
                    pair_log_details['req_ticks'] = required_ticks
                    slope_ok = False
                    if current_spread > 0: # Short Spread
                        slope_ok = self._check_ma_slope_streak_retroactive("decreasing", required_ticks, self.ma_slope_entry_threshold, pair_log_details)
                        if slope_ok:
                            active_signal = f"->SHORT SPREAD (S>{self.abs_spread_entry_threshold:.0f},MA Slope decr x{required_ticks})"
                            target_pb1 = -self.position_limits[self.pb1]
                            target_pb2 = +self.position_limits[self.pb2]
                    elif current_spread < 0: # Long Spread
                         slope_ok = self._check_ma_slope_streak_retroactive("increasing", required_ticks, self.ma_slope_entry_threshold, pair_log_details)
                         if slope_ok:
                            active_signal = f"->LONG SPREAD (S<{-self.abs_spread_entry_threshold:.0f},MA Slope incr x{required_ticks})"
                            target_pb1 = +self.position_limits[self.pb1]
                            target_pb2 = -self.position_limits[self.pb2]
                    # If slope_ok is False, active_signal remains HOLD

            # Apply limits
            target_positions[self.pb1] = np.clip(target_pb1, -self.position_limits[self.pb1], self.position_limits[self.pb1])
            target_positions[self.pb2] = np.clip(target_pb2, -self.position_limits[self.pb2], self.position_limits[self.pb2])
        else: # Missing spread
            active_signal = "HOLD (No Spread)"
            # Maintain positions if spread fails
            target_positions[self.pb1] = current_positions.get(self.pb1, 0)
            target_positions[self.pb2] = current_positions.get(self.pb2, 0)

        pair_log_details['pair_signal'] = active_signal
        pair_log_details[f"tgt_{self.pb1[:2].upper()}"] = target_positions[self.pb1]
        pair_log_details[f"tgt_{self.pb2[:2].upper()}"] = target_positions[self.pb2]
        return target_positions, pair_log_details

    # --- Main Execution Method ---
    def run_strategy(self, state: TradingState, incoming_trader_data_dict: Dict[str, Any]) -> Tuple[Dict[Symbol, List[Order]], int, Dict[str, Any], Dict[str, Any]]:
        """ Executes the Basket Pair trading logic. """
        final_orders: Dict[Symbol, List[Order]] = {asset: [] for asset in self.assets}
        conversions = 0
        detail_log_dict = {}

        self._load_state(incoming_trader_data_dict) # Load history

        current_positions = {asset: state.position.get(asset, 0) for asset in self.assets}
        prices = self._calculate_prices(state)
        current_spread = self._update_spread_history(prices) # Updates self.spread_history

        final_target_positions, pair_log_details = self._get_target_positions(current_positions, current_spread)
        detail_log_dict.update(pair_log_details)

        for symbol in self.assets:
            target_pos = final_target_positions.get(symbol, 0)
            current_pos = current_positions.get(symbol, 0)
            order_depth = state.order_depths.get(symbol)

            # Determine if trade is needed (target change or instant liquidate)
            trade_needed = (target_pos != current_pos) or (self.instant_liquidate and current_pos != 0)

            if trade_needed:
                # Use the global helper function for orders
                symbol_orders = generate_orders_with_slippage(symbol, target_pos, current_pos, order_depth, self.slippage_ticks)
                if symbol_orders:
                    final_orders[symbol].extend(symbol_orders)

        detail_log_dict[f"pos_{self.pb1[:2].upper()}"] = current_positions.get(self.pb1, 0)
        detail_log_dict[f"pos_{self.pb2[:2].upper()}"] = current_positions.get(self.pb2, 0)
        detail_log_dict['hist_len'] = len(self.spread_history)

        outgoing_trader_data_dict = self._save_state() # Save updated history

        final_orders_filtered = {sym: ord_list for sym, ord_list in final_orders.items() if ord_list}
        return final_orders_filtered, conversions, detail_log_dict, outgoing_trader_data_dict

class MarketMakePB2Strategy:
    def __init__(self):
        """Initializes the MarketMakePB2 Strategy."""
        logger.print("Initializing MarketMakePB2Strategy...")
        self.symbol: Symbol = "PICNIC_BASKET2"
        self.position_limit: int = 100
        self.target_volume: int = 5
        logger.print(f"Strategy: MarketMakePB2 for {self.symbol}")
        logger.print(f"Position Limit: +/- {self.position_limit}, Target Volume: {self.target_volume}")
        logger.print("MarketMakePB2Strategy Initialized.")
        # No persistent state needed

    def run_strategy(self, state: TradingState, incoming_trader_data_dict: Dict[str, Any]) -> Tuple[Dict[Symbol, List[Order]], int, Dict[str, Any], Dict[str, Any]]:
        """
        Executes the PICNIC_BASKET2 market making logic.
        Returns: orders, conversions, detail_log, new_state_dict
        """
        orders: List[Order] = []
        conversions: int = 0
        detail_log_dict: Dict[str, Any] = {}
        outgoing_trader_data_dict: Dict[str, Any] = {} # Stateless

        order_depth: Optional[OrderDepth] = state.order_depths.get(self.symbol)
        current_position: int = state.position.get(self.symbol, 0)
        detail_log_dict['pos_PB2'] = current_position

        if order_depth is None:
            detail_log_dict['status'] = "No order depth"
            return {}, conversions, detail_log_dict, outgoing_trader_data_dict

        # Get best bid/ask prices
        best_bid_price = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else None
        best_ask_price = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None
        detail_log_dict['best_bid'] = best_bid_price
        detail_log_dict['best_ask'] = best_ask_price

        # Apply mild skewing logic (simplified from marketmake.py to remove redundancy)
        bid_skew = 0
        ask_skew = 0
        if current_position > 15: # Covers > 30 as well
            bid_skew = -1
        if current_position < -15: # Covers < -30 as well
            ask_skew = 1
        detail_log_dict['bid_skew'] = bid_skew
        detail_log_dict['ask_skew'] = ask_skew

        # Place BID order
        if best_bid_price is not None:
            bid_price = int(best_bid_price + bid_skew) # Ensure integer price
            if current_position > -self.position_limit:
                buy_volume = min(self.target_volume, self.position_limit - current_position) # Max volume respecting limit
                if buy_volume > 0:
                    orders.append(Order(self.symbol, bid_price, buy_volume))
                    detail_log_dict['bid_order'] = f"{buy_volume}@{bid_price}"
                else:
                    detail_log_dict['bid_order'] = "Blocked (Pos Limit)"

        # Place ASK order
        if best_ask_price is not None:
            ask_price = int(best_ask_price + ask_skew) # Ensure integer price
            if current_position < self.position_limit:
                 sell_volume = min(self.target_volume, self.position_limit + current_position) # Max volume respecting limit
                 if sell_volume > 0:
                    orders.append(Order(self.symbol, ask_price, -sell_volume))
                    detail_log_dict['ask_order'] = f"{-sell_volume}@{ask_price}"
                 else:
                    detail_log_dict['ask_order'] = "Blocked (Neg Limit)"


        final_orders = {self.symbol: orders} if orders else {}
        return final_orders, conversions, detail_log_dict, outgoing_trader_data_dict

# --- End of MarketMakePB2Strategy class ---

class OptionsStrategy:
    # --- TTE Calculation (Specific to this strategy as per options.py) ---
    # NOTE: This uses 5.0, differing from the global 4.0 in round4v0.py
    # This is kept to preserve the EXACT logic from options.py
    def _calculate_specific_tte(self, timestamp):
        tte = (3.0 - (timestamp / 1_000_000)) / 30.0
        return max(tte, 1e-9)

    def __init__(self):
        # Keep original prints, slightly adapted for context
        logger.print("Initializing OptionsStrategy (Ported from options.py)...")
        # Strategy Parameters (Copied directly from options.py Trader.__init__)
        self.position_limit_per_option = 200
        self.order_size = 35

        # Product Info (Copied directly)
        self.underlying_symbol = "VOLCANIC_ROCK"
        self.option_symbols_dict = {
            'VOLCANIC_ROCK_VOUCHER_9750': 9750.0,
            'VOLCANIC_ROCK_VOUCHER_10000': 10000.0,
            'VOLCANIC_ROCK_VOUCHER_10250': 10250.0,
            'VOLCANIC_ROCK_VOUCHER_10500': 10500.0,
        }
        self.atm_option_symbol = "VOLCANIC_ROCK_VOUCHER_10000"
        self.atm_option_strike = 10000.0

        self.fixed_shape_coeffs = [0.7780, 0.0091] # [a, b]
        self.initial_base_iv_guess = 0.0512 # [c]

    # --- Helper Methods (Copied directly from options.py Trader) ---
    def _get_mid_price(self, symbol: Symbol, state: TradingState) -> Optional[float]:
        order_depth = state.order_depths.get(symbol)
        if not order_depth: return None
        buy_orders = order_depth.buy_orders; sell_orders = order_depth.sell_orders
        if not buy_orders or not sell_orders: return None
        best_bid = max(buy_orders.keys()); best_ask = min(sell_orders.keys())
        # Added check for crossed book from global helper - minimal safe change
        if best_bid >= best_ask: return None
        return (best_bid + best_ask) / 2.0

    def _get_best_prices(self, symbol: Symbol, state: TradingState) -> Tuple[Optional[float], Optional[float]]:
        order_depth = state.order_depths.get(symbol)
        if not order_depth: return None, None
        best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else None
        best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None
        return best_bid, best_ask

    # Note: _option_delta_call was defined but not used in options.py run method.
    # Keeping it here for exactness but it won't be called by run_strategy.
    def _option_delta_call(self, S, K, T, sigma):
        # Black-Scholes call delta: N(d1)
        if S is None or K is None or T is None or sigma is None:
            return 0.0
        try:
            # Use global BS functions defined in round4v0.py
            d1 = bs_d1(S, K, T, sigma)
            if not math.isfinite(d1):
                return 0.0
            return normal_cdf(d1) # Use global normal_cdf
        except Exception:
            return 0.0

    def _fit_vol_curve(self, state, current_St, current_TTE):
        m_list = []
        iv_list = []
        # Use global IV function defined in round4v0.py
        sqrt_TTE = math.sqrt(current_TTE + 1e-9) # Precalculate
        if sqrt_TTE < 1e-9: return None

        for symbol, strike in self.option_symbols_dict.items():
            option_mid = self._get_mid_price(symbol, state)
            # Added St > 1e-9 check for safety
            if option_mid is None or current_St is None or current_St <= 1e-9 or current_TTE <= 1e-9:
                continue

            # Use global IV function
            iv = implied_volatility_bisection(option_mid, current_St, strike, current_TTE)

            # Original filtering from options.py
            if iv is None or not math.isfinite(iv) or iv < 0.001 or iv > 2.0:
                continue
            try:
                 # Added strike > 1e-9 check for safety
                if strike <= 1e-9: continue
                log_K_S = math.log(strike / current_St)
                m = log_K_S / sqrt_TTE # Use precalculated sqrt_TTE
            except Exception:
                continue

            if math.isfinite(m):
                m_list.append(m)
                iv_list.append(iv)

        if len(m_list) >= 3:
            try: # Added try-except around polyfit
                coeffs = np.polyfit(m_list, iv_list, 2)
                # Added basic sanity check on coefficients
                if all(abs(c) < 100 for c in coeffs):
                    return np.poly1d(coeffs)
                else: return None # Fit failed - extreme coeffs
            except Exception:
                return None # Fit failed
        return None # Not enough points

    # --- Main Execution Method (WITH NEW LOGIC v2) ---
    def run_strategy(self, state: TradingState, incoming_trader_data_dict: Dict[str, Any]) -> Tuple[Dict[Symbol, List[Order]], int, Dict[str, Any], Dict[str, Any]]:
        final_orders: Dict[Symbol, List[Order]] = {}
        conversions: int = 0
        detail_log_dict = {} # For logging output this tick
        persistent_cache = {}

        # --- Load Cache ---
        if incoming_trader_data_dict and isinstance(incoming_trader_data_dict, dict):
            persistent_cache = incoming_trader_data_dict.copy()

        timestamp = state.timestamp
        detail_log_dict['ts'] = timestamp

        # --- Load cached curve ---
        last_vol_fit_ts = persistent_cache.get('last_vol_fit_ts', -1)
        cached_coeffs = persistent_cache.get('vol_curve_coeffs', None)
        vol_curve = None
        if cached_coeffs is not None and isinstance(cached_coeffs, list) and len(cached_coeffs) == 3:
             try: vol_curve = np.poly1d(cached_coeffs)
             except Exception: vol_curve = None

        # --- Calculate Current Inputs ---
        current_St = self._get_mid_price(self.underlying_symbol, state)
        # Use the TTE calculation method LOCAL to this class (5.0 constant)
        current_TTE = self._calculate_specific_tte(timestamp)
        sqrt_TTE = math.sqrt(current_TTE + 1e-9) if current_TTE > 1e-9 else 0.0
        detail_log_dict['St'] = round(current_St, 2) if current_St is not None else None
        detail_log_dict['TTE'] = round(current_TTE, 6)

        # --- Refit vol curve periodically ---
        refit = False
        # Use self.vol_refit_frequency defined in __init__
        vol_refit_frequency = getattr(self, 'vol_refit_frequency', 5000) # Default if not set
        if (vol_curve is None or (timestamp - last_vol_fit_ts >= vol_refit_frequency)) and current_St is not None and current_TTE > 1e-9:
            new_vol_curve = self._fit_vol_curve(state, current_St, current_TTE)
            if new_vol_curve is not None:
                vol_curve = new_vol_curve
                persistent_cache['vol_curve_coeffs'] = vol_curve.coefficients.tolist()
                refit = True
            persistent_cache['last_vol_fit_ts'] = timestamp

        detail_log_dict['vol_curve'] = vol_curve.coefficients.tolist() if vol_curve is not None else None
        detail_log_dict['refit'] = refit

        # --- Trading Logic ---
        net_delta = 0.0

        if current_St is not None and current_TTE > 1e-9 and vol_curve is not None:
            for symbol, strike in self.option_symbols_dict.items():
                option_log_key = f"opt_{int(strike)}"
                opt_logs = {}
                orders_placed_this_symbol = [] # Track orders for this symbol this tick
                current_pos = state.position.get(symbol, 0)
                market_bid, market_ask = self._get_best_prices(symbol, state)
                market_mid = self._get_mid_price(symbol, state)
                order_depth = state.order_depths.get(symbol) # Get order depth once

                opt_logs['pos'] = current_pos
                opt_logs['bid'] = market_bid
                opt_logs['ask'] = market_ask
                opt_logs['mid'] = market_mid

                # Calculate model price and delta
                model_price = np.nan
                option_delta = 0.0
                fitted_iv = np.nan

                try:
                    if strike > 1e-9 and sqrt_TTE > 1e-9:
                        log_K_S = math.log(strike / current_St)
                        m = log_K_S / sqrt_TTE
                        if math.isfinite(m):
                            fitted_iv = vol_curve(m)
                            fitted_iv = max(0.001, min(fitted_iv, 2.0))
                            if math.isfinite(fitted_iv):
                                model_price = bs_call_price(current_St, strike, current_TTE, fitted_iv)
                                d1 = bs_d1(current_St, strike, current_TTE, fitted_iv)
                                if math.isfinite(d1): option_delta = normal_cdf(d1)
                except Exception: pass

                opt_logs['model_px'] = round(model_price, 2) if math.isfinite(model_price) else None
                opt_logs['fitted_iv'] = round(fitted_iv, 4) if math.isfinite(fitted_iv) else None
                opt_logs['delta'] = round(option_delta, 3)

                net_delta += current_pos * option_delta

                # --- NEW ORDER LOGIC ---
                reduction_order_placed = False

                # Check if model price calculation was successful
                if not math.isfinite(model_price):
                    opt_logs['status'] = "No Model Price"
                    detail_log_dict[option_log_key] = opt_logs
                    continue # Skip trading this symbol

                # ** Priority 1: Position Reduction at Fair Value **
                if current_pos > 0 and market_bid is not None and market_bid >= model_price:
                    # Have LONG position, market BID is at or above our fair value -> SELL to reduce
                    if order_depth and market_bid in order_depth.buy_orders:
                        available_volume = order_depth.buy_orders[market_bid]
                        qty_to_sell = min(current_pos, available_volume, self.order_size)
                        if qty_to_sell > 0:
                            sell_order = Order(symbol, int(market_bid), -qty_to_sell)
                            orders_placed_this_symbol.append(sell_order)
                            opt_logs['order_reduce'] = f"REDUCE_L {-qty_to_sell}@{int(market_bid)}"
                            reduction_order_placed = True

                elif current_pos < 0 and market_ask is not None and market_ask <= model_price:
                    # Have SHORT position, market ASK is at or below our fair value -> BUY to reduce
                     if order_depth and market_ask in order_depth.sell_orders:
                        available_volume = abs(order_depth.sell_orders[market_ask])
                        qty_to_buy = min(abs(current_pos), available_volume, self.order_size)
                        if qty_to_buy > 0:
                            buy_order = Order(symbol, int(market_ask), qty_to_buy)
                            orders_placed_this_symbol.append(buy_order)
                            opt_logs['order_reduce'] = f"REDUCE_S {qty_to_buy}@{int(market_ask)}"
                            reduction_order_placed = True

                # ** Priority 2: Market Making / Taking (Only if no reduction trade was placed) **
                if not reduction_order_placed:
                    # Check if market mid price exists for comparison
                    if market_mid is not None:
                        price_diff = model_price - market_mid

                        # --- Market Making ---
                        if abs(price_diff) <= 1.0:
                            opt_logs['action'] = "MM"
                            # Place BUY order at best bid
                            if market_bid is not None and current_pos < self.position_limit_per_option:
                                qty_to_buy = min(self.order_size, self.position_limit_per_option - current_pos)
                                if qty_to_buy > 0:
                                    buy_order = Order(symbol, int(market_bid), qty_to_buy)
                                    orders_placed_this_symbol.append(buy_order)
                                    opt_logs['order_mm_buy'] = f"{qty_to_buy}@{int(market_bid)}"

                            # Place SELL order at best ask
                            if market_ask is not None and current_pos > -self.position_limit_per_option:
                                qty_to_sell = min(self.order_size, self.position_limit_per_option + current_pos)
                                if qty_to_sell > 0:
                                    sell_order = Order(symbol, int(market_ask), -qty_to_sell)
                                    orders_placed_this_symbol.append(sell_order)
                                    opt_logs['order_mm_sell'] = f"{-qty_to_sell}@{int(market_ask)}"

                        # --- Market Taking (Buy) ---
                        elif price_diff > 1.0: # Model price significantly higher than market mid
                            opt_logs['action'] = "MT_BUY"
                            if market_ask is not None and current_pos < self.position_limit_per_option:
                                if order_depth and market_ask in order_depth.sell_orders:
                                    available_volume = abs(order_depth.sell_orders[market_ask])
                                    qty_to_buy = min(self.order_size, self.position_limit_per_option - current_pos, available_volume)
                                    if qty_to_buy > 0:
                                        buy_order = Order(symbol, int(market_ask), qty_to_buy) # Buy at ask
                                        orders_placed_this_symbol.append(buy_order)
                                        opt_logs['order_mt'] = f"BUY {qty_to_buy}@{int(market_ask)}"

                        # --- Market Taking (Sell) ---
                        elif price_diff < -1.0: # Model price significantly lower than market mid
                            opt_logs['action'] = "MT_SELL"
                            if market_bid is not None and current_pos > -self.position_limit_per_option:
                                if order_depth and market_bid in order_depth.buy_orders:
                                    available_volume = order_depth.buy_orders[market_bid]
                                    qty_to_sell = min(self.order_size, self.position_limit_per_option + current_pos, available_volume)
                                    if qty_to_sell > 0:
                                        sell_order = Order(symbol, int(market_bid), -qty_to_sell) # Sell at bid
                                        orders_placed_this_symbol.append(sell_order)
                                        opt_logs['order_mt'] = f"SELL {-qty_to_sell}@{int(market_bid)}"
                    else:
                        opt_logs['status'] = "No Market Mid"

                # Add collected orders for this symbol to the main dictionary
                if orders_placed_this_symbol:
                    if symbol not in final_orders: final_orders[symbol] = []
                    final_orders[symbol].extend(orders_placed_this_symbol)

                detail_log_dict[option_log_key] = opt_logs # Add this option's log dict

            # --- Delta Hedging (Keep original logic) ---
            delta_threshold = getattr(self, 'delta_hedge_threshold', 25) # Use class attribute or default
            max_hedge_size = getattr(self, 'max_hedge_size', 200)       # Use class attribute or default
            hedge_order = None
            underlying_bid, underlying_ask = self._get_best_prices(self.underlying_symbol, state)
            hedge_qty = 0
            if abs(net_delta) > delta_threshold:
                hedge_qty = int(round(-net_delta))
                hedge_qty = max(-max_hedge_size, min(max_hedge_size, hedge_qty)) # Clamp qty
                if hedge_qty > 0 and underlying_ask is not None:
                    hedge_price = int(underlying_ask + 1) # Aggressive pricing
                    hedge_order = Order(self.underlying_symbol, hedge_price, hedge_qty)
                elif hedge_qty < 0 and underlying_bid is not None:
                    hedge_price = int(underlying_bid - 1) # Aggressive pricing
                    hedge_order = Order(self.underlying_symbol, hedge_price, hedge_qty)

                if hedge_order:
                    if self.underlying_symbol not in final_orders:
                        final_orders[self.underlying_symbol] = []
                    # Avoid adding duplicate hedge orders if symbol already has orders
                    # Check if a hedge order for the underlying already exists from option loops (unlikely but safe)
                    if not any(o.symbol == self.underlying_symbol for o_list in final_orders.values() for o in o_list):
                         final_orders[self.underlying_symbol].append(hedge_order)
                         detail_log_dict['hedge_order'] = f"{hedge_qty}@{hedge_order.price}" # Log hedge order

            detail_log_dict['net_delta'] = round(net_delta, 2)
            detail_log_dict['hedge_qty'] = hedge_qty # Log intended hedge qty

        else:
             # Log reason for no trading if inputs were missing at the start
            if current_St is None: detail_log_dict["status"] = "No Trade (No St)"
            elif current_TTE <= 1e-9: detail_log_dict["status"] = "No Trade (TTE Zero)"
            elif vol_curve is None: detail_log_dict["status"] = "No Trade (No Vol Curve)"

        # Filter empty order lists before returning
        final_orders_filtered = {sym: ord_list for sym, ord_list in final_orders.items() if ord_list}

        outgoing_trader_data_dict = persistent_cache
        return final_orders_filtered, conversions, detail_log_dict, outgoing_trader_data_dict

class Trader:
    def __init__(self):
        """ Initializes the main trader orchestrator. """
        logger.print("Initializing Master Trader...")
        self.trios_strategy = TriosStrategy()
        self.resin_strategy = ResinStrategy()
        self.ink_strategy = InkStrategy()
        self.basket_strategy = BasketStrategy()
        self.options_strategy = OptionsStrategy()
        self.macaron_strategy = MacaronStrategy()
        self.kelp_strategy = KelpStrategy()
        self.croissant_strategy = CroissantStrategy()
        self.mm_strategy = MarketMakePB2Strategy()
        logger.print("Master Trader Initialized.")
        # self.round_num = 0 # Optional tracking

    def run(self, state: TradingState) -> Tuple[Dict[Symbol, List[Order]], int, str]:
        """ Main entry point for the trading logic. """
        # self.round_num += 1
        # logger.print(f"\n--- Round {self.round_num} | Timestamp: {state.timestamp} ---")

        all_orders: Dict[Symbol, List[Order]] = {}
        total_conversions: int = 0
        combined_outgoing_state: Dict[str, Any] = {}
        all_detail_logs: Dict[str, Any] = {}

        # 1. Load and Parse Previous Combined State
        previous_state_dict: Dict[str, Any] = {}
        if state.traderData:
            try:
                previous_state_dict = json.loads(state.traderData)
            except Exception as e:
                logger.print(f"CRITICAL: Error loading traderData: {e}. Data: '{state.traderData[:200]}...'")
                previous_state_dict = {} # Reset state on error

        # 2. Run Each Strategy
        strategy_runners = {
            "TRIOS": (self.trios_strategy.run_strategy, previous_state_dict.get("trios_state", "")),
            "RESIN": (self.resin_strategy.run_strategy, ""), # Resin uses no persistent state string
            "INK": (self.ink_strategy.run_strategy, previous_state_dict.get("ink_state", {})),
            "BASKET": (self.basket_strategy.run_strategy, previous_state_dict.get("basket_state", {})),
            "OPTIONS": (self.options_strategy.run_strategy, previous_state_dict.get("options_state", {})),
            "MACARON": (self.macaron_strategy.run_strategy, previous_state_dict.get("macaron_state", {})),
            "KELP": (self.kelp_strategy.run_strategy, previous_state_dict.get("kelp_state", {})),
            "CROISSANTS": (self.croissant_strategy.run_strategy, previous_state_dict.get("croissant_state", {})),
            "PICNIC_BASKET2": (self.mm_strategy.run_strategy, previous_state_dict.get("mm_state", {}))
        }
        strategy_results = {}

        for name, (runner_func, incoming_state) in strategy_runners.items():
            try:
                orders, conv, detail_log, new_state = runner_func(state, incoming_state)
                # Store results including the state key name for saving later
                strategy_results[name] = (orders, conv, detail_log, new_state, name.lower() + "_state")
            except Exception as e:
                logger.print(f"ERROR running {name} Strategy: {e}")
                # import traceback
                # logger.print(traceback.format_exc()) # Uncomment for detailed trace

        # 3. Aggregate Results
        for strategy_name, result_tuple in strategy_results.items():
            orders, conv, detail_log, new_state, state_key = result_tuple
            for symbol, order_list in orders.items():
                if symbol not in all_orders: all_orders[symbol] = []
                all_orders[symbol].extend(order_list) # Assumes no symbol conflicts
            total_conversions += conv
            if detail_log: all_detail_logs[strategy_name] = detail_log
            if new_state or isinstance(new_state, (dict, str)): # Ensure non-empty state gets saved
                 combined_outgoing_state[state_key] = new_state

        # 5. Serialize Final Combined State
        final_trader_data_str = ""
        try:
            final_trader_data_str = json.dumps(combined_outgoing_state, default=logger.default_serializer, separators=(",", ":"))
        except Exception as e:
            logger.print(f"CRITICAL: Error serializing final combined trader data: {e}")
            final_trader_data_str = "{}"

        logger.flush(state, all_orders, total_conversions, final_trader_data_str)

        return all_orders, total_conversions, final_trader_data_str