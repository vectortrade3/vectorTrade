import sys
import json
import os
import time
import warnings
warnings.filterwarnings('ignore')
import yfinance as yf
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from datetime import datetime

BASE_DIR = Path(__file__).resolve().parent
OUT_FILE = str(BASE_DIR / 'live_predictions.json')
NIFTY_CACHE_FILE = str(BASE_DIR / 'nifty50_cache.pkl')

def calc_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))

def calc_atr(df, period=14):
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    return true_range.rolling(period).mean()

def get_nifty50_cached():
    """Cache Nifty 50 data for 15 minutes to save bandwidth and speed up low-end laptops"""
    now = time.time()
    if os.path.exists(NIFTY_CACHE_FILE):
        try:
            mtime = os.path.getmtime(NIFTY_CACHE_FILE)
            if (now - mtime) < 900: # 15 minutes
                return pd.read_pickle(NIFTY_CACHE_FILE)
        except Exception:
            pass
    try:
        nifty = yf.download('^NSEI', period='200d', progress=False)
        if isinstance(nifty.columns, pd.MultiIndex):
            nifty.columns = nifty.columns.get_level_values(0)
        nifty.to_pickle(NIFTY_CACHE_FILE)
        return nifty
    except Exception:
        if os.path.exists(NIFTY_CACHE_FILE):
            try: return pd.read_pickle(NIFTY_CACHE_FILE)
            except Exception: pass
        return None

def calculate_stock_features(df, nifty_df=None):
    df = df.copy()
    
    # 1. Moving Averages
    for ma in [15, 30, 50, 100, 150, 200]:
        df[f'MA{ma}'] = df['Close'].rolling(window=ma).mean()
        df[f'MA{ma}_slope'] = df[f'MA{ma}'].diff(1) / (df[f'MA{ma}'].shift(1) + 1e-9) * 100
        df[f'Close_vs_MA{ma}'] = df['Close'] / (df[f'MA{ma}'] + 1e-9)
        
    df['MA15_vs_MA30'] = df['MA15'] / (df['MA30'] + 1e-9)
    df['MA30_vs_MA50'] = df['MA30'] / (df['MA50'] + 1e-9)
    df['MA50_vs_MA100'] = df['MA50'] / (df['MA100'] + 1e-9)
    df['MA50_vs_MA150'] = df['MA50'] / (df['MA150'] + 1e-9)
    df['MA100_vs_MA200'] = df['MA100'] / (df['MA200'] + 1e-9)
    
    # 2. Bollinger Bands (20, 2)
    df['BB_Middle'] = df['Close'].rolling(window=20).mean()
    bb_std = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
    df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
    df['BB_Width'] = df['BB_Upper'] - df['BB_Lower']
    df['BB_Width_pct'] = df['BB_Width'] / (df['BB_Middle'] + 1e-9)
    df['BB_pctB'] = (df['Close'] - df['BB_Lower']) / (df['BB_Width'] + 1e-9)
    df['Close_dist_BB_Upper'] = df['Close'] - df['BB_Upper']
    df['Close_dist_BB_Middle'] = df['Close'] - df['BB_Middle']
    df['Close_dist_BB_Lower'] = df['Close'] - df['BB_Lower']
    df['BB_Width_change'] = df['BB_Width'].diff(1)
    
    # 3. RSI
    for r in [7, 14, 21]:
        df[f'RSI_{r}'] = calc_rsi(df['Close'], period=r)
    df['RSI_slope'] = df['RSI_14'].diff(1)
    df['RSI_change_3D'] = df['RSI_14'].diff(3)
    df['RSI_change_5D'] = df['RSI_14'].diff(5)
    
    # 4. Returns
    for d in [1, 3, 5, 7, 10, 20, 50]:
        df[f'Ret_{d}D'] = df['Close'].pct_change(d) * 100
        
    # 5. Price Structure
    for w in [5, 20]:
        df[f'{w}D_High'] = df['High'].rolling(w).max()
        df[f'{w}D_Low'] = df['Low'].rolling(w).min()
        df[f'Dist_{w}D_High'] = (df['Close'] - df[f'{w}D_High']) / (df[f'{w}D_High'] + 1e-9) * 100
        df[f'Dist_{w}D_Low'] = (df['Close'] - df[f'{w}D_Low']) / (df[f'{w}D_Low'] + 1e-9) * 100
        
    df['52W_High'] = df['High'].rolling(200).max()
    df['52W_Low'] = df['Low'].rolling(200).min()
    df['Dist_52W_High'] = (df['Close'] - df['52W_High']) / (df['52W_High'] + 1e-9) * 100
    df['Dist_52W_Low'] = (df['Close'] - df['52W_Low']) / (df['52W_Low'] + 1e-9) * 100
    
    # 6. Candlestick Features
    df['Body'] = np.abs(df['Close'] - df['Open'])
    df['Range'] = df['High'] - df['Low']
    df['Upper_Wick'] = df['High'] - np.maximum(df['Close'], df['Open'])
    df['Lower_Wick'] = np.minimum(df['Close'], df['Open']) - df['Low']
    df['Body_to_Range'] = df['Body'] / (df['Range'] + 1e-9)
    df['Upper_Wick_to_Range'] = df['Upper_Wick'] / (df['Range'] + 1e-9)
    df['Lower_Wick_to_Range'] = df['Lower_Wick'] / (df['Range'] + 1e-9)
    df['Close_Pos_in_Range'] = (df['Close'] - df['Low']) / (df['Range'] + 1e-9)
    df['Gap_pct'] = (df['Open'] - df['Close'].shift(1)) / (df['Close'].shift(1) + 1e-9) * 100
    df['Is_Hammer'] = (df['Lower_Wick'] > (2 * df['Body'])) & (df['Upper_Wick'] < (0.2 * df['Body']))
    df['Is_Inside_Candle'] = (df['High'] < df['High'].shift(1)) & (df['Low'] > df['Low'].shift(1))
    df['Is_Outside_Candle'] = (df['High'] > df['High'].shift(1)) & (df['Low'] < df['Low'].shift(1))
    
    # 7. Volume Features
    for v in [5, 20, 50]:
        df[f'Vol_MA{v}'] = df['Volume'].rolling(v).mean()
        df[f'Vol_Ratio_{v}D'] = df['Volume'] / (df[f'Vol_MA{v}'] + 1e-9)
    df['Vol_Change_1D'] = df['Volume'].pct_change(1) * 100
    df['Vol_Change_5D'] = df['Volume'].pct_change(5) * 100
    
    # OBV
    obv_dir = np.sign(df['Close'].diff().fillna(0))
    df['OBV'] = (obv_dir * df['Volume']).cumsum()
    df['OBV_slope'] = df['OBV'].diff(5)
    
    # 8. Volatility
    df['ATR14'] = calc_atr(df, 14)
    df['ATR_pct'] = df['ATR14'] / (df['Close'] + 1e-9) * 100
    for d in [5, 10, 20, 50]:
        df[f'Volat_{d}D'] = df['Ret_1D'].rolling(d).std()
    df['Volat_change'] = df['Volat_10D'].diff()
    
    # 9. Trend Structure
    df['Trend_15_30'] = df['MA15'] > df['MA30']
    df['Trend_30_50'] = df['MA30'] > df['MA50']
    df['Trend_50_100'] = df['MA50'] > df['MA100']
    df['Trend_100_200'] = df['MA100'] > df['MA200']
    df['Trend_15_30_50_Up'] = df['Trend_15_30'] & df['Trend_30_50']
    df['Trend_15_30_50_Dn'] = (df['MA15'] < df['MA30']) & (df['MA30'] < df['MA50'])
    df['MA15_cross_MA50'] = (df['MA15'].shift(1) <= df['MA50'].shift(1)) & (df['MA15'] > df['MA50'])
    df['MA100_above_MA200'] = df['MA100'] > df['MA200']
    
    # 10. Support / Resistance
    df['Support_dist'] = df['Dist_20D_Low']
    df['Resistance_dist'] = df['Dist_20D_High']
    df['Support_touches'] = (df['Low'] <= (df['20D_Low'] * 1.01)).rolling(20).sum()
    df['Resistance_touches'] = (df['High'] >= (df['20D_High'] * 0.99)).rolling(20).sum()
    
    # 11. Market Context (Nifty 50)
    if nifty_df is not None and not nifty_df.empty:
        n_close = nifty_df['Close'].reindex(df.index).ffill()
        for d in [1, 5, 7, 20]:
            df[f'Nifty_{d}D_ret'] = n_close.pct_change(d) * 100
        df['Nifty_MA50'] = n_close.rolling(50).mean()
        df['Nifty_MA200'] = n_close.rolling(200).mean()
        df['Nifty_vs_MA50'] = n_close / (df['Nifty_MA50'] + 1e-9)
        df['Nifty_vs_MA200'] = n_close / (df['Nifty_MA200'] + 1e-9)
        df['Nifty_RSI'] = calc_rsi(n_close)
        df['Nifty_Trend'] = df['Nifty_MA50'] > df['Nifty_MA200']
        for d in [5, 7, 20]:
            df[f'Rel_str_Nifty_{d}D'] = df[f'Ret_{d}D'] - df[f'Nifty_{d}D_ret']
            
    # 12. Base Strategy Signal
    df['MA50_trend_7D_up'] = (df['MA50'] > df['MA50'].shift(7)) & (df['MA50'] > df['MA50'].shift(1))
    df['Dist_Low_BB'] = df['Low'] - df['BB_Lower']
    df['BB_near_Low'] = (df['Dist_Low_BB'] >= -10) & (df['Dist_Low_BB'] <= 20)
    df['Candle_Green'] = df['Close'] > df['Open']
    df['Dist_Low_MA50'] = df['Low'] - df['MA50']
    df['Setup_1'] = (df['Dist_Low_MA50'] >= -10) & (df['Dist_Low_MA50'] <= 30)
    df['Setup_2'] = df['Dist_Low_MA50'] > 30
    df['Setup_3'] = df['Dist_Low_MA50'] < -10
    df['Base_Signal'] = df['MA50_trend_7D_up'] & df['BB_near_Low'] & df['Candle_Green'] & (df['Setup_1'] | df['Setup_2'] | df['Setup_3'])
    
    return df


COLUMN_MAPPING = {
    'Date': 'date', 'Stock Symbol': 'symbol', 'Sector': 'sector',
    'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close',
    'Adj Close': 'adjusted_close', 'Volume': 'volume',
    'MA15': 'ma15', 'MA30': 'ma30', 'MA50': 'ma50', 'MA100': 'ma100', 'MA150': 'ma150', 'MA200': 'ma200',
    'MA15_slope': 'ma15_slope', 'MA30_slope': 'ma30_slope', 'MA50_slope': 'ma50_slope',
    'MA100_slope': 'ma100_slope', 'MA150_slope': 'ma150_slope', 'MA200_slope': 'ma200_slope',
    'Close_vs_MA15': 'close_ma15_ratio', 'Close_vs_MA30': 'close_ma30_ratio', 'Close_vs_MA50': 'close_ma50_ratio',
    'Close_vs_MA100': 'close_ma100_ratio', 'Close_vs_MA150': 'close_ma150_ratio',
    'MA15_vs_MA30': 'ma15_ma30_ratio', 'MA30_vs_MA50': 'ma30_ma50_ratio',
    'MA50_vs_MA100': 'ma50_ma100_ratio', 'MA50_vs_MA150': 'ma50_ma150_ratio', 'MA100_vs_MA200': 'ma100_ma200_ratio',
    'BB_Middle': 'bb_middle', 'BB_Upper': 'bb_upper', 'BB_Lower': 'bb_lower',
    'BB_Width': 'bb_width', 'BB_Width_pct': 'bb_width_pct', 'BB_pctB': 'bb_percent_b',
    'Close_dist_BB_Upper': 'close_bb_upper_dist', 'Close_dist_BB_Middle': 'close_bb_middle_dist', 'Close_dist_BB_Lower': 'close_bb_lower_dist',
    'BB_Width_change': 'bb_width_change',
    'RSI_7': 'rsi7', 'RSI_14': 'rsi14', 'RSI_21': 'rsi21',
    'RSI_slope': 'rsi_slope', 'RSI_change_3D': 'rsi_change_3d', 'RSI_change_5D': 'rsi_change_5d',
    'Ret_1D': 'return_1d', 'Ret_3D': 'return_3d', 'Ret_5D': 'return_5d', 'Ret_7D': 'return_7d',
    'Ret_10D': 'return_10d', 'Ret_20D': 'return_20d', 'Ret_50D': 'return_50d',
    '5D_Low': 'low_5d', 'Dist_5D_High': 'dist_high_5d', 'Dist_5D_Low': 'dist_low_5d',
    'Dist_20D_High': 'dist_high_20d', 'Dist_20D_Low': 'dist_low_20d',
    'Dist_52W_High': 'dist_high_52w', 'Dist_52W_Low': 'dist_low_52w',
    'Body': 'candle_body', 'Range': 'candle_range', 'Upper_Wick': 'upper_wick', 'Lower_Wick': 'lower_wick',
    'Body_to_Range': 'body_range_ratio', 'Upper_Wick_to_Range': 'upper_wick_range_ratio', 'Lower_Wick_to_Range': 'lower_wick_range_ratio',
    'Close_Pos_in_Range': 'close_position', 'Gap_pct': 'gap_pct',
    'Is_Hammer': 'is_hammer', 'Is_Inside_Candle': 'is_inside_candle', 'Is_Outside_Candle': 'is_outside_candle',
    'Vol_MA5': 'volume_avg_5d', 'Vol_MA20': 'volume_avg_20d', 'Vol_MA50': 'volume_avg_50d',
    'Vol_Ratio_20D': 'volume_ratio_20d', 'Vol_Ratio_50D': 'volume_ratio_50d',
    'Vol_Change_1D': 'volume_change_1d', 'Vol_Change_5D': 'volume_change_5d',
    'OBV': 'obv', 'OBV_slope': 'obv_slope', 'ATR_pct': 'atr_pct',
    'Volat_5D': 'volatility_5d', 'Volat_10D': 'volatility_10d', 'Volat_20D': 'volatility_20d', 'Volat_50D': 'volatility_50d',
    'Volat_change': 'volatility_change',
    'MA100_above_MA200': 'ma100_above_ma200', 'MA15_cross_MA50': 'ma_crossover',
    'Support_touches': 'support_touches', 'Resistance_touches': 'resistance_touches',
    'Nifty_1D_ret': 'nifty_return_1d', 'Nifty_5D_ret': 'nifty_return_5d', 'Nifty_7D_ret': 'nifty_return_7d', 'Nifty_20D_ret': 'nifty_return_20d',
    'Nifty_MA50': 'nifty_ma50', 'Nifty_MA200': 'nifty_ma200', 'Nifty_vs_MA50': 'nifty_ma50_distance', 'Nifty_vs_MA200': 'nifty_ma200_distance',
    'Nifty_RSI': 'nifty_rsi', 'Nifty_Trend': 'nifty_trend',
    'Rel_str_Nifty_5D': 'relative_strength_5d', 'Rel_str_Nifty_7D': 'relative_strength_7d', 'Rel_str_Nifty_20D': 'relative_strength_20d',
    'Base_Signal': 'base_signal'
}


def run_predictor(symbol_arg: str = None):
    raw_symbol = symbol_arg if symbol_arg else (sys.argv[1].strip() if len(sys.argv) > 1 and sys.argv[1].strip() else 'RELIANCE')
    
    # Load stock metadata (names and sectors)
    stock_meta = {}
    for p in [Path('stock_names.json'), BASE_DIR / 'stock_names.json']:
        if p.exists():
            try:
                with open(p, 'r', encoding='utf-8') as f:
                    stock_meta = json.load(f)
                break
            except Exception:
                pass

    # Locate models folder (supports models/ and offline/models/)
    def get_model_file(fn):
        for p in [Path("models") / fn, Path("offline/models") / fn, Path(__file__).resolve().parent / "offline" / "models" / fn, Path(__file__).resolve().parent / "models" / fn]:
            if p.exists():
                return p
        return Path("models") / fn

    def get_feature_file(fn):
        for p in [Path(fn), Path(__file__).resolve().parent / fn]:
            if p.exists():
                return p
        return Path(fn)

    # Load Saved Models (Lightweight < 300KB each)
    try:
        strat_model = joblib.load(str(get_model_file('5_year_xgboost_model.pkl')))
        strat_scaler = joblib.load(str(get_model_file('5_year_scaler.pkl')))
        strat_imputer = joblib.load(str(get_model_file('5_year_imputer.pkl')))
        feat_path = get_feature_file('final_features.txt')
        with open(feat_path, 'r') as f:
            strat_feats = [line.strip() for line in f if line.strip()]
    except Exception as e:
        print(f"Error loading strategy models: {e}")
        sys.exit(1)

    try:
        gain_model = joblib.load(str(get_model_file('5_year_global_gain_model.pkl')))
        drop_model = joblib.load(str(get_model_file('5_year_global_drop_model.pkl')))
        glob_scaler = joblib.load(str(get_model_file('5_year_global_scaler.pkl')))
        glob_imputer = joblib.load(str(get_model_file('5_year_global_imputer.pkl')))
        glob_feat_path = get_feature_file('global_features.txt')
        with open(glob_feat_path, 'r') as f:
            glob_feats = [line.strip() for line in f if line.strip()]
        has_global = True
    except Exception as e:
        print(f"Warning: Global models not loaded: {e}")
        has_global = False

    # Determine symbols
    if raw_symbol.upper() == 'ALL':
        if os.path.exists('dataset_metadata.json'):
            with open('dataset_metadata.json') as f:
                meta = json.load(f)
            all_syms = list(meta.get('symbols', {}).keys())
            symbols = all_syms[:25] if len(all_syms) > 25 else all_syms
        else:
            symbols = ['RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'ICICIBANK', 'SBIN', 'BHARTIARTL', 'TATAMOTORS']
    else:
        clean = raw_symbol.upper().replace('.NS', '').strip()
        symbols = [clean]

    print(f"Fetching market data for {len(symbols)} symbol(s)...")

    # Fast cached Nifty 50 retrieval
    nifty = get_nifty50_cached()

    # Load existing predictions to update/prepend
    predictions = {}
    if os.path.exists(OUT_FILE):
        try:
            with open(OUT_FILE, 'r') as f:
                existing = json.load(f)
                predictions = existing.get('predictions', {})
        except Exception:
            predictions = {}

    for sym in symbols:
        try:
            yf_ticker = f"{sym}.NS"
            df = yf.download(yf_ticker, period="200d", progress=False)
            if df.empty or len(df) < 30:
                print(f"Warning: Not enough data for {sym}")
                continue

            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            df['Stock Symbol'] = sym
            df['Sector'] = 'General'
            if 'Adj Close' not in df.columns:
                df['Adj Close'] = df['Close']
            df = df.reset_index()

            # Technical Feature Calculation
            df_feat = calculate_stock_features(df, nifty)
            df_feat.rename(columns=COLUMN_MAPPING, inplace=True)

            for col in df_feat.columns:
                if col not in COLUMN_MAPPING.values() and col != 'Date':
                    df_feat.rename(columns={col: col.lower()}, inplace=True)

            if 'stock symbol' in df_feat.columns:
                df_feat.rename(columns={'stock symbol': 'symbol'}, inplace=True)
            if 'adjusted_close' not in df_feat.columns:
                df_feat['adjusted_close'] = df_feat['close']

            if 'ma_crossover' in df_feat.columns:
                crossover_id = df_feat['ma_crossover'].cumsum()
                df_feat['days_since_ma_crossover'] = df_feat.groupby(crossover_id).cumcount()

            # Ensure expected features are aligned
            for col in strat_feats:
                if col not in df_feat.columns:
                    df_feat[col] = np.nan
            if has_global:
                for col in glob_feats:
                    if col not in df_feat.columns:
                        df_feat[col] = np.nan

            df_feat.replace([np.inf, -np.inf], np.nan, inplace=True)
            last_row = df_feat.iloc[[-1]].copy()

            # 1. Strategy Model Inference
            X_strat = last_row[strat_feats]
            X_strat_s = strat_scaler.transform(strat_imputer.transform(X_strat))
            pred_r = float(strat_model.predict(X_strat_s)[0])
            is_triggered = bool(last_row.get('base_signal', pd.Series([False])).iloc[0])

            # 2. Global Model Inference (points in ₹)
            if has_global:
                X_glob = last_row[glob_feats]
                X_glob_s = glob_scaler.transform(glob_imputer.transform(X_glob))
                pred_gain_pts = max(0.5, float(gain_model.predict(X_glob_s)[0]))
                pred_drop_pts = max(0.5, float(drop_model.predict(X_glob_s)[0]))
            else:
                pred_gain_pts = 0.0
                pred_drop_pts = 0.0

            # Price data
            close_price = round(float(last_row['close'].iloc[0]), 2)
            high_price = round(float(last_row['high'].iloc[0]), 2)
            low_price = round(float(last_row['low'].iloc[0]), 2)
            atr_val = round(float(last_row.get('atr14', pd.Series([close_price * 0.015])).iloc[0]), 2)
            ma50_val = round(float(last_row.get('ma50', pd.Series([close_price])).iloc[0]), 2)
            bb_lower_val = round(float(last_row.get('bb_lower', pd.Series([close_price * 0.98])).iloc[0]), 2)

            # Percentage calculations
            gain_pct = round((pred_gain_pts / close_price) * 100, 2)
            drop_pct = round((pred_drop_pts / close_price) * 100, 2)

            # =========================================================================
            # INTEGRATED DUAL-MODEL TRADE PLAN SYNTHESIS
            # =========================================================================
            # Standard minimum risk buffer based on volatility
            default_risk = max(pred_drop_pts, 1.5 * atr_val, close_price * 0.015)

            if is_triggered:
                # Base strategy actively triggered today!
                if pred_r >= 1.0:
                    verdict = "BUY (CONFIRMED BREAKOUT)"
                    verdict_badge = "buy"
                    buy_range_low = round(close_price, 2)
                    buy_range_high = round(max(close_price * 1.004, high_price + 1.0), 2)
                    entry_ref = round((buy_range_low + buy_range_high) / 2, 2)
                    risk_amount = max(round(entry_ref - max(low_price - 1.0, entry_ref - (1.5 * atr_val)), 2), round(entry_ref * 0.015, 2))
                    sl_price = round(entry_ref - risk_amount, 2)
                    target_price = round(entry_ref + (2.0 * risk_amount), 2)
                    explanation = f"50MA + Bollinger setup triggered today. AI confirms high win potential with {pred_r:.2f}R payoff multiple."
                else:
                    verdict = "SKIP (LOW PAYOFF)"
                    verdict_badge = "skip"
                    buy_range_low = round(close_price * 0.985, 2)
                    buy_range_high = round(close_price, 2)
                    entry_ref = close_price
                    risk_amount = round(default_risk, 2)
                    sl_price = round(entry_ref - risk_amount, 2)
                    target_price = round(entry_ref + (1.0 * risk_amount), 2)
                    explanation = f"50MA setup triggered, but AI filters it out (predicted only {pred_r:.2f}R). Wait for a cleaner signal."
            else:
                # Regular market conditions - rely on Global 7-day model + technical levels
                if gain_pct >= (drop_pct * 1.25) and gain_pct >= 2.0:
                    verdict = "BUY ON PULLBACK (BULLISH)"
                    verdict_badge = "bullish"
                    # Accumulate on dip near support
                    buy_range_low = round(close_price * 0.985, 2)
                    buy_range_high = round(close_price, 2)
                    entry_ref = close_price
                    risk_amount = round(default_risk, 2)
                    sl_price = round(entry_ref - risk_amount, 2)
                    target_price = round(entry_ref + max(pred_gain_pts, 2.0 * risk_amount), 2)
                    explanation = f"Strong 7-day upward potential (+{gain_pct}%). Accumulate on minor pullbacks between Rs {buy_range_low} and Rs {buy_range_high}."
                elif drop_pct >= (gain_pct * 1.3):
                    verdict = "AVOID (HIGH DOWNSIDE)"
                    verdict_badge = "avoid"
                    buy_range_low = round(close_price * 0.95, 2)
                    buy_range_high = round(close_price * 0.97, 2)
                    entry_ref = close_price
                    risk_amount = round(default_risk, 2)
                    sl_price = round(entry_ref - risk_amount, 2)
                    target_price = round(entry_ref + max(pred_gain_pts, risk_amount), 2)
                    explanation = f"Downside risk outweighs upside (-{drop_pct}% vs +{gain_pct}%). Avoid fresh longs until support forms."
                else:
                    verdict = "NEUTRAL / RANGEBOUND"
                    verdict_badge = "neutral"
                    buy_range_low = round(close_price * 0.99, 2)
                    buy_range_high = round(close_price * 1.005, 2)
                    entry_ref = close_price
                    risk_amount = round(default_risk, 2)
                    sl_price = round(entry_ref - risk_amount, 2)
                    target_price = round(entry_ref + max(pred_gain_pts, 1.5 * risk_amount), 2)
                    explanation = f"Consolidating within a narrow band (+{gain_pct}% / -{drop_pct}%). Wait for clear breakout before entering."

            # Calculate Exact P&L Figures
            profit_per_share = round(max(1.0, target_price - entry_ref), 2)
            profit_pct = round((profit_per_share / entry_ref) * 100, 2)

            loss_per_share = round(max(1.0, entry_ref - sl_price), 2)
            loss_pct = round((loss_per_share / entry_ref) * 100, 2)

            rr_ratio = round(profit_per_share / loss_per_share, 2)
            profit_100_shares = round(profit_per_share * 100, 2)
            loss_100_shares = round(loss_per_share * 100, 2)

            date_val = str(last_row['Date'].iloc[0]) if 'Date' in last_row.columns else str(last_row['date'].iloc[0])

            # Prepend/update prediction item
            company_name = stock_meta.get(sym, {}).get('name', sym)
            sector_name = stock_meta.get(sym, {}).get('sector', 'NSE Equity')

            new_pred_item = {
                "symbol": sym,
                "company_name": company_name,
                "sector": sector_name,
                "date": date_val[:10],
                "close": close_price,
                "strategy_triggered": is_triggered,
                "predicted_r": round(pred_r, 2),
                "predicted_gain_pct": gain_pct,
                "predicted_drop_pct": drop_pct,
                "verdict": verdict,
                "verdict_badge": verdict_badge,
                "explanation": explanation,
                # Clear Price Action Levels
                "buy_range_low": buy_range_low,
                "buy_range_high": buy_range_high,
                "entry_ref": entry_ref,
                "target_price": target_price,
                "sl_price": sl_price,
                # Profit / Loss Breakdown
                "profit_per_share": profit_per_share,
                "profit_pct": profit_pct,
                "profit_100_shares": profit_100_shares,
                "loss_per_share": loss_per_share,
                "loss_pct": loss_pct,
                "loss_100_shares": loss_100_shares,
                "rr_ratio": rr_ratio,
                "updated_at": datetime.now().strftime("%I:%M %p, %d %b")
            }

            # Put newest at the front
            predictions = {sym: new_pred_item, **{k: v for k, v in predictions.items() if k != sym}}
            print(f"[OK] Predicted {sym}: Rs {close_price} | {verdict} | Buy: Rs {buy_range_low}-Rs {buy_range_high} | Target: Rs {target_price} | SL: Rs {sl_price}")

        except Exception as e:
            print(f"Error predicting {sym}: {e}")

    # Save to JSON
    out_data = {
        "timestamp": datetime.now().isoformat(),
        "predictions": predictions
    }
    with open(OUT_FILE, 'w') as f:
        json.dump(out_data, f, indent=2)

    print("Live prediction task completed successfully!")
    return predictions


if __name__ == '__main__':
    run_predictor()
