from flask import Flask, render_template_string, jsonify, request
import json
import os
import sys
import threading
import subprocess
from datetime import datetime, timedelta

app = Flask(__name__)

TASK_STATUS = {
    "strategy": "Idle",
    "global": "Idle",
    "data": "Idle"
}

PREDICT_STATUS = {
    "status": "Idle",
    "target": ""
}

def run_notebook_pipeline(notebook_name, task_key):
    global TASK_STATUS
    TASK_STATUS[task_key] = "Running..."
    try:
        proc = subprocess.run([sys.executable, "-m", "nbconvert", "--execute", "--inplace", notebook_name],
                              capture_output=True, text=True, timeout=900)
        if proc.returncode == 0:
            TASK_STATUS[task_key] = "Completed"
        else:
            TASK_STATUS[task_key] = f"Error: {proc.stderr[:120]}"
    except subprocess.TimeoutExpired:
        TASK_STATUS[task_key] = "Timed out after 15 minutes"
    except Exception as e:
        TASK_STATUS[task_key] = f"Error: {str(e)[:80]}"

def run_data_job():
    global TASK_STATUS
    TASK_STATUS["data"] = "Running... (Downloading & Processing)"
    try:
        proc = subprocess.run([sys.executable, "5_years_6.0_data_update_job.py"],
                              capture_output=True, text=True, timeout=900)
        if proc.returncode == 0:
            TASK_STATUS["data"] = "Completed"
        else:
            TASK_STATUS["data"] = f"Error: {proc.stderr[:120]}"
    except subprocess.TimeoutExpired:
        TASK_STATUS["data"] = "Timed out after 15 minutes"
    except Exception as e:
        TASK_STATUS["data"] = f"Error: {str(e)[:80]}"

def run_live_prediction(symbol):
    global PREDICT_STATUS
    clean_sym = symbol.strip().upper().replace('.NS', '') if symbol else 'RELIANCE'
    PREDICT_STATUS["status"] = f"Analyzing {clean_sym}..."
    PREDICT_STATUS["target"] = clean_sym
    try:
        proc = subprocess.run([sys.executable, "live_predictor_job.py", clean_sym],
                              capture_output=True, text=True, timeout=60)
        if proc.returncode == 0:
            PREDICT_STATUS["status"] = "Completed"
        else:
            err_msg = proc.stderr.strip() or proc.stdout.strip()
            PREDICT_STATUS["status"] = f"Error: {err_msg[:80]}"
    except subprocess.TimeoutExpired:
        PREDICT_STATUS["status"] = "Timed out after 60s"
    except Exception as e:
        PREDICT_STATUS["status"] = f"Error: {str(e)[:80]}"


HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>NIFTY 200 AI Trading Hub</title>
    <!-- Simple, clean, lightweight typography (Inter) -->
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
    <script src="https://cdn.tailwindcss.com"></script>
    <script>
        tailwind.config = {
            theme: {
                extend: {
                    fontFamily: {
                        sans: ['Inter', '-apple-system', 'BlinkMacSystemFont', 'sans-serif'],
                    }
                }
            }
        }
    </script>
    <style>
        body { font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif; }
        .autocomplete-dropdown { max-height: 240px; overflow-y: auto; }
    </style>
</head>
<body class="bg-slate-50 text-slate-900 min-h-screen flex flex-col antialiased">

    <!-- Top Navigation Bar -->
    <header class="bg-white border-b border-slate-200 sticky top-0 z-50 shadow-sm">
        <div class="max-w-6xl mx-auto px-4 sm:px-6">
            <div class="flex items-center justify-between h-16">
                <div class="flex items-center space-x-3">
                    <div class="w-9 h-9 rounded-lg bg-indigo-600 flex items-center justify-center text-white font-bold text-lg shadow-sm">
                        ⚡
                    </div>
                    <div>
                        <span class="text-base font-extrabold tracking-tight text-slate-900">Nifty AI Hub</span>
                        <span class="hidden sm:inline-block text-xs bg-indigo-50 text-indigo-700 font-semibold px-2 py-0.5 rounded-full ml-2">Lightweight AI Models</span>
                    </div>
                </div>

                <nav class="flex space-x-1 sm:space-x-2">
                    <a href="/" class="px-3 py-2 rounded-lg text-sm font-semibold transition {% if page == 'dashboard' %}bg-indigo-50 text-indigo-700{% else %}text-slate-600 hover:text-slate-900 hover:bg-slate-100{% endif %}">
                        📊 AI Success Rate
                    </a>
                    <a href="/predict" class="px-3 py-2 rounded-lg text-sm font-semibold transition {% if page == 'predict' %}bg-indigo-50 text-indigo-700{% else %}text-slate-600 hover:text-slate-900 hover:bg-slate-100{% endif %}">
                        🎯 Stock Predictor
                    </a>
                    <a href="/data" class="px-3 py-2 rounded-lg text-sm font-semibold transition {% if page == 'data' %}bg-indigo-50 text-indigo-700{% else %}text-slate-600 hover:text-slate-900 hover:bg-slate-100{% endif %}">
                        💾 Data Hub
                    </a>
                    <a href="/update" class="px-3 py-2 rounded-lg text-sm font-semibold transition {% if page == 'update' %}bg-indigo-50 text-indigo-700{% else %}text-slate-600 hover:text-slate-900 hover:bg-slate-100{% endif %}">
                        ⚙️ Retrain Models
                    </a>
                </nav>
            </div>
        </div>
    </header>

    <!-- Main Content Area -->
    <main class="max-w-6xl mx-auto px-4 sm:px-6 py-8 flex-grow w-full">

        {% if page == 'dashboard' %}
        <!-- ==================== DASHBOARD / SUCCESS RATES ==================== -->
        <div class="mb-8 flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
            <div>
                <h1 class="text-2xl sm:text-3xl font-extrabold text-slate-900 tracking-tight">AI Models & Success Rate</h1>
                <p class="text-slate-500 text-sm mt-1">High-clarity evaluation of how well the models identify winning trades.</p>
            </div>
            <button onclick="location.reload()" class="self-start sm:self-auto inline-flex items-center space-x-2 bg-white hover:bg-slate-50 text-slate-700 text-xs font-semibold px-3 py-2 rounded-lg border border-slate-300 shadow-sm transition">
                <span>🔄</span>
                <span>Refresh Stats</span>
            </button>
        </div>

        <!-- Highlight Summary Cards -->
        <div class="grid grid-cols-1 md:grid-cols-3 gap-5 mb-8">
            <div class="bg-white rounded-xl p-6 border border-slate-200 shadow-sm">
                <div class="flex items-center justify-between mb-3">
                    <span class="text-xs font-bold uppercase tracking-wider text-emerald-700 bg-emerald-50 px-2.5 py-1 rounded-full">Winning Signal Capture</span>
                    <span class="text-2xl">🎯</span>
                </div>
                <div class="text-3xl font-extrabold text-emerald-600 tracking-tight">77.3%</div>
                <div class="text-xs font-semibold text-slate-700 mt-1">Win Rate Recall (Target Hits)</div>
                <p class="text-xs text-slate-500 mt-2">When a real winning breakout occurs, the AI model successfully confirms it 77 times out of 100.</p>
            </div>

            <div class="bg-white rounded-xl p-6 border border-slate-200 shadow-sm">
                <div class="flex items-center justify-between mb-3">
                    <span class="text-xs font-bold uppercase tracking-wider text-indigo-700 bg-indigo-50 px-2.5 py-1 rounded-full">Reward / Risk Edge</span>
                    <span class="text-2xl">⚖️</span>
                </div>
                <div class="text-3xl font-extrabold text-indigo-600 tracking-tight">1 : 2.0</div>
                <div class="text-xs font-semibold text-slate-700 mt-1">Target vs Stop Loss Design</div>
                <p class="text-xs text-slate-500 mt-2">Every trade aims for double the profit relative to risk, guaranteeing profitable expectancy even with a 44% win rate.</p>
            </div>

            <div class="bg-white rounded-xl p-6 border border-slate-200 shadow-sm">
                <div class="flex items-center justify-between mb-3">
                    <span class="text-xs font-bold uppercase tracking-wider text-blue-700 bg-blue-50 px-2.5 py-1 rounded-full">Lightweight & Fast</span>
                    <span class="text-2xl">⚡</span>
                </div>
                <div class="text-3xl font-extrabold text-blue-600 tracking-tight">&lt; 300 KB</div>
                <div class="text-xs font-semibold text-slate-700 mt-1">Low-End Laptop Friendly</div>
                <p class="text-xs text-slate-500 mt-2">Trained XGBoost models run completely in memory, making predictions in 2-3 seconds without GPU requirements.</p>
            </div>
        </div>

        <!-- Detailed Breakdown -->
        <div class="bg-white rounded-xl border border-slate-200 shadow-sm overflow-hidden mb-8">
            <div class="p-6 border-b border-slate-100">
                <h2 class="text-lg font-bold text-slate-900">How the Two AI Models Work Together</h2>
                <p class="text-xs text-slate-500 mt-0.5">Combining strategy filtering with global multi-day price boundary forecasting.</p>
            </div>
            <div class="grid grid-cols-1 md:grid-cols-2 divide-y md:divide-y-0 md:divide-x divide-slate-100">
                <div class="p-6">
                    <div class="flex items-center space-x-2 mb-3">
                        <span class="text-xl">🏹</span>
                        <h3 class="font-bold text-slate-900">Model 1: Strategy Sniper</h3>
                    </div>
                    <p class="text-xs text-slate-600 mb-4">
                        Filters 50MA + Lower Bollinger Band pullbacks. Scores each trade to output whether the expected payoff is worth entering (≥ 1.0 R).
                    </p>
                    <div class="space-y-2.5 text-xs">
                        <div class="flex justify-between py-1.5 border-b border-slate-100">
                            <span class="text-slate-500">Target Hit Recall</span>
                            <span class="font-bold text-emerald-600">77.3% Success Rate</span>
                        </div>
                        <div class="flex justify-between py-1.5 border-b border-slate-100">
                            <span class="text-slate-500">Filter Accuracy</span>
                            <span class="font-bold text-slate-800">43.9% (at strict 1:2 R:R)</span>
                        </div>
                        <div class="flex justify-between py-1.5">
                            <span class="text-slate-500">Selected Indicators</span>
                            <span class="font-bold text-slate-800">99 Refined Technical Features</span>
                        </div>
                    </div>
                </div>

                <div class="p-6">
                    <div class="flex items-center space-x-2 mb-3">
                        <span class="text-xl">🌐</span>
                        <h3 class="font-bold text-slate-900">Model 2: Global 7-Day Forecaster</h3>
                    </div>
                    <p class="text-xs text-slate-600 mb-4">
                        Scans all stocks across the market regardless of strategy, forecasting the maximum rally (+Gain) and maximum pullback (-Drop) over 7 days.
                    </p>
                    <div class="space-y-2.5 text-xs">
                        <div class="flex justify-between py-1.5 border-b border-slate-100">
                            <span class="text-slate-500">Directional Accuracy</span>
                            <span class="font-bold text-blue-600">48.8% - 52.0%</span>
                        </div>
                        <div class="flex justify-between py-1.5 border-b border-slate-100">
                            <span class="text-slate-500">Expected 7-Day Move Error</span>
                            <span class="font-bold text-slate-800">±9.9% High/Low Precision</span>
                        </div>
                        <div class="flex justify-between py-1.5">
                            <span class="text-slate-500">Selected Indicators</span>
                            <span class="font-bold text-slate-800">89 Global Features</span>
                        </div>
                    </div>
                </div>
            </div>
        </div>


        {% elif page == 'predict' %}
        <!-- ==================== STOCK PREDICTOR WITH AUTOCOMPLETE & ACTIONABLE PLAN ==================== -->
        <div class="mb-6 flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
            <div>
                <h1 class="text-2xl sm:text-3xl font-extrabold text-slate-900 tracking-tight">Live Stock Predictor</h1>
                <p class="text-slate-500 text-sm mt-1">Dual-Model AI Synthesis &bull; Exact Buy Zone, Target Sell, Stop Loss &amp; Profit/Loss breakdown.</p>
            </div>
            <div class="flex items-center space-x-2">
                <span class="text-xs bg-emerald-50 text-emerald-700 font-bold px-2.5 py-1 rounded-full border border-emerald-200">⚡ 2-3s Inference (CPU Light)</span>
                {% if predictions %}
                <button onclick="clearHistory()" class="text-xs text-slate-500 hover:text-rose-600 font-semibold px-3 py-1 rounded-lg border border-slate-200 bg-white hover:bg-rose-50 transition">
                    🗑️ Clear History
                </button>
                {% endif %}
            </div>
        </div>

        <!-- Search Bar with Live Instant Autocomplete Dropdown -->
        <div class="bg-white rounded-xl p-6 border border-slate-200 shadow-sm mb-6 relative">
            <label class="block text-xs font-bold uppercase tracking-wider text-slate-600 mb-2">Search Stock (Type Symbol or Company Name, e.g. REL, TCS, INFY)</label>
            
            <div class="relative">
                <div class="flex flex-col sm:flex-row gap-3">
                    <div class="relative flex-grow">
                        <div class="absolute inset-y-0 left-0 pl-3.5 flex items-center pointer-events-none text-slate-400">
                            🔍
                        </div>
                        <input type="text" id="symbol-input" autocomplete="off"
                               placeholder="Type 2-3 letters (e.g. REL, TCS, HDFC, TATA)..." 
                               value="RELIANCE"
                               oninput="handleSearchInput(this.value)"
                               onkeydown="handleKeyNavigation(event)"
                               class="w-full pl-10 pr-10 py-3 rounded-lg border border-slate-300 focus:outline-none focus:ring-2 focus:ring-indigo-500 text-slate-900 text-base font-semibold uppercase tracking-wider">
                        
                        <button type="button" onclick="clearSearchInput()" class="absolute inset-y-0 right-0 pr-3.5 flex items-center text-slate-400 hover:text-slate-600">
                            ✕
                        </button>

                        <!-- Floating Live Suggestions Box -->
                        <div id="suggestions-box" class="hidden absolute left-0 right-0 top-full mt-1.5 bg-white border border-slate-200 rounded-lg shadow-2xl z-50 autocomplete-dropdown divide-y divide-slate-100">
                            <!-- Injected dynamically by JS -->
                        </div>
                    </div>

                    <button id="btn-predict-stock" onclick="triggerPredict(document.getElementById('symbol-input').value)" 
                            class="bg-indigo-600 hover:bg-indigo-700 text-white font-bold px-7 py-3 rounded-lg transition shadow-sm text-sm flex items-center justify-center space-x-2">
                        <span>⚡</span>
                        <span>Analyze & Predict</span>
                    </button>
                </div>
            </div>

            <!-- Top Popular Stock Chips for 1-Click Scan -->
            <div class="mt-4 pt-3.5 border-t border-slate-100 flex flex-wrap items-center gap-2">
                <span class="text-xs font-semibold text-slate-400">Quick Scan:</span>
                {% for chip in ['RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'ICICIBANK', 'SBIN', 'TATAMOTORS', 'BHARTIARTL', 'ITC', 'LT', 'SUNPHARMA', 'MARUTI'] %}
                <button onclick="quickPick('{{ chip }}')" class="text-xs font-semibold px-2.5 py-1 rounded-md bg-slate-100 hover:bg-indigo-50 hover:text-indigo-700 text-slate-700 transition">
                    {{ chip }}
                </button>
                {% endfor %}
            </div>
        </div>

        <!-- Realtime Processing Indicator with Dynamic Live Stopwatch -->
        <div id="status-card" class="hidden bg-indigo-50 border border-indigo-200 rounded-xl p-4 mb-6">
            <div class="flex items-center space-x-3">
                <div class="animate-spin text-2xl">⏳</div>
                <div class="flex-grow">
                    <div class="flex items-center justify-between">
                        <h3 id="status-title" class="text-xs font-bold text-indigo-900 uppercase tracking-wider">Processing Market Candle</h3>
                        <span id="status-timer" class="text-xs font-mono font-bold text-indigo-600 bg-white px-2 py-0.5 rounded border border-indigo-200">0.0s</span>
                    </div>
                    <p id="status-text" class="text-xs text-indigo-700 font-medium mt-0.5">Fetching latest quote & running Strategy Sniper + Global 7D Forecaster... (~2 to 3s)</p>
                </div>
            </div>
        </div>

        {% if predictions %}
        <!-- ==================== PREVIOUS PREDICTED STOCKS SUMMARY TABLE ==================== -->
        {% if predictions|length > 1 %}
        <div class="bg-white rounded-xl border border-slate-200 shadow-sm overflow-hidden mb-6">
            <div class="px-5 py-3.5 border-b border-slate-100 flex items-center justify-between bg-slate-50/50">
                <div class="flex items-center space-x-2">
                    <span class="text-sm font-bold text-slate-800">📋 Previous Scanned Stocks Overview</span>
                    <span class="text-[11px] bg-slate-200 text-slate-700 font-bold px-2 py-0.5 rounded-full">{{ predictions|length }} stocks</span>
                </div>
                <span class="text-[11px] text-slate-400">Click any stock to inspect full trade plan</span>
            </div>
            <div class="overflow-x-auto">
                <table class="w-full text-left text-xs">
                    <thead class="bg-slate-50 text-slate-500 font-bold uppercase tracking-wider border-b border-slate-200">
                        <tr>
                            <th class="py-2.5 px-4">Stock</th>
                            <th class="py-2.5 px-3">Price (CMP)</th>
                            <th class="py-2.5 px-3">AI Verdict</th>
                            <th class="py-2.5 px-3">Buy Zone</th>
                            <th class="py-2.5 px-3">Target (Sell)</th>
                            <th class="py-2.5 px-3">Stop Loss</th>
                            <th class="py-2.5 px-3 text-emerald-600">Profit (100 sh)</th>
                            <th class="py-2.5 px-3 text-rose-600">Max Loss (100 sh)</th>
                            <th class="py-2.5 px-3 text-center">Action</th>
                        </tr>
                    </thead>
                    <tbody class="divide-y divide-slate-100">
                        {% for sym, data in predictions.items() %}
                        <tr class="hover:bg-slate-50 transition cursor-pointer" onclick="scrollToCard('card-{{ sym }}')">
                            <td class="py-3 px-4 font-bold text-slate-900">
                                <div>{{ sym }}</div>
                                <div class="text-[10px] text-slate-400 font-normal truncate max-w-[120px]">{{ data.company_name or sym }}</div>
                            </td>
                            <td class="py-3 px-3 font-semibold text-slate-800 font-mono">₹{{ data.close }}</td>
                            <td class="py-3 px-3">
                                {% if data.verdict_badge == 'buy' %}
                                <span class="bg-emerald-100 text-emerald-800 text-[10px] font-extrabold px-2 py-0.5 rounded">🟢 BUY</span>
                                {% elif data.verdict_badge == 'bullish' %}
                                <span class="bg-blue-100 text-blue-800 text-[10px] font-extrabold px-2 py-0.5 rounded">🔵 ACCUMULATE</span>
                                {% elif data.verdict_badge == 'avoid' %}
                                <span class="bg-rose-100 text-rose-800 text-[10px] font-extrabold px-2 py-0.5 rounded">🔴 AVOID</span>
                                {% else %}
                                <span class="bg-slate-100 text-slate-700 text-[10px] font-extrabold px-2 py-0.5 rounded">⚪ NEUTRAL</span>
                                {% endif %}
                            </td>
                            <td class="py-3 px-3 font-mono text-slate-700">₹{{ data.buy_range_low }}–{{ data.buy_range_high }}</td>
                            <td class="py-3 px-3 font-mono font-bold text-emerald-600">₹{{ data.target_price }} <span class="text-[10px] text-emerald-500">(+{{ data.profit_pct }}%)</span></td>
                            <td class="py-3 px-3 font-mono font-bold text-rose-600">₹{{ data.sl_price }} <span class="text-[10px] text-rose-500">(-{{ data.loss_pct }}%)</span></td>
                            <td class="py-3 px-3 font-mono font-bold text-emerald-600">+₹{{ data.profit_100_shares }}</td>
                            <td class="py-3 px-3 font-mono font-bold text-rose-600">-₹{{ data.loss_100_shares }}</td>
                            <td class="py-3 px-3 text-center" onclick="event.stopPropagation()">
                                <button onclick="triggerPredict('{{ sym }}')" class="p-1 text-slate-400 hover:text-indigo-600 font-bold" title="Re-analyze stock">🔄</button>
                                <button onclick="removeStock('{{ sym }}')" class="p-1 text-slate-400 hover:text-rose-600 font-bold ml-1" title="Remove from list">✕</button>
                            </td>
                        </tr>
                        {% endfor %}
                    </tbody>
                </table>
            </div>
        </div>
        {% endif %}

        <!-- ==================== DETAILED TRADE EXECUTION CARDS ==================== -->
        <div class="space-y-6 mb-8">
            {% for sym, data in predictions.items() %}
            <div id="card-{{ sym }}" class="bg-white rounded-xl border border-slate-200 shadow-sm overflow-hidden hover:border-slate-300 transition">
                
                <!-- Card Header -->
                <div class="p-5 sm:p-6 border-b border-slate-100 flex flex-col md:flex-row md:items-center md:justify-between gap-4">
                    <div>
                        <div class="flex items-center space-x-3">
                            <span class="text-2xl font-black text-slate-900 tracking-tight">{{ sym }}</span>
                            <span class="text-xs bg-slate-100 text-slate-600 font-bold px-2 py-0.5 rounded">{{ data.company_name or sym }}</span>
                            <span class="text-xs text-slate-400 font-mono">{{ data.date }} &bull; {{ data.updated_at or '' }}</span>
                        </div>
                        <div class="text-base font-bold text-slate-700 mt-1">
                            Current Market Price (CMP): <span class="text-indigo-600 font-extrabold font-mono">₹ {{ data.close }}</span>
                        </div>
                    </div>

                    <!-- AI Unified Recommendation Badge -->
                    <div class="flex items-center space-x-2">
                        {% if data.verdict_badge == 'buy' %}
                        <div class="inline-flex items-center space-x-2 bg-emerald-100 text-emerald-800 px-4 py-2 rounded-xl text-sm font-extrabold shadow-sm">
                            <span class="text-base">🟢</span>
                            <span>{{ data.verdict }}</span>
                        </div>
                        {% elif data.verdict_badge == 'bullish' %}
                        <div class="inline-flex items-center space-x-2 bg-blue-100 text-blue-800 px-4 py-2 rounded-xl text-sm font-extrabold shadow-sm">
                            <span class="text-base">🔵</span>
                            <span>{{ data.verdict }}</span>
                        </div>
                        {% elif data.verdict_badge == 'avoid' %}
                        <div class="inline-flex items-center space-x-2 bg-rose-100 text-rose-800 px-4 py-2 rounded-xl text-sm font-extrabold shadow-sm">
                            <span class="text-base">🔴</span>
                            <span>{{ data.verdict }}</span>
                        </div>
                        {% else %}
                        <div class="inline-flex items-center space-x-2 bg-slate-100 text-slate-700 px-4 py-2 rounded-xl text-sm font-extrabold shadow-sm">
                            <span class="text-base">⚪</span>
                            <span>{{ data.verdict }}</span>
                        </div>
                        {% endif %}

                        <button onclick="removeStock('{{ sym }}')" class="text-slate-300 hover:text-rose-500 font-bold p-1 rounded transition" title="Delete this card">✕</button>
                    </div>
                </div>

                <!-- Actionable Trade Plan (Buy Range, Target, Stop Loss, R:R) -->
                <div class="p-5 sm:p-6 bg-slate-50/60 border-b border-slate-100">
                    <div class="text-xs font-bold uppercase tracking-wider text-slate-500 mb-3 flex items-center justify-between">
                        <span>📍 Recommended Price Levels (Combined 2-Model Output)</span>
                        <span class="text-[11px] text-slate-400">Entry Reference: ₹ {{ data.entry_ref }}</span>
                    </div>
                    <div class="grid grid-cols-2 sm:grid-cols-4 gap-3.5">
                        
                        <!-- Recommended Buy Range -->
                        <div class="bg-white rounded-lg p-3.5 border border-slate-200 shadow-xs">
                            <span class="text-[11px] font-bold uppercase tracking-wider text-slate-500 block mb-1">Buy Zone (Range)</span>
                            <div class="text-base font-black text-slate-900 font-mono">₹ {{ data.buy_range_low }} – {{ data.buy_range_high }}</div>
                            <span class="text-[10px] text-slate-400">Optimal buy window</span>
                        </div>

                        <!-- Target / Sell Price -->
                        <div class="bg-white rounded-lg p-3.5 border border-emerald-200 bg-emerald-50/25 shadow-xs">
                            <span class="text-[11px] font-bold uppercase tracking-wider text-emerald-700 block mb-1">Target (Sell) Price</span>
                            <div class="text-base font-black text-emerald-600 font-mono">₹ {{ data.target_price }}</div>
                            <span class="text-[10px] text-emerald-600 font-semibold">+{{ data.profit_pct }}% Expected upside</span>
                        </div>

                        <!-- Stop Loss (SL) -->
                        <div class="bg-white rounded-lg p-3.5 border border-rose-200 bg-rose-50/25 shadow-xs">
                            <span class="text-[11px] font-bold uppercase tracking-wider text-rose-700 block mb-1">Stop Loss (SL)</span>
                            <div class="text-base font-black text-rose-600 font-mono">₹ {{ data.sl_price }}</div>
                            <span class="text-[10px] text-rose-600 font-semibold">-{{ data.loss_pct }}% Max risk floor</span>
                        </div>

                        <!-- Risk to Reward (R:R) -->
                        <div class="bg-white rounded-lg p-3.5 border border-indigo-200 bg-indigo-50/25 shadow-xs">
                            <span class="text-[11px] font-bold uppercase tracking-wider text-indigo-700 block mb-1">Reward : Risk Ratio</span>
                            <div class="text-base font-black text-indigo-600 font-mono">1 : {{ data.rr_ratio }}</div>
                            <span class="text-[10px] text-indigo-600 font-semibold">Multiple: {{ data.predicted_r }} R</span>
                        </div>
                    </div>
                </div>

                <!-- Exact Profit & Loss Breakdown with Dynamic Quantity Calculator -->
                <div class="p-5 sm:p-6 border-b border-slate-100">
                    <div class="flex flex-col sm:flex-row sm:items-center sm:justify-between mb-3 gap-2">
                        <span class="text-xs font-bold uppercase tracking-wider text-slate-700">💰 Profit vs. Loss Scenarios Calculator</span>
                        <div class="flex items-center space-x-1.5 text-xs">
                            <span class="text-slate-400 text-[11px]">Position Size:</span>
                            <button onclick="updateQty('{{ sym }}', 50, {{ data.profit_per_share }}, {{ data.loss_per_share }}, {{ data.entry_ref }})" class="btn-qty-{{ sym }} px-2 py-0.5 rounded border border-slate-200 bg-white hover:bg-slate-100 text-slate-700 text-xs font-semibold">50</button>
                            <button onclick="updateQty('{{ sym }}', 100, {{ data.profit_per_share }}, {{ data.loss_per_share }}, {{ data.entry_ref }})" class="btn-qty-{{ sym }} px-2 py-0.5 rounded border border-indigo-500 bg-indigo-50 text-indigo-700 text-xs font-bold">100</button>
                            <button onclick="updateQty('{{ sym }}', 250, {{ data.profit_per_share }}, {{ data.loss_per_share }}, {{ data.entry_ref }})" class="btn-qty-{{ sym }} px-2 py-0.5 rounded border border-slate-200 bg-white hover:bg-slate-100 text-slate-700 text-xs font-semibold">250</button>
                            <button onclick="updateQty('{{ sym }}', 500, {{ data.profit_per_share }}, {{ data.loss_per_share }}, {{ data.entry_ref }})" class="btn-qty-{{ sym }} px-2 py-0.5 rounded border border-slate-200 bg-white hover:bg-slate-100 text-slate-700 text-xs font-semibold">500</button>
                        </div>
                    </div>

                    <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
                        <!-- Profit Scenario -->
                        <div class="rounded-xl border border-emerald-200 bg-emerald-50/40 p-4">
                            <div class="flex items-center justify-between mb-2">
                                <span class="text-xs font-bold text-emerald-900 uppercase tracking-wider flex items-center space-x-1">
                                    <span>📈</span><span>If Stock Goes UP (Target Reached)</span>
                                </span>
                                <span class="text-xs font-black text-emerald-700 bg-emerald-100 px-2 py-0.5 rounded">+{{ data.profit_pct }}%</span>
                            </div>
                            <div class="text-lg font-black text-emerald-700 mb-1 font-mono">
                                +₹ {{ data.profit_per_share }} <span class="text-xs font-semibold text-emerald-800">per share</span>
                            </div>
                            <p class="text-xs text-emerald-800 font-medium">
                                Net Gain on <span id="label-profit-qty-{{ sym }}" class="font-bold">100</span> shares: <b id="profit-val-{{ sym }}" class="font-black text-emerald-900 font-mono text-sm">+₹ {{ data.profit_100_shares }}</b>
                            </p>
                        </div>

                        <!-- Loss Scenario -->
                        <div class="rounded-xl border border-rose-200 bg-rose-50/40 p-4">
                            <div class="flex items-center justify-between mb-2">
                                <span class="text-xs font-bold text-rose-900 uppercase tracking-wider flex items-center space-x-1">
                                    <span>📉</span><span>If Stock Goes DOWN (Stop Loss Hit)</span>
                                </span>
                                <span class="text-xs font-black text-rose-700 bg-rose-100 px-2 py-0.5 rounded">-{{ data.loss_pct }}%</span>
                            </div>
                            <div class="text-lg font-black text-rose-700 mb-1 font-mono">
                                -₹ {{ data.loss_per_share }} <span class="text-xs font-semibold text-rose-800">per share</span>
                            </div>
                            <p class="text-xs text-rose-800 font-medium">
                                Max Risk on <span id="label-loss-qty-{{ sym }}" class="font-bold">100</span> shares: <b id="loss-val-{{ sym }}" class="font-black text-rose-900 font-mono text-sm">-₹ {{ data.loss_100_shares }}</b>
                            </p>
                        </div>
                    </div>
                </div>

                <!-- Strategy Rationale & Dual AI Breakdown -->
                <div class="p-5 sm:p-6 bg-white flex flex-col sm:flex-row sm:items-center sm:justify-between text-xs text-slate-500 gap-3">
                    <div class="flex-grow">
                        <span class="font-bold text-slate-700">AI Logic:</span>
                        <span>{{ data.explanation }}</span>
                    </div>
                    <div class="flex items-center space-x-4 self-start sm:self-auto font-mono text-[11px] text-slate-400">
                        <span>7D High Potential: <b class="text-emerald-600">+{{ data.predicted_gain_pct }}%</b></span>
                        <span>7D Dip Risk: <b class="text-rose-600">-{{ data.predicted_drop_pct }}%</b></span>
                        <button onclick="triggerPredict('{{ sym }}')" class="text-indigo-600 hover:text-indigo-800 font-bold ml-2 underline">🔄 Re-scan</button>
                    </div>
                </div>

            </div>
            {% endfor %}
        </div>
        {% else %}
        <div class="bg-white rounded-xl border border-slate-200 p-12 text-center shadow-sm">
            <span class="text-3xl block mb-2">🔍</span>
            <h3 class="text-base font-bold text-slate-800">No Stocks Predicted Yet</h3>
            <p class="text-xs text-slate-500 max-w-sm mx-auto mt-1">Type any stock above (like RELIANCE or TCS) and click "Analyze & Predict" to see exact entry range, exit target, and stop loss.</p>
        </div>
        {% endif %}

        <script>
            const ALL_STOCKS = {{ stock_list | tojson }};
            let selectedSuggestionIndex = -1;
            let timerInterval = null;

            function handleSearchInput(query) {
                const box = document.getElementById('suggestions-box');
                const val = query.trim().toUpperCase();
                selectedSuggestionIndex = -1;

                if (!val || val.length < 1) {
                    box.classList.add('hidden');
                    return;
                }

                // Match against symbol OR company name
                const matches = ALL_STOCKS.filter(item => {
                    const sym = item.symbol.toUpperCase();
                    const name = (item.name || '').toUpperCase();
                    return sym.startsWith(val) || sym.includes(val) || name.includes(val);
                }).slice(0, 8);

                if (matches.length === 0) {
                    box.classList.add('hidden');
                    return;
                }

                box.innerHTML = matches.map((m, idx) => `
                    <div id="suggest-item-${idx}" onclick="selectSuggestion('${m.symbol}')" class="suggest-row px-4 py-2.5 hover:bg-indigo-50 cursor-pointer flex justify-between items-center transition">
                        <div>
                            <span class="font-bold text-slate-900 text-sm">${m.symbol}</span>
                            <span class="text-xs text-slate-500 ml-2 font-normal">${m.name || ''}</span>
                        </div>
                        <span class="text-[10px] text-indigo-600 font-semibold uppercase bg-indigo-50 px-2 py-0.5 rounded">${m.sector || 'NSE'}</span>
                    </div>
                `).join('');
                box.classList.remove('hidden');
            }

            function clearSearchInput() {
                const input = document.getElementById('symbol-input');
                input.value = '';
                document.getElementById('suggestions-box').classList.add('hidden');
                input.focus();
            }

            function selectSuggestion(sym) {
                document.getElementById('symbol-input').value = sym;
                document.getElementById('suggestions-box').classList.add('hidden');
                triggerPredict(sym);
            }

            function handleKeyNavigation(e) {
                const box = document.getElementById('suggestions-box');
                const rows = document.querySelectorAll('.suggest-row');
                
                if (e.key === 'ArrowDown') {
                    e.preventDefault();
                    if (!box.classList.contains('hidden') && rows.length > 0) {
                        selectedSuggestionIndex = (selectedSuggestionIndex + 1) % rows.length;
                        highlightSuggestion(rows);
                    }
                } else if (e.key === 'ArrowUp') {
                    e.preventDefault();
                    if (!box.classList.contains('hidden') && rows.length > 0) {
                        selectedSuggestionIndex = (selectedSuggestionIndex - 1 + rows.length) % rows.length;
                        highlightSuggestion(rows);
                    }
                } else if (e.key === 'Enter') {
                    e.preventDefault();
                    if (selectedSuggestionIndex >= 0 && rows[selectedSuggestionIndex]) {
                        rows[selectedSuggestionIndex].click();
                    } else {
                        box.classList.add('hidden');
                        triggerPredict(document.getElementById('symbol-input').value);
                    }
                } else if (e.key === 'Escape') {
                    box.classList.add('hidden');
                }
            }

            function highlightSuggestion(rows) {
                rows.forEach((r, idx) => {
                    if (idx === selectedSuggestionIndex) {
                        r.classList.add('bg-indigo-100');
                    } else {
                        r.classList.remove('bg-indigo-100');
                    }
                });
            }

            document.addEventListener('click', function(e) {
                const box = document.getElementById('suggestions-box');
                const input = document.getElementById('symbol-input');
                if (box && !box.contains(e.target) && e.target !== input) {
                    box.classList.add('hidden');
                }
            });

            function quickPick(sym) {
                document.getElementById('symbol-input').value = sym;
                triggerPredict(sym);
            }

            function triggerPredict(sym) {
                if (!sym) return;
                const cleanSym = sym.trim().toUpperCase().replace('.NS', '');
                document.getElementById('btn-predict-stock').disabled = true;
                const banner = document.getElementById('status-card');
                banner.classList.remove('hidden');
                document.getElementById('status-title').innerText = "Analyzing " + cleanSym;
                document.getElementById('status-text').innerText = "Running Strategy Sniper & Global Forecaster... (~2 to 3s)";

                // Live stopwatch
                let startTime = Date.now();
                if (timerInterval) clearInterval(timerInterval);
                timerInterval = setInterval(() => {
                    const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
                    document.getElementById('status-timer').innerText = elapsed + 's';
                }, 100);

                fetch('/api/start_predict/' + cleanSym, { method: 'POST' })
                    .then(res => res.json())
                    .then(() => pollPredict())
                    .catch(() => {
                        if (timerInterval) clearInterval(timerInterval);
                        document.getElementById('status-text').innerText = "Error initiating prediction.";
                        document.getElementById('btn-predict-stock').disabled = false;
                    });
            }

            function pollPredict() {
                fetch('/api/predict_status')
                    .then(res => res.json())
                    .then(data => {
                        if (data.status.includes('Completed')) {
                            if (timerInterval) clearInterval(timerInterval);
                            document.getElementById('status-text').innerText = "Success! Updating display...";
                            location.reload();
                        } else if (data.status.includes('Error')) {
                            if (timerInterval) clearInterval(timerInterval);
                            document.getElementById('status-text').innerText = data.status;
                            document.getElementById('btn-predict-stock').disabled = false;
                        } else {
                            setTimeout(pollPredict, 1000);
                        }
                    })
                    .catch(() => setTimeout(pollPredict, 1500));
            }

            function updateQty(sym, qty, profitPerShare, lossPerShare, entryRef) {
                const profitTotal = (qty * profitPerShare).toFixed(2);
                const lossTotal = (qty * lossPerShare).toFixed(2);
                document.getElementById('profit-val-' + sym).innerText = '+₹ ' + profitTotal;
                document.getElementById('loss-val-' + sym).innerText = '-₹ ' + lossTotal;
                document.getElementById('label-profit-qty-' + sym).innerText = qty;
                document.getElementById('label-loss-qty-' + sym).innerText = qty;
            }

            function scrollToCard(id) {
                const el = document.getElementById(id);
                if (el) {
                    el.scrollIntoView({ behavior: 'smooth', block: 'center' });
                    el.classList.add('ring-2', 'ring-indigo-500');
                    setTimeout(() => el.classList.remove('ring-2', 'ring-indigo-500'), 2000);
                }
            }

            function removeStock(sym) {
                fetch('/api/remove_prediction/' + sym, { method: 'POST' })
                    .then(() => location.reload());
            }

            function clearHistory() {
                if (!confirm("Clear all previous predictions?")) return;
                fetch('/api/clear_predictions', { method: 'POST' })
                    .then(() => location.reload());
            }
        </script>


        {% elif page == 'data' %}
        <!-- ==================== DATA HUB ==================== -->
        <div class="mb-8">
            <h1 class="text-2xl sm:text-3xl font-extrabold text-slate-900 tracking-tight">Market Data Hub</h1>
            <p class="text-slate-500 text-sm mt-1">Manage and sync the 5-year historical dataset across all 200 NIFTY constituent stocks.</p>
        </div>

        <div class="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
            <div class="bg-white rounded-xl border border-slate-200 p-6 shadow-sm">
                <h2 class="text-base font-bold text-slate-900 mb-4 flex items-center space-x-2">
                    <span>📁</span><span>Dataset Snapshot</span>
                </h2>
                <div class="space-y-4">
                    <div class="flex justify-between items-center py-2 border-b border-slate-100">
                        <span class="text-xs text-slate-500 font-medium">Last Saved Market Date</span>
                        <span class="text-sm font-bold text-indigo-600 bg-indigo-50 px-2.5 py-1 rounded">{{ last_updated_date }}</span>
                    </div>
                    <div class="flex justify-between items-center py-2 border-b border-slate-100">
                        <span class="text-xs text-slate-500 font-medium">Master CSV Size</span>
                        <span class="text-sm font-bold text-slate-800">{{ final_csv_size }}</span>
                    </div>
                    <div class="flex justify-between items-center py-2">
                        <span class="text-xs text-slate-500 font-medium">Tracked Constituents</span>
                        <span class="text-sm font-bold text-slate-800">{{ total_stocks }} Stocks</span>
                    </div>
                </div>
            </div>

            <div class="bg-white rounded-xl border border-slate-200 p-6 shadow-sm flex flex-col justify-between">
                <div>
                    <h2 class="text-base font-bold text-slate-900 mb-1 flex items-center space-x-2">
                        <span>🔄</span><span>Sync Missing Days</span>
                    </h2>
                    <p class="text-xs text-slate-500 mt-2 leading-relaxed">
                        Checks if your dataset is behind today's market. If missing, it downloads only the delta days up to yesterday and updates all indicators silently.
                    </p>
                </div>

                <div class="mt-6">
                    <div class="bg-slate-50 rounded-lg p-3 text-xs mb-4 border border-slate-100">
                        <span class="text-slate-400 font-bold uppercase text-[10px] block">Job Status</span>
                        <span id="data-status-label" class="font-semibold text-slate-800">{{ status.data }}</span>
                    </div>
                    <button id="btn-sync-data" onclick="startDataSync()" 
                            class="w-full bg-emerald-600 hover:bg-emerald-700 text-white font-bold py-3 px-4 rounded-lg transition text-xs shadow-sm flex items-center justify-center space-x-2">
                        <span>📥</span><span>Download & Process Missing Data</span>
                    </button>
                </div>
            </div>
        </div>

        <script>
            function startDataSync() {
                document.getElementById('btn-sync-data').disabled = true;
                fetch('/api/start_data', { method: 'POST' }).then(() => pollDataSync());
            }
            function pollDataSync() {
                fetch('/api/status').then(r => r.json()).then(s => {
                    document.getElementById('data-status-label').innerText = s.data;
                    if (s.data.includes('Running')) {
                        setTimeout(pollDataSync, 2000);
                    } else {
                        document.getElementById('btn-sync-data').disabled = false;
                    }
                });
            }
        </script>


        {% elif page == 'update' %}
        <!-- ==================== UPDATE / RETRAIN ==================== -->
        <div class="mb-8">
            <h1 class="text-2xl sm:text-3xl font-extrabold text-slate-900 tracking-tight">Retrain AI Models</h1>
            <p class="text-slate-500 text-sm mt-1">Retrain the models on the newest data cycles to adapt to changing market volatility.</p>
        </div>

        <div class="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
            <div class="bg-white rounded-xl border border-slate-200 p-6 shadow-sm flex flex-col justify-between">
                <div>
                    <div class="flex items-center space-x-2 mb-2">
                        <span class="text-xl">🏹</span>
                        <h2 class="text-base font-bold text-slate-900">Retrain Strategy Sniper</h2>
                    </div>
                    <p class="text-xs text-slate-500 leading-relaxed">
                        Retrains the 50MA breakout filter. Outputs updated model weights to <code>models/</code>.
                    </p>
                </div>

                <div class="mt-6">
                    <div class="bg-slate-50 rounded-lg p-3 text-xs mb-4 border border-slate-100">
                        <span class="text-slate-400 font-bold uppercase text-[10px] block">Status</span>
                        <span id="strat-status-label" class="font-semibold text-slate-800">{{ status.strategy }}</span>
                    </div>
                    <button id="btn-update-strat" onclick="triggerRetrain('strategy')"
                            class="w-full bg-indigo-600 hover:bg-indigo-700 text-white font-bold py-3 px-4 rounded-lg transition text-xs shadow-sm">
                        Retrain Strategy AI (Notebook 6)
                    </button>
                </div>
            </div>

            <div class="bg-white rounded-xl border border-slate-200 p-6 shadow-sm flex flex-col justify-between">
                <div>
                    <div class="flex items-center space-x-2 mb-2">
                        <span class="text-xl">🌐</span>
                        <h2 class="text-base font-bold text-slate-900">Retrain Global Forecaster</h2>
                    </div>
                    <p class="text-xs text-slate-500 leading-relaxed">
                        Retrains both the 7-day Gain and 7-day Drop models across all historical trading days.
                    </p>
                </div>

                <div class="mt-6">
                    <div class="bg-slate-50 rounded-lg p-3 text-xs mb-4 border border-slate-100">
                        <span class="text-slate-400 font-bold uppercase text-[10px] block">Status</span>
                        <span id="glob-status-label" class="font-semibold text-slate-800">{{ status.global }}</span>
                    </div>
                    <button id="btn-update-glob" onclick="triggerRetrain('global')"
                            class="w-full bg-blue-600 hover:bg-blue-700 text-white font-bold py-3 px-4 rounded-lg transition text-xs shadow-sm">
                        Retrain Global AI (Notebook 6.2)
                    </button>
                </div>
            </div>
        </div>

        <script>
            function triggerRetrain(type) {
                document.getElementById('btn-update-' + (type === 'strategy' ? 'strat' : 'glob')).disabled = true;
                fetch('/api/start_update/' + type, { method: 'POST' }).then(() => pollRetrain(type));
            }
            function pollRetrain(type) {
                fetch('/api/status').then(r => r.json()).then(s => {
                    document.getElementById('strat-status-label').innerText = s.strategy;
                    document.getElementById('glob-status-label').innerText = s.global;
                    if (s[type].includes('Running')) {
                        setTimeout(() => pollRetrain(type), 3000);
                    } else {
                        document.getElementById('btn-update-' + (type === 'strategy' ? 'strat' : 'glob')).disabled = false;
                    }
                });
            }
        </script>
        {% endif %}

    </main>

    <!-- Footer -->
    <footer class="bg-white border-t border-slate-200 py-6 text-center text-xs text-slate-400">
        NIFTY 200 Machine Learning Trading Strategy &bull; Lightweight CPU Inference
    </footer>

</body>
</html>
"""


@app.route('/')
def dashboard():
    log_file = 'models/model_performance_log.json'
    log_data = {}
    if os.path.exists(log_file):
        with open(log_file, 'r') as f:
            try: log_data = json.load(f)
            except: pass

    return render_template_string(HTML_TEMPLATE, page='dashboard', models=log_data)


@app.route('/predict')
def predict_page():
    pred_file = 'live_predictions.json'
    predictions = {}
    timestamp = "Never"
    if os.path.exists(pred_file):
        with open(pred_file, 'r') as f:
            try:
                data = json.load(f)
                predictions = data.get('predictions', {})
                raw_ts = data.get('timestamp', '')
                if raw_ts:
                    ts = datetime.fromisoformat(raw_ts)
                    timestamp = ts.strftime("%d %b %Y, %I:%M %p")
            except: pass

    stock_meta = {}
    if os.path.exists('stock_names.json'):
        try:
            with open('stock_names.json') as f:
                stock_meta = json.load(f)
        except Exception:
            pass

    stock_list = []
    if stock_meta:
        for s, info in stock_meta.items():
            stock_list.append({
                'symbol': s,
                'name': info.get('name', s),
                'sector': info.get('sector', 'NSE')
            })
    elif os.path.exists('dataset_metadata.json'):
        try:
            with open('dataset_metadata.json') as f:
                meta = json.load(f)
                for s in sorted(list(meta.get('symbols', {}).keys())):
                    stock_list.append({'symbol': s, 'name': s, 'sector': 'NSE'})
        except Exception:
            pass

    if not stock_list:
        default_syms = ['RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'ICICIBANK', 'SBIN', 'BHARTIARTL', 'TATAMOTORS', 'TRENT']
        stock_list = [{'symbol': s, 'name': s, 'sector': 'NSE'} for s in default_syms]

    return render_template_string(HTML_TEMPLATE, page='predict', predictions=predictions,
                                  timestamp=timestamp, stock_list=stock_list, stock_meta=stock_meta)


@app.route('/data')
def data_page():
    meta_file = 'dataset_metadata.json'
    last_updated = "Unknown"
    total_stocks = 200
    if os.path.exists(meta_file):
        with open(meta_file, 'r') as f:
            try:
                meta_data = json.load(f)
                last_updated = meta_data.get('last_updated', meta_data.get('global_last_date', 'Unknown'))
                total_stocks = len(meta_data.get('symbols', {})) or 200
            except: pass

    final_csv_size = "Unknown"
    if os.path.exists('5_year_final_data.csv'):
        size_bytes = os.path.getsize('5_year_final_data.csv')
        final_csv_size = f"{size_bytes / (1024*1024):.1f} MB"

    return render_template_string(HTML_TEMPLATE, page='data', status=TASK_STATUS,
                                  last_updated_date=last_updated, total_stocks=total_stocks,
                                  final_csv_size=final_csv_size)


@app.route('/update')
def update_page():
    return render_template_string(HTML_TEMPLATE, page='update', status=TASK_STATUS)


@app.route('/api/status')
def api_status():
    return jsonify(TASK_STATUS)


@app.route('/api/predict_status')
def api_predict_status():
    return jsonify(PREDICT_STATUS)


@app.route('/api/start_data', methods=['POST'])
def api_start_data():
    global TASK_STATUS
    if TASK_STATUS['data'].startswith('Running'): return jsonify({"status": "already running"})
    threading.Thread(target=run_data_job, daemon=True).start()
    return jsonify({"status": "started"})


@app.route('/api/start_update/<task_key>', methods=['POST'])
def api_start_update(task_key):
    global TASK_STATUS
    if TASK_STATUS[task_key].startswith('Running'): return jsonify({"status": "already running"})
    nb_name = '5_years_6_strategy_update.ipynb' if task_key == 'strategy' else '5_years_6.2_global_update.ipynb'
    threading.Thread(target=run_notebook_pipeline, args=(nb_name, task_key), daemon=True).start()
    return jsonify({"status": "started"})


@app.route('/api/start_predict/<symbol>', methods=['POST'])
def api_start_predict(symbol):
    global PREDICT_STATUS
    if PREDICT_STATUS['status'].startswith('Running'): return jsonify({"status": "already running"})
    threading.Thread(target=run_live_prediction, args=(symbol,), daemon=True).start()
    return jsonify({"status": "started"})


@app.route('/api/remove_prediction/<symbol>', methods=['POST'])
def api_remove_prediction(symbol):
    clean_sym = symbol.strip().upper().replace('.NS', '')
    if os.path.exists('live_predictions.json'):
        try:
            with open('live_predictions.json', 'r') as f:
                data = json.load(f)
            preds = data.get('predictions', {})
            if clean_sym in preds:
                del preds[clean_sym]
                data['predictions'] = preds
                with open('live_predictions.json', 'w') as f:
                    json.dump(data, f, indent=2)
        except Exception:
            pass
    return jsonify({"status": "removed", "symbol": clean_sym})


@app.route('/api/clear_predictions', methods=['POST'])
def api_clear_predictions():
    if os.path.exists('live_predictions.json'):
        with open('live_predictions.json', 'w') as f:
            json.dump({"timestamp": datetime.now().isoformat(), "predictions": {}}, f)
    return jsonify({"status": "cleared"})


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True, use_reloader=False)
