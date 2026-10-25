# 🚀 NIFTY 200 Trading AI: Official User Guide

Welcome to your automated Machine Learning trading factory! This system is designed to download stock market data, calculate hundreds of complex technical indicators, and train two highly advanced Artificial Intelligence "Brains" to predict future stock prices. 

This guide explains exactly how to use the files in this folder.

---

## 🧠 The Two AI Brains

You have two completely separate AI systems at your disposal:

1. **The Strategy Sniper (Files 4 & 5):** 
   * **What it does:** It *only* looks at trades where your specific 50MA Breakout strategy triggered. 
   * **Goal:** It predicts exactly how high a breakout will go before it crashes, telling you whether to hold for a 1:2 profit, take profit early at 1:1, or skip the trade entirely.

2. **The Global Forecaster (Files 4.2 & 5.2):** 
   * **What it does:** It looks at *every single stock on every single day*, regardless of any strategy.
   * **Goal:** It predicts the Maximum % Gain and the Maximum % Drop over the next 7 days, telling you if a stock is a "Strong Buy" or if you should "Avoid" it.

---

## 🛠️ Phase 1: First-Time Setup (Building the Database)
*If you haven't downloaded any data yet, or if you ever want to completely wipe the system and rebuild it from scratch, run these three files in order:*

* **Step 1:** Open `5_years_1_download.ipynb` and click **Run All**. It will download 5 years of daily history for all 200 NIFTY stocks.
* **Step 2:** Open `5_years_2_feature_generation.ipynb` and click **Run All**. It calculates 130+ technical indicators (RSI, Moving Averages, etc.) for every stock.
* **Step 3:** Open `5_years_3_data_cleaning.ipynb` and click **Run All**. It cleans up the math, combines everything, and outputs your master database: `5_year_final_data.csv`.

*(Note: You only need to run Files 1, 2, and 3 manually once!)*

---

## 🎯 Phase 2: Training & Testing the AI
*Once your database is built, you can train and test your AI models.*

### To use the Strategy Sniper:
* Open `5_years_4_model_training.ipynb` and **Run All**. The AI will train itself, fine-tune its settings, and save its brain into the `models/` folder.
* Open `5_years_5_inference.ipynb` and **Run All**. This will test the AI on unseen data, show you its accuracy percentage, and give you live trade advice.

### To use the Global Forecaster:
* Open `5_years_4.2_global_model_training.ipynb` and **Run All**. The AI will train two separate high/low models and save them.
* Open `5_years_5.2_global_inference.ipynb` and **Run All**. It will grade its own directional accuracy and show you percentage-based forecasts.

---

## 🔄 Phase 3: The Daily / Weekly Routine (Auto-Updating)
*The stock market changes every day. You don't have to manually run Steps 1-5 ever again. Instead, you just use the "Auto-Updaters".*

* **To update your Strategy AI:** Open `5_years_6_strategy_update.ipynb` and click **Run All**.
* **To update your Global AI:** Open `5_years_6.2_global_update.ipynb` and click **Run All**.

**What the Updaters do automatically:**
1. Check the date of your last saved data.
2. Download only the missing days up to yesterday.
3. Stitch the new data to the bottom of your files.
4. Run Notebooks 2 and 3 silently in the background to update the math.
5. Run Notebooks 4 and 5 silently in the background to retrain the AI on the new data.

Just run Notebook 6 or 6.2, grab a cup of coffee, and when it finishes, open Notebook 5 or 5.2 to see the newly updated market forecasts!

---

## 📦 Using the AI in Production (Live Bots)
Whenever the AI trains (Notebooks 4 and 4.2), it saves its brain into the **`models/`** folder as `.pkl` files. 

If you ever want to connect this AI to a live broker (like Zerodha or Angel One) or a website, you don't need to copy any notebooks. You just need to copy the `models/` folder to your live server and load it using the Python `joblib` library!
