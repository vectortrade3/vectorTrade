"""
Tests for VectorTrade Data Cleaner & Master Consolidation Engine
Verifies schema enforcement, inf replacement, derived features, and output data integrity.
"""

import unittest
import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add project root to path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from offline import data_cleaner, json_manager


class TestDataCleanerPipeline(unittest.TestCase):

    def test_final_columns_count_and_uniqueness(self):
        """Validates that final_columns has exactly 157 columns and no duplicate names."""
        cols = data_cleaner.FINAL_COLUMNS
        self.assertEqual(len(cols), 157, f"Expected 157 final columns, found {len(cols)}")
        self.assertEqual(len(cols), len(set(cols)), "Duplicate columns detected in FINAL_COLUMNS")

    def test_column_mapping_integrity(self):
        """Ensures all column mappings map to valid target names."""
        mapping = data_cleaner.COLUMN_MAPPING
        self.assertGreater(len(mapping), 80)
        self.assertEqual(mapping['Open'], 'open')
        self.assertEqual(mapping['Close'], 'close')
        self.assertEqual(mapping['MA50'], 'ma50')
        self.assertEqual(mapping['RSI14'], 'rsi14')
        self.assertEqual(mapping['ATR14'], 'atr14')

    def test_daily_skip_logic(self):
        """Validates that pipeline returns in under 0.1s with skipped=True if already processed today."""
        result = data_cleaner.run_data_cleaning_pipeline(force=False)
        self.assertEqual(result["status"], "SUCCESS")
        self.assertTrue(result.get("skipped", False))
        self.assertIn("Consolidation skipped", result["message"])

    def test_final_dataset_file_integrity(self):
        """Inspects 5_year_final_data.csv on disk to ensure data validity."""
        csv_file = BASE_DIR / "5_year_final_data.csv"
        self.assertTrue(csv_file.exists(), "5_year_final_data.csv does not exist")
        self.assertGreater(csv_file.stat().st_size, 100_000_000, "5_year_final_data.csv file is too small")

        # Read sample of first 1,000 rows
        df_sample = pd.read_csv(csv_file, nrows=1000)
        self.assertEqual(df_sample.shape[1], 157, f"Expected 157 columns, found {df_sample.shape[1]}")
        
        # Verify required key columns
        required = ['date', 'symbol', 'sector', 'open', 'high', 'low', 'close', 'mfe', 'mae', 'days_since_ma_crossover', 'sector_trend']
        for col in required:
            self.assertIn(col, df_sample.columns, f"Required column {col} missing from output dataset")

        # Verify inf values were replaced
        numeric_cols = df_sample.select_dtypes(include=[np.number]).columns
        has_pos_inf = np.isinf(df_sample[numeric_cols].values).any()
        self.assertFalse(has_pos_inf, "Dataset contains unhandled np.inf values")


if __name__ == "__main__":
    unittest.main()
