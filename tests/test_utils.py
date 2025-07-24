"""
Unit tests for utility functions
"""

import unittest
import numpy as np
import pandas as pd
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.utils import create_page_matrix, select_rank_by_energy, calculate_metrics


class TestUtils(unittest.TestCase):
    """Test utility functions."""
    
    def setUp(self):
        """Set up test data."""
        self.test_ts = np.arange(12)  # [0, 1, 2, ..., 11]
        self.L = 3
        
    def test_create_page_matrix(self):
        """Test page matrix creation."""
        page_matrix = create_page_matrix(self.test_ts, self.L)
        
        # Check shape
        expected_shape = (3, 4)  # L=3, T//L=12//3=4
        self.assertEqual(page_matrix.shape, expected_shape)
        
        # Check values
        expected_first_col = [0, 1, 2]
        np.testing.assert_array_equal(page_matrix[:, 0], expected_first_col)
        
    def test_page_matrix_error_handling(self):
        """Test page matrix error cases."""
        with self.assertRaises(ValueError):
            create_page_matrix(np.array([1, 2]), 5)  # L > T
            
    def test_select_rank_by_energy(self):
        """Test rank selection by energy."""
        # Create test singular values
        s = np.array([10, 5, 2, 1])  # Decreasing singular values
        
        # Test 99% threshold
        rank = select_rank_by_energy(s, threshold=0.99)
        self.assertIsInstance(rank, int)
        self.assertGreater(rank, 0)
        self.assertLessEqual(rank, len(s))
        
    def test_calculate_metrics(self):
        """Test metrics calculation."""
        # Create test data
        actual = pd.Series([1, 2, 3, 4, 5])
        predicted = pd.Series([1.1, 1.9, 3.1, 3.9, 5.1])
        
        metrics = calculate_metrics(actual, predicted)
        
        # Check that all metrics are calculated
        expected_keys = ['mse', 'rmse', 'mae', 'mape', 'valid_predictions']
        for key in expected_keys:
            self.assertIn(key, metrics)
            self.assertIsInstance(metrics[key], (int, float))
        
        # Check that metrics are reasonable
        self.assertGreater(metrics['mse'], 0)
        self.assertEqual(metrics['valid_predictions'], 5)
        
    def test_calculate_metrics_with_nan(self):
        """Test metrics calculation with NaN values."""
        actual = pd.Series([1, 2, np.nan, 4, 5])
        predicted = pd.Series([1.1, np.nan, 3.1, 3.9, 5.1])
        
        metrics = calculate_metrics(actual, predicted)
        
        # Should have 3 valid predictions (indices 0, 3, 4)
        self.assertEqual(metrics['valid_predictions'], 3)
        self.assertGreater(metrics['mse'], 0)


if __name__ == '__main__':
    unittest.main()