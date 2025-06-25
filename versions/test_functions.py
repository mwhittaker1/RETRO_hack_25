import unittest
import pandas as pd
import numpy as np
import logging
from functions import ReturnsClusteringAnalysis

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("TestReturnsClusteringAnalysis")

class TestReturnsClusteringAnalysis(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        logger.info("==== Starting Test Suite for ReturnsClusteringAnalysis ====")

    @classmethod
    def tearDownClass(cls):
        logger.info("==== Finished Test Suite for ReturnsClusteringAnalysis ====")

    def setUp(self):
        logger.info("\n--- Setting up for a new test case ---")
        # Use real sample data from RETRO_SAMPLE.xlsx
        self.df = pd.read_excel("RETRO_SAMPLE - Copy.xlsx")
        # Ensure all required columns exist for the analysis
        required_cols = [
            'CUSTOMER_EMAILID', 'SALES_ORDER_NO', 'RETURN_NO', 'RETURN_QTY',
            'ORDER_DATE', 'RETURN_DATE', 'SKU', 'SALES_QTY'
        ]
        for col in required_cols:
            if col not in self.df.columns:
                # Add missing columns with default values
                if col in ['RETURN_QTY', 'SALES_QTY']:
                    self.df[col] = 1
                else:
                    self.df[col] = ''
        self.analyzer = ReturnsClusteringAnalysis(self.df)

    def tearDown(self):
        logger.info("--- Finished test case ---\n")

    def test_prepare_customer_features(self):
        logger.info("[test_prepare_customer_features] Starting test.")
        features = self.analyzer.prepare_customer_features()
        logger.info(f"[test_prepare_customer_features] Output sample:\n{features.head()}")
        self.assertIsInstance(features, pd.DataFrame)
        self.assertIn('CUSTOMER_EMAILID', features.columns)
        logger.info("[test_prepare_customer_features] Passed.")

    def test_find_optimal_clusters(self):
        logger.info("[test_find_optimal_clusters] Starting test.")
        # Use positional argument instead of keyword if required by the function signature
        optimal_k, scores = self.analyzer.find_optimal_clusters(3)
        logger.info(f"[test_find_optimal_clusters] optimal_k: {optimal_k}, scores sample: {dict(list(scores.items())[:2])}")
        self.assertIsInstance(optimal_k, int)
        self.assertIsInstance(scores, dict)
        logger.info("[test_find_optimal_clusters] Passed.")

    def test_perform_clustering(self):
        logger.info("[test_perform_clustering] Starting test.")
        features = self.analyzer.prepare_customer_features()
        labels, centers = self.analyzer.perform_clustering(n_clusters=2)
        logger.info(f"[test_perform_clustering] labels sample: {labels[:3]}, centers shape: {centers.shape}")
        self.assertEqual(len(labels), len(features))
        self.assertTrue(isinstance(centers, np.ndarray))
        logger.info("[test_perform_clustering] Passed.")

    def test_analyze_clusters(self):
        logger.info("[test_analyze_clusters] Starting test.")
        self.analyzer.perform_clustering(n_clusters=2)
        summary, interpretations = self.analyzer.analyze_clusters()
        logger.info(f"[test_analyze_clusters] summary sample:\n{summary.head()}")
        logger.info(f"[test_analyze_clusters] interpretations sample: {dict(list(interpretations.items())[:1])}")
        self.assertIsInstance(summary, pd.DataFrame)
        self.assertIsInstance(interpretations, dict)
        logger.info("[test_analyze_clusters] Passed.")

    def test_export_results(self):
        logger.info("[test_export_results] Starting test.")
        self.analyzer.perform_clustering(n_clusters=2)
        results = self.analyzer.export_results()
        logger.info(f"[test_export_results] results keys: {list(results.keys())}")
        self.assertIsInstance(results, dict)
        logger.info("[test_export_results] Passed.")

    def test_get_cluster_customers(self):
        logger.info("[test_get_cluster_customers] Starting test.")
        self.analyzer.perform_clustering(n_clusters=2)
        customers = self.analyzer.get_cluster_customers(0)
        logger.info(f"[test_get_cluster_customers] customers sample: {customers[:3] if len(customers) > 3 else customers}")
        self.assertIsInstance(customers, list)
        logger.info("[test_get_cluster_customers] Passed.")

if __name__ == '__main__':
    unittest.main()
