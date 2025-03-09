import unittest
import logging
import sys
import os
from test_kan_vs_mlp_depths import TestKANvsMLPDepths

def main():
    # Create results_js directory if it doesn't exist
    os.makedirs("results_js", exist_ok=True)
    os.makedirs("models_janestreet", exist_ok=True)
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('optimization_comparison.log')
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info("Starting optimization method comparison...")
    
    # Create a test suite with just the optimization comparison test
    suite = unittest.TestSuite()
    suite.addTest(TestKANvsMLPDepths('test_4_alternative_optimization'))
    
    # Run the test
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Report results
    if result.wasSuccessful():
        logger.info("All tests passed!")
    else:
        logger.error("Some tests failed!")
        for failure in result.failures:
            logger.error(f"Failure: {failure[0]}")
            logger.error(f"Details: {failure[1]}")
    
    logger.info("Optimization method comparison completed.")

if __name__ == "__main__":
    main() 