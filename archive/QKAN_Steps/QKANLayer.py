import time
import unittest

import numpy as np

from archive.QKAN_Steps.ChebyshevStep import ChebyshevStep
from archive.QKAN_Steps.LCUStep import LCUStep
from archive.QKAN_Steps.MulStep import MulStep
from archive.QKAN_Steps.SUMStep import SUMStep


class QKANLayer:
    def __init__(self, N: int, K: int, max_degree:int):
        """
        Initialize a single QKAN layer.
        :param N:  Input dimension
        :param K:  Output dimension
        :param max_degree: Maximum degree of Chebyshev polynomials
        """

        self.N = N
        self.K = K
        self.max_degree = max_degree

        self.cheb_step = ChebyshevStep(max_degree)
        self.mul_step = MulStep(max_degree, N*K)
        self.lcu_step = LCUStep(max_degree)
        self.sum_step = SUMStep()

    def get_intermediate_matrices(
            self,
            x: np.ndarray,
            weights: list[np.ndarray]
    ) -> dict[str, np.ndarray]:
        """
        Get intermediate matrices from each step for debugging.
        :param x: Input vector [N]
        :param weights: List of weight vectors for each degree [max_degree+1, N*K]
        :return: Dictionary containing intermediate matrices from each step
        """
        if len(x) != self.N:
            raise ValueError(f"Expected input dimension {self.N}, got {len(x)}")
        if len(weights) != self.max_degree + 1:
            raise ValueError(f"Expected {self.max_degree + 1} weight vectors")

            # Set weights in MulStep
        for degree, w in enumerate(weights):
            if len(w) != self.N * self.K:
                raise ValueError(f"Expected weight dimension {self.N * self.K}")
            self.mul_step.set_weights(degree, w)

            # Step 1-2: Get Chebyshev matrices
        results = {"input": x}
        results["cheb"] = {
            d: self.cheb_step.create_dilated_chebyshev(x, self.K)
            for d in range(self.max_degree + 1)
        }

        # Step 3: Get weighted matrices
        results["weighted"] = {
            d: self.mul_step.get_weighted_polynomial_matrix(x, self.K, d)
            for d in range(self.max_degree + 1)
        }

        # Step 4: Get LCU combined matrix
        results["lcu"] = self.lcu_step.get_combined_matrix(x, self.mul_step, self.K)

        # Step 5: Extract diagonal and reshape for SUM step
        lcu_diag = np.diag(results["lcu"])
        results["reshaped"] = lcu_diag.reshape(self.N, self.K, order='F')
        # Sum over inputs and divide by N
        summed = np.sum(results["reshaped"], axis=0) / self.N
        results["final"] = summed

        return results

    def forward(
            self,
            x: np.ndarray,
            weights: list[np.ndarray],
            verbose:bool=False
    ) -> np.ndarray:
        """
        Forward pass throuh QKAN layer.
        :param x: Input vector [N]
        :param weights: List of weight vectors for each degree [max_degree+1, N*K]
        :param verbose: Whether to print intermediate matrices
        :return: Output vector [K]
        """
        if verbose:
            # Get and print all intermediate matrices
            matrices = self.get_intermediate_matrices(x, weights)

            print("\nQKAN Layer Forward Pass:")
            print(f"Input x: {matrices['input']}")

            print("\nStep 1-2 (DILATE + CHEB):")
            for d, mat in matrices["cheb"].items():
                print(f"Chebyshev matrix degree {d}:")
                print(mat)
                print(f"Matrix diagonal:", np.diag(mat))

            print("\nStep 3 (MUL):")
            for d, mat in matrices["weighted"].items():
                print(f"Weighted matrix degree {d}:")
                print(mat)
                print(f"Matrix diagonal:", np.diag(mat))

            print("\nStep 4 (LCU):")
            print("Combined matrix:")
            print(matrices["lcu"])
            print("Matrix diagonal:", np.diag(matrices["lcu"]))

            print("\nStep 5 (SUM):")
            print("Reshaped (NxK):")
            print(matrices["reshaped"])
            print("Final output (summed over inputs):")
            print(matrices["final"])

            return matrices["final"]

        else:
            # Set weights
            for degree, w in enumerate(weights):
                self.mul_step.set_weights(degree, w)

            # Step 4: LCU combine
            lcu_matrix = self.lcu_step.get_combined_matrix(x, self.mul_step, self.K)

            # Step 5: Final sum
            lcu_diag = np.diag(lcu_matrix)
            reshaped = lcu_diag.reshape(self.N, self.K, order='F')
            summed = np.sum(reshaped, axis=0) / self.N

            return summed


class TestQKANLayer(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures"""
        np.random.seed(42)
        self.N, self.K = 4, 4
        self.max_degree = 3
        self.layer = QKANLayer(self.N, self.K, self.max_degree)

    def test_simple_forward(self):
        """Test basic forward pass"""
        # Create random input and weights
        x = np.random.uniform(-1, 1, self.N)
        weights = [
            np.random.uniform(-1, 1, self.N * self.K)
            for _ in range(self.max_degree + 1)
        ]

        # Forward pass with verbosity
        output = self.layer.forward(x, weights, verbose=True)

        # Check output shape and bounds
        self.assertEqual(len(output), self.K)
        self.assertTrue(np.all(np.abs(output) <= 1))

    def test_intermediate_matrices(self):
        """Test intermediate matrix shapes and properties"""
        x = np.random.uniform(-1, 1, self.N)
        weights = [
            np.random.uniform(-1, 1, self.N * self.K)
            for _ in range(self.max_degree + 1)
        ]

        matrices = self.layer.get_intermediate_matrices(x, weights)

        # Check shapes
        self.assertEqual(matrices["cheb"][0].shape, (self.N * self.K, self.N * self.K))
        self.assertEqual(matrices["weighted"][0].shape, (self.N * self.K, self.N * self.K))
        self.assertEqual(matrices["lcu"].shape, (self.N * self.K, self.N * self.K))
        self.assertEqual(len(matrices["final"]), self.K)

        # Check properties
        for d in range(self.max_degree + 1):
            self.assertTrue(np.all(np.abs(matrices["weighted"][d]) <= 1))
        self.assertTrue(np.all(np.abs(matrices["lcu"]) <= 1))
        self.assertTrue(np.all(np.abs(matrices["final"]) <= 1))

    def test_power_of_two_dimensions(self):
        """Test various power-of-2 dimensions"""
        configs = [
            {"N": 4, "K": 4, "d": 3, "name": "4x4 Basic"},
            {"N": 4, "K": 8, "d": 2, "name": "4x8 Wide"},
            {"N": 8, "K": 4, "d": 2, "name": "8x4 Tall"}
        ]

        for config in configs:
            N, K, d = config["N"], config["K"], config["d"]
            print(f"\nTesting {config['name']}")

            layer = QKANLayer(N, K, d)
            x = np.random.uniform(-1, 1, N)
            weights = [
                np.random.uniform(-1, 1, N * K)
                for _ in range(d + 1)
            ]

            start_time = time.time()
            output = layer.forward(x, weights)
            compute_time = time.time() - start_time

            print(f"Shape: {N}x{K}, Degree: {d}")
            print(f"Computation time: {compute_time:.4f}s")
            print("Output:", output)

            self.assertEqual(len(output), K)
            self.assertTrue(np.all(np.abs(output) <= 1))

    def test_edge_cases(self):
        """Test edge cases"""
        cases = {
            "zero_input": {
                "x": np.zeros(self.N),
                "desc": "All zero input"
            },
            "boundary_input": {
                "x": np.array([-1.0] * (self.N // 2) + [1.0] * (self.N // 2)),
                "desc": "Boundary inputs"
            },
            "uniform_input": {
                "x": np.ones(self.N) * 0.5,
                "desc": "Uniform inputs"
            }
        }

        for case_name, case_data in cases.items():
            print(f"\nTesting {case_name}: {case_data['desc']}")

            # Random weights
            weights = [
                np.random.uniform(-1, 1, self.N * self.K)
                for _ in range(self.max_degree + 1)
            ]

            start_time = time.time()
            output = self.layer.forward(case_data["x"], weights)
            compute_time = time.time() - start_time

            print(f"Computation time: {compute_time:.4f}s")
            print("Output:", output)

            self.assertEqual(len(output), self.K)
            self.assertTrue(np.all(np.abs(output) <= 1))

            if case_name == "zero_input":
                # Output should be small for zero input
                self.assertTrue(np.allclose(output, 0, atol=1e-6))

    def test_numerical_stability(self):
        """Test numerical stability with different degrees"""
        x = np.random.uniform(-1, 1, self.N)

        # Test increasing degrees
        for d in [1, 3, 5, 10]:
            print(f"\nTesting degree {d}")
            layer = QKANLayer(self.N, self.K, d)

            # Scale weights for stability
            weights = [
                np.random.uniform(-1 / (deg + 1), 1 / (deg + 1), self.N * self.K)
                for deg in range(d + 1)
            ]

            output = layer.forward(x, weights, verbose=True)
            self.assertTrue(np.all(np.abs(output) <= 1))


if __name__ == "__main__":
    unittest.main(argv=['first-arg-is-ignored'], exit=False)