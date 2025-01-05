import unittest

from fable import fable
from qiskit import QuantumCircuit, QuantumRegister, transpile
from qiskit_aer import Aer

from ChebyshevStep import ChebyshevStep
import numpy as np


class MulStep(ChebyshevStep):
    def __init__(self, degree: int, num_weights: int):
        """
        Initialize MUL step for QKAN.

        Args:
            degree: Maximum degree of Chebyshev polynomials
            num_weights: Number of weights per polynomial (N*K)
        """
        super().__init__(degree)
        self.num_weights = num_weights
        self._weights = np.zeros((degree + 1, num_weights))

    def set_weights(self, degree: int, weights: np.ndarray):
        """
        Set weights for a specific polynomial degree.

        Args:
            degree: Degree of the polynomial (0 to max degree)
            weights: Weight vector of size N*K with values in [-1,1]
        """
        if degree < 0 or degree > self.degree:
            raise ValueError(f"Degree must be between 0 and {self.degree}")
        if len(weights) != self.num_weights:
            raise ValueError(f"Expected {self.num_weights} weights, got {len(weights)}")
        if not np.all(np.abs(weights) <= 1):
            raise ValueError("Weight magnitudes must be <= 1 for unitarity")

        self._weights[degree] = weights

    def get_weighted_polynomial_matrix(
            self,
            x: np.ndarray,
            K: int,
            degree: int
    ) -> np.ndarray:
        """
                Get weighted Chebyshev polynomial matrix before quantum encoding.

                Args:
                    x: Input vector with values in [-1,1]
                    K: Number of copies for dilation
                    degree: Degree of Chebyshev polynomial

                Returns:
                    Diagonal matrix containing weighted polynomial values
                """
        # First create Chebyshev matrix
        cheb_matrix = self.create_dilated_chebyshev(x, K)

        # Ensure weight vector size matches N*K
        N = len(x)
        expected_weights = N * K
        if self.num_weights != expected_weights:
            raise ValueError(f"Weight vector size {self.num_weights} does not match "
                             f"expected size {expected_weights} = {N}*{K}")

        # Get weights for this degree
        weights = self._weights[degree]

        # Return weighted polynomial matrix
        return np.diag(np.diag(cheb_matrix) * weights)

    def create_weighted_chebyshev(
            self,
            x: np.ndarray,
            K: int,
            degree: int
    ) -> tuple[QuantumCircuit, float]:
        """
        Create block encoding for weighted Chebyshev polynomial.

        Args:
            x: Input vector with values in [-1,1]
            K: Number of copies for dilation
            degree: Degree of Chebyshev polynomial

        Returns:
            Quantum circuit implementing weighted polynomial and scaling factor
        """
        # First create Chebyshev matrix
        cheb_matrix = self.create_dilated_chebyshev(x, K)

        # Ensure weight vector size matches N*K
        N = len(x)
        expected_weights = N * K
        if self.num_weights != expected_weights:
            raise ValueError(f"Weight vector size {self.num_weights} does not match "
                             f"expected size {expected_weights} = {N}*{K}")

        # Get weights for this degree
        weights = self._weights[degree]

        # Multiply matrices (element-wise since both are diagonal)
        result_matrix = np.diag(np.diag(cheb_matrix) * weights)

        return fable(result_matrix, 0)


class TestMulStep(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures"""
        self.simulator = Aer.get_backend('unitary_simulator')

    def verify_unitary(self, circuit, expected_matrix, scale, test_name=""):
        """Helper to verify circuit implements expected matrix"""
        print(f"\n=== Testing {test_name} ===")

        # Get circuit unitary
        compiled = transpile(circuit, self.simulator)
        result = self.simulator.run(compiled).result()
        unitary = np.asarray(result.get_unitary(compiled))

        # Extract top-left block
        block_size = expected_matrix.shape[0]
        actual = unitary[:block_size, :block_size] * scale * block_size

        print("\nExpected matrix:")
        print(np.array2string(expected_matrix, precision=4, suppress_small=True))
        print("\nActual matrix (scaled):")
        print(np.array2string(actual, precision=4, suppress_small=True))

        # For zero matrices, use absolute difference instead of relative
        if np.allclose(expected_matrix, 0):
            print("\nZero matrix detected, using absolute difference")
            elem_diff = np.abs(actual - expected_matrix)
            diff = np.linalg.norm(actual - expected_matrix)
        else:
            # Element-wise relative differences
            with np.errstate(divide='ignore', invalid='ignore'):
                elem_diff = np.abs((actual - expected_matrix) / expected_matrix)
                elem_diff = np.nan_to_num(elem_diff)  # Replace inf/nan with 0
            diff = np.linalg.norm(actual - expected_matrix) / np.linalg.norm(expected_matrix)

        print("\nElement-wise differences:")
        print(np.array2string(elem_diff, precision=4, suppress_small=True))
        print(f"\nOverall difference: {diff:.2e}")
        print(f"Scale factor: {scale}")
        print(f"Circuit depth: {circuit.depth()}")
        print(f"Number of qubits: {circuit.num_qubits}")

        # Check if structures match by comparing non-zero patterns
        expected_pattern = np.abs(expected_matrix) > 1e-10
        actual_pattern = np.abs(actual) > 1e-10
        structure_matches = np.array_equal(expected_pattern, actual_pattern)
        print(f"Matrix structure matches: {structure_matches}")

        if not structure_matches:
            print("\nExpected non-zero pattern:")
            print(expected_pattern.astype(int))
            print("\nActual non-zero pattern:")
            print(actual_pattern.astype(int))

        # Verify within tolerance
        self.assertTrue(diff < 1e-6, f"Relative difference too high: {diff}")
        self.assertTrue(structure_matches, "Matrix structure does not match expected pattern")

    def test_weight_validation(self):
        """Test weight vector validation"""
        mul = MulStep(degree=2, num_weights=4)

        # Valid weights
        valid_weights = np.array([0.5, -0.3, 0.1, 0.7])
        mul.set_weights(1, valid_weights)

        # Invalid magnitude
        invalid_weights = np.array([1.5, 0.5, 0.5, 0.5])
        with self.assertRaises(ValueError):
            mul.set_weights(1, invalid_weights)

        # Wrong size
        wrong_size = np.array([0.5, 0.5])
        with self.assertRaises(ValueError):
            mul.set_weights(1, wrong_size)

    def test_weighted_chebyshev_degree1(self):
        """Test weighted Chebyshev polynomial of degree 1"""
        N, K = 2, 2
        mul = MulStep(degree=1, num_weights=N * K)

        # Input vector
        x = np.array([0.5, -0.5])

        # Weights for degree 1
        weights = np.array([1.0, 0.5, -0.5, -1.0])
        mul.set_weights(1, weights)

        # Expected: T_1(x) = x weighted by w
        expected_cheb = np.array([0.5, 0.5, -0.5, -0.5])  # Dilated T_1
        expected = np.diag(expected_cheb * weights)

        print("\nDiagnostic info:")
        print("Input x:", x)
        print("Dilated Chebyshev values:", expected_cheb)
        print("Weights:", weights)
        print("Expected diagonal:", expected_cheb * weights)

        circuit, scale = mul.create_weighted_chebyshev(x, K=2, degree=1)
        self.verify_unitary(circuit, expected, scale, "Weighted Chebyshev Degree 1")

    def test_weighted_chebyshev_degree2(self):
        """Test weighted Chebyshev polynomial of degree 2"""
        N, K = 2, 2
        mul = MulStep(degree=2, num_weights=N * K)

        # Input vector
        x = np.array([0.5, -0.5])

        # Weights for degree 2
        weights = np.array([0.5, 0.5, -0.5, -0.5])
        mul.set_weights(2, weights)

        # Expected: T_2(x) = 2x^2 - 1 weighted by w
        expected_cheb = np.array([-0.5, -0.5, -0.5, -0.5])  # Dilated T_2
        expected = np.diag(expected_cheb * weights)

        print("\nDiagnostic info:")
        print("Input x:", x)
        print("Dilated Chebyshev values:", expected_cheb)
        print("Weights:", weights)
        print("Expected diagonal:", expected_cheb * weights)

        circuit, scale = mul.create_weighted_chebyshev(x, K=2, degree=2)
        self.verify_unitary(circuit, expected, scale, "Weighted Chebyshev Degree 2")

    def test_dimension_mismatch(self):
        """Test error handling for dimension mismatches"""
        # Create MulStep with wrong number of weights
        mul = MulStep(degree=1, num_weights=6)  # Should be 4 for N=2, K=2

        x = np.array([0.5, -0.5])
        weights = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
        mul.set_weights(1, weights)

        # Should raise error due to dimension mismatch
        with self.assertRaises(ValueError):
            mul.create_weighted_chebyshev(x, K=2, degree=1)

    def test_zero_weights(self):
        """Test behavior with zero weights"""
        N, K = 2, 2
        mul = MulStep(degree=1, num_weights=N * K)

        x = np.array([0.5, -0.5])
        weights = np.zeros(N * K)
        mul.set_weights(1, weights)

        # Should give zero matrix
        expected = np.zeros((N * K, N * K))
        print("\nDiagnostic info for zero weights:")
        print("Expected zero matrix shape:", expected.shape)

        circuit, scale = mul.create_weighted_chebyshev(x, K=2, degree=1)
        self.verify_unitary(circuit, expected, scale, "Zero Weights")
