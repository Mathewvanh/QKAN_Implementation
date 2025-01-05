import unittest
import numpy as np
from fable import fable
from qiskit import ClassicalRegister, transpile
from qiskit_aer import Aer


class ChebyshevStep:
    def __init__(self, degree: int):
        """
        Initialize Chebyshev transformation step.
        :param degree: Degree of chebyshev polynomial
        """
        if degree < 0:
            raise ValueError("Degree must be positive integer.")
        self.degree = degree

    def apply_chebyshev(self, x: float) -> float:
        """
        Apply Chebyshev polynomial to input value.
        :param x(float): Input value in [-1, 1].
        :return: float: T_d(x) = cos(degree * arccos(x)).
        """
        # Add small tolerance and clip
        eps = 1e-8
        if not np.all((x >= -1-eps) & (x <= 1+eps)):
            raise ValueError("Input value must be between -1 and 1.")

        x = np.clip(x, -1, 1)
        return np.cos(self.degree * np.arccos(x))

    def transform_diagonal(self, x: np.ndarray) -> np.ndarray:
        """
        Apply Chebyshev polynomial to input value.
        :param x: Input vector with values in [-1, 1].
        :return: Vector of Chebyshev polynomial values.
        """
        # Add small tolerance for floating point precision
        eps = 1e-8

        # Print debug info
        #print(f"Shape of input array: {x.shape}")
        #print(f"First few values: {x[:5]}")
        #print(f"Min value: {np.min(x)}, Max value: {np.max(x)}")

        # Check for values significantly outside [-1,1]
        violations = x[~(-1-eps <= x) | ~(x <= 1+eps)]
        if len(violations) > 0:
            print(f"Values outside [-1,1] range: {violations[:5]}")

        # Clip values that are just slightly outside range
        x = np.clip(x, -1, 1)
        return np.array([self.apply_chebyshev(xi) for xi in x])

    def create_dilated_chebyshev(self, x: np.ndarray, K: int) -> np.ndarray:
        """
        Create dilated matrix of Chebyshev transformations.
        :param x: (np.ndarray) Input vector with values in [-1, 1].
        :param K: Number of copies for dilation
        :return: np.ndarray of Chebyshev polynomial values.
        """
        chebyshev_values = self.transform_diagonal(x)

        dilated_values = np.repeat(chebyshev_values, K)
        return np.diag(dilated_values)


class TestChebyshevStep(unittest.TestCase):
    def test_simple_chebyshev(self):
        """Test simple chebyshev polynomial values"""
        cheb = ChebyshevStep(degree=1)

        x_test = 0.5
        np.testing.assert_almost_equal(
            cheb.apply_chebyshev(x_test),
            x_test)
        cheb2 = ChebyshevStep(degree=2)
        np.testing.assert_almost_equal(
            cheb2.apply_chebyshev(x_test),
            2 * x_test ** 2 - 1
        )

    def test_vector_transform(self):
        """Test Chebyshev transformation on vector"""
        cheb = ChebyshevStep(degree=2)
        x = np.array([0.5, -0.5, 0.0])

        result = cheb.transform_diagonal(x)
        expected = 2 * x ** 2 - 1
        print(result)
        np.testing.assert_almost_equal(result, expected)

    def test_dilation(self):
        """Test Chebyshev polynomial with dilation"""
        cheb = ChebyshevStep(degree=1)
        x = np.array([0.5, -0.5])
        K = 2

        result = cheb.create_dilated_chebyshev(x, K)
        expected = np.diag([0.5, 0.5, -0.5, -0.5])

        np.testing.assert_almost_equal(result, expected)

    def test_input_validation(self):
        """Test input validation"""
        cheb = ChebyshevStep(degree=1)

        with self.assertRaises(ValueError):
            cheb.apply_chebyshev(1.5)

        with self.assertRaises(ValueError):
            cheb.transform_diagonal(np.array([1.5, 0.5]))

        with self.assertRaises(ValueError):
            ChebyshevStep(degree=-1)

    def test_dilated_block_encoding_different_sizes(self):
        cheb = ChebyshevStep(degree=8)
        x = np.array(np.array(np.random.uniform(low=-1, high=1, size=4)))
        K = 1
        A = cheb.create_dilated_chebyshev(x, K)
        N = A.shape[0]
        n = int(np.ceil(np.log2(N)))
        circ, alpha = fable(A, 0)
        simulator = Aer.get_backend('unitary_simulator')
        compiled_circuit = transpile(circ, simulator)
        result = simulator.run(compiled_circuit).result()
        unitary = np.asarray(result.get_unitary(compiled_circuit))
        block_size = N
        top_left_block = unitary[:block_size, :block_size]
        reconstructed_A = top_left_block * alpha * N
        difference = np.linalg.norm(reconstructed_A - A) / np.linalg.norm(A)
        self.assertTrue(difference < 1e-15, f"Relative difference too high: {difference}")
        print(f"Test passed for input size {len(x)} and K={K}.")
