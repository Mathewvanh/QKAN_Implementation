import time
import unittest

import numpy as np
from fable import fable
from qiskit import QuantumCircuit, transpile
from qiskit_aer import Aer


class SUMStep():
    def __init__(self):
        """Initialize the SUM step for QKAN."""
        pass

    def apply_sum(
            self,
            matrix: np.ndarray,
            N: int,
            K:int
    ) -> tuple[QuantumCircuit, float]:
        """
        Apply summation over N inputs for each of K outputs.
        :param matrix: Input matrix from LCU step
        :param N: Number of input nodes
        :param K: Number of output nodes
        :return: Quantum circuit implementing final summer output and scaling factor
        """
        diag_elements = np.diag(matrix).reshape(N, K, order='F')
        summed = np.sum(diag_elements, axis=0) / N
        output = np.diag(summed)
        return fable(output, 0)


class TestSUMStep(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures"""
        self.simulator = Aer.get_backend('unitary_simulator')
        np.random.seed(42)  # For reproducibility

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

        # For zero matrices, use absolute difference
        if np.allclose(expected_matrix, 0):
            print("\nZero matrix detected, using absolute difference")
            elem_diff = np.abs(actual - expected_matrix)
            diff = np.linalg.norm(actual - expected_matrix)
        else:
            # Element-wise relative differences
            with np.errstate(divide='ignore', invalid='ignore'):
                elem_diff = np.abs((actual - expected_matrix) / expected_matrix)
                elem_diff = np.nan_to_num(elem_diff)
            diff = np.linalg.norm(actual - expected_matrix) / np.linalg.norm(expected_matrix)

        print("\nElement-wise differences:")
        print(np.array2string(elem_diff, precision=4, suppress_small=True))
        print(f"\nOverall difference: {diff:.2e}")
        print(f"Scale factor: {scale}")
        print(f"Circuit depth: {circuit.depth()}")
        print(f"Number of qubits: {circuit.num_qubits}")

        # Verify within tolerance
        self.assertTrue(diff < 1e-6, f"Difference too high: {diff}")

    def test_simple_sum(self):
        """Test simple summation case"""
        N, K = 2, 2

        # Create test input matrix
        # Values arranged so each output sums specific inputs
        input_vals = np.array([1.0, 0.5, -0.5, -1.0])  # NK values
        input_matrix = np.diag(input_vals)

        # Create SUMStep and apply
        sum_step = SUMStep()
        circuit, scale = sum_step.apply_sum(input_matrix, N, K)

        # Expected: each output gets average of its inputs
        expected = np.diag([0.75, -0.75])  # (1.0 + 0.5)/2, (-0.5 + -1.0)/2

        print("\nTest matrix structure:")
        print("Input diagonal:", input_vals)
        print("Reshaped (NxK):")
        print(input_vals.reshape(N, K, order='F'))
        print("Expected output:", np.diag(expected))

        self.verify_unitary(circuit, expected, scale, "Simple Sum")

    def test_power_of_two(self):
        """Test power of 2 dimensions"""
        configs = [
            {"N": 4, "K": 4, "name": "4x4 Square"},
            {"N": 4, "K": 8, "name": "4x8 Wide"},
            {"N": 8, "K": 4, "name": "8x4 Tall"},
        ]

        for config in configs:
            N, K = config["N"], config["K"]
            print(f"\nTesting {config['name']}")

            # Create random input diagonal matrix
            input_vals = np.random.uniform(-1, 1, N * K)
            input_matrix = np.diag(input_vals)

            # Expected: sum over N inputs for each K outputs
            reshaped = input_vals.reshape(N, K, order='F')
            expected = np.diag(np.sum(reshaped, axis=0) / N)

            # Apply SUM step
            sum_step = SUMStep()
            circuit, scale = sum_step.apply_sum(input_matrix, N, K)

            print("Input shape:", input_matrix.shape)
            print("Output shape:", expected.shape)
            self.verify_unitary(circuit, expected, scale, config["name"])

    def test_edge_cases(self):
        """Test edge cases"""
        N, K = 4, 4
        sum_step = SUMStep()

        cases = {
            "zeros": {
                "input": np.zeros(N * K),
                "desc": "All zero inputs",
            },
            "ones": {
                "input": np.ones(N * K),
                "desc": "All one inputs",
            },
            "alternating": {
                "input": np.array([1, -1] * (N * K // 2)),
                "desc": "Alternating +1/-1",
            },
        }

        for case_name, case_data in cases.items():
            print(f"\nTesting {case_name}: {case_data['desc']}")

            input_matrix = np.diag(case_data["input"])
            circuit, scale = sum_step.apply_sum(input_matrix, N, K)

            # Calculate expected
            reshaped = case_data["input"].reshape(N, K, order='F')
            expected = np.diag(np.sum(reshaped, axis=0) / N)

            print("Input values:", case_data["input"])
            print("Reshaped (NxK):")
            print(reshaped)
            print("Expected output:", np.diag(expected))

            self.verify_unitary(circuit, expected, scale, f"Edge Case: {case_name}")

    def test_numerical_stability(self):
        """Test numerical stability with different scales"""
        N, K = 4, 4
        sum_step = SUMStep()

        scales = [1e-3, 1e-2, 1e-1, 1, 1e1, 1e2]
        for scale in scales:
            print(f"\nTesting scale {scale}")

            # Random values scaled
            input_vals = np.random.uniform(-1, 1, N * K) * scale
            input_matrix = np.diag(input_vals)

            # Calculate expected
            reshaped = input_vals.reshape(N, K, order='F')
            expected = np.diag(np.sum(reshaped, axis=0) / N)

            circuit, circ_scale = sum_step.apply_sum(input_matrix, N, K)
            self.verify_unitary(circuit, expected, circ_scale, f"Scale {scale}")