import unittest
import networkx as nx
import numpy as np
import sys
import os

# Add the parent directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from quantum_ai.problem_encoding import ProblemEncoder

class TestProblemEncoder(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.max_cut_encoder = ProblemEncoder(problem_type='max-cut')
        self.tsp_encoder = ProblemEncoder(problem_type='tsp')
        
        # Create a simple test graph for Max-Cut
        self.test_graph = nx.Graph()
        self.test_graph.add_edges_from([(0, 1), (1, 2), (2, 0)])
        
        # Create a simple distance matrix for TSP
        self.test_distance_matrix = np.array([
            [0, 10, 15],
            [10, 0, 20],
            [15, 20, 0]
        ])
    
    def test_encode_max_cut(self):
        """Test the Max-Cut encoding functionality."""
        hamiltonian, n_qubits = self.max_cut_encoder.encode_max_cut(self.test_graph)
        
        # Verify the number of qubits is correct
        self.assertEqual(n_qubits, 3, "Should encode to 3 qubits for a 3-node graph")
        
        # Verify the Hamiltonian is a SparsePauliOp
        from qiskit.quantum_info import SparsePauliOp
        self.assertIsInstance(hamiltonian, SparsePauliOp, 
                            "Hamiltonian should be a SparsePauliOp")
        
        # Check that we have the right number of terms for a 3-edge graph
        # Each edge contributes 2 terms: I and ZZ
        self.assertEqual(len(hamiltonian.paulis), 6, 
                        "Should have 6 Pauli terms for a 3-edge graph")
    
    def test_encode_tsp(self):
        """Test the TSP encoding functionality."""
        hamiltonian, n_qubits = self.tsp_encoder.encode_tsp(self.test_distance_matrix)
        
        # Verify the number of qubits is correct (n^2 for TSP with n cities)
        self.assertEqual(n_qubits, 9, "Should encode to 9 qubits for a 3-city TSP")
        
        # Verify the Hamiltonian is a SparsePauliOp
        from qiskit.quantum_info import SparsePauliOp
        self.assertIsInstance(hamiltonian, SparsePauliOp, 
                            "Hamiltonian should be a SparsePauliOp")
    
    def test_encode_problem_dispatcher(self):
        """Test that the encode_problem method correctly dispatches based on problem type."""
        hamiltonian_max_cut, n_qubits_max_cut = self.max_cut_encoder.encode_problem(self.test_graph)
        hamiltonian_tsp, n_qubits_tsp = self.tsp_encoder.encode_problem(self.test_distance_matrix)
        
        self.assertEqual(n_qubits_max_cut, 3, "Max-Cut should use 3 qubits")
        self.assertEqual(n_qubits_tsp, 9, "TSP should use 9 qubits")
    
    def test_invalid_problem_type(self):
        """Test that an invalid problem type raises an error."""
        with self.assertRaises(ValueError):
            invalid_encoder = ProblemEncoder(problem_type='invalid_type')
            # The error should be raised during the encode_problem call
            invalid_encoder.encode_problem(self.test_graph)
    
    def test_invalid_problem_data(self):
        """Test that incompatible problem data raises an error."""
        with self.assertRaises(ValueError):
            # Trying to encode a distance matrix with Max-Cut encoder
            self.max_cut_encoder.encode_problem(self.test_distance_matrix)
        
        with self.assertRaises(ValueError):
            # Trying to encode a graph with TSP encoder
            self.tsp_encoder.encode_problem(self.test_graph)

if __name__ == '__main__':
    unittest.main()

