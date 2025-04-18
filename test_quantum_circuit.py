import unittest
import numpy as np
import sys
import os

# Add the parent directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.quantum_info import SparsePauliOp
from quantum_ai.quantum_circuit import QuantumOptimizer

class TestQuantumOptimizer(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.optimizer = QuantumOptimizer(backend='statevector_simulator', shots=100)
        
        # Create a simple mock Hamiltonian using SparsePauliOp
        # For a 2-qubit system, with terms: 0.5 I⊗I - 0.5 Z⊗Z
        pauli_list = ['II', 'ZZ']
        coeffs = [0.5, -0.5]
        self.mock_hamiltonian = SparsePauliOp(pauli_list, coeffs)
    
    def test_init(self):
        """Test initialization of the QuantumOptimizer."""
        self.assertEqual(self.optimizer.shots, 100,
                         "Optimizer should use the specified number of shots")
    
    def test_create_qaoa_circuit(self):
        """Test the creation of a QAOA circuit."""
        n_qubits = 2
        p = 1  # QAOA depth
        
        circuit = self.optimizer.create_qaoa_circuit(self.mock_hamiltonian, n_qubits, p)
        
        # Verify circuit has the correct number of qubits
        self.assertEqual(circuit.num_qubits, n_qubits, 
                        f"Circuit should have {n_qubits} qubits")
        
        # Verify circuit has parameters (gammas and betas)
        self.assertEqual(len(circuit.parameters), 2*p, 
                        f"Circuit should have {2*p} parameters for p={p}")
        
        # Check if the circuit has measurement operations
        has_measure = False
        for inst, _, _ in circuit.data:
            if inst.name == 'measure':
                has_measure = True
                break
        self.assertTrue(has_measure, "Circuit should include measurement operations")
    
    def test_execute_circuit(self):
        """Test the execution of a quantum circuit."""
        # Create a simple circuit
        circuit = QuantumCircuit(1, 1)
        circuit.h(0)  # Put qubit in superposition
        param = Parameter('θ')
        circuit.rz(param, 0)
        circuit.measure(0, 0)
        
        # Execute with a parameter value
        counts = self.optimizer.execute_circuit(circuit, [np.pi])
        
        # Verify that we get measurement results
        self.assertTrue(len(counts) > 0, "Circuit execution should return counts")
        
    def test_optimize_parameters_small(self):
        """Test parameter optimization with a small circuit."""
        # Create a simple parameterized circuit
        circuit = QuantumCircuit(1, 1)
        circuit.h(0)
        param = Parameter('θ')
        circuit.rz(param, 0)
        circuit.measure(0, 0)
        
        # Simple Hamiltonian for a single qubit: 0.5*I - 0.5*Z
        pauli_list = ['I', 'Z']
        coeffs = [0.5, -0.5]
        simple_hamiltonian = SparsePauliOp(pauli_list, coeffs)
        
        # Perform optimization
        initial_params = [0.0]
        opt_params, opt_value = self.optimizer.optimize_parameters(
            circuit, 
            simple_hamiltonian, 
            initial_params=initial_params,
            optimizer='COBYLA',
            maxiter=3  # Small maxiter for faster tests
        )
        
        # Verify that optimization returns parameters and value
        self.assertEqual(len(opt_params), len(initial_params), 
                        "Optimized parameters should have same dimension as initial")
        self.assertIsInstance(opt_value, float, 
                            "Optimized value should be a float")

if __name__ == '__main__':
    unittest.main()


