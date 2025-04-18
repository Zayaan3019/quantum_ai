import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import time

from quantum_ai.problem_encoding import ProblemEncoder
from quantum_ai.quantum_circuit import QuantumOptimizer
from quantum_ai.ai_optimization import QuantumCircuitOptimizer
from quantum_ai.neural_architecture import NASAgent

class HybridFramework:
    """Hybrid framework combining AI techniques with quantum variational algorithms."""
    
    def __init__(self, problem_type='max-cut'):
        """Initialize the hybrid framework.
        
        Args:
            problem_type (str): Type of NP-hard problem ('max-cut', 'tsp', 'graph-partition')
        """
        self.problem_encoder = ProblemEncoder(problem_type)
        self.quantum_optimizer = QuantumOptimizer(backend='qasm_simulator', shots=1024)
        self.problem_type = problem_type
    
    def solve(self, problem_data, method='rl', nas_generations=5, rl_episodes=300):
        """Solve the given NP-hard problem using the hybrid framework.
        
        Args:
            problem_data: Problem-specific data structure
            method (str): Method to use ('rl', 'nas', 'hybrid')
            nas_generations (int): Number of NAS generations
            rl_episodes (int): Number of RL episodes
            
        Returns:
            solution: Solution to the problem
            energy: Energy value of the solution
            performance_data: Additional performance data
        """
        start_time = time.time()
        
        # Encode problem
        hamiltonian, n_qubits = self.problem_encoder.encode_problem(problem_data)
        print(f"Problem encoded using {n_qubits} qubits")
        
        # Method selection
        if method == 'rl':
            # Create QAOA circuit
            circuit = self.quantum_optimizer.create_qaoa_circuit(hamiltonian, n_qubits, p=1)
            
            # Use RL to optimize parameters
            quantum_circuit_optimizer = QuantumCircuitOptimizer(self.quantum_optimizer)
            opt_params, opt_energy = quantum_circuit_optimizer.optimize_qaoa_parameters(
                circuit, hamiltonian, len(circuit.parameters), episodes=rl_episodes
            )
            
            # Execute optimized circuit
            counts = self.quantum_optimizer.execute_circuit(circuit, opt_params, shots=10000)
            
            # Interpret results
            solution = self._interpret_results(counts, n_qubits)
        
        elif method == 'nas':
            # Use NAS to design quantum circuit
            nas_agent = NASAgent(n_qubits, self.quantum_optimizer, hamiltonian)
            best_structure, opt_energy = nas_agent.run_evolution(generations=nas_generations)
            
            # Convert to circuit and optimize parameters
            circuit, params = best_structure.to_circuit()
            initial_params = np.random.uniform(0, 2*np.pi, len(params))
            opt_params, _ = self.quantum_optimizer.optimize_parameters(
                circuit, hamiltonian, initial_params=initial_params
            )
            
            # Execute optimized circuit
            counts = self.quantum_optimizer.execute_circuit(circuit, opt_params, shots=10000)
            
            # Interpret results
            solution = self._interpret_results(counts, n_qubits)
        
        elif method == 'hybrid':
            # Use NAS to design quantum circuit
            nas_agent = NASAgent(n_qubits, self.quantum_optimizer, hamiltonian)
            best_structure, _ = nas_agent.run_evolution(generations=nas_generations)
            
            # Convert to circuit
            circuit, params = best_structure.to_circuit()
            
            # Use RL to optimize parameters
            quantum_circuit_optimizer = QuantumCircuitOptimizer(self.quantum_optimizer)
            opt_params, opt_energy = quantum_circuit_optimizer.optimize_qaoa_parameters(
                circuit, hamiltonian, len(params), episodes=rl_episodes
            )
            
            # Execute optimized circuit
            counts = self.quantum_optimizer.execute_circuit(circuit, opt_params, shots=10000)
            
            # Interpret results
            solution = self._interpret_results(counts, n_qubits)
        
        else:
            raise ValueError(f"Method '{method}' not supported")
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Collect performance data
        performance_data = {
            'execution_time': execution_time,
            'n_qubits': n_qubits,
            'method': method,
            'optimal_energy': opt_energy
        }
        
        return solution, opt_energy, performance_data
    
    def _interpret_results(self, counts, n_qubits):
        """Interpret measurement results based on the problem type.
        
        Args:
            counts (dict): Measurement counts from quantum circuit
            n_qubits (int): Number of qubits
            
        Returns:
            solution: Problem-specific solution
        """
        # Find most frequent measurement
        max_count = 0
        max_bitstring = None
        for bitstring, count in counts.items():
            if count > max_count:
                max_count = count
                max_bitstring = bitstring
        
        # Ensure correct length and order
        # In Qiskit 1.4.2, bitstrings are LSB first, so reverse for proper mapping
        if len(max_bitstring) < n_qubits:
            max_bitstring = '0' * (n_qubits - len(max_bitstring)) + max_bitstring
        max_bitstring = max_bitstring[::-1]  # Reverse to get MSB first
        
        # Convert to solution based on problem type
        if self.problem_type == 'max-cut':
            # Solution is a binary assignment for each node
            solution = [int(bit) for bit in max_bitstring[:n_qubits]]
        
        elif self.problem_type == 'tsp':
            # Solution is a permutation of cities
            # This is a simplified interpretation
            solution = []
            n_cities = int(np.sqrt(n_qubits))
            
            for i in range(n_cities):
                for j in range(n_cities):
                    idx = i * n_cities + j
                    if idx < len(max_bitstring) and max_bitstring[idx] == '1':
                        solution.append(j)
        
        else:
            # Default interpretation
            solution = [int(bit) for bit in max_bitstring[:n_qubits]]
        
        return solution

