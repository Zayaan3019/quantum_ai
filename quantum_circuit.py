from qiskit import Aer, transpile, QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.algorithms.optimizers import COBYLA, SPSA
from qiskit.utils import QuantumInstance
import numpy as np

class QuantumOptimizer:
    def __init__(self, backend='qasm_simulator', shots=1024):
        """Initialize the quantum optimizer.
        
        Args:
            backend (str): Qiskit backend to use
            shots (int): Number of shots for quantum circuit execution
        """
        self.backend = Aer.get_backend(backend)
        self.quantum_instance = QuantumInstance(self.backend, shots=shots)
        self.shots = shots
        
    def create_qaoa_circuit(self, hamiltonian, n_qubits, p=1):
        """Create a QAOA circuit for the given Hamiltonian.
        
        Args:
            hamiltonian: Problem Hamiltonian (SparsePauliOp)
            n_qubits (int): Number of qubits
            p (int): QAOA depth parameter
            
        Returns:
            QuantumCircuit: Parameterized QAOA circuit
        """
        # Create a parameterized circuit
        circuit = QuantumCircuit(n_qubits, n_qubits)
        
        # Apply initial Hadamard gates
        for qubit in range(n_qubits):
            circuit.h(qubit)
        
        # Define parameters
        gammas = [Parameter(f'γ_{i}') for i in range(p)]
        betas = [Parameter(f'β_{i}') for i in range(p)]
        
        # Apply QAOA layers
        for layer in range(p):
            # Problem Hamiltonian evolution
            for pauli_str, coef in zip(hamiltonian.paulis.to_labels(), hamiltonian.coeffs):
                # Apply the Pauli string
                # Skip identity terms (they only add global phase)
                if pauli_str == 'I' * n_qubits:
                    continue
                    
                # Track which qubits have non-identity operations
                active_qubits = [i for i, p in enumerate(pauli_str) if p != 'I']
                
                # Handle different Pauli terms
                # For simplicity, only handling Z and ZZ terms which are the most common
                if all(p == 'Z' for p in pauli_str if p != 'I'):
                    if len(active_qubits) == 1:
                        # Single Z term
                        circuit.rz(2 * gammas[layer] * coef, active_qubits[0])
                    elif len(active_qubits) == 2:
                        # ZZ term
                        circuit.cx(active_qubits[0], active_qubits[1])
                        circuit.rz(2 * gammas[layer] * coef, active_qubits[1])
                        circuit.cx(active_qubits[0], active_qubits[1])
            
            # Mixing Hamiltonian evolution
            for qubit in range(n_qubits):
                circuit.rx(2 * betas[layer], qubit)
        
        # Measurement
        circuit.measure_all()
        
        return circuit
    
    def execute_circuit(self, circuit, parameters, shots=None):
        """Execute a quantum circuit with the given parameters.
        
        Args:
            circuit (QuantumCircuit): Parameterized quantum circuit
            parameters (list): Parameter values
            shots (int, optional): Number of shots
            
        Returns:
            dict: Measurement counts
        """
        # Bind parameters
        bound_circuit = circuit.bind_parameters(parameters)
        
        # Execute circuit
        compiled_circuit = transpile(bound_circuit, self.backend)
        job = self.backend.run(compiled_circuit, shots=shots or self.shots)
        result = job.result()
        
        return result.get_counts(compiled_circuit)
    
    def optimize_parameters(self, circuit, hamiltonian, initial_params=None, optimizer='COBYLA', maxiter=100):
        """Optimize QAOA parameters for the given circuit.
        
        Args:
            circuit (QuantumCircuit): Parameterized QAOA circuit
            hamiltonian: Problem Hamiltonian (SparsePauliOp)
            initial_params (list, optional): Initial parameters
            optimizer (str): Optimizer to use ('COBYLA', 'SPSA')
            maxiter (int): Maximum iterations
            
        Returns:
            optimal_params (list): Optimized parameters
            optimal_value (float): Optimal function value
        """
        # Define the objective function
        def objective_function(params):
            counts = self.execute_circuit(circuit, params)
            
            # Calculate expectation value of the Hamiltonian
            energy = 0
            for bitstring, count in counts.items():
                # Convert bitstring to spin configuration (Qiskit 1.4.2 returns LSB first)
                spins = [1 if bit == '0' else -1 for bit in bitstring[::-1]]
                
                # Calculate energy for this configuration
                bitstring_vector = np.array(spins)
                # Evaluate each Pauli term
                for pauli_str, coef in zip(hamiltonian.paulis.to_labels(), hamiltonian.coeffs):
                    term_value = coef
                    for i, p in enumerate(pauli_str):
                        if p == 'Z':
                            term_value *= bitstring_vector[i]
                    energy += (count / sum(counts.values())) * term_value
            
            return energy
        
        # Set optimizer
        if optimizer == 'COBYLA':
            opt = COBYLA(maxiter=maxiter)
        elif optimizer == 'SPSA':
            opt = SPSA(maxiter=maxiter)
        else:
            raise ValueError(f"Optimizer '{optimizer}' not supported")
        
        # Set initial parameters if not provided
        if initial_params is None:
            num_params = len(circuit.parameters)
            initial_params = np.random.uniform(0, 2*np.pi, num_params)
        
        # Run optimization
        result = opt.minimize(objective_function, initial_params)
        
        return result.x, result.fun


