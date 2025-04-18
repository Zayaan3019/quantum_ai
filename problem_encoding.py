import networkx as nx
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from pyqubo import Binary

class ProblemEncoder:
    def __init__(self, problem_type='max-cut'):
        """Initialize the problem encoder.
        
        Args:
            problem_type (str): Type of NP-hard problem ('max-cut', 'tsp', 'graph-partition')
        """
        self.problem_type = problem_type
        self.qubit_mapping = {}
        
    def encode_max_cut(self, graph):
        """Encode Max-Cut problem as an Ising Hamiltonian.
        
        Args:
            graph (nx.Graph): Input graph for Max-Cut problem
            
        Returns:
            SparsePauliOp: Hamiltonian operator for the problem
            int: Number of qubits required
        """
        num_qubits = len(graph.nodes())
        pauli_list = []
        coeffs = []
        
        # Construct Ising Hamiltonian for Max-Cut
        # For each edge (i,j), we add a term 0.5(I - Z_i Z_j)
        for i, j in graph.edges():
            # Identity term with coefficient 0.5
            identity_str = 'I' * num_qubits
            pauli_list.append(identity_str)
            coeffs.append(0.5)
            
            # Z_i Z_j term with coefficient -0.5
            zz_str = list('I' * num_qubits)
            zz_str[i] = 'Z'
            zz_str[j] = 'Z'
            pauli_list.append(''.join(zz_str))
            coeffs.append(-0.5)
        
        # Create the SparsePauliOp directly
        if pauli_list:  # Only create if we have terms
            hamiltonian = SparsePauliOp(pauli_list, coeffs)
        else:
            # Create an empty Hamiltonian for an empty graph
            hamiltonian = SparsePauliOp(['I' * num_qubits], [0.0])
            
        return hamiltonian, num_qubits
    
    def encode_tsp(self, distance_matrix):
        """Encode Traveling Salesman Problem as an Ising Hamiltonian.
        
        Args:
            distance_matrix (np.ndarray): Distance matrix between cities
            
        Returns:
            SparsePauliOp: Hamiltonian operator for the problem
            int: Number of qubits required
        """
        n_cities = distance_matrix.shape[0]
        n_qubits = n_cities ** 2
        
        # Create binary variables for TSP
        x = {}
        for i in range(n_cities):
            for j in range(n_cities):
                x[(i, j)] = Binary(f'x_{i}_{j}')
        
        # Construct Hamiltonian for TSP (simplified version)
        H = 0
        
        # Each city must be visited exactly once
        for i in range(n_cities):
            H += (sum(x[(i, j)] for j in range(n_cities)) - 1)**2
        
        # Each position must be filled exactly once
        for j in range(n_cities):
            H += (sum(x[(i, j)] for i in range(n_cities)) - 1)**2
        
        # Add distance costs
        for i in range(n_cities):
            for j in range(n_cities):
                for k in range(n_cities):
                    if k != n_cities - 1:
                        H += distance_matrix[i, j] * x[(i, k)] * x[(j, (k+1) % n_cities)]
                    else:
                        H += distance_matrix[i, j] * x[(i, k)] * x[(j, 0)]
        
        # Create a mapping from variable names to qubit indices
        var_to_index = {}
        index = 0
        for i in range(n_cities):
            for j in range(n_cities):
                var_to_index[f'x_{i}_{j}'] = index
                index += 1
        
        # Convert to quantum Hamiltonian
        model = H.compile()
        qubo, offset = model.to_qubo()
        
        # Map QUBO to Pauli operators
        pauli_list = []
        coeffs = []
        
        # Add offset as identity term
        if offset != 0:
            pauli_list.append('I' * n_qubits)
            coeffs.append(offset)
        
        # Add QUBO terms
        for (i, j), coef in qubo.items():
            # Convert variable names to indices
            idx_i = var_to_index.get(str(i), -1) if isinstance(i, str) else -1
            idx_j = var_to_index.get(str(j), -1) if isinstance(j, str) else -1
            
            # Skip if variables not in our mapping
            if idx_i == -1 or idx_j == -1:
                continue
                
            if i == j:  # Diagonal terms (Z)
                z_str = list('I' * n_qubits)
                z_str[idx_i] = 'Z'
                pauli_list.append(''.join(z_str))
                coeffs.append(coef/2)
                
                # Add identity component
                pauli_list.append('I' * n_qubits)
                coeffs.append(coef/2)
            else:  # Off-diagonal terms (ZZ)
                zz_str = list('I' * n_qubits)
                zz_str[idx_i] = 'Z'
                zz_str[idx_j] = 'Z'
                pauli_list.append(''.join(zz_str))
                coeffs.append(coef/4)
                
                # Add corresponding Z terms
                z_i_str = list('I' * n_qubits)
                z_i_str[idx_i] = 'Z'
                pauli_list.append(''.join(z_i_str))
                coeffs.append(coef/4)
                
                z_j_str = list('I' * n_qubits)
                z_j_str[idx_j] = 'Z'
                pauli_list.append(''.join(z_j_str))
                coeffs.append(coef/4)
                
                # Add identity component
                pauli_list.append('I' * n_qubits)
                coeffs.append(coef/4)
        
        # Create SparsePauliOp (ensure we have at least one term)
        if not pauli_list:
            pauli_list = ['I' * n_qubits]
            coeffs = [0.0]
            
        hamiltonian = SparsePauliOp(pauli_list, coeffs)
        
        return hamiltonian, n_qubits
    
    def encode_problem(self, problem_data):
        """Encode the given problem into a quantum Hamiltonian.
        
        Args:
            problem_data: Problem-specific data structure
            
        Returns:
            hamiltonian: Quantum Hamiltonian representing the problem
            n_qubits: Number of qubits required
        """
        if self.problem_type == 'max-cut':
            if not isinstance(problem_data, nx.Graph):
                raise ValueError("For Max-Cut problems, problem_data must be a NetworkX graph")
            return self.encode_max_cut(problem_data)
        
        elif self.problem_type == 'tsp':
            if not isinstance(problem_data, np.ndarray):
                raise ValueError("For TSP problems, problem_data must be a distance matrix (numpy array)")
            return self.encode_tsp(problem_data)
        
        elif self.problem_type == 'graph-partition':
            # Implement graph partitioning encoding
            raise NotImplementedError("Graph partition encoding not yet implemented")
        
        else:
            raise ValueError(f"Problem type '{self.problem_type}' not supported")


