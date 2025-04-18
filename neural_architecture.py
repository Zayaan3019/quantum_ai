import numpy as np
import random
from collections import deque
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class CircuitStructure:
    """Representation of a quantum circuit structure."""
    
    def __init__(self, n_qubits, max_depth=10):
        self.n_qubits = n_qubits
        self.max_depth = max_depth
        self.structure = []
        
    def add_layer(self, gate_type, qubits, parameters=None):
        """Add a layer to the circuit structure.
        
        Args:
            gate_type (str): Type of gate ('h', 'x', 'cx', 'rz', 'rx', 'ry')
            qubits (list): Qubits to apply the gate to
            parameters (list, optional): Gate parameters
        """
        self.structure.append({
            'gate': gate_type,
            'qubits': qubits,
            'parameters': parameters
        })
    
    def to_circuit(self):
        """Convert structure to a Qiskit quantum circuit.
        
        Returns:
            QuantumCircuit: Qiskit quantum circuit
        """
        from qiskit import QuantumCircuit
        from qiskit.circuit import Parameter
        
        circuit = QuantumCircuit(self.n_qubits, self.n_qubits)
        param_count = 0
        params = []
        
        for layer in self.structure:
            gate = layer['gate']
            qubits = layer['qubits']
            
            if gate == 'h':
                for q in qubits:
                    circuit.h(q)
            
            elif gate == 'x':
                for q in qubits:
                    circuit.x(q)
            
            elif gate == 'cx':
                if len(qubits) >= 2:
                    circuit.cx(qubits[0], qubits[1])
            
            elif gate == 'rz':
                for q in qubits:
                    param = Parameter(f'p_{param_count}')
                    params.append(param)
                    circuit.rz(param, q)
                    param_count += 1
            
            elif gate == 'rx':
                for q in qubits:
                    param = Parameter(f'p_{param_count}')
                    params.append(param)
                    circuit.rx(param, q)
                    param_count += 1
            
            elif gate == 'ry':
                for q in qubits:
                    param = Parameter(f'p_{param_count}')
                    params.append(param)
                    circuit.ry(param, q)
                    param_count += 1
        
        # Add measurement to all qubits
        circuit.measure_all()
        
        return circuit, params
    
    def mutate(self):
        """Perform a random mutation on the circuit structure.
        
        Returns:
            CircuitStructure: Mutated circuit structure
        """
        # Create a copy
        mutated = CircuitStructure(self.n_qubits, self.max_depth)
        mutated.structure = self.structure.copy()
        
        # Perform random mutation
        mutation_type = random.choice(['add', 'remove', 'modify'])
        
        if mutation_type == 'add' and len(mutated.structure) < self.max_depth:
            # Add a new layer
            gate = random.choice(['h', 'x', 'cx', 'rz', 'rx', 'ry'])
            if gate == 'cx':
                q1 = random.randint(0, self.n_qubits - 1)
                q2 = random.randint(0, self.n_qubits - 1)
                while q2 == q1:
                    q2 = random.randint(0, self.n_qubits - 1)
                qubits = [q1, q2]
            else:
                num_qubits = random.randint(1, self.n_qubits)
                qubits = random.sample(range(self.n_qubits), num_qubits)
            
            mutated.add_layer(gate, qubits)
        
        elif mutation_type == 'remove' and len(mutated.structure) > 0:
            # Remove a random layer
            idx = random.randint(0, len(mutated.structure) - 1)
            mutated.structure.pop(idx)
        
        elif mutation_type == 'modify' and len(mutated.structure) > 0:
            # Modify a random layer
            idx = random.randint(0, len(mutated.structure) - 1)
            layer = mutated.structure[idx]
            
            # Change gate type with some probability
            if random.random() < 0.3:
                layer['gate'] = random.choice(['h', 'x', 'cx', 'rz', 'rx', 'ry'])
            
            # Change qubits with some probability
            if random.random() < 0.3:
                if layer['gate'] == 'cx':
                    q1 = random.randint(0, self.n_qubits - 1)
                    q2 = random.randint(0, self.n_qubits - 1)
                    while q2 == q1:
                        q2 = random.randint(0, self.n_qubits - 1)
                    layer['qubits'] = [q1, q2]
                else:
                    num_qubits = random.randint(1, self.n_qubits)
                    layer['qubits'] = random.sample(range(self.n_qubits), num_qubits)
        
        return mutated

class NASAgent:
    """Neural Architecture Search Agent for quantum circuits."""
    
    def __init__(self, n_qubits, quantum_optimizer, hamiltonian, population_size=20):
        self.n_qubits = n_qubits
        self.quantum_optimizer = quantum_optimizer
        self.hamiltonian = hamiltonian
        self.population_size = population_size
        self.population = []
        self.best_structure = None
        self.best_energy = float('inf')
    
    def initialize_population(self):
        """Initialize a random population of circuit structures."""
        self.population = []
        
        for _ in range(self.population_size):
            circuit = CircuitStructure(self.n_qubits)
            
            # Add initial Hadamard layer
            circuit.add_layer('h', list(range(self.n_qubits)))
            
            # Add random layers
            depth = random.randint(1, 5)
            for _ in range(depth):
                gate = random.choice(['cx', 'rz', 'rx'])
                if gate == 'cx':
                    q1 = random.randint(0, self.n_qubits - 1)
                    q2 = random.randint(0, self.n_qubits - 1)
                    while q2 == q1:
                        q2 = random.randint(0, self.n_qubits - 1)
                    qubits = [q1, q2]
                else:
                    qubits = list(range(self.n_qubits))
                
                circuit.add_layer(gate, qubits)
            
            self.population.append(circuit)
    
    def evaluate_circuit(self, circuit_structure):
        """Evaluate a circuit structure using the quantum optimizer.
        
        Args:
            circuit_structure (CircuitStructure): Circuit structure to evaluate
            
        Returns:
            float: Energy value (lower is better)
        """
        # Convert structure to Qiskit circuit
        circuit, params = circuit_structure.to_circuit()
        
        # Initialize parameters
        initial_params = np.random.uniform(0, 2*np.pi, len(params))
        
        # Optimize parameters
        opt_params, opt_value = self.quantum_optimizer.optimize_parameters(
            circuit, self.hamiltonian, initial_params=initial_params, maxiter=50
        )
        
        return opt_value
    
    def run_evolution(self, generations=10):
        """Run evolutionary search for optimal circuit structure.
        
        Args:
            generations (int): Number of generations
            
        Returns:
            CircuitStructure: Best circuit structure found
            float: Best energy value
        """
        # Initialize population
        self.initialize_population()
        
        for gen in range(generations):
            print(f"Generation {gen+1}/{generations}")
            
            # Evaluate population
            fitness = []
            for circuit in self.population:
                energy = self.evaluate_circuit(circuit)
                fitness.append(-energy)  # Higher fitness for lower energy
            
            # Find best circuit
            best_idx = np.argmax(fitness)
            if -fitness[best_idx] < self.best_energy:
                self.best_energy = -fitness[best_idx]
                self.best_structure = self.population[best_idx]
                print(f"New best energy: {self.best_energy:.4f}")
            
            # Create new population
            new_population = [self.population[best_idx]]  # Keep best (elitism)
            
            # Tournament selection and mutation
            while len(new_population) < self.population_size:
                # Tournament selection
                candidates = random.sample(range(self.population_size), 3)
                tournament_fitness = [fitness[i] for i in candidates]
                winner_idx = candidates[np.argmax(tournament_fitness)]
                
                # Mutate winner
                mutated = self.population[winner_idx].mutate()
                new_population.append(mutated)
            
            self.population = new_population
        
        return self.best_structure, self.best_energy
