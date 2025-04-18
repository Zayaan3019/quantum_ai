import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import gym
from gym import spaces
import random
from collections import deque

class QuantumParameterEnvironment(gym.Env):
    """Custom Environment for QAOA parameter optimization using RL."""
    
    def __init__(self, quantum_optimizer, circuit, hamiltonian, n_params):
        super(QuantumParameterEnvironment, self).__init__()
        
        self.quantum_optimizer = quantum_optimizer
        self.circuit = circuit
        self.hamiltonian = hamiltonian
        self.n_params = n_params
        
        # Define action and observation spaces
        self.action_space = spaces.Box(low=-0.1, high=0.1, shape=(n_params,), dtype=np.float32)
        self.observation_space = spaces.Box(low=0, high=2*np.pi, shape=(n_params,), dtype=np.float32)
        
        # Initial state
        self.state = np.random.uniform(0, 2*np.pi, n_params)
        self.best_energy = float('inf')
        self.steps = 0
        self.max_steps = 100
    
    def reset(self):
        """Reset the environment to initial state."""
        self.state = np.random.uniform(0, 2*np.pi, self.n_params)
        self.best_energy = float('inf')
        self.steps = 0
        return self.state
    
    def step(self, action):
        """Take a step in the environment."""
        # Update parameters
        new_state = self.state + action
        
        # Ensure parameters are within bounds [0, 2π]
        new_state = np.mod(new_state, 2*np.pi)
        
        # Execute quantum circuit with new parameters
        counts = self.quantum_optimizer.execute_circuit(self.circuit, new_state)
        
        # Calculate energy
        energy = 0
        for bitstring, count in counts.items():
            # Convert bitstring to spin configuration (Qiskit 1.4.2 returns LSB first)
            spins = [1 if bit == '0' else -1 for bit in bitstring[::-1]]
                
            # Calculate energy for this configuration
            bitstring_vector = np.array(spins)
            # Evaluate each Pauli term
            for pauli_str, coef in zip(self.hamiltonian.paulis.to_labels(), self.hamiltonian.coeffs):
                term_value = coef
                for i, p in enumerate(pauli_str):
                    if p == 'Z':
                        term_value *= bitstring_vector[i]
                energy += (count / sum(counts.values())) * term_value
        
        # Calculate reward
        if energy < self.best_energy:
            reward = self.best_energy - energy  # Positive reward for improvement
            self.best_energy = energy
        else:
            reward = 0  # No improvement
        
        # Update state
        self.state = new_state
        self.steps += 1
        
        # Check if done
        done = self.steps >= self.max_steps
        
        return self.state, reward, done, {"energy": energy}

class DQNAgent:
    """Deep Q-Network Agent for optimizing quantum circuit parameters."""
    
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0   # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()
    
    def _build_model(self):
        """Build a neural network model for DQN."""
        model = keras.Sequential([
            layers.Dense(24, input_dim=self.state_size, activation='relu'),
            layers.Dense(24, activation='relu'),
            layers.Dense(self.action_size, activation='linear')
        ])
        model.compile(loss='mse', optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate))
        return model
    
    def update_target_model(self):
        """Update target model with weights from main model."""
        self.target_model.set_weights(self.model.get_weights())
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in memory."""
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state):
        """Choose action based on epsilon-greedy policy."""
        if np.random.rand() <= self.epsilon:
            return np.random.uniform(-0.1, 0.1, self.action_size)
        act_values = self.model.predict(state.reshape(1, -1), verbose=0)
        return act_values[0]
    
    def replay(self, batch_size):
        """Train the model using experience replay."""
        if len(self.memory) < batch_size:
            return
        
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.target_model.predict(next_state.reshape(1, -1), verbose=0)[0])
            target_f = self.model.predict(state.reshape(1, -1), verbose=0)
            target_f[0] = target
            self.model.fit(state.reshape(1, -1), target_f, epochs=1, verbose=0)
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

class QuantumCircuitOptimizer:
    """Class for optimizing quantum circuits using RL."""
    
    def __init__(self, quantum_optimizer):
        self.quantum_optimizer = quantum_optimizer
    
    def optimize_qaoa_parameters(self, circuit, hamiltonian, n_params, episodes=500, batch_size=32):
        """Optimize QAOA parameters using reinforcement learning.
        
        Args:
            circuit (QuantumCircuit): Parameterized QAOA circuit
            hamiltonian: Problem Hamiltonian
            n_params (int): Number of parameters
            episodes (int): Number of RL episodes
            batch_size (int): Batch size for experience replay
            
        Returns:
            optimal_params (list): Optimized parameters
            optimal_value (float): Optimal function value
        """
        # Create environment
        env = QuantumParameterEnvironment(self.quantum_optimizer, circuit, hamiltonian, n_params)
        
        # Create RL agent
        agent = DQNAgent(n_params, n_params)
        
        # Training loop
        best_params = None
        best_energy = float('inf')
        
        for episode in range(episodes):
            state = env.reset()
            done = False
            
            while not done:
                action = agent.act(state)
                next_state, reward, done, info = env.step(action)
                agent.remember(state, action, reward, next_state, done)
                state = next_state
                
                # Update best solution
                if info['energy'] < best_energy:
                    best_energy = info['energy']
                    best_params = state.copy()
            
            # Train the agent
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)
            
            # Update target model periodically
            if episode % 10 == 0:
                agent.update_target_model()
            
            if episode % 50 == 0:
                print(f"Episode: {episode}, Best Energy: {best_energy:.4f}, Epsilon: {agent.epsilon:.2f}")
        
        return best_params, best_energy

