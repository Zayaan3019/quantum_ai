import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import time
import sys
import os

# Add the parent directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from quantum_ai.framework import HybridFramework

def solve_max_cut_classical(graph):
    """Solve Max-Cut using a classical greedy algorithm."""
    start_time = time.time()
    
    # Random initial assignment
    solution = np.random.randint(0, 2, len(graph.nodes()))
    
    # Greedy local search
    improved = True
    iterations = 0
    max_iterations = 1000
    
    while improved and iterations < max_iterations:
        improved = False
        for node in graph.nodes():
            # Calculate cut value with current assignment
            current_cut = 0
            for neighbor in graph.neighbors(node):
                if solution[node] != solution[neighbor]:
                    current_cut += 1
            
            # Calculate cut value with flipped assignment
            flipped_cut = 0
            for neighbor in graph.neighbors(node):
                if (1 - solution[node]) != solution[neighbor]:
                    flipped_cut += 1
            
            # Flip if it improves cut value
            if flipped_cut > current_cut:
                solution[node] = 1 - solution[node]
                improved = True
        
        iterations += 1
    
    # Calculate final cut value
    cut_value = 0
    for u, v in graph.edges():
        if solution[u] != solution[v]:
            cut_value += 1
    
    end_time = time.time()
    
    return solution, cut_value, end_time - start_time

def benchmark_max_cut(n_instances=2, node_sizes=[4, 6]):
    """Benchmark Max-Cut solvers."""
    results = {
        'classical': {'time': [], 'cut_value': []},
        'rl': {'time': [], 'cut_value': []}
    }
    
    for n_nodes in node_sizes:
        print(f"\nBenchmarking with {n_nodes} nodes")
        
        classical_times = []
        classical_cuts = []
        rl_times = []
        rl_cuts = []
        
        for i in range(n_instances):
            print(f"Instance {i+1}/{n_instances}")
            
            # Generate a random graph
            graph = nx.gnp_random_graph(n_nodes, 0.3)
            
            # Classical solution
            print("Running classical solver...")
            classical_solution, classical_cut, classical_time = solve_max_cut_classical(graph)
            classical_times.append(classical_time)
            classical_cuts.append(classical_cut)
            
            # RL solution
            print("Running quantum RL solver...")
            framework = HybridFramework(problem_type='max-cut')
            
            # Reduced episodes for faster execution
            rl_episodes = max(20, 30 - n_nodes * 2)
            rl_solution, rl_energy, rl_performance = framework.solve(
                graph, method='rl', rl_episodes=rl_episodes
            )
            
            # Calculate RL cut value
            rl_cut = 0
            for u, v in graph.edges():
                if rl_solution[u] != rl_solution[v]:
                    rl_cut += 1
            
            rl_times.append(rl_performance['execution_time'])
            rl_cuts.append(rl_cut)
        
        # Average results
        results['classical']['time'].append(np.mean(classical_times))
        results['classical']['cut_value'].append(np.mean(classical_cuts))
        results['rl']['time'].append(np.mean(rl_times))
        results['rl']['cut_value'].append(np.mean(rl_cuts))
        
        # Print summary
        print(f"Classical: Avg cut = {np.mean(classical_cuts):.2f}, Avg time = {np.mean(classical_times):.2f}s")
        print(f"RL: Avg cut = {np.mean(rl_cuts):.2f}, Avg time = {np.mean(rl_times):.2f}s")
    
    # Plot results
    plt.figure(figsize=(12, 5))
    
    # Cut value comparison
    plt.subplot(1, 2, 1)
    plt.plot(node_sizes, results['classical']['cut_value'], 'o-', label='Classical Greedy')
    plt.plot(node_sizes, results['rl']['cut_value'], 's-', label='RL-QAOA')
    
    plt.xlabel('Number of Nodes')
    plt.ylabel('Average Cut Value')
    plt.title('Cut Value Comparison')
    plt.legend()
    plt.grid(True)
    
    # Time comparison
    plt.subplot(1, 2, 2)
    plt.plot(node_sizes, results['classical']['time'], 'o-', label='Classical Greedy')
    plt.plot(node_sizes, results['rl']['time'], 's-', label='RL-QAOA')
    
    plt.xlabel('Number of Nodes')
    plt.ylabel('Average Execution Time (s)')
    plt.title('Execution Time Comparison')
    plt.legend()
    plt.grid(True)
    plt.yscale('log')
    
    plt.tight_layout()
    plt.savefig('benchmark_results.png')
    plt.show()
    
    return results

# Run the benchmark
if __name__ == "__main__":
    # Use smaller problem sizes and fewer instances for Qiskit 2.0
    benchmark_results = benchmark_max_cut(n_instances=1, node_sizes=[4, 5])

