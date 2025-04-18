import networkx as nx
import matplotlib.pyplot as plt
import sys
import os

# Add the parent directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from quantum_ai.framework import HybridFramework

# Create a random graph
def generate_random_graph(n_nodes, edge_probability=0.3):
    G = nx.gnp_random_graph(n_nodes, edge_probability)
    return G

# Visualize the graph and solution
def visualize_max_cut(graph, solution):
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(graph)
    
    # Node colors based on solution
    node_colors = ['skyblue' if solution[i] == 0 else 'salmon' for i in range(len(solution))]
    
    # Draw the graph
    nx.draw(graph, pos, with_labels=True, node_color=node_colors, 
            node_size=800, font_size=15, font_weight='bold', width=2)
    
    # Count cut edges
    cut_value = 0
    for u, v in graph.edges():
        if solution[u] != solution[v]:
            cut_value += 1
    
    plt.title(f"Max-Cut Solution: {cut_value} edges cut", fontsize=16)
    plt.show()

# Main execution
if __name__ == "__main__":
    # Generate a problem instance
    n_nodes = 4  # Start small for faster execution with Qiskit 2.0
    graph = generate_random_graph(n_nodes)
    
    # Create the hybrid framework
    framework = HybridFramework(problem_type='max-cut')
    
    # Solve the problem using RL
    print("Solving Max-Cut problem using reinforcement learning...")
    solution, energy, performance = framework.solve(graph, method='rl', rl_episodes=50)
    
    # Display results
    print(f"Solution: {solution}")
    print(f"Energy: {energy}")
    print(f"Execution time: {performance['execution_time']:.2f} seconds")
    
    # Visualize the solution
    visualize_max_cut(graph, solution)
    
    print("\nNote: Running the hybrid approach will take substantially longer.")
    run_hybrid = input("Do you want to run the hybrid NAS+RL approach too? (y/n): ")
    if run_hybrid.lower() == 'y':
        # Solve the same problem using hybrid approach
        print("\nSolving Max-Cut problem using hybrid NAS+RL approach...")
        hybrid_solution, hybrid_energy, hybrid_performance = framework.solve(
            graph, method='hybrid', nas_generations=2, rl_episodes=30
        )
        
        # Display results
        print(f"Solution: {hybrid_solution}")
        print(f"Energy: {hybrid_energy}")
        print(f"Execution time: {hybrid_performance['execution_time']:.2f} seconds")
        
        # Visualize the hybrid solution
        visualize_max_cut(graph, hybrid_solution)
