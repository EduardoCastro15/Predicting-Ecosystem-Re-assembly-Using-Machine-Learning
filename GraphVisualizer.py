import os
import csv
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

class GraphVisualizer:
    """Class to handle the visualization of NetworkX graphs."""
    
    def __init__(self, graph=None):
        """
        Initialize the GraphVisualizer with a NetworkX graph.

        :param graph: NetworkX graph to visualize.
        """
        self.graph = graph
    
    def draw_graph(self, pos=None, with_labels=True, node_color='green', node_size=500, font_size=8, font_color='black',
                   edge_color='gray', width=1, style='dashed', alpha=0.9, linewidths=2, edge_cmap=plt.cm.Blues,
                   edge_vmin=0, edge_vmax=1, show_info=True):
        """
        Draws the graph with the given layout and styling options.

        :param pos: Position layout for nodes (default: spring_layout).
        :param with_labels: Whether to display labels on nodes (default: True).
        :param node_color: Color of the nodes (default: 'green').
        :param node_size: Size of the nodes (default: 500).
        :param font_size: Font size for node labels (default: 8).
        :param font_color: Font color for node labels (default: 'black').
        :param edge_color: Color of the edges (default: 'gray').
        :param width: Width of the edges (default: 1).
        :param style: Style of the edges (default: 'dashed').
        :param alpha: Transparency level for nodes and edges (default: 0.9).
        :param linewidths: Border width of the nodes (default: 2).
        :param edge_cmap: Colormap for the edges (default: plt.cm.Blues).
        :param edge_vmin: Minimum value for edge colormap normalization (default: 0).
        :param edge_vmax: Maximum value for edge colormap normalization (default: 1).
        """
        if pos is None:
            pos = nx.spring_layout(self.graph)
        
        # Draw the graph with the specified properties
        nx.draw(
            self.graph,
            pos,
            with_labels=with_labels,
            node_color=node_color,
            node_size=node_size,
            font_size=font_size,
            font_color=font_color,
            edge_color=edge_color,
            width=width,
            style=style,
            alpha=alpha,
            linewidths=linewidths,
            edge_cmap=edge_cmap,
            edge_vmin=edge_vmin,
            edge_vmax=edge_vmax
        )
        if show_info:
            # Get the number of nodes and edges
            num_nodes = self.graph.number_of_nodes()
            num_edges = self.graph.number_of_edges()
            
            # Add text with the number of nodes and edges to the plot
            plt.text(0.05, 0.95, f'Nodes: {num_nodes}\nEdges: {num_edges}', 
                     fontsize=12, color='black', ha='left', va='top', transform=plt.gca().transAxes,
                     bbox=dict(facecolor='white', alpha=0.5))
        
        # Show the plot
        plt.show()
    
    def plot_auc_by_size(self, csv_file_path):
        """
        Plot the AUC scores over iterations and generate separate curves by size.
        
        :param csv_file_path: Path to the CSV file containing iteration results.
        """
        iterations_list = []
        auc_list = []
        size_list = []

        # Read the CSV file and extract data
        with open(csv_file_path, mode='r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                iterations_list.append(int(row["Iteration"]))
                auc_list.append(float(row["AUC"]))
                if row["Size"] is not None and row["Size"] != '':
                    size_list.append(int(row["Size"]))
                else:
                    size_list.append(0)  # Handle missing size values
        
        # Get unique sizes to plot separate curves
        unique_sizes = sorted(set(size_list))
        colors = plt.cm.viridis(np.linspace(0, 1, len(unique_sizes)))  # Generate distinct colors

        # Plot the AUC for each size
        for i, unique_size in enumerate(unique_sizes):
            indices = [j for j, s in enumerate(size_list) if s == unique_size]
            auc_for_size = [auc_list[j] for j in indices]
            iterations_for_size = [iterations_list[j] for j in indices]
            plt.plot(iterations_for_size, auc_for_size, marker='o', linestyle='-', color=colors[i], label=f'Size {unique_size}')

        # Add labels, title, and legend
        plt.xlabel('Iteration')
        plt.ylabel('AUC Score')
        plt.title('AUC Score Over Iterations by Size')
        plt.legend(title='Size')
        plt.grid(True)
        plt.show()

        # Print the size of the CSV file
        csv_file_size = os.path.getsize(csv_file_path) / 1024  # Convert to KB
        print(f"Size of the CSV file: {csv_file_size:.2f} KB")
    