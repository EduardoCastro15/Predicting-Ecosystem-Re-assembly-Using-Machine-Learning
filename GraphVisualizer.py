import networkx as nx
import matplotlib.pyplot as plt

class GraphVisualizer:
    """Class to handle the visualization of NetworkX graphs."""
    
    def __init__(self, graph):
        """
        Initialize the GraphVisualizer with a NetworkX graph.

        :param graph: NetworkX graph to visualize.
        """
        self.graph = graph
    
    def draw_graph(self, pos=None, with_labels=True, node_color='green', node_size=500, font_size=8, font_color='black',
                   edge_color='gray', width=1, style='dashed', alpha=0.9, linewidths=2, edge_cmap=plt.cm.Blues,
                   edge_vmin=0, edge_vmax=1):
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
        plt.show()
